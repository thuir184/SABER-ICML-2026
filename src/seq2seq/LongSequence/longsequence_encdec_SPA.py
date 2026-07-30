import torch
from torch import nn
import pandas as pd
import numpy as np
import time
import random

from datasets.formatting.formatting import NumpyArrowExtractor

from tqdm.auto import tqdm
import logging, os, argparse

import t5_dataset
from itertools import cycle
from copy import deepcopy
from torch.optim import AdamW
from transformers import T5Tokenizer, T5ForConditionalGeneration
from sklearn.metrics import matthews_corrcoef, f1_score


class ResMLP(torch.nn.Module):
    # Initialize MLP re-parameterization module
    def __init__(self, 
                 bottleneck_size,
                 module_type='MLP1',
                 emb_dimension=512,
                 residual=True,
                 ):
        super().__init__()
        if module_type=='MLP1':
            if layer_norm:
                self.module = nn.Sequential(
                    nn.Linear(emb_dimension, bottleneck_size),
                    nn.ReLU(),
                    nn.Linear(bottleneck_size, emb_dimension),
                    nn.LayerNorm(emb_dimension),
                )
            else:
                self.module = nn.Sequential(
                    nn.Linear(emb_dimension, bottleneck_size),
                    nn.Tanh(),
                    nn.Linear(bottleneck_size, emb_dimension),
                )

        elif module_type=='MLP2':
            self.module = nn.Sequential(
                nn.Linear(emb_dimension, bottleneck_size),
                nn.ReLU(),
                nn.Linear(bottleneck_size, bottleneck_size // 2),
                nn.Tanh(),
                nn.Linear(bottleneck_size // 2, emb_dimension),
            )

        elif module_type=='transformer':
            device = 'cuda'
            self.encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dimension, nhead=2, dropout=0.05).to(device)
            self.module = nn.TransformerEncoder(self.encoder_layer, num_layers=2).to(device)

        self.residual = residual
        if self.residual:
            print('Using skip connection in MLP')

    def forward(self, inputs):
        if self.residual:
            return self.module(inputs) + inputs
        else:
            return self.module(inputs)



class T5ContinualLearner:
    # Initialize continual learner and model (Big P)
    def __init__(self,
                 model_name,
                 cache_dir,
                 task_list,
                 batch_size=8,
                 select_k_per_class=-1,
                 prefix_len=0,
                 prefix_path=None,
                 global_prefix_len=None,
                 global_prefix_path=None,
                 freeze_weights=True,
                 freeze_except='shared',
                 lr=0.3,
                 weight_decay=1e-5,
                 seq_len=512,
                 early_stopping=True,
                 prefix_MLP='None',
                 bottleneck_size=800,
                 mlp_lr=None,
                 mlp_layer_norm=False,
                 weight_decay_mlp=None,
                 get_test_subset=True,
                 memory_perc=0.0,
                 pre_processed=False,
                 global_prompt_train_scale=0.5,
                 global_prompt_lr_scale=0.1,
                 seed=42,
                 ):
        
        self.glue_datasets = ['cola', 'sst2', 'mrpc', 'qqp', 'stsb', 'mnli', \
                              'mnli_mismatched', 'mnli_matched', 'qnli', 'rte', 'wnli', 'ax']
        self.superglue_datasets = ['copa', 'boolq', 'wic', 'wsc', 'wsc_bool', 'cb', 'record', 'multirc', 'rte_superglue']
        self.task_to_target_len = {
            'rte': 5,
            'mrpc': 5,
            'sst2': 2,
            'qqp': 5,
            'cola': 5,
            'qnli': 5,
            'mnli': 5,
            'stsb': 3,

            'wic': 2,
            'boolq': 2,
            'copa': 2,
            'wsc': 3,
            'wsc_bool': 2,
            'cb': 5,
            'multirc': 5,
            'record': 10,
            'rte_superglue': 5,

            'imdb': 2,

            'ag_news': 2,
            'yahoo_answers_topics': 5,
            'dbpedia_14': 5,
            'amazon': 2,
            'yelp_review_full': 2,
        }
        self.task_list = task_list
        self.seed = seed

        self.freeze_weights = freeze_weights
        self.lr = lr
        self.seq_len = seq_len
        self.batch_size = batch_size

        self.select_k_per_class = select_k_per_class
        self.pre_processed = pre_processed
        self.early_stopping = early_stopping

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        try:
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception as e:
            print('Deterministic setup warning:', e)

        self.model_name = model_name
        self.cache_dir = cache_dir
        self.model = T5ForConditionalGeneration.from_pretrained(model_name, cache_dir=cache_dir,local_files_only=False)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name, cache_dir=cache_dir,local_files_only=False)
        if freeze_weights:
            print('Freezing weights')
            self.do_freeze_weights(except_condition=freeze_except)
           
        self.prefix_len = prefix_len
        self.global_prefix_len = prefix_len if global_prefix_len is None else global_prefix_len
        print('self.global_prefix_len',self.global_prefix_len)
        self.global_prompt_train_scale = 0.5
        self.global_prompt_lr_scale = 0.1
        self.task_prompts = {}
        self.learned_tasks = []
        self.task_to_global_snapshot = {}
        self.warmup_prev_frac = 0.2
        self.prev_prompt_lr_scale = 0.1
        self.prev_update_active = False
        self.prev_prompt_param = None
        self.task_to_svd_basis = {}
        self.prev_svd_topk = 3
        self.prev_svd_batches = 10
        self.reverse_phase_active = False
        self.previous_prompts_param = None
        self._reverse_basis_list = []
        self.previous_prompts_updated = None
        self.selection_method = getattr(self, "selection_method", "proj_cos")
        self.cumulative_restrict_basis = {}


        if prefix_len>0:
            self.model.prompt = nn.Parameter(torch.tensor(self.init_new_prompt(prefix_len),
                                                          requires_grad=True))
            if prefix_path==None:
                self.previous_prompts = torch.zeros([0, self.model.prompt.shape[1]],
                                                    requires_grad=False).to(self.device)
            else:
                print('Using pre-trained progressive prompt - ' + prefix_path)
                self.previous_prompts = torch.tensor(np.load(prefix_path), requires_grad = False).to(self.device)
        
        # Creating a global shared prompt (optional)
        if self.global_prefix_len and self.global_prefix_len>0:
            if global_prefix_path is None:
                self.global_prompt = nn.Parameter(
                    torch.tensor(self.init_new_prompt(self.global_prefix_len), dtype=torch.float32).to(self.device)
                )
            else:
                print('Using pre-trained GLOBAL prompt - ' + global_prefix_path)
                self.global_prompt = nn.Parameter(
                    torch.tensor(np.load(global_prefix_path), dtype=torch.float32).to(self.device)
                )
        else:
            self.global_prompt = None

        # Model to cuda
        self.model.to(self.device) 
        # Create MLP (if prompt re-parameterization is requested)
        self.get_MLP(prefix_MLP, bottleneck_size) # adds prompt MLP reparametrization (and puts to cuda)

        self.lr = lr
        self.weight_decay = weight_decay
        self.mlp_lr = mlp_lr
        self.weight_decay_mlp = weight_decay_mlp
        self.optimizer = self.get_optimizer(lr, weight_decay,
                                            task=self.task_list[0],
                                            mlp_lr=mlp_lr,
                                            weight_decay_mlp=weight_decay_mlp)
        
        # Create best prompt/model copy for early stopping
        if self.early_stopping:
            if self.prefix_len>0:
                # prompt tuning
                self.best_prompt = self.model.prompt.detach().cpu().numpy()
            else:
                # model tuning
                self.best_model = deepcopy(self.model.state_dict()) # saving best model
            self.best_acc = 0.0 # best avg accuracy on seen tasks

        # Get task -> data dictionary for CL training
        self.get_test_subset = get_test_subset
        # print(memory_perc)
        self.tasks_data_dict = self.get_tasks_data_dict(memory_perc=memory_perc)


    # Create optimizer 
    def get_optimizer(self, lr, weight_decay,
                      task=None, mlp_lr=None, weight_decay_mlp=None): # task is used for MLP

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
                "lr": lr,
            },

            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
                "lr": lr,
            },
        ]

        if task!=None and self.prefix_MLPs!=None:
            if weight_decay_mlp==None:
                weight_decay_mlp = weight_decay
            if mlp_lr==None:
                mlp_lr = lr

            optimizer_grouped_parameters.append({
                "params": [p for n, p in self.prefix_MLPs[task].named_parameters()],# if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay_mlp,
                "lr": mlp_lr,
            })
        # Include global shared prompt params (if any)
        if getattr(self, 'global_prompt', None) is not None:
            optimizer_grouped_parameters.append({
                "params": [self.global_prompt],
                "weight_decay": weight_decay,
                "lr": lr * self.global_prompt_lr_scale,
            })

        optimizer = AdamW(optimizer_grouped_parameters, eps=1e-8)
        return optimizer

    
    # Create MLP for prompt tuning
    def get_MLP(self, prefix_MLP, bottleneck_size, layer_norm=False):
        if prefix_MLP == 'None':
            self.prefix_MLPs = None
        else:
            print('Using MLP reparametrization with bottleneck = ', bottleneck_size)
            N = self.model.encoder.embed_tokens.weight.shape[1]
            self.prefix_MLPs = {t: ResMLP(bottleneck_size=bottleneck_size,
                                          module_type=prefix_MLP,
                                          #layer_norm=layer_norm,
                                          emb_dimension=N) for t in self.task_list}
        if self.prefix_MLPs!=None:
            for t in self.task_list:
                self.prefix_MLPs[t].to(self.device)

    
    # Initialize new task prompt from random vocab. tokens
    def init_new_prompt(self, prompt_len):
        model = self.model
        N = model.encoder.embed_tokens.weight.shape[0]
        prompt_weigths = []

        for i in range(prompt_len):
            with torch.no_grad():
                j = np.random.randint(N) # random token
                w = deepcopy(model.encoder.embed_tokens.weight[j].detach().cpu().numpy())
                prompt_weigths.append(w)
        prompt_weigths = np.array(prompt_weigths)
        return prompt_weigths



    # Concatenate newly learned prompt to the joint "Progressive Prompts"
    def progress_previous_prompts(self, task=None):
        if self.early_stopping: # use best val acc prompt & MLP
            new_prompt = self.best_prompt # prompt has already passed MLP
        else: # use last prompt
            if task!=None and self.prefix_MLPs!=None:
                new_prompt = self.prefix_MLPs[task](self.model.prompt)
            else:
                new_prompt = self.model.prompt
            new_prompt = new_prompt.detach().cpu().numpy()

        new_prompt = torch.tensor(new_prompt, requires_grad = False).to(self.device)
        self.previous_prompts = torch.concat([new_prompt, self.previous_prompts], axis=0)
        self.task_prompts[task] = new_prompt
        if task is not None:
            self.learned_tasks.insert(0, task)
            # store global prompt snapshot used during this task's training (if exists)
            if getattr(self, 'global_prompt', None) is not None:
                self.task_to_global_snapshot[task] = self.global_prompt.detach().clone().to(self.device)
        print('task prompt',new_prompt)
        print('Updated progressive prompts ', self.previous_prompts.shape)
        # After finishing training this task, compute and store SVD basis (top-k) of its prompt gradients on its own data
        if task is not None and self.prefix_len > 0 and task in self.tasks_data_dict:
            V_k = self._compute_task_svd_basis(task, max_batches=self.prev_svd_batches, topk=self.prev_svd_topk)
            self.task_to_svd_basis[task] = V_k
            print('Stored SVD basis for task', task, 'shape', tuple(V_k.shape))

        # (Removed) mean grad storage


    # Update best prompt/model based on val. score
    def update_best_model(self, acc, task=None):
        if acc>self.best_acc:
            # getting best prompt
            if self.prefix_len>0:
                best_prompt = self.model.prompt
                if self.prefix_MLPs!=None:
                    self.prefix_MLPs[task].eval()
                    best_prompt = self.prefix_MLPs[task](best_prompt)

                self.best_prompt = best_prompt.detach().cpu().numpy()

            # getting best model
            else:
                self.best_model = deepcopy(self.model.state_dict()) # saving best model
            self.best_acc = acc # best avg accuracy on seen tasks


    # Restrieve best-performing model (for early stopping)
    def restore_best_model(self):
        if self.prefix_len>0:
            self.model.prompt = nn.Parameter(torch.tensor(self.best_prompt,
                                                          requires_grad=True))
            self.model.to(self.device)

            print("restored best prompt")
        else:
            self.model.load_state_dict(deepcopy(self.model.state_dict()))
            print("restored best model")

            
    # Create Dictionary of task_name -> dataloader (for CL experiments)
    def get_tasks_data_dict(self, memory_perc=0):
        tasks_data_dict = {}

        for task in self.task_list:
            tasks_data_dict[task] = {}
            print(task)
            data_params = {'task': task,
                           'batch_size': self.batch_size,
                           'max_length': self.seq_len,
                           'target_len': self.task_to_target_len[task],
                           'prefix_list': [],
                           }
            ds2 = t5_dataset.T5Dataset(self.tokenizer, task, self.cache_dir, self.pre_processed, seed=self.seed)
            if task not in ['mrpc', 'cola', 'copa', 'rte', 'rte_superglue', 'cb', 'wsc', 'wsc_bool']:
                k = self.select_k_per_class
                k_val = max(500, int(0.2*k)) if task!='sst2' else 400
            else:
                k = self.select_k_per_class if (self.select_k_per_class<=500 and task not in ['cb', 'copa', 'wsc', 'wsc_bool']) else -1
                k_val = -1
            time.sleep(2)
    
            if self.get_test_subset==False: k_val = -1
            
            dataloader_train = ds2.get_final_ds(**data_params, k=k, split='train')
            print('k = ', k, '  k-val = ',k_val)
            val_split = 'validation' if (task in self.glue_datasets) or (task in self.superglue_datasets) else 'test'
            dataloaders = ds2.get_final_ds(**data_params, k=k_val,
                                           split=val_split, return_test=self.get_test_subset)

            tasks_data_dict[task]['train'] = dataloader_train

            if memory_perc>0:
                k_mem = max(1, int(len(dataloader_train) * self.batch_size * memory_perc) )
                dataloader_mem = ds2.get_final_ds(**data_params, k=k_mem, split='train')
                tasks_data_dict[task]['train_mem'] = dataloader_mem

            if self.get_test_subset:
                dataloader_val, dataloader_test = dataloaders[0], dataloaders[1]
                tasks_data_dict[task]['val'] = dataloader_val
                tasks_data_dict[task]['test'] = dataloader_test
            else:
                tasks_data_dict[task]['val'] = dataloaders

            if task == 'multirc' and k_val==-1:
                self.multirc_idx = ds2.multirc_idx
            else: 
                self.multirc_idx = None
        return tasks_data_dict


    # Prev prompt grad-norm on current task
    def _compute_prev_slice_grad_norm_on_current(self, prompt_slice, dataloader, max_batches=1):
        model = self.model
        tokenizer = self.tokenizer
        model.eval()
        param_prompt = nn.Parameter(prompt_slice.clone().detach().to(self.device), requires_grad=True)
        batches_seen = 0
        for batch in dataloader:
            batch = {k: batch[k].to(self.device) for k in batch}
            lm_labels = batch["target_ids"].clone()
            lm_labels[lm_labels[:, :] == tokenizer.pad_token_id] = -100
            inputs_embeds = model.encoder.embed_tokens(batch["source_ids"])  
            k = inputs_embeds.shape[0]
            # Prefix ONLY with this previous prompt slice
            combined_inputs = torch.concat([param_prompt.repeat(k, 1, 1), inputs_embeds], dim=1)[:, :self.seq_len]
            full_prefix_len = param_prompt.shape[0]
            source_mask_updated = torch.concat((batch["source_mask"][0][0].repeat(k, full_prefix_len),
                                                batch["source_mask"]), dim=1)[:, :self.seq_len]
            encoder_outputs = model.encoder(attention_mask=source_mask_updated,
                                            inputs_embeds=combined_inputs,
                                            return_dict=None)
            outputs = model(input_ids=batch["source_ids"],
                            attention_mask=source_mask_updated,
                            labels=lm_labels,
                            decoder_attention_mask=batch['target_mask'],
                            encoder_outputs=encoder_outputs)
            loss = outputs[0]
            if param_prompt.grad is not None:
                param_prompt.grad.zero_()
            loss.backward()
            batches_seen += 1
            if batches_seen >= max_batches:
                break
        if param_prompt.grad is None:
            return 0.0
        return float(param_prompt.grad.detach().norm().cpu().item())

    # Prev prompt grad-direction on current task
    def _compute_prev_slice_grad_dir_on_current(self, prompt_slice, dataloader, max_batches=1):
        model = self.model
        tokenizer = self.tokenizer
        model.eval()
        param_prompt = nn.Parameter(prompt_slice.clone().detach().to(self.device), requires_grad=True)
        batches_seen = 0
        g_accum = None
        for batch in dataloader:
            batch = {k: batch[k].to(self.device) for k in batch}
            lm_labels = batch["target_ids"].clone()
            lm_labels[lm_labels[:, :] == tokenizer.pad_token_id] = -100
            inputs_embeds = model.encoder.embed_tokens(batch["source_ids"])
            k = inputs_embeds.shape[0]
            combined_inputs = torch.concat([param_prompt.repeat(k, 1, 1), inputs_embeds], dim=1)[:, :self.seq_len]
            full_prefix_len = param_prompt.shape[0]
            source_mask_updated = torch.concat((batch["source_mask"][0][0].repeat(k, full_prefix_len),
                                                batch["source_mask"]), dim=1)[:, :self.seq_len]
            encoder_outputs = model.encoder(attention_mask=source_mask_updated, inputs_embeds=combined_inputs, return_dict=None)
            outputs = model(input_ids=batch["source_ids"], attention_mask=source_mask_updated,
                            labels=lm_labels, decoder_attention_mask=batch['target_mask'], encoder_outputs=encoder_outputs)
            loss = outputs[0]
            if param_prompt.grad is not None:
                param_prompt.grad.zero_()
            loss.backward()
            g = param_prompt.grad.detach().reshape(-1)
            g_accum = g.clone() if g_accum is None else (g_accum + g)
            batches_seen += 1
            if batches_seen >= max_batches:
                break
        if g_accum is None:
            return None
        g_norm = torch.norm(g_accum)
        if g_norm.item() == 0:
            return None
        return (g_accum / g_norm).detach().cpu()

    # Loss distribution utilities
    def _compute_loss_distribution(self, dataloader, task, prompt=None, max_batches=200):
        model = self.model
        tokenizer = self.tokenizer
        device = self.device
        model.eval()
        losses = []
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= max_batches:
                    break
                b = {k: batch[k].to(device) for k in batch}
                inputs_embeds = model.encoder.embed_tokens(b["source_ids"]).to(device)
                if prompt is not None:
                    k = inputs_embeds.shape[0]
                    inputs_embeds = torch.cat([prompt.repeat(k, 1, 1),
                                               inputs_embeds], dim=1)[:, :self.seq_len]
                    full_prefix_len = prompt.shape[0]
                    source_mask_updated = torch.cat((b["source_mask"][0][0].repeat(k, full_prefix_len),
                                                     b["source_mask"]), dim=1)[:, :self.seq_len]
                else:
                    source_mask_updated = b["source_mask"]
                lm_labels = b["target_ids"]
                lm_labels[lm_labels[:, :] == tokenizer.pad_token_id] = -100
                encoder_outputs = model.encoder(attention_mask=source_mask_updated,
                                                inputs_embeds=inputs_embeds,
                                                return_dict=None)
                outputs = model(input_ids=b["source_ids"],
                                attention_mask=source_mask_updated,
                                labels=lm_labels,
                                decoder_attention_mask=b['target_mask'],
                                encoder_outputs=encoder_outputs)
                loss = outputs[0]
                losses.append(float(loss.detach().cpu().item()))
        return losses

    def _compute_loss_based_similarity(self, current_task, print_results=True, max_batches=200):
        # Returns: prev_task -> {'dis_prime': x, 'dis': y, 'similar': bool}
        def wdist(a, b):
            try:
                from scipy.stats import wasserstein_distance
                return float(wasserstein_distance(a, b))
            except Exception:
                a_sorted = np.sort(np.array(a))
                b_sorted = np.sort(np.array(b))
                n = max(len(a_sorted), len(b_sorted))
                if n == 0:
                    return 0.0
                a_interp = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(a_sorted)), a_sorted)
                b_interp = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(b_sorted)), b_sorted)
                return float(np.mean(np.abs(np.cumsum(a_interp) - np.cumsum(b_interp))) / n)

        sim = {}
        tasks_dict = self.tasks_data_dict
        if current_task not in tasks_dict:
            return sim
        # base losses for current (no prompt)
        L_t_base = self._compute_loss_distribution(tasks_dict[current_task]['train'], current_task, prompt=None, max_batches=max_batches)
        try:
            curr_idx = self.task_list.index(current_task)
        except ValueError:
            curr_idx = len(self.task_list)
        prev_tasks = [t for t in self.task_list[:curr_idx] if t in self.task_prompts]
        for prev_task in prev_tasks:
            L_i_base = self._compute_loss_distribution(tasks_dict[prev_task]['train'], prev_task, prompt=None, max_batches=max_batches)
            prompt_prev = self.task_prompts[prev_task].to(self.device)
            L_i_self = self._compute_loss_distribution(tasks_dict[prev_task]['train'], prev_task, prompt=prompt_prev, max_batches=max_batches)
            L_t_with_i = self._compute_loss_distribution(tasks_dict[current_task]['train'], current_task, prompt=prompt_prev, max_batches=max_batches)
            dis_prime = wdist(L_i_base, L_t_base)
            dis = wdist(L_i_self, L_t_with_i)
            is_similar = dis < dis_prime
            sim[prev_task] = {'dis_prime': dis_prime, 'dis': dis, 'similar': is_similar}
            if print_results:
                try:
                    print(f"[LossSim] current={current_task} prev={prev_task} dis_self_vs_cross={dis:.6f} dis_base_vs_base={dis_prime:.6f} similar={is_similar}")
                except Exception:
                    pass
        return sim

    # Train step (prompt tuning)
    def train_step_lester(self,
                          batch,
                          task=None,
                          progressive=True):
        prefix_len = self.prefix_len
        model = self.model
        embed_prompt = self.prefix_MLPs!=None
        if embed_prompt:
            assert task!=None
            mlp = self.prefix_MLPs[task]
        tokenizer = self.tokenizer

        batch = {k: batch[k].to(self.device) for k in batch}
        lm_labels = batch["target_ids"]
        lm_labels[lm_labels[:, :] == tokenizer.pad_token_id] = -100

        inputs_embeds = model.encoder.embed_tokens(batch["source_ids"])

        k = inputs_embeds.shape[0]
        if self.reverse_phase_active and self.previous_prompts_param is not None:
            with torch.no_grad():
                prompt = model.prompt.detach()
        else:
            if embed_prompt:
                prompt = mlp(model.prompt)
            else:
                prompt = model.prompt

        # Forward transfer: use GLOBAL prompt + current TASK prompt; optionally include selected previous prompt
        prefix_segments = []
        if (not self.reverse_phase_active) and progressive and getattr(self, 'global_prompt', None) is not None:
            prefix_segments.append(self.global_prompt * self.global_prompt_train_scale)
        # Always include current task prompt during training
        prefix_segments.append(prompt)
        # In reverse phase, include the trainable previous prompts block
        if self.reverse_phase_active and self.previous_prompts_param is not None:
            prefix_segments.append(self.previous_prompts_param)

        if len(prefix_segments)>0:
            combined_prefix = torch.concat(prefix_segments, dim=0)
            inputs_embeds = torch.concat([combined_prefix.repeat(k, 1, 1),
                                          inputs_embeds], axis=1)[:,:self.seq_len]
            full_prefix_len = combined_prefix.shape[0]
        else:
            full_prefix_len = 0

        source_mask_updated = torch.concat( (batch["source_mask"][0][0].repeat(k,full_prefix_len),
                                             batch["source_mask"]), axis=1)[:,:self.seq_len]

        encoder_outputs = model.encoder(
                                attention_mask=source_mask_updated,
                                inputs_embeds=inputs_embeds,
                                head_mask=None,  
                                output_attentions=None,  
                                output_hidden_states=None, 
                                return_dict=None,  
                            )

        outputs = model(
            input_ids=batch["source_ids"],
            attention_mask=source_mask_updated, 
            labels=lm_labels,
            decoder_attention_mask=batch['target_mask'],
            encoder_outputs=encoder_outputs,
        )
        loss = outputs[0]

        return loss




    # Train step (full model)
    def train_step(self, batch):
        model = self.model
        tokenizer = self.tokenizer

        batch = {k: batch[k].to(self.device) for k in batch}
        lm_labels = batch["target_ids"]
        lm_labels[lm_labels[:, :] == tokenizer.pad_token_id] = -100

        inputs_embeds = model.encoder.embed_tokens(batch["source_ids"])
        encoder_outputs = model.encoder(
                                attention_mask=batch["source_mask"],
                                inputs_embeds=inputs_embeds,
                                head_mask=None,
                                output_attentions=None,
                                output_hidden_states=None,
                                return_dict=None, 
                            )

        outputs = model(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            labels=lm_labels,
            decoder_attention_mask=batch['target_mask'],
            encoder_outputs=encoder_outputs,
        )
        loss = outputs[0]

        return loss



    # Normalize and clean text
    def normalize_text(self, s):
        import string, re

        def remove_articles(text):
            regex = re.compile(r"\b(a|an|the|)\b", re.UNICODE)
            return re.sub(regex, " ", text)

        def white_space_fix(text):
            return " ".join(text.split())

        def remove_punc(text):
            text2 = text.replace('<pad>', '').replace('</s>', '')
            exclude = set(string.punctuation)
            return "".join(ch for ch in text2 if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))
    



    # Compute EM score used for some SuperGLUE tasks
    def compute_exact_match(self, prediction, truth):
        return int(self.normalize_text(prediction) == self.normalize_text(truth))


    # Compute F1 score used for some GLUE & SuperGLUE tasks
    def compute_f1(self, prediction, truth):
        pred_tokens = self.normalize_text(prediction).split()
        truth_tokens = self.normalize_text(truth).split()

        # if either the prediction or the truth is no-answer then f1 = 1 if they agree, 0 otherwise
        if len(pred_tokens) == 0 or len(truth_tokens) == 0:
            return int(pred_tokens == truth_tokens)

        common_tokens = set(pred_tokens) & set(truth_tokens)

        # if there are no common tokens then f1 = 0
        if len(common_tokens) == 0:
            return 0

        prec = len(common_tokens) / len(pred_tokens)
        rec = len(common_tokens) / len(truth_tokens)

        return 2 * (prec * rec) / (prec + rec)


    # # Compute task metrics on a validation (test) set
    def validate(self,
                 dataloader_val,
                 task,
                 prompt=None,
                 target_len=2,
                 print_outputs=False,
                 use_global_prompt=True,
                ):
        model = self.model
        max_length = target_len
        tokenizer = self.tokenizer
        model.eval()

        corr, total, f1 = 0, 0, 0
        y_true, y_pred = [], []

        # Optional debug
        if not use_global_prompt:
            print(f"Test-eval | task={task} | using only task-specific prompt (no Big P)")

        for i, batch in enumerate(tqdm(dataloader_val)):
            batch = {k:batch[k].to(self.device) for k in batch}
            inputs_embeds = model.encoder.embed_tokens(batch["source_ids"]).to(self.device)

            # Build prefix
            prefix_segments = []
            if use_global_prompt and getattr(self, 'global_prompt', None) is not None:
                prefix_segments.append(self.global_prompt)

            # Determine task-specific prompt to use
            if prompt is not None:
                task_prompt_eval = prompt
            elif task in self.task_prompts:
                task_prompt_eval = self.task_prompts[task]
            elif self.prefix_len>0:
                task_prompt_eval = self.model.prompt
            else:
                task_prompt_eval = None

            if task_prompt_eval is not None:
                prefix_segments.append(task_prompt_eval)

            if len(prefix_segments)>0:
                k = inputs_embeds.shape[0]
                combined_prefix = torch.concat(prefix_segments, dim=0)
                inputs_embeds = torch.concat([combined_prefix.repeat(k, 1, 1),
                                              inputs_embeds], axis=1)[:,:self.seq_len]
                full_prefix_len = combined_prefix.shape[0]
                source_mask_updated = torch.concat( (batch["source_mask"][0][0].repeat(k,full_prefix_len),
                                                     batch["source_mask"]), axis=1)[:,:self.seq_len]
            else:
                # No prompt case
                source_mask_updated = batch["source_mask"]


            encoder_outputs = model.encoder(
                                    attention_mask=source_mask_updated,
                                    inputs_embeds=inputs_embeds,
                                    head_mask=None,  
                                    output_attentions=None, 
                                    output_hidden_states=None,  
                                    return_dict=None, 
                                )

            outs = model.generate(
                input_ids=batch["source_ids"],
                attention_mask=source_mask_updated,
                encoder_outputs=encoder_outputs,
                max_length=max_length,
            )
            dec = [tokenizer.decode(ids) for ids in outs]
            texts = [tokenizer.decode(ids) for ids in batch['source_ids']]
            targets = [tokenizer.decode(ids) for ids in batch['target_ids']]

            if task in ['stsb', 'cola', 'cb', 'multirc']:
                row_true = [self.normalize_text(x) for x in targets]
                row_pred = [self.normalize_text(x) for x in dec]
                if task=='stsb':
                    row_true = [float(x) if any(c.isalpha() for c in x)==False else 0.0 for x in row_true] # convert digits to float, convert letters to 0
                    row_pred = [float(x) if any(c.isalpha() for c in x)==False else 0.0 for x in row_pred]
                y_true += row_true
                y_pred += row_pred

            elif task=='record':
                # multiple answers
                for x,y in zip(dec, targets):
                    corr += max([self.compute_exact_match(x, yi) for yi in y.split(';')])
                    f1 += max([self.compute_f1(x, yi) for yi in y.split(';')])
                total += batch['source_ids'].shape[0]

            else:
                corr += np.sum([self.normalize_text(x)==self.normalize_text(y) for x,y in zip(dec, targets)])
                total += batch['source_ids'].shape[0]

            
        if task=='cola':
            return matthews_corrcoef(y_true, y_pred)

        elif task=='stsb':
            return np.corrcoef(y_true, y_pred)[0,1]

        elif task=='cb':
            return np.mean(np.array(y_true) == np.array(y_pred)), f1_score(y_true, y_pred, average='macro')

        elif task=='multirc':
            if self.multirc_idx!=None:
                em = []
                for idx in set(self.multirc_idx):
                    k = np.where(self.multirc_idx==idx)[0]
                    score = (np.array(y_true)[k] == np.array(y_pred)[k]).all()
                    em.append(score)
                return np.mean(em), f1_score(y_true, y_pred, average='micro')
            else:
                return f1_score(y_true, y_pred, average='micro')

        elif task=='record':
            return corr/total, f1/total
        return corr/total

    # Compute top-k SVD basis for gradients of a task's prompt on its own data (no Big P)
    def _compute_task_svd_basis(self, task, max_batches=10, topk=3):
        dataloader = self.tasks_data_dict[task]['val'] if 'val' in self.tasks_data_dict[task] else self.tasks_data_dict[task]['train']
        model = self.model
        tokenizer = self.tokenizer
        model.eval()
        prompt_tensor = self.task_prompts[task].detach().to(self.device)
        rows = []
        seen = 0
        for batch in dataloader:
            batch = {k: batch[k].to(self.device) for k in batch}
            lm_labels = batch['target_ids'].clone()
            lm_labels[lm_labels[:, :] == tokenizer.pad_token_id] = -100
            k = batch['source_ids'].shape[0]
            # fresh leaf param for grad
            prompt_param = nn.Parameter(prompt_tensor.clone(), requires_grad=True)
            inputs_embeds = model.encoder.embed_tokens(batch['source_ids'])
            combined_inputs = torch.concat([prompt_param.repeat(k, 1, 1), inputs_embeds], dim=1)[:, :self.seq_len]
            source_mask_updated = torch.concat((batch['source_mask'][0][0].repeat(k, prompt_tensor.shape[0]), batch['source_mask']), dim=1)[:, :self.seq_len]
            encoder_outputs = model.encoder(attention_mask=source_mask_updated, inputs_embeds=combined_inputs, return_dict=None)
            outputs = model(input_ids=batch['source_ids'], attention_mask=source_mask_updated,
                            labels=lm_labels, decoder_attention_mask=batch['target_mask'], encoder_outputs=encoder_outputs)
            loss = outputs[0]
            loss.backward()
            g = prompt_param.grad.detach().reshape(-1)
            rows.append(g)
            seen += 1
            if seen >= max_batches:
                break
        G = torch.stack(rows, dim=0)  # M x D
        # Thin SVD
        U, S, Vh = torch.linalg.svd(G, full_matrices=False)
        k = max(1, int(topk))
        V_k = Vh[:k, :].T  # D x k
        return V_k.detach()

    # (Removed) mean gradient computation helper




    # Freeze model weights
    def do_freeze_weights(self, except_condition='shared'):
        model = self.model
        for name, param in model.named_parameters():
            if param.requires_grad == True and except_condition not in name:
                param.requires_grad = False


    # Freeze / unfreeze MLPs for given tasks (when requires_grad==False then freezing)
    def freeze_unfreeze_mlps(self, tasks, requires_grad=False):
        assert self.prefix_MLPs != None

        for t in tasks:
            #for name, param in self.prefix_MLPs[t].named_parameters():
            for name, param in self.prefix_MLPs[t].named_parameters():
                if param.requires_grad != requires_grad:
                    param.requires_grad = requires_grad
                    param.grad = None # remove old gradient


    # Create replay buffers for data replay in CL
    def create_memory_replay_generators(self, task, split='train_mem'): # creating previous tasks memory buffers
        print('Creating generators for previous tasks ...')
        tasks_to_generators = {}
        curr_task_num = self.task_list.index(task)
        for idx in np.arange(curr_task_num):
            prev_task = self.task_list[idx]
            print(prev_task)
            tasks_to_generators[prev_task] = iter(self.tasks_data_dict[prev_task][split])
        return tasks_to_generators


    # Perfor memory replay from past tasks
    def memory_replay(self, tasks_to_generators, progressive):
        print("Rehearsal on " + str((', ').join(list(tasks_to_generators)) ))
        for prev_task in tasks_to_generators:
            generator_mem1 = tasks_to_generators[prev_task]
            try:
                b = next(generator_mem1)
            except StopIteration:
                # restart the generator if the previous generator is exhausted.
                generator_mem1 = iter(self.tasks_data_dict[prev_task]['train_mem'])
                tasks_to_generators[prev_task] = generator_mem1
                b = next(generator_mem1)

            b = {k: v.to(self.device) for k, v in b.items()}
            if self.prefix_len>0: # prompt tuning
                loss = self.train_step_lester(b,
                                              task=prev_task if self.prefix_MLPs!=None else None,
                                              progressive=progressive)
            else:
                loss = self.train_step(b)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
    
    def train_one_task(self,
                   task,
                   epochs=40,
                   progressive=True,
                   eval_every_N=1,
                   eval_on_all_tasks=False,
                   data_replay_freq=-1,
                   save_path=None):

        print('task = ', task)
        print('progressive', progressive)
        if progressive:
            assert self.prefix_len > 0  # can only do progressive prompts when prompt tuning
            print('progressive prompts')
        if self.early_stopping:
            self.best_acc = 0.0  # re-setting best acc

        if self.prefix_MLPs != None:
            print('Freezing all MLPs except for ', task)
            mlp = self.prefix_MLPs[task]
            self.freeze_unfreeze_mlps([x for x in self.task_list if x != task], requires_grad=False)
            self.freeze_unfreeze_mlps([task], requires_grad=True)  # unfreezing current task

        model = self.model

        with torch.no_grad():
            model.prompt = nn.Parameter(torch.tensor(self.init_new_prompt(self.prefix_len),
                                                    requires_grad=True))
            self.optimizer = self.get_optimizer(self.lr, self.weight_decay,
                                                task=task)
        model.to(self.device)
        target_len = self.task_to_target_len[task]
        dataloader_train = self.tasks_data_dict[task]['train']
        dataloader_val = self.tasks_data_dict[task]['val']

        # Two-phase training (parity with decoder/continual): 10 normal + 2 reverse epochs
        # Normal: update Big P (scaled lr) + current task prompt; Reverse: freeze both; update selected previous prompts orthogonally
        normal_phase_epochs = 10
        reverse_phase_epochs = 2
        has_prev_prompts = self.previous_prompts is not None and self.previous_prompts.shape[0] > 0
        effective_epochs = epochs if has_prev_prompts else min(epochs, normal_phase_epochs)

        val_acc = []

        # Print gradient norms and direction similarity (own-task SVD v1 vs current-task grad dir) for previous prompts
        try:
            if has_prev_prompts and self.prefix_len > 0:
                num_prev = int(self.previous_prompts.shape[0] // self.prefix_len)
                for idx_prev in range(num_prev):
                    s = idx_prev * self.prefix_len
                    e = s + self.prefix_len
                    prompt_slice = self.previous_prompts[s:e, :].detach()
                    gn = self._compute_prev_slice_grad_norm_on_current(prompt_slice, dataloader_train, max_batches=1)
                    print(f"[PrevGradNorm] current={task} prev_idx={idx_prev} norm={gn:.6f}")
                    # direction similarity if we have own-task basis and learned task name
                    if idx_prev < len(self.learned_tasks):
                        prev_task_name = self.learned_tasks[idx_prev]
                        V = self.task_to_svd_basis.get(prev_task_name, None)
                        g_dir = self._compute_prev_slice_grad_dir_on_current(prompt_slice, dataloader_train, max_batches=1)
                        if V is not None and g_dir is not None and V.shape[0] == g_dir.shape[0]:
                            v1 = V[:, 0].detach().cpu()
                            # ensure unit
                            v1 = v1 / (torch.norm(v1) + 1e-12)
                            cos_sim = float(torch.clamp(torch.dot(v1, g_dir), -1.0, 1.0).item())
                            angle_deg = float(np.degrees(np.arccos(cos_sim)))
                            print(f"[DirAlign] prev={prev_task_name} vs current={task} -> cosine={cos_sim:.4f}, angle={angle_deg:.2f} deg")
        except Exception as e:
            print("Warning: prev prompt diagnostics failed:", e)

        for epoch in range(effective_epochs):
            print(epoch)
            model.train()
            if self.prefix_MLPs != None:
                mlp.train()

            # Capture previous prompt state at start of epoch for per-epoch logging
            if getattr(self, 'prev_update_active', False) and getattr(self, 'prev_prompt_param', None) is not None:
                with torch.no_grad():
                    epoch_prev_vec_before = self.prev_prompt_param.detach().reshape(-1).cpu()
                    epoch_prev_norm_before = float(self.prev_prompt_param.detach().norm().cpu().item())
            else:
                epoch_prev_vec_before = None
                epoch_prev_norm_before = None


            if data_replay_freq != -1:
                tasks_to_generators = self.create_memory_replay_generators(task, split='train_mem')

            # Phase toggles
            in_normal_phase = epoch < normal_phase_epochs or epoch >= (normal_phase_epochs + reverse_phase_epochs)
            in_reverse_phase = has_prev_prompts and (epoch >= normal_phase_epochs) and (epoch < (normal_phase_epochs + reverse_phase_epochs))

            if epoch == normal_phase_epochs and has_prev_prompts:
                # enter reverse: freeze current prompt (and keep Big P frozen by excluding it), make SELECTED previous prompts trainable
                self.reverse_phase_active = True
                if hasattr(model, 'prompt'):
                    model.prompt.requires_grad = False
                # Build newest-first previous tasks order
                try:
                    curr_idx = self.task_list.index(task)
                except ValueError:
                    curr_idx = len(self.task_list)
                prev_tasks_order_full = [t for t in reversed(self.task_list[:curr_idx]) if t in self.task_prompts]
                # Compute per-prev-task current-task grad dir and cosine vs own-task SVD v1
                proj_scores = {}
                cos_scores = {}
                if has_prev_prompts and self.prefix_len > 0:
                    num_prev = int(self.previous_prompts.shape[0] // self.prefix_len)
                    for idx_prev, t_prev in enumerate(prev_tasks_order_full):
                        if idx_prev >= num_prev:
                            break
                        s = idx_prev * self.prefix_len
                        e = s + self.prefix_len
                        prompt_slice = self.previous_prompts[s:e, :].detach()
                        g_dir = self._compute_prev_slice_grad_dir_on_current(prompt_slice, dataloader_train, max_batches=10)
                        V = self.task_to_svd_basis.get(t_prev, None)
                        if g_dir is not None and V is not None and V.shape[0] == g_dir.shape[0] and V.shape[1] > 0:
                            # projection norm onto span(V)
                            V_t = V.to(self.device)
                            coeff = V_t.t() @ g_dir.to(self.device)
                            proj_scores[t_prev] = float(torch.norm(coeff).item())
                            # cosine with top singular vector
                            v1 = V_t[:, 0]
                            v1 = v1 / (torch.norm(v1) + 1e-12)
                            cos_scores[t_prev] = float(torch.clamp(torch.dot(v1, g_dir.to(self.device)), -1.0, 1.0).item())
                        else:
                            proj_scores[t_prev] = 0.0
                            cos_scores[t_prev] = -1.0
                        try:
                            print(f"[BigPProjScore] current={task} prev={t_prev} proj_norm={proj_scores[t_prev]:.6f} cos_v1={cos_scores[t_prev]:.6f}")
                        except Exception:
                            pass
                # Optional: Wasserstein similarity
                loss_sim = {}
                if getattr(self, "selection_method", "proj_cos") == "wasserstein":
                    try:
                        loss_sim = self._compute_loss_based_similarity(task, print_results=True)
                    except Exception as e_ls:
                        print("Warning: BigP Wasserstein similarity failed:", e_ls)
                # Select according to method
                selected_set = set()
                if getattr(self, "selection_method", "proj_cos") == "wasserstein":
                    TH = 0.2
                    for t_prev in prev_tasks_order_full:
                        s = loss_sim.get(t_prev, None)
                        if s is None:
                            continue
                        score = float(s.get('dis_prime', 0.0)) - float(s.get('dis', 0.0))
                        if s.get('similar', False) and score > TH:
                            selected_set.add(t_prev)
                    if len(selected_set) == 0:
                        selected_set = set(prev_tasks_order_full)
                    try:
                        print(f"[BigPSelectByWass] current={task} selected_prev={list(selected_set)} thresh=0.2")
                    except Exception:
                        pass
                else:
                    # proj_cos thresholds
                    for t_prev in prev_tasks_order_full:
                        if cos_scores.get(t_prev, -1.0) > 0.0 and proj_scores.get(t_prev, 0.0) > 0.1:
                            selected_set.add(t_prev)
                    if len(selected_set) == 0:
                        selected_set = set(prev_tasks_order_full)
                    try:
                        print(f"[BigPSelectByProjCos] current={task} selected_prev={list(selected_set)}")
                    except Exception:
                        pass
                # Slice selected previous prompts into a trainable block
                orig_chunks = torch.split(self.previous_prompts.detach().to(self.device), self.prefix_len, dim=0)
                selected_indices = [idx for idx, t_prev in enumerate(prev_tasks_order_full) if t_prev in selected_set]
                sel_chunks = [orig_chunks[idx] for idx in selected_indices] if len(selected_indices) > 0 else []
                if len(sel_chunks) == 0:
                    selected_indices = list(range(len(orig_chunks)))
                    sel_chunks = list(orig_chunks)
                sel_block = torch.cat(sel_chunks, dim=0).to(self.device)
                self.previous_prompts_param = nn.Parameter(sel_block.clone(), requires_grad=True)
                self.optimizer.add_param_group({
                    "params": [self.previous_prompts_param],
                    "weight_decay": self.weight_decay,
                    "lr": self.lr,
                })
                self._reverse_selected_order = [prev_tasks_order_full[i] for i in selected_indices]
                self._reverse_selected_indices = selected_indices
                # Build per-task bases (newest-first) for SELECTED set, including cumulative restricted basis
                prev_tasks_order = self._reverse_selected_order
                basis_list = []
                emb_dim = self.model.encoder.embed_tokens.weight.shape[1]
                D = self.prefix_len * emb_dim
                for t_prev in prev_tasks_order:
                    parts = []
                    V_main = self.task_to_svd_basis.get(t_prev, None)
                    if V_main is not None:
                        parts.append(V_main.to(self.device))
                    V_cum_np = self.cumulative_restrict_basis.get(t_prev, None)
                    if V_cum_np is not None and getattr(V_cum_np, "size", 0) > 0:
                        parts.append(torch.tensor(V_cum_np, dtype=self.previous_prompts_param.dtype, device=self.device))
                    if len(parts) == 0:
                        # fallback: use first singular vector if available, else empty basis
                        if V_main is not None and V_main.shape[1] > 0:
                            v1 = V_main[:, 0].to(self.device).unsqueeze(1)
                            parts.append(v1 / (torch.norm(v1) + 1e-12))
                        else:
                            parts.append(torch.zeros((D, 0), device=self.device, dtype=self.previous_prompts_param.dtype))
                    Vcat = torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]
                    try:
                        Q, _ = torch.linalg.qr(Vcat, mode="reduced")
                        if Q.shape[1] > 6:
                            Q = Q[:, :6]
                        basis_list.append(Q)
                    except Exception:
                        basis_list.append(Vcat)
                self._reverse_basis_list = basis_list

            if epoch == (normal_phase_epochs + reverse_phase_epochs) and self.reverse_phase_active:
                # exit reverse: store updated copy; unfreeze current prompt
                self.reverse_phase_active = False
                if self.previous_prompts_param is not None:
                    # Merge updated selected blocks back into full previous prompts and update cumulative restricted basis
                    try:
                        full_chunks = list(torch.split(self.previous_prompts.detach().to(self.device), self.prefix_len, dim=0))
                        upd_chunks = list(torch.split(self.previous_prompts_param.detach().to(self.device), self.prefix_len, dim=0))
                        selected_indices = getattr(self, '_reverse_selected_indices', list(range(len(full_chunks))))
                        # update cumulative restriction with net deltas
                        try:
                            for j, idx_full in enumerate(selected_indices):
                                if j < len(upd_chunks) and idx_full < len(full_chunks):
                                    t_prev = self._reverse_selected_order[j] if j < len(self._reverse_selected_order) else None
                                    if t_prev is None:
                                        continue
                                    delta = (upd_chunks[j] - full_chunks[idx_full]).detach()
                                    dflat = delta.reshape(-1)
                                    nrm = torch.norm(dflat)
                                    if nrm is not None and float(nrm.item()) > 0.0:
                                        v = (dflat / nrm).to(torch.float32).cpu().numpy().reshape(-1, 1)
                                        exist = self.cumulative_restrict_basis.get(t_prev, None)
                                        if exist is None or getattr(exist, "size", 0) == 0:
                                            Vnew = v
                                        else:
                                            Vnew = np.concatenate([exist, v], axis=1)
                                            Vtorch = torch.tensor(Vnew, dtype=torch.float32)
                                            try:
                                                Q, _ = torch.linalg.qr(Vtorch, mode="reduced")
                                                Vnew = Q.cpu().numpy()
                                            except Exception:
                                                Vnew = Vtorch.cpu().numpy()
                                        if Vnew.shape[1] > 6:
                                            Vnew = Vnew[:, :6]
                                        self.cumulative_restrict_basis[t_prev] = Vnew
                                        try:
                                            print(f"[BigPCumRestrict] prev_task={t_prev} added=1 total={Vnew.shape[1]}")
                                        except Exception:
                                            pass
                        except Exception as e_cum:
                            print("Warning: BigP cumulative restrict update failed:", e_cum)
                        # merge updated
                        for j, idx_full in enumerate(selected_indices):
                            if j < len(upd_chunks) and idx_full < len(full_chunks):
                                full_chunks[idx_full] = upd_chunks[j]
                        self.previous_prompts_updated = torch.cat(full_chunks, dim=0).to(self.device)
                        self.previous_prompts_updated.requires_grad = False
                    except Exception:
                        self.previous_prompts_updated = self.previous_prompts_param.detach().to(self.device)
                        self.previous_prompts_updated.requires_grad = False
                if hasattr(model, 'prompt'):
                    model.prompt.requires_grad = True
                self.previous_prompts_param = None
                self._reverse_basis_list = []

            for i, batch in enumerate(tqdm(dataloader_train)):
                batch = {k: batch[k].to('cuda') for k in batch}

                if self.prefix_len>0: # prompt tuning
                    loss = self.train_step_lester(batch,
                                                  task=task if self.prefix_MLPs!=None else None,
                                                  progressive=progressive)
                else:
                    loss = self.train_step(batch)

                loss.backward()
                # OGD-style projection for all previous prompts (per-chunk) during reverse phase
                if self.reverse_phase_active and self.previous_prompts_param is not None and self.previous_prompts_param.grad is not None and len(self._reverse_basis_list)>0:
                    with torch.no_grad():
                        g = self.previous_prompts_param.grad  # (sum_len, emb_dim)
                        chunks_g = torch.split(g, self.prefix_len, dim=0)
                        for idx_chunk, gi in enumerate(chunks_g):
                            if idx_chunk >= len(self._reverse_basis_list):
                                break
                            V = self._reverse_basis_list[idx_chunk]  # (D, k)
                            if V is None or V.numel()==0:
                                continue
                            gi_flat = gi.reshape(-1)
                            proj = V @ (V.transpose(0,1) @ gi_flat)
                            gi_flat -= proj
                            gi.copy_(gi_flat.view_as(gi))
                
                
                self.optimizer.step()
                self.optimizer.zero_grad()

                # performing data replay on all previous tasks
                if data_replay_freq != -1 and i % data_replay_freq == 0:
                    self.memory_replay(tasks_to_generators, progressive)

            # evaluate accuracy after each epoch
            if self.prefix_MLPs != None:
                mlp.eval()
                prompt = mlp(model.prompt)
            else:
                if self.prefix_len > 0:
                    prompt = model.prompt
                    print(prompt.shape)
                else:
                    prompt = None
            if progressive:
                # In evaluation, we always use [GLOBAL + TASK PROMPT] if available. No concatenation with previous prompts.
                pass

            if epoch % eval_every_N == 0:
                overall_acc = []
                if eval_on_all_tasks:
                    # eval current model/prompt on all tasks (for approaches that suffer from catastrophic forgetting)
                    for eval_task in self.task_list:
                        acc = self.validate(self.tasks_data_dict[eval_task]['val'],
                                            eval_task,
                                            prompt=prompt, target_len=self.task_to_target_len[eval_task],
                                            print_outputs=False)
                        overall_acc.append(np.mean(acc))
                        if eval_task == task:  # record val accuracy for the current task
                            val_acc.append(np.mean(acc))
                    acc = np.mean(overall_acc)
                else:
                    acc = self.validate(dataloader_val, task,
                                        prompt=prompt, target_len=target_len, print_outputs=True)
                    if task in ['record', 'cb'] or (task == 'multirc' and self.multirc_idx != None):
                        acc = np.mean(acc)  # averaging 2 scores
                    val_acc.append(acc)

                if self.early_stopping:
                    self.update_best_model(acc, task=task)
                print(epoch, task, '->', val_acc[-1])
                self._save_partial_state(save_path, task, epoch, val_acc)

            # During reverse phase, evaluate previous tasks after each epoch:
            if self.reverse_phase_active and self.previous_prompts_param is not None:
                try:
                    # newest-first previous tasks
                    try:
                        curr_idx = self.task_list.index(task)
                    except ValueError:
                        curr_idx = len(self.task_list)
                    prev_tasks_order = [t for t in reversed(self.task_list[:curr_idx]) if t in self.tasks_data_dict and 'test' in self.tasks_data_dict[t]]
                    chunks_upd = torch.split(self.previous_prompts_param.detach(), self.prefix_len, dim=0)
                    # original chunks for historical queue if needed
                    orig_chunks = torch.split(self.previous_prompts.detach(), self.prefix_len, dim=0) if (self.previous_prompts is not None and self.previous_prompts.shape[0] > 0) else []
                    for idx_chunk, prev_task in enumerate(prev_tasks_order):
                        if idx_chunk >= len(chunks_upd):
                            break
                        # 1) Only updated task-specific prompt
                        prompt_prev_only = chunks_upd[idx_chunk].to(self.device)
                        acc_prev = self.validate(self.tasks_data_dict[prev_task]['test'],
                                                 prev_task,
                                                 prompt_prev_only,
                                                 self.task_to_target_len[prev_task],
                                                 print_outputs=False,
                                                 use_global_prompt=False)
                        acc_prev_mean = float(np.mean(acc_prev)) if isinstance(acc_prev, (list, tuple, np.ndarray)) else float(acc_prev)
                        print(f"[ReverseEval] epoch={epoch} prev_task={prev_task} test_acc={acc_prev_mean:.4f}")
                        # 2) Updated prompt + the same Big P snapshot used when that task was trained
                        try:
                            if prev_task in self.task_to_global_snapshot:
                                gp_snap = self.task_to_global_snapshot[prev_task]
                                prompt_with_gp = torch.cat([gp_snap, prompt_prev_only], dim=0).to(self.device)
                                acc_prev_gp = self.validate(self.tasks_data_dict[prev_task]['test'],
                                                            prev_task,
                                                            prompt_with_gp,
                                                            self.task_to_target_len[prev_task],
                                                            print_outputs=False,
                                                            use_global_prompt=False)
                                acc_prev_gp_mean = float(np.mean(acc_prev_gp)) if isinstance(acc_prev_gp, (list, tuple, np.ndarray)) else float(acc_prev_gp)
                                print(f"[ReverseEval+BigP] epoch={epoch} prev_task={prev_task} test_acc={acc_prev_gp_mean:.4f}")
                        except Exception as e2:
                            print('Warning: reverse phase BigP eval failed:', e2)
                except Exception as e:
                    print('Warning: reverse phase test eval failed:', e)

        # Ensure reverse-phase updates are stored even if training ended mid-phase
        if self.reverse_phase_active:
            self.reverse_phase_active = False
            if self.previous_prompts_param is not None:
                self.previous_prompts_updated = self.previous_prompts_param.detach().to(self.device)
                self.previous_prompts_updated.requires_grad = False
            if hasattr(model, 'prompt'):
                model.prompt.requires_grad = True
            self.previous_prompts_param = None
            self._reverse_basis_list = []

        # After finishing task: print per-task prompt norms from original and updated queues
        try:
            try:
                curr_idx = self.task_list.index(task)
            except ValueError:
                curr_idx = len(self.task_list)
            prev_tasks_order = [t for t in reversed(self.task_list[:curr_idx]) if t in self.task_prompts]
            if self.previous_prompts is not None and self.previous_prompts.shape[0] > 0 and len(prev_tasks_order) > 0:
                orig_chunks = torch.split(self.previous_prompts, self.prefix_len, dim=0)
                if self.previous_prompts_updated is not None and self.previous_prompts_updated.shape[0] == self.previous_prompts.shape[0]:
                    upd_chunks = torch.split(self.previous_prompts_updated, self.prefix_len, dim=0)
                else:
                    upd_chunks = [None] * len(orig_chunks)
                for idx, t_prev in enumerate(prev_tasks_order):
                    if idx >= len(orig_chunks):
                        break
                    orig_norm = float(torch.norm(orig_chunks[idx]).detach().cpu().item())
                    if upd_chunks[idx] is not None:
                        upd_norm = float(torch.norm(upd_chunks[idx]).detach().cpu().item())
                        print(f"[PromptNorm] task={t_prev} original={orig_norm:.6f} updated={upd_norm:.6f}")
                    else:
                        print(f"[PromptNorm] task={t_prev} original={orig_norm:.6f} updated=NA")
            else:
                print("[PromptNorm] No previous prompts to report.")
        except Exception as e:
            print("Warning: failed to print prompt norms:", e)

        if progressive:
            self.progress_previous_prompts(task=task)
        else:
            if self.early_stopping:
                self.restore_best_model()

        # Print global (big) prompt norm after finishing training this task
        if getattr(self, 'global_prompt', None) is not None:
            try:
                print('Global prompt L2 norm after task', task, ':', float(self.global_prompt.detach().norm().cpu().item()))
            except Exception as e:
                print('Global prompt norm print failed:', e)
        return val_acc

    def _save_partial_state(self, save_path, task, epoch, val_acc):
        if save_path is None:
            return
        os.makedirs(save_path, exist_ok=True)
        task_prompts = {}
        for name, prompt in getattr(self, 'task_prompts', {}).items():
            task_prompts[name] = prompt.detach().cpu().numpy()
        state = {
            'task': task,
            'epoch': epoch,
            'val_acc': val_acc,
            'previous_prompts': self.previous_prompts.detach().cpu().numpy() if self.previous_prompts is not None else None,
            'current_prompt': self.model.prompt.detach().cpu().numpy() if hasattr(self.model, 'prompt') else None,
            'task_prompts': task_prompts,
        }
        np.save(os.path.join(save_path, 'partial_state.npy'), state, allow_pickle=True)
        np.save(os.path.join(save_path, f'partial_{task}_epoch_{epoch}.npy'), state, allow_pickle=True)
    
    # Train model continually
    def train_continual(self,
                        task_list,
                        epochs=40,
                        save_path=None,
                        progressive=True,
                        eval_every_N=1,
                        test_eval_after_every_task=False, # only needed for methods with catastrophic forgetting
                        data_replay_freq=-1,
                        ):
        results_dict = {}
        if self.get_test_subset: results_dict['test'] = {}

        for num, task in enumerate(task_list):
            eval_on_all_tasks = False if progressive or len(task_list)==1 else True
            eval_frq = eval_every_N if not eval_on_all_tasks else int(epochs//3)
            val_acc = self.train_one_task(task, epochs,
                                          progressive=progressive,
                                          eval_every_N=eval_frq,
                                          #eval_on_all_tasks=False, # too slow
                                          data_replay_freq=data_replay_freq,
                                          eval_on_all_tasks=eval_on_all_tasks,
                                          save_path=save_path,
                                          )
            print(task, val_acc)
            results_dict[task] = val_acc

            print('Calculating test acc ...')
            print('test_eval_after_every_task',test_eval_after_every_task)
            if self.get_test_subset:
                if test_eval_after_every_task:
                    # Print global (big) prompt norm before starting test-eval across tasks
                    if getattr(self, 'global_prompt', None) is not None:
                        try:
                            print('Global prompt L2 norm before test-eval (after task', task, '):', float(self.global_prompt.detach().norm().cpu().item()))
                        except Exception as e:
                            print('Global prompt norm print failed:', e)
                    # eval test accuracy for seen tasks only
                    results_dict['test'][num] = {}
                    for test_task in task_list[:num+1]:
                            acc = self.validate(self.tasks_data_dict[test_task]['test'],
                                                test_task,
                                                None,
                                                self.task_to_target_len[test_task],
                                                print_outputs=True,
                                                use_global_prompt=False)
                            results_dict['test'][num][test_task] = acc
                            # print single-prompt test accuracy tag for parity
                            try:
                                acc_mean = float(np.mean(acc)) if isinstance(acc, (list, tuple, np.ndarray)) else float(acc)
                                print(f"[TestSinglePrompt] step={num} task={test_task} acc={acc_mean:.4f}")
                                # also evaluate with this task's historical Big P snapshot if available
                                if test_task in self.task_to_global_snapshot:
                                    gp_snap = self.task_to_global_snapshot[test_task]
                                    prompt_with_gp = torch.cat([gp_snap, self.task_prompts[test_task]], dim=0).to(self.device)
                                    acc_with_gp = self.validate(self.tasks_data_dict[test_task]['test'],
                                                                test_task,
                                                                prompt_with_gp,
                                                                self.task_to_target_len[test_task],
                                                                print_outputs=False,
                                                                use_global_prompt=False)
                                    acc_with_gp_mean = float(np.mean(acc_with_gp)) if isinstance(acc_with_gp, (list, tuple, np.ndarray)) else float(acc_with_gp)
                                    print(f"[TestWithBigP] step={num} task={test_task} acc={acc_with_gp_mean:.4f}")
                            except Exception as e:
                                print('Warning: test single-prompt/BigP eval failed:', e)

                else:
                    acc = self.validate(self.tasks_data_dict[task]['test'],
                                        task,
                                        None,
                                        self.task_to_target_len[task],
                                        print_outputs=True,
                                        use_global_prompt=False)
                    results_dict['test'][task] = acc
            np.save(os.path.join(save_path, 'results_dict.npy'), results_dict)

        return results_dict




    # Perform multi-task training
    def multi_task_training(self, num_epochs=5, progressive=False, save_path=''):
        tasks_data_dict = self.tasks_data_dict
        val_scores = {x: [] for x in list(tasks_data_dict)}
        task_lengths = [len(tasks_data_dict[t]['train'])*self.batch_size for t in list(tasks_data_dict)]
        idx_biggest_task = np.argmax(task_lengths)
        n_tasks = len(list(tasks_data_dict))

        results_dict = {'test': {}}
        device = self.device

        for epoch in range(num_epochs):
            print(epoch)

            dataloaders_list = [tasks_data_dict[t]['train'] if j==idx_biggest_task else cycle(tasks_data_dict[t]['train']) \
                                for j, t in enumerate(tasks_data_dict)]
            mlt_dataloader = zip(*dataloaders_list)

            max_task = np.max([len(tasks_data_dict[t]['train']) for t in list(tasks_data_dict)])
            pbar = tqdm(total=max_task)

            for i, batch_combined in enumerate(mlt_dataloader):
                loss_combined = 0

                for task_num in range(n_tasks):
                    batch = {k: v.to(device) for k, v in batch_combined[task_num].items()}
                    if self.prefix_len>0: 
                        loss = self.train_step_lester(batch,
                                                      task=task if self.prefix_MLPs!=None else None,
                                                      progressive=progressive)
                    else:
                        loss = self.train_step(batch)

                    loss_combined += loss

                loss_combined.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                pbar.update(1)



            results_dict['test'][epoch] = {}
            curr_prompt = None
            for test_task in self.task_list:
                acc = self.validate(self.tasks_data_dict[test_task]['test'],
                                    test_task,
                                    curr_prompt,
                                    self.task_to_target_len[test_task],
                                    print_outputs=True)
                results_dict['test'][epoch][test_task] = acc

            if save_path!='':
                np.save(os.path.join(save_path, 'results_dict.npy'), results_dict)
            pbar.close()

        return results_dict




