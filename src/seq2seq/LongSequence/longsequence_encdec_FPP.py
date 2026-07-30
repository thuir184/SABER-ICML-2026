import torch
from torch import nn
import pandas as pd
import numpy as np
import time
import random
import math

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
    # Initialize continual learner and model
    def __init__(self,
                 model_name,
                 cache_dir,
                 task_list,
                 batch_size=8,
                 select_k_per_class=-1,
                 prefix_len=0,
                 prefix_path=None, 
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
        # Freezing model weights for prompt tuning
        if freeze_weights:
            print('Freezing weights')
            self.do_freeze_weights(except_condition=freeze_except)
           
        self.prefix_len = prefix_len

        # intializing dictionary to store prompts
        self.task_prompts = {}
        self.prompt_grad_dirs = {}
        self.prev_prompt_grad_dirs = {}
        self.task_prompts_updated = {}
        self.previous_prompts_updated = None
        self.task_to_svd_basis = {}
        self.reverse_phase_active = False
        self.previous_prompts_param = None
        self._reverse_dir_stack = None
        # Decoder parity: selection method and cumulative restricted bases
        self.selection_method = getattr(self, "selection_method", "proj_cos") 
        self.cumulative_restrict_basis = {}


        # Creating a trainable soft prompt
        if prefix_len>0:
            self.model.prompt = nn.Parameter(torch.tensor(self.init_new_prompt(prefix_len),
                                                          requires_grad=True))
            if prefix_path==None:
                self.previous_prompts = torch.zeros([0, self.model.prompt.shape[1]],
                                                    requires_grad=False).to(self.device)
                # self.selected_prompts = None  # Will be updated after Task 6
                                        
                                                    
            else: # initializing previous prompts from the path
                print('Using pre-trained progressive prompt - ' + prefix_path)
                self.previous_prompts = torch.tensor(np.load(prefix_path), requires_grad = False).to(self.device)
        
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
        
        # Storage for epoch-2 current prompt gradient directions (per task)
        self.epoch2_prompt_grad_dirs = {}
        self.epoch2_prompt_svd = {}  
        self._epoch2_accum = {}  
        self.epoch2_subspace_sim = {}  
        self.epoch2_dir_sim = {}  
        self.epoch2_combined_sim = {}  
        self._epoch2_done = {}  
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
        self.tasks_data_dict = self.get_tasks_data_dict(memory_perc=memory_perc)


    # Create optimizer 
    def get_optimizer(self, lr, weight_decay,
                      task=None, mlp_lr=None, weight_decay_mlp=None):

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
                "params": [p for n, p in self.prefix_MLPs[task].named_parameters()],
                "weight_decay": weight_decay_mlp,
                "lr": mlp_lr,
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
        if self.early_stopping: 
            new_prompt = self.best_prompt 
        else: # use last prompt
            if task!=None and self.prefix_MLPs!=None:
                new_prompt = self.prefix_MLPs[task](self.model.prompt)
            else:
                new_prompt = self.model.prompt
            new_prompt = new_prompt.detach().cpu().numpy()

        new_prompt = torch.tensor(new_prompt, requires_grad = False).to(self.device)
        self.previous_prompts = torch.concat([new_prompt, self.previous_prompts], axis=0)
        self.task_prompts[task] = new_prompt
        print('task prompt',new_prompt)
        print('Updated progressive prompts ', self.previous_prompts.shape)


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
            # ds2 = t5_dataset.T5Dataset(self.tokenizer, task, self.cache_dir, self.pre_processed)
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
            else: self.multirc_idx = None
        return tasks_data_dict


    # Perform one train step for prompt tuning (following Lester et al.)
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

        if progressive:
            prev_block = self.previous_prompts_param if (self.reverse_phase_active and self.previous_prompts_param is not None) else self.previous_prompts
            inputs_embeds = torch.concat([prompt.repeat(k, 1, 1),
                                          prev_block.repeat(k, 1, 1),
                                          inputs_embeds], axis=1)[:,:self.seq_len]
            full_prefix_len = prev_block.shape[0] + prompt.shape[0] # prefix including all previous tasks
        else:
            inputs_embeds = torch.concat([prompt.repeat(k, 1, 1),
                                          inputs_embeds], axis=1)[:,:self.seq_len]
            full_prefix_len = prompt.shape[0]

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

    # --------- Loss distribution utilities for Wasserstein similarity (parity with decoder/orth) ---------
    def compute_loss_distribution(self, dataloader, task, prompt=None, max_batches=200):
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
                    inputs_embeds = torch.concat([prompt.repeat(k, 1, 1),
                                                  inputs_embeds], axis=1)[:, :self.seq_len]
                    full_prefix_len = prompt.shape[0]
                    source_mask_updated = torch.concat((b["source_mask"][0][0].repeat(k, full_prefix_len),
                                                        b["source_mask"]), axis=1)[:, :self.seq_len]
                else:
                    source_mask_updated = b["source_mask"]
                lm_labels = b["target_ids"]
                lm_labels[lm_labels[:, :] == tokenizer.pad_token_id] = -100
                encoder_outputs = model.encoder(attention_mask=source_mask_updated,
                                                inputs_embeds=inputs_embeds,
                                                head_mask=None,
                                                output_attentions=None,
                                                output_hidden_states=None,
                                                return_dict=None)
                outputs = model(input_ids=b["source_ids"],
                                attention_mask=source_mask_updated,
                                labels=lm_labels,
                                decoder_attention_mask=b['target_mask'],
                                encoder_outputs=encoder_outputs)
                loss = outputs[0]
                losses.append(float(loss.detach().cpu().item()))
        return losses

    def compute_loss_based_similarity(self, current_task, print_results=True, max_batches=200):
        # Returns dict: prev_task -> {'dis_prime': x, 'dis': y, 'similar': bool}
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

        similar = {}
        tasks_dict = self.tasks_data_dict
        if current_task not in tasks_dict:
            return similar
        # base losses for current task (no prompt)
        L_t_base = self.compute_loss_distribution(tasks_dict[current_task]['train'], current_task, prompt=None, max_batches=max_batches)
        # previous tasks are those before current in task_list
        try:
            curr_idx = self.task_list.index(current_task)
        except ValueError:
            curr_idx = len(self.task_list)
        prev_tasks = [t for t in self.task_list[:curr_idx] if t in self.task_prompts]
        for prev_task in prev_tasks:
            # base(prev), self(prev), cross(current with prev prompt)
            L_i_base = self.compute_loss_distribution(tasks_dict[prev_task]['train'], prev_task, prompt=None, max_batches=max_batches)
            prompt_prev = self.task_prompts[prev_task].to(self.device)
            L_i_self = self.compute_loss_distribution(tasks_dict[prev_task]['train'], prev_task, prompt=prompt_prev, max_batches=max_batches)
            L_t_with_i = self.compute_loss_distribution(tasks_dict[current_task]['train'], current_task, prompt=prompt_prev, max_batches=max_batches)
            dis_prime = wdist(L_i_base, L_t_base)
            dis = wdist(L_i_self, L_t_with_i)
            is_similar = dis < dis_prime
            similar[prev_task] = {'dis_prime': dis_prime, 'dis': dis, 'similar': is_similar}
            if print_results:
                try:
                    print(f"[LossSim] current={current_task} prev={prev_task} dis_self_vs_cross={dis:.6f} dis_base_vs_base={dis_prime:.6f} similar={is_similar}")
                except Exception:
                    pass
        return similar



    # Perform one train step for full model training
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



    # Process string for validation (remove pad and end tokens)

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
                 print_outputs=False
                ):
        model = self.model
        prefix_len = self.prefix_len
        max_length = target_len
        tokenizer = self.tokenizer
        model.eval()

        corr, total, f1 = 0, 0, 0
        y_true, y_pred = [], []

        for i, batch in enumerate(tqdm(dataloader_val)):
            batch = {k:batch[k].to(self.device) for k in batch}
            inputs_embeds = model.encoder.embed_tokens(batch["source_ids"]).to(self.device)

            if prompt!=None:
                k = inputs_embeds.shape[0]
                inputs_embeds = torch.concat([prompt.repeat(k, 1, 1),
                                              inputs_embeds], axis=1)[:,:self.seq_len]

                full_prefix_len = prompt.shape[0] # prompt is inputted by user
                source_mask_updated = torch.concat( (batch["source_mask"][0][0].repeat(k,full_prefix_len),
                                                     batch["source_mask"]), axis=1)[:,:self.seq_len]

            else: # full model fine tuning, no prompt added
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





    # Freeze model weights
    def do_freeze_weights(self, except_condition='shared'):
        model = self.model
        for name, param in model.named_parameters():
            if param.requires_grad == True and except_condition not in name:
                param.requires_grad = False

    # Collect per-batch current prompt gradients during epoch 2 and compute avg direction.
    # Additionally, store per-batch normalized gradient directions and compute top-k SVD basis at finalize.
    def collect_current_prompt_grad_direction(self, task, prompt_grad, target_batches=30):
        if task not in self._epoch2_accum:
            self._epoch2_accum[task] = {
                'sum': torch.zeros_like(prompt_grad.detach()),
                'count': 0,
                'target': int(target_batches),
                'rows': [],  
            }
        acc = self._epoch2_accum[task]
        if acc['count'] < acc['target']:
            g = prompt_grad.detach()
            acc['sum'] += g
            acc['count'] += 1
            g_flat = g.reshape(-1)
            g_norm = torch.norm(g_flat)
            if g_norm > 0:
                acc['rows'].append((g_flat / g_norm).cpu())
            else:
                acc['rows'].append(g_flat.cpu())

    def finalize_current_prompt_grad_direction(self, task, topk=3):
        if task not in self._epoch2_accum:
            return None
        acc = self._epoch2_accum[task]
        if acc['count'] == 0:
            return None
        avg_grad = acc['sum'] / acc['count']
        norm = torch.norm(avg_grad)
        direction = (avg_grad / norm) if norm > 0 else avg_grad
        self.epoch2_prompt_grad_dirs[task] = direction.detach().cpu().numpy()
        # compute SVD of collected directions if available
        try:
            rows = acc.get('rows', [])
            if len(rows) >= 2:
                X = torch.stack(rows, dim=0)  # m x D
                X = X.to(self.device)
                U, S, Vh = torch.linalg.svd(X, full_matrices=False)  # Vh: r x D
                V = Vh.transpose(0, 1)  # D x r
                k = min(int(topk), V.shape[1])
                self.epoch2_prompt_svd[task] = V[:, : k].detach().cpu().numpy()
                top_sv = float(S[0].detach().cpu().item()) if S.numel() > 0 else 0.0
                print(f"[Epoch2PromptSVD] task={task} rows={len(rows)} topk={k} top_singular={top_sv:.6f}")
        except Exception:
            pass
        print(f"[Epoch2PromptDir] task={task} batches={acc['count']} norm={float(norm.detach().cpu().item()):.6f}")
        return self.epoch2_prompt_grad_dirs[task]

    # Compute similarity of current task epoch-2 direction to each previous task's epoch-2 SVD subspace.
    # Similarity = ||Proj_{span(V_prev)}(u_curr)||_2 where u_curr is unit-norm direction.
    def compute_epoch2_subspace_similarities(self, current_task):
        sims = {}
        if current_task not in self.epoch2_prompt_grad_dirs:
            return sims
        u_np = self.epoch2_prompt_grad_dirs[current_task]
        if u_np is None:
            return sims
        u = torch.as_tensor(u_np, device=self.device, dtype=self.model.encoder.embed_tokens.weight.dtype).reshape(-1)
        for prev_task, V_np in self.epoch2_prompt_svd.items():
            if prev_task == current_task:
                continue
            if V_np is None:
                continue
            V = torch.as_tensor(V_np, device=self.device, dtype=self.model.encoder.embed_tokens.weight.dtype)
            if V.dim() != 2:
                continue
            if V.shape[0] != u.numel() and V.shape[1] == u.numel():
                V = V.transpose(0, 1).contiguous()
            if u.numel() != V.shape[0]:
                print(f"[Epoch2SubspaceSim] skip prev={prev_task}: shape mismatch u={u.numel()} vs V_rows={V.shape[0]}")
                continue
            proj = V @ (V.t() @ u)
            proj_norm = float(torch.norm(proj).item())
            sims[prev_task] = {'proj_norm': proj_norm, 'k': int(V.shape[1])}
            print(f"[Epoch2SubspaceSim] cur={current_task} prev={prev_task} proj_norm={proj_norm:.6f} k={int(V.shape[1])}")
        self.epoch2_subspace_sim[current_task] = sims
        return sims

    # Compute pairwise cosine similarity between current task's epoch-2 direction and each previous task's epoch-2 direction.
    def compute_epoch2_pairwise_direction_similarities(self, current_task):
        sims = {}
        if current_task not in self.epoch2_prompt_grad_dirs:
            return sims
        u_np = self.epoch2_prompt_grad_dirs[current_task]
        if u_np is None:
            return sims
        u = torch.as_tensor(u_np, dtype=self.model.encoder.embed_tokens.weight.dtype).reshape(-1)
        for prev_task, v_np in self.epoch2_prompt_grad_dirs.items():
            if prev_task == current_task:
                continue
            if v_np is None:
                continue
            v = torch.as_tensor(v_np, dtype=self.model.encoder.embed_tokens.weight.dtype).reshape(-1)
            if v.numel() != u.numel():
                continue
            u_n = u / (u.norm() + 1e-12)
            v_n = v / (v.norm() + 1e-12)
            cos = float(torch.dot(u_n, v_n).item())
            sims[prev_task] = cos
            print(f"[Epoch2DirCos] cur={current_task} prev={prev_task} cos={cos:.6f}")
        key = f"pairwise:{current_task}"
        self.epoch2_dir_sim[key] = sims
        return sims

    # Compute combined similarity: projection norm times pairwise cosine
    def compute_epoch2_combined_similarity(self, current_task):
        combined = {}
        sub = self.epoch2_subspace_sim.get(current_task, {})
        pair = self.epoch2_dir_sim.get(f"pairwise:{current_task}", {})
        if not sub or not pair:
            self.epoch2_combined_sim[current_task] = combined
            return combined
        for prev_task, subdict in sub.items():
            proj = float(subdict.get('proj_norm', 0.0))
            cos = float(pair.get(prev_task, 0.0))
            score = proj * cos
            combined[prev_task] = {'combined': score, 'proj_norm': proj, 'cos': cos}
            print(f"[Epoch2CombinedSim] cur={current_task} prev={prev_task} combined={score:.6f} (proj={proj:.4f}, cos={cos:.4f})")
        self.epoch2_combined_sim[current_task] = combined
        return combined

    # Print top-k most similar previous tasks to current_task by combined epoch-2 similarity
    def print_topk_epoch2_similar_tasks(self, current_task, k=7):
        combined = self.epoch2_combined_sim.get(current_task, {})
        if not combined:
            print(f"[Epoch2TopK] cur={current_task} no combined similarities available")
            return []
        # sort by combined score descending
        sorted_items = sorted(combined.items(), key=lambda x: float(x[1].get('combined', 0.0)), reverse=True)
        topk = sorted_items[: int(k)]
        pretty = ", ".join([f"{t}:{float(v.get('combined', 0.0)):.6f}" for t, v in topk])
        print(f"[Epoch2TopK] cur={current_task} k={int(k)} -> {pretty}")
        return topk

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
        # for each memory buffer in tasks_to_generators perform memory replay
        print("Rehearsal on " + str((', ').join(list(tasks_to_generators)) ))
        for prev_task in tasks_to_generators:
            generator_mem1 = tasks_to_generators[prev_task]
            try:
                # Samples the batch
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
    
    # Compute, for the current task, the gradient direction for each previous prompt
    # Only the single previous prompt under consideration is prepended (no others)
    def compute_prev_prompts_gradient_directions(self, dataloader_val, current_task, max_batches=10):
        if len(self.task_prompts) == 0:
            return {}
        model = self.model
        tokenizer = self.tokenizer
        device = self.device
        model.eval()
        results = {}
        # iterate previous prompts by their task keys
        for prev_task, prev_prompt_tensor in self.task_prompts.items():
            total_grad = None
            num = 0
            for i, batch in enumerate(dataloader_val):
                if i >= max_batches:
                    break
                batch = {k: batch[k].to(device) for k in batch}
                base_embeds = model.encoder.embed_tokens(batch["source_ids"])
                # treat the stored previous prompt embedding as the variable
                prompt_eff = prev_prompt_tensor.detach().requires_grad_(True)
                k = base_embeds.shape[0]
                inputs_embeds = torch.concat([prompt_eff.repeat(k, 1, 1),
                                              base_embeds], axis=1)[:, :self.seq_len]
                full_prefix_len = prompt_eff.shape[0]
                source_mask_updated = torch.concat((batch["source_mask"][0][0].repeat(k, full_prefix_len),
                                                    batch["source_mask"]), axis=1)[:, :self.seq_len]
                lm_labels = batch["target_ids"]
                lm_labels[lm_labels[:, :] == tokenizer.pad_token_id] = -100
                
                model.zero_grad(set_to_none=True)
                encoder_outputs = model.encoder(attention_mask=source_mask_updated,
                                                inputs_embeds=inputs_embeds,
                                                head_mask=None,
                                                output_attentions=None,
                                                output_hidden_states=None,
                                                return_dict=None)
                outputs = model(input_ids=batch["source_ids"],
                                attention_mask=source_mask_updated,
                                labels=lm_labels,
                                decoder_attention_mask=batch["target_mask"],
                                encoder_outputs=encoder_outputs)
                loss = outputs[0]
                loss.backward()

                grad = prompt_eff.grad.detach()
                if total_grad is None:
                    total_grad = grad.clone()
                else:
                    total_grad += grad
                num += 1

            if num == 0:
                continue
            avg_grad = total_grad / num
            norm = torch.norm(avg_grad)
            if norm > 0:
                direction = avg_grad / norm
            else:
                direction = avg_grad
            results[prev_task] = {
                'direction': direction.detach().cpu().numpy(),
                'norm': float(norm.detach().cpu().item()) if torch.is_tensor(norm) else float(norm)
            }
            # Print norm of gradient for previous prompt wrt current task
            try:
                print(f"[PrevGradNorm] current={current_task} prev={prev_task} norm={results[prev_task]['norm']:.6f}")
            except Exception:
                pass
        self.prev_prompt_grad_dirs[current_task] = results
        return results
    
    # Compare each previous prompt's own-task direction vs. current-task direction (cosine and angle)
    def compute_prev_prompt_direction_similarity(self, current_task, print_results=True):
        if current_task not in self.prev_prompt_grad_dirs:
            return {}
        results = {}
        for prev_task in self.prev_prompt_grad_dirs[current_task]:
            dir_curr = self.prev_prompt_grad_dirs[current_task][prev_task].get('direction', None)
            dir_self = self.prompt_grad_dirs.get(prev_task, None)
            if dir_curr is None or dir_self is None:
                continue
            # Both are already L2-normalized (Frobenius). Flatten and compute cosine.
            a = dir_self.reshape(-1)
            b = dir_curr.reshape(-1)
            # numerical clamp
            cos_sim = float(np.clip(np.dot(a, b), -1.0, 1.0))
            angle_deg = float(180.0 * math.acos(cos_sim) / math.pi)
            results[prev_task] = {'cosine': cos_sim, 'angle_deg': angle_deg}
            # attach to stored structure
            self.prev_prompt_grad_dirs[current_task][prev_task]['cosine'] = cos_sim
            self.prev_prompt_grad_dirs[current_task][prev_task]['angle_deg'] = angle_deg
            if print_results:
                print(f"[DirAlign] prev={prev_task} vs current={current_task} -> cosine={cos_sim:.4f}, angle={angle_deg:.2f} deg")
        return results

    # Compute and store average gradient direction of the learned prompt on a validation set
    def compute_prompt_gradient_direction(self, dataloader_val, task, progressive=True, max_batches=10):
        model = self.model
        tokenizer = self.tokenizer
        device = self.device
        model.eval()
        if self.prefix_MLPs != None:
            self.prefix_MLPs[task].eval()

        total_grad = None
        grad_samples = []
        num = 0
        for i, batch in enumerate(dataloader_val):
            if i >= max_batches:
                break
            batch = {k: batch[k].to(device) for k in batch}
            
            # Build inputs_embeds like in validate(), but keep prompt tensor as a leaf with grad
            base_embeds = model.encoder.embed_tokens(batch["source_ids"])
            if self.prefix_MLPs != None:
                with torch.no_grad():
                    prompt_eff = self.prefix_MLPs[task](self.model.prompt.detach())
            else:
                with torch.no_grad():
                    prompt_eff = self.model.prompt.detach()
            prompt_eff = prompt_eff.requires_grad_(True)

            k = base_embeds.shape[0]
            if progressive:
                inputs_embeds = torch.concat([prompt_eff.repeat(k, 1, 1),
                                             self.previous_prompts.repeat(k, 1, 1),
                                             base_embeds], axis=1)[:, :self.seq_len]
                full_prefix_len = self.previous_prompts.shape[0] + prompt_eff.shape[0]
            else:
                inputs_embeds = torch.concat([prompt_eff.repeat(k, 1, 1),
                                             base_embeds], axis=1)[:, :self.seq_len]
                full_prefix_len = prompt_eff.shape[0]

            source_mask_updated = torch.concat((batch["source_mask"][0][0].repeat(k, full_prefix_len),
                                                batch["source_mask"]), axis=1)[:, :self.seq_len]

            lm_labels = batch["target_ids"]
            lm_labels[lm_labels[:, :] == tokenizer.pad_token_id] = -100

            model.zero_grad(set_to_none=True)
            encoder_outputs = model.encoder(attention_mask=source_mask_updated,
                                            inputs_embeds=inputs_embeds,
                                            head_mask=None,
                                            output_attentions=None,
                                            output_hidden_states=None,
                                            return_dict=None)
            outputs = model(input_ids=batch["source_ids"],
                            attention_mask=source_mask_updated,
                            labels=lm_labels,
                            decoder_attention_mask=batch["target_mask"],
                            encoder_outputs=encoder_outputs)
            loss = outputs[0]
            loss.backward()
            
            grad = prompt_eff.grad.detach()
            grad_samples.append(grad.reshape(-1).detach().cpu())
            if total_grad is None:
                total_grad = grad.clone()
            else:
                total_grad += grad
            num += 1
            
        if num == 0:
            return None
        avg_grad = total_grad / num
        norm = torch.norm(avg_grad)
        if norm > 0:
            direction = avg_grad / norm
        else:
            direction = avg_grad
        self.prompt_grad_dirs[task] = direction.detach().cpu().numpy()
        # Compute SVD basis (top-3 right singular vectors) from collected gradient samples
        try:
            if len(grad_samples) >= 2:
                X = torch.stack(grad_samples, dim=0)  # n x D
                # move to device for SVD, but careful with memory
                X = X.to(self.device)
                U, S, Vh = torch.linalg.svd(X, full_matrices=False)
                V = Vh.transpose(0, 1)  # D x r
                k = min(3, V.shape[1])
                V_topk = V[:, :k].detach().cpu().numpy()
                self.task_to_svd_basis[task] = V_topk
        except Exception as e:
            print('Warning: SVD basis computation failed:', e)
        return direction
    
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

        # Before training current task: compute gradient directions for each previous prompt wrt current task
        try:
            self.compute_prev_prompts_gradient_directions(dataloader_val, current_task=task, max_batches=10)
            # Also compute direction alignment between each previous prompt's own-task direction and current-task direction
            self.compute_prev_prompt_direction_similarity(task, print_results=True)
        except Exception as e:
            print('Warning: prev prompts grad direction computation failed:', e)

        val_acc = []

        # Two-phase training control (align with decoder: 10 normal + 2 reverse)
        normal_phase_epochs = 10
        reverse_phase_epochs = 2

        # If there are no previous prompts (e.g., first task), cap training to normal_phase_epochs
        has_prev_prompts = self.previous_prompts is not None and self.previous_prompts.shape[0] > 0
        effective_epochs = epochs if has_prev_prompts else min(epochs, normal_phase_epochs)

        for epoch in range(effective_epochs):
            print(epoch)
            # Determine phase for this epoch (10 normal, 5 reverse, then normal)
            in_normal_phase = epoch < normal_phase_epochs or epoch >= (normal_phase_epochs + reverse_phase_epochs)
            in_reverse_phase = (epoch >= normal_phase_epochs) and (epoch < (normal_phase_epochs + reverse_phase_epochs))

            # Phase entry/exit toggles
            if epoch == normal_phase_epochs:
                # enter reverse only if there are previous prompts; otherwise stay in normal phase
                if self.previous_prompts is not None and self.previous_prompts.shape[0] > 0:
                    self.reverse_phase_active = True
                    if hasattr(model, 'prompt'):
                        model.prompt.requires_grad = False
                    print(f"[ReversePhaseStart] epoch={epoch} task={task} entering reverse phase; updating previous prompts orthogonally and printing per-task updated accuracies")
                    # Determine newest-first full previous task order
                    try:
                        curr_idx = self.task_list.index(task)
                    except ValueError:
                        curr_idx = len(self.task_list)
                    prev_tasks_order_full = [t for t in reversed(self.task_list[:curr_idx]) if t in self.task_prompts]
                    # Decoder parity: selection by method (proj_cos thresholds or wasserstein) and update ONLY selected
                    # Prepare signals: ensure current task mean dir and prev-prompts current-task dirs are available
                    try:
                        # current task mean direction (flattened, normalized)
                        cur_mean_dir = self.compute_prompt_gradient_direction(dataloader_val, task, progressive=True, max_batches=20)
                    except Exception:
                        cur_mean_dir = None
                    # prev prompts' mean gradients wrt current task were computed pre-training
                    prev_grad_current = self.prev_prompt_grad_dirs.get(task, {})
                    # Build projection scores: project cur_mean_dir onto each prev task's SVD subspace
                    proj_scores = {}
                    if cur_mean_dir is not None:
                        u = torch.as_tensor(cur_mean_dir.detach().cpu().numpy() if torch.is_tensor(cur_mean_dir) else cur_mean_dir,
                                            device=self.device, dtype=self.model.encoder.embed_tokens.weight.dtype).reshape(-1)
                        un = torch.norm(u)
                        if un > 0:
                            u = u / un
                        for t_prev in prev_tasks_order_full:
                            V_np = self.task_to_svd_basis.get(t_prev, None)
                            if V_np is None:
                                proj_scores[t_prev] = 0.0
                                continue
                            V = torch.as_tensor(V_np, device=self.device, dtype=self.model.encoder.embed_tokens.weight.dtype)
                            if V.dim() == 2 and V.shape[1] > 0:
                                # proj norm = ||V^T u||
                                coeff = V.t() @ u
                                proj_scores[t_prev] = float(torch.norm(coeff).item())
                            else:
                                proj_scores[t_prev] = 0.0
                            try:
                                print(f"[ProjScore] current={task} prev={t_prev} proj_norm={proj_scores[t_prev]:.6f}")
                            except Exception:
                                pass
                    # Cosine alignment from prev_prompt_grad_dirs (own-task vs current-task)
                    cosines = {}
                    for t_prev, dat in prev_grad_current.items():
                        c = dat.get('cosine', None)
                        if c is not None:
                            cosines[t_prev] = float(c)
                    # Wasserstein similarity if requested
                    loss_sim = {}
                    if getattr(self, 'selection_method', 'proj_cos') == 'wasserstein':
                        try:
                            loss_sim = self.compute_loss_based_similarity(task, print_results=True)
                        except Exception as e_ls:
                            print('Warning: Wasserstein similarity failed:', e_ls)
                    # Select according to method
                    selected_set = set()
                    if getattr(self, 'selection_method', 'proj_cos') == 'wasserstein':
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
                            print(f"[SelectByWass] current={task} selected_prev={list(selected_set)} thresh=0.2")
                        except Exception:
                            pass
                    else:
                        # proj_cos: proj_norm > 0.1 and cosine > 0
                        for t_prev in prev_tasks_order_full:
                            if cosines.get(t_prev, -1.0) > 0.0 and proj_scores.get(t_prev, 0.0) > 0.1:
                                selected_set.add(t_prev)
                        if len(selected_set) == 0:
                            selected_set = set(prev_tasks_order_full)
                        try:
                            print(f"[SelectByProjCos] current={task} selected_prev={list(selected_set)}")
                        except Exception:
                            pass
                    # Preserve training order but filter to selected
                    selected_order = [t for t in prev_tasks_order_full if t in selected_set]
                    # Build selected previous prompts param by slicing original blocks
                    orig_chunks = torch.split(self.previous_prompts.detach(), self.prefix_len, dim=0)
                    # Map from full order index to chunk index (same indexing)
                    selected_indices = [idx for idx, t_prev in enumerate(prev_tasks_order_full) if t_prev in selected_set]
                    selected_chunks = [orig_chunks[idx] for idx in selected_indices] if len(selected_indices) > 0 else []
                    if len(selected_chunks) == 0:
                        # fallback to all
                        selected_order = prev_tasks_order_full
                        selected_indices = list(range(len(prev_tasks_order_full)))
                        selected_chunks = list(orig_chunks)
                    selected_block = torch.cat(selected_chunks, dim=0).to(self.device)
                    self.previous_prompts_param = nn.Parameter(selected_block.clone(), requires_grad=True)
                    # Track selection for projection/eval/merge
                    self._reverse_selected_order = selected_order
                    self._reverse_selected_indices = selected_indices
                    # add to optimizer
                    self.optimizer.add_param_group({
                        "params": [self.previous_prompts_param],
                        "weight_decay": self.weight_decay if hasattr(self, 'weight_decay') else 0.0,
                        "lr": self.lr,
                    })
                    # Build projection bases for each SELECTED previous prompt (newest-first among selected)
                    prev_tasks_order = selected_order
                    # Build list of per-task bases V (D x k) or fallback single directions (D x 1).
                    # Decoder parity: include cumulative restricted basis per task.
                    self._reverse_prev_tasks_order = prev_tasks_order
                    basis_list = []
                    emb_dim = self.model.encoder.embed_tokens.weight.shape[1]
                    D = self.prefix_len * emb_dim
                    for t_prev in prev_tasks_order:
                        parts = []
                        V_main_np = self.task_to_svd_basis.get(t_prev, None)
                        if V_main_np is not None:
                            parts.append(torch.tensor(V_main_np, dtype=self.previous_prompts_param.dtype, device=self.device))  # [D,k]
                        V_cum_np = self.cumulative_restrict_basis.get(t_prev, None)
                        if V_cum_np is not None and V_cum_np.size > 0:
                            parts.append(torch.tensor(V_cum_np, dtype=self.previous_prompts_param.dtype, device=self.device))  # [D,m]
                        if len(parts) == 0:
                            d_np = self.prompt_grad_dirs.get(t_prev, None)
                            if d_np is None:
                                basis_list.append(torch.zeros((D, 0), device=self.device, dtype=self.previous_prompts_param.dtype))
                            else:
                                d = torch.tensor(d_np, dtype=self.previous_prompts_param.dtype, device=self.device).reshape(-1)
                                dn = torch.norm(d)
                                if dn > 0:
                                    d = d / dn
                                basis_list.append(d.view(-1, 1))  # [D,1]
                        else:
                            Vcat = torch.cat(parts, dim=1)  # [D, k+m]
                            try:
                                Q, _ = torch.linalg.qr(Vcat, mode='reduced')
                                # cap columns to 6 for stability/memory
                                if Q.shape[1] > 6:
                                    Q = Q[:, :6]
                                basis_list.append(Q)
                            except Exception:
                                basis_list.append(Vcat)
                    self._reverse_basis_list = basis_list
                else:
                    self.reverse_phase_active = False
                    self.previous_prompts_param = None
                    self._reverse_prev_tasks_order = []
                    self._reverse_basis_list = []
            elif epoch == (normal_phase_epochs + reverse_phase_epochs):
                # exit reverse: store updated copy; unfreeze current prompt
                self.reverse_phase_active = False
                if self.previous_prompts_param is not None:
                    # Merge updated SELECTED blocks back into full previous prompts and update cumulative restriction bases
                    try:
                        full_chunks = list(torch.split(self.previous_prompts.detach().to(self.device), self.prefix_len, dim=0))
                        upd_chunks = list(torch.split(self.previous_prompts_param.detach().to(self.device), self.prefix_len, dim=0))
                        selected_indices = getattr(self, '_reverse_selected_indices', list(range(len(full_chunks))))
                        # Update cumulative restricted basis with net update directions per selected previous task
                        try:
                            for j, idx_full in enumerate(selected_indices):
                                if j < len(upd_chunks) and idx_full < len(full_chunks):
                                    t_prev = self._reverse_selected_order[j] if j < len(self._reverse_selected_order) else None
                                    if t_prev is None:
                                        continue
                                    delta = (upd_chunks[j] - full_chunks[idx_full]).detach()  # [L, H]
                                    dflat = delta.reshape(-1)
                                    nrm = torch.norm(dflat)
                                    if nrm is not None and float(nrm.item()) > 0.0:
                                        v = (dflat / nrm).to(torch.float32).cpu().numpy().reshape(-1, 1)  # [D,1]
                                        exist = self.cumulative_restrict_basis.get(t_prev, None)
                                        if exist is None or getattr(exist, 'size', 0) == 0:
                                            Vnew = v
                                        else:
                                            Vnew = np.concatenate([exist, v], axis=1)
                                            Vtorch = torch.tensor(Vnew, dtype=torch.float32)
                                            try:
                                                Q, _ = torch.linalg.qr(Vtorch, mode='reduced')
                                                Vnew = Q.cpu().numpy()
                                            except Exception:
                                                Vnew = Vtorch.cpu().numpy()
                                        if Vnew.shape[1] > 6:
                                            Vnew = Vnew[:, :6]
                                        self.cumulative_restrict_basis[t_prev] = Vnew
                                        try:
                                            print(f"[CumRestrict] prev_task={t_prev} added=1 total={Vnew.shape[1]}")
                                        except Exception:
                                            pass
                        except Exception as e_cum:
                            print('Warning: cumulative restrict basis update failed:', e_cum)
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
                self._reverse_prev_tasks_order = []
                self._reverse_basis_list = []
                self._reverse_selected_order = []
                self._reverse_selected_indices = []

            model.train()
            if self.prefix_MLPs != None:
                mlp.train()

            if data_replay_freq != -1:
                tasks_to_generators = self.create_memory_replay_generators(task, split='train_mem')

            for i, batch in enumerate(tqdm(dataloader_train)):
                batch = {k: batch[k].to('cuda') for k in batch}

                # Single path: train_step_lester handles which blocks are trainable
                if self.prefix_len>0: # prompt tuning
                    loss = self.train_step_lester(batch,
                                                  task=task if self.prefix_MLPs!=None else None,
                                                  progressive=progressive)
                else:
                    loss = self.train_step(batch)

                loss.backward()
                # During epoch 2 (index==1) before reverse phase, collect current prompt grads for first N batches
                if epoch == 1 and (not in_reverse_phase) and self.prefix_len > 0 and hasattr(model, 'prompt') and model.prompt.grad is not None:
                    self.collect_current_prompt_grad_direction(task, model.prompt.grad, target_batches=30)
                    # If we've collected enough samples in epoch 2, finalize and compute similarities on-the-fly (once)
                    try:
                        acc_e2 = self._epoch2_accum.get(task, None)
                        if acc_e2 is not None and acc_e2.get('count', 0) >= acc_e2.get('target', 0) and not self._epoch2_done.get(task, False):
                            self.finalize_current_prompt_grad_direction(task)
                            self.compute_epoch2_subspace_similarities(task)
                            self.compute_epoch2_pairwise_direction_similarities(task)
                            self.compute_epoch2_combined_similarity(task)
                            # Print top-7 similar tasks based on combined similarity
                            self.print_topk_epoch2_similar_tasks(task, k=7)
                            self._epoch2_done[task] = True
                    except Exception as e:
                        print('Warning: epoch-2 on-the-fly similarity computation failed:', e)
                # If in reverse phase, project grad of previous_prompts_param orthogonally to top-k bases before stepping
                if self.reverse_phase_active and self.previous_prompts_param is not None and self.previous_prompts_param.grad is not None:
                    basis_list = getattr(self, '_reverse_basis_list', [])
                    if len(basis_list) > 0 and self.prefix_len > 0 and self.previous_prompts_param.grad.shape[0] % self.prefix_len == 0:
                        with torch.no_grad():
                            g = self.previous_prompts_param.grad  # shape: (sum_prompts_len, emb_dim)
                            chunks_g = torch.split(g, self.prefix_len, dim=0)  # list of (prefix_len, emb_dim)
                            for idx, gi in enumerate(chunks_g):
                                if idx >= len(basis_list):
                                    break
                                V = basis_list[idx]  # (D, k)
                                if V is None or V.numel() == 0:
                                    continue
                                D = V.shape[0]
                                gi_flat = gi.reshape(-1)  # (D,)
                                # proj = V @ (V^T @ gi_flat)
                                proj = V @ (V.transpose(0, 1) @ gi_flat)
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
                prompt = torch.concat([prompt, self.previous_prompts], axis=0)

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
                self._print_epoch_score(task, epoch, val_acc[-1])
                self._save_partial_state(save_path, task, epoch, val_acc)

            # During reverse phase, also evaluate previous tasks' test accuracy
            # using only that task's updated prompt (no other prompts)
            if in_reverse_phase and self.previous_prompts_param is not None:
                try:
                    # selected previous tasks order (newest-first among selected)
                    prev_tasks_order = getattr(self, '_reverse_selected_order', [])
                    if not prev_tasks_order:
                        # fallback to full order
                        try:
                            curr_idx = self.task_list.index(task)
                        except ValueError:
                            curr_idx = len(self.task_list)
                        prev_tasks_order = [t for t in reversed(self.task_list[:curr_idx]) if t in self.tasks_data_dict and 'test' in self.tasks_data_dict[t]]
                    selected_indices = getattr(self, '_reverse_selected_indices', list(range(len(prev_tasks_order))))
                    # split updated block per previous prompt
                    chunks_upd = torch.split(self.previous_prompts_param.detach(), self.prefix_len, dim=0)
                    orig_chunks = torch.split(self.previous_prompts.detach(), self.prefix_len, dim=0) if (self.previous_prompts is not None and self.previous_prompts.shape[0] > 0) else []
                    for j, prev_task in enumerate(prev_tasks_order):
                        if j >= len(chunks_upd):
                            break
                        prompt_prev_only = chunks_upd[j].to(self.device)
                        # Evaluate on TEST
                        acc_prev = self.validate(self.tasks_data_dict[prev_task]['test'],
                                                 prev_task,
                                                 prompt_prev_only,
                                                 self.task_to_target_len[prev_task],
                                                 print_outputs=False)
                        acc_prev_mean = float(np.mean(acc_prev)) if isinstance(acc_prev, (list, tuple, np.ndarray)) else float(acc_prev)
                        print(f"[ReverseEval] epoch={epoch} prev_task={prev_task} test_acc={acc_prev_mean:.4f}")
                        # (removed for parity) no per-epoch validation eval for updated previous prompt
                        # Also evaluate with the same historical queue (earlier prompts) that existed when this task was learnt:
                        # queue = [updated prev_task prompt] + [original prompts of tasks earlier than prev_task]
                        try:
                            # earlier prompts are those after the FULL index of this task (newest-first order)
                            if len(orig_chunks) > 0:
                                idx_full = selected_indices[j] if j < len(selected_indices) else None
                                if idx_full is not None and idx_full < len(orig_chunks):
                                    if idx_full + 1 < len(orig_chunks):
                                        hist_queue_chunks = [chunks_upd[j]] + list(orig_chunks[idx_full+1:])
                                    else:
                                        hist_queue_chunks = [chunks_upd[j]]
                                else:
                                    hist_queue_chunks = [chunks_upd[j]]
                                prompt_with_queue = torch.cat(hist_queue_chunks, dim=0).to(self.device)
                                # Evaluate on TEST with historical queue
                                acc_prev_hist = self.validate(self.tasks_data_dict[prev_task]['test'],
                                                              prev_task,
                                                              prompt_with_queue,
                                                              self.task_to_target_len[prev_task],
                                                              print_outputs=False)
                                acc_prev_hist_mean = float(np.mean(acc_prev_hist)) if isinstance(acc_prev_hist, (list, tuple, np.ndarray)) else float(acc_prev_hist)
                                print(f"[ReverseEvalQueue] epoch={epoch} prev_task={prev_task} test_acc={acc_prev_hist_mean:.4f}")
                                # (removed for parity) no per-epoch validation eval with historical queue
                        except Exception as e2:
                            print('Warning: reverse phase queue eval failed:', e2)
                except Exception as e:
                    print('Warning: reverse phase test eval failed:', e)

        # Ensure reverse-phase updates are stored even if training ended mid-phase
        if self.reverse_phase_active:
            self.reverse_phase_active = False
            if self.previous_prompts_param is not None:
                # Merge updated SELECTED blocks back into full previous prompts
                try:
                    full_chunks = list(torch.split(self.previous_prompts.detach().to(self.device), self.prefix_len, dim=0))
                    upd_chunks = list(torch.split(self.previous_prompts_param.detach().to(self.device), self.prefix_len, dim=0))
                    selected_indices = getattr(self, '_reverse_selected_indices', list(range(len(full_chunks))))
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
            self._reverse_prev_tasks_order = []
            self._reverse_basis_list = []
            self._reverse_selected_order = []
            self._reverse_selected_indices = []

        # Compute and store average gradient direction of the learned prompt on validation data
        try:
            self.compute_prompt_gradient_direction(dataloader_val, task, progressive=progressive, max_batches=10)
        except Exception as e:
            print('Warning: prompt grad direction computation failed:', e)
        # Finalize and store epoch-2 current prompt direction; then compute similarities
        try:
            if not self._epoch2_done.get(task, False):
                if task in self._epoch2_accum and self._epoch2_accum[task]['count'] > 0:
                    self.finalize_current_prompt_grad_direction(task)
        except Exception as e:
            print('Warning: finalize epoch-2 prompt grad direction failed:', e)
        try:
            if not self._epoch2_done.get(task, False):
                self.compute_epoch2_subspace_similarities(task)
        except Exception as e:
            print('Warning: epoch-2 subspace similarity computation failed:', e)
        try:
            if not self._epoch2_done.get(task, False):
                self.compute_epoch2_pairwise_direction_similarities(task)
        except Exception as e:
            print('Warning: epoch-2 pairwise direction similarity computation failed:', e)
        try:
            if not self._epoch2_done.get(task, False):
                self.compute_epoch2_combined_similarity(task)
                # Print top-7 similar tasks based on combined similarity
                self.print_topk_epoch2_similar_tasks(task, k=7)
                self._epoch2_done[task] = True
        except Exception as e:
            print('Warning: epoch-2 combined similarity computation failed:', e)

        # After finishing task: print per-task prompt norms from original and updated queues
        try:
            # reconstruct previous tasks order (newest-first)
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
        return val_acc

    def _format_score(self, score):
        if isinstance(score, (list, tuple, np.ndarray)):
            try:
                return float(np.mean(score))
            except Exception:
                return score
        try:
            return float(score)
        except Exception:
            return score

    def _print_epoch_score(self, task, epoch, score):
        score_out = self._format_score(score)
        best_out = self._format_score(getattr(self, 'best_acc', score_out))
        if isinstance(score_out, float):
            print(f"[EpochScore] task={task} epoch={epoch} val={score_out:.6f} best={best_out:.6f}")
        else:
            print(f"[EpochScore] task={task} epoch={epoch} val={score_out} best={best_out}")

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
                if progressive:
                    curr_prompt = torch.tensor(self.previous_prompts, requires_grad=False).to(self.device)
                else:
                    if self.prefix_len>0:
                        curr_prompt = self.model.prompt
                    else:
                        curr_prompt = None

                if test_eval_after_every_task:
                    # eval test accuracy for all tasks
                    results_dict['test'][num] = {}
                    # Only evaluate tasks seen so far (up to current index)
                    for test_task in task_list[:num+1]:
                            # Prefer updated task-specific prompt if available; else fallback to progressive prompt
                            used_updated = False
                            prompt_for_eval = None
                            if self.previous_prompts_updated is not None and self.previous_prompts is not None and self.previous_prompts.shape[0] == self.previous_prompts_updated.shape[0]:
                                try:
                                    prev_order_full = [t for t in reversed(task_list[:num]) if t in self.task_prompts]
                                    if test_task in prev_order_full:
                                        idx_full = prev_order_full.index(test_task)
                                        upd_chunks = torch.split(self.previous_prompts_updated.detach().to(self.device), self.prefix_len, dim=0)
                                        if idx_full < len(upd_chunks):
                                            prompt_for_eval = upd_chunks[idx_full]
                                            used_updated = True
                                except Exception:
                                    used_updated = False
                            if used_updated and prompt_for_eval is not None:
                                try:
                                    acc = self.validate(self.tasks_data_dict[test_task]['test'],
                                                        test_task,
                                                        prompt_for_eval,
                                                        self.task_to_target_len[test_task],
                                                        print_outputs=True)
                                    results_dict['test'][num][test_task] = acc
                                    acc_mean = float(np.mean(acc)) if isinstance(acc, (list, tuple, np.ndarray)) else float(acc)
                                    print(f"[TestSinglePromptUpdated] step={num} task={test_task} acc={acc_mean:.4f}")
                                except Exception as e_upd:
                                    print('Warning: updated prompt test eval failed, falling back:', e_upd)
                                    acc = self.validate(self.tasks_data_dict[test_task]['test'],
                                                        test_task,
                                                        curr_prompt,
                                                        self.task_to_target_len[test_task],
                                                        print_outputs=True)
                                    results_dict['test'][num][test_task] = acc
                            else:
                                acc = self.validate(self.tasks_data_dict[test_task]['test'],
                                                    test_task,
                                                    curr_prompt,
                                                    self.task_to_target_len[test_task],
                                                    print_outputs=True)
                                results_dict['test'][num][test_task] = acc

                else:
                    acc = self.validate(self.tasks_data_dict[task]['test'],
                                        task,
                                        curr_prompt,
                                        self.task_to_target_len[task],
                                        print_outputs=True)
                    results_dict['test'][task] = acc
            # saving results dict after each task
            np.save(os.path.join(save_path, 'results_dict.npy'), results_dict)
            # persist prompt gradient direction for this task if available
            if save_path is not None and task in self.prompt_grad_dirs:
                np.save(os.path.join(save_path, f'prompt_grad_{task}.npy'), self.prompt_grad_dirs[task])
            # persist previous prompts' gradient directions wrt current task if available
            if save_path is not None and task in self.prev_prompt_grad_dirs:
                for prev_task, data in self.prev_prompt_grad_dirs[task].items():
                    np.save(os.path.join(save_path, f'prompt_grad_prev_{task}_from_{prev_task}.npy'), data['direction'])
                    # also save the norm as a small txt for readability
                    try:
                        with open(os.path.join(save_path, f'prompt_grad_prev_{task}_from_{prev_task}_norm.txt'), 'w') as f:
                            f.write(str(data.get('norm', '')))
                        # save cosine similarity and angle if available
                        with open(os.path.join(save_path, f'prompt_grad_prev_{task}_from_{prev_task}_similarity.txt'), 'w') as f:
                            f.write(f"cosine={data.get('cosine','')}\nangle_deg={data.get('angle_deg','')}\n")
                    except Exception as e:
                        print('Warning: failed to save grad norm:', e)

        return results_dict





    # Perform multi-task training
    def multi_task_training(self, num_epochs=5, progressive=False, save_path=''):
        tasks_data_dict = self.tasks_data_dict
        val_scores = {x: [] for x in list(tasks_data_dict)}
        # getting index of the largest dataset (other datasets will be cycled)
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
                    #loss = self.trainer.pass_batch(batch, list(tasks_data_dict)[task_num], self.device, cls_idx=cls_idx, only_output_loss=True)
                    if self.prefix_len>0: # prompt tuning
                        loss = self.train_step_lester(batch,
                                                      task=task if self.prefix_MLPs!=None else None,
                                                      progressive=progressive)
                    else:
                        loss = self.train_step(batch)

                    # loss.backward()
                    # self.optimizer.step()
                    # self.optimizer.zero_grad()
                    loss_combined += loss

                loss_combined.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                pbar.update(1)

            #val_scores = self.eval_on_tasks(val_scores, prompt_tuning=False, original_task_id=None)
            #results_dict[epoch] = val_scores

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




