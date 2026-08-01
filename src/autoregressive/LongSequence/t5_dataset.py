import pandas as pd
import numpy as np
import json
import os

if not hasattr(np, "object"):
    np.object = object

from datasets.formatting.formatting import NumpyArrowExtractor

def patched_arrow_array_to_numpy(self, array):
    return np.asarray(array)

NumpyArrowExtractor._arrow_array_to_numpy = patched_arrow_array_to_numpy
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import datasets
from datasets import Dataset
from transformers import AutoTokenizer, AutoModel
import torch


class T5Dataset:
    def __init__(self, tokenizer, task, cache_dir, pre_processed, seed=42):
        """Dataset class for T5 model experiments.
        Args:
            task (str): Name of the downstream task.
            tokenizer (HuggingFace Tokenizer): T5 model tokenizer to use.
        """
        
        self.tokenizer = tokenizer
        self.cache_dir = cache_dir
        self.pre_processed=pre_processed
        self.seed = seed
        self.glue_datasets = ['cola', 'sst2', 'mrpc', 'qqp', 'stsb', 'mnli', \
                              'mnli_mismatched', 'mnli_matched', 'qnli', 'rte', 'wnli', 'ax']
        self.superglue_datasets = ['copa', 'boolq', 'wic', 'wsc', 'cb', 'record', 'multirc', 'rte_superglue', 'wsc_bool']
        
        # Load configurations from JSON files
        self._load_configurations()
    
    def _load_configurations(self):
        """Load task configurations from JSON files."""
        # Get the directory of the current script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Load key mappings
        key_map_path = os.path.join(script_dir, 'key_map.json')
        with open(key_map_path, 'r') as f:
            key_map = json.load(f)
        
        # Load label mappings  
        label_map_path = os.path.join(script_dir, 'label_map.json')
        with open(label_map_path, 'r') as f:
            label_map = json.load(f)
        
        # Convert to the expected format (tuples for keys, handling None values)
        self.task_to_keys = {}
        for task, keys in key_map.items():
            if len(keys) == 1:
                self.task_to_keys[task] = (keys[0], None)
            else:
                self.task_to_keys[task] = tuple(keys)
        
        # Convert to the expected format (tuples for labels)
        self.task_to_labels = {}
        for task, labels in label_map.items():
            self.task_to_labels[task] = tuple(labels)

        self.task = task
        self.label_key = 'label'
        if 'yahoo_' in task: self.label_key = 'topic'
        if 'stsb' in task: self.label_key = 'similarity_score'
        if task=='record': self.label_key = 'answers'

    
    def save_multirc_questions_idx(self, val_ds):
        """Save idx of multirc questions (needed later for test metric computation)"""
        idx = []
        i = 0
        x_prev, y_prev= val_ds['paragraph'][0], val_ds['question'][0]

        for x,y in zip(val_ds['paragraph'], val_ds['question']):
            if x_prev!=x or y_prev!=y:
                i += 1
            x_prev = x
            y_prev = y
            idx.append(i)
        self.multirc_idx = np.array(idx)

    
    def select_subset_ds(self, ds, k=2000, seed=0):
        """Select a subset of k samples per class in a dataset"""
        if self.task in ['stsb', 'record', 'wsc']:
            idx_total = np.random.choice(np.arange(ds.shape[0]), min(k,ds.shape[0]), replace=False)
        else:
            label_key = self.label_key
            N = len(ds[label_key])
            idx_total = np.array([], dtype='int64')

            for l in set(ds[label_key]):
                idx = np.where(np.array(ds[label_key]) == l)[0]
                idx_total = np.concatenate([idx_total, 
                                            np.random.choice(idx, min(k, idx.shape[0]), replace=False)])

        np.random.seed(seed)
        np.random.shuffle(idx_total)
        return ds.select(idx_total)

    def process_wsc(self, wsc_row):
        """WSC task function to preprocess raw input & label text into tokenized dictionary"""
        text_proc = wsc_row['text'].split(' ')
        target = text_proc[wsc_row['span1_index']]
        text_proc[wsc_row['span2_index']] = '*' + text_proc[wsc_row['span2_index']] + '*'
        text_proc = (' ').join(text_proc)
        return text_proc, target

    
    def preprocess_function(self, examples, task, max_length=512, max_length_target=2, prefix_list=[]):
        """Function to preprocess raw input & label text into tokenized dictionary"""
        tokenizer = self.tokenizer
        keys = self.task_to_keys[task]
        label_key = self.label_key

        if keys[1]!=None:
            if task=='record':
                text = 'passage : ' + str(examples['passage']) + ' query: ' + str(examples['query']) + ' entities: ' + ('; ').join((examples['entities']))
            elif task=='wsc':
                text, target = self.process_wsc(examples)
            else:
                text = ''
                for key in keys:
                    text += key + ': ' + str(examples[key]) + ' '
        else:
            text = examples[keys[0]]

        if len(prefix_list)>0:
            text = (' ').join(prefix_list) + ' ' + text
        source = tokenizer(text.strip()+' </s>', truncation=True, padding='max_length', max_length=max_length)

        if task=='stsb':
            target = str(examples[label_key])[:3]
        elif task=='record':
            target = '; '.join(examples[label_key])
        elif task=='wsc':
            pass
        else:
            target = self.task_to_labels[task][examples[label_key]]
        target += ' </s>'
        target = tokenizer(target, truncation=True, padding='max_length', max_length=max_length_target)

        dict_final = {"source_ids": source['input_ids'],
                      "source_mask": source['attention_mask'],
                      "target_ids": target['input_ids'],
                      "target_mask": target['attention_mask']}
        return dict_final


    
    def get_final_ds(self, task, split, batch_size, k=-1, seed=0, return_test=False,
                     target_len=2, max_length=512, prefix_list=[]):
        """Function that returns final T5 dataloader."""
        cache_dir = self.cache_dir
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        src_root = os.path.dirname(os.path.dirname(script_dir))
        fixed_train_path = os.path.join(src_root, "data", "train")
        fixed_test_path = os.path.join(src_root, "data", "test")

        fixed_train_files = {
            "imdb": "imdb_1k.npy",
            "sst2": "sst2_1k.npy",
            "yelp_review_full": "yelp_1k.npy",
            "amazon": "amazon_1k.npy",
            "rte": "rte_1k.npy",
            "wic": "wic_1k.npy",
            "multirc": "multirc_1k.npy",
            "boolq": "boolq_1k.npy",
            "ag_news": "ag_news_1k.npy",
            "dbpedia_14": "dbpedia_14_1k.npy",
            "qqp": "qqp_1k.npy",
            "mnli": "mnli_1k.npy",
            "yahoo_answers_topics": "yahoo_1k.npy",
        }
        fixed_eval_files = {
            "amazon": "amazon.npy",
            "mnli": "mnli_test.npy",
            "wic": "super_glue_wic.npy",
            "multirc": "super_glue_multirc.npy",
            "sst2": "sst2.npy",
            "boolq": "super_glue_boolq.npy",
            "yelp_review_full": "yelp_review_full.npy",
            "imdb": "imdb.npy",
            "qqp": "glue_qqp.npy",
            "dbpedia_14": "dbpedia_14.npy",
            "ag_news": "ag_news.npy",
            "yahoo_answers_topics": "yahoo_answers_topics.npy",
        }

        dataset = None
        if self.pre_processed and split == 'train' and task in fixed_train_files:
            local_path = os.path.join(fixed_train_path, fixed_train_files[task])
            if os.path.exists(local_path):
                dataset = Dataset.from_dict(np.load(local_path, allow_pickle=True).item())
        if dataset is None and split != 'train' and task in fixed_eval_files:
            local_path = os.path.join(fixed_test_path, fixed_eval_files[task])
            if os.path.exists(local_path):
                dataset = Dataset.from_dict(np.load(local_path, allow_pickle=True).item())

        if dataset is not None:
            pass
        elif task in ['amazon']:
            df = pd.read_csv(os.path.join(src_root, "data", task, f"{split}.csv"), header=None)
            df = df.rename(columns={0: "label", 1: "title", 2: "content"})
            df['label'] = df['label'] - 1
            dataset = datasets.Dataset.from_pandas(df)
        elif task == 'mnli':
            dataset = load_dataset('LysandreJik/glue-mnli-train', split=split, cache_dir=cache_dir)
        elif task == 'qnli':
            dataset = load_dataset('SetFit/qnli', split=split, cache_dir=cache_dir)
        elif task == 'stsb':
            dataset = load_dataset('stsb_multi_mt', name='en', split=split if split=='train' else 'dev', cache_dir=cache_dir)
        else:
            if task not in self.glue_datasets and task not in self.superglue_datasets:
                dataset = load_dataset(task, split=split, cache_dir=cache_dir)
            else:
                benchmark = 'glue' if task not in self.superglue_datasets else 'super_glue'
                dataset = load_dataset(benchmark, task.replace('_superglue', '').replace('_bool', ''),
                                       split=split, cache_dir=cache_dir)

        flag = True
        
        if self.pre_processed and split=='train':
            if task == 'imdb':
                loaded_train_dict = np.load(os.path.join(fixed_train_path, "imdb_1k.npy"), allow_pickle=True).item()
            elif task == 'sst2':
                loaded_train_dict = np.load(os.path.join(fixed_train_path, "sst2_1k.npy"), allow_pickle=True).item()
            elif task == 'yelp_review_full':
                loaded_train_dict = np.load(os.path.join(fixed_train_path, "yelp_1k.npy"), allow_pickle=True).item()
            elif task == 'amazon':
                loaded_train_dict = np.load(os.path.join(fixed_train_path, "amazon_1k.npy"), allow_pickle=True).item()
            elif task == 'rte':
                loaded_train_dict = np.load(os.path.join(fixed_train_path, "rte_1k.npy"), allow_pickle=True).item()
            elif task == 'wic':
                loaded_train_dict = np.load(os.path.join(fixed_train_path, "wic_1k.npy"), allow_pickle=True).item()
            elif task == 'multirc':
                loaded_train_dict = np.load(os.path.join(fixed_train_path, "multirc_1k.npy"), allow_pickle=True).item()
            elif task == 'boolq':
                loaded_train_dict = np.load(os.path.join(fixed_train_path, "boolq_1k.npy"), allow_pickle=True).item()
            elif task == 'ag_news':
                loaded_train_dict = np.load(os.path.join(fixed_train_path, "ag_news_1k.npy"), allow_pickle=True).item()
            elif task == 'dbpedia_14':
                loaded_train_dict = np.load(os.path.join(fixed_train_path, "dbpedia_14_1k.npy"), allow_pickle=True).item()
            elif task == 'qqp':
                loaded_train_dict = np.load(os.path.join(fixed_train_path, "qqp_1k.npy"), allow_pickle=True).item()
            elif task == 'mnli':
                loaded_train_dict = np.load(os.path.join(fixed_train_path, "mnli_1k.npy"), allow_pickle=True).item()
            elif task == 'yahoo_answers_topics':
                loaded_train_dict = np.load(os.path.join(fixed_train_path, "yahoo_1k.npy"), allow_pickle=True).item()
            
            if task == 'cb' or task == "copa":
                flag = False

            if flag:
                dataset = Dataset.from_dict(loaded_train_dict)

        if split=='test':
            if task=='amazon':
                loaded_dict = np.load(os.path.join(fixed_test_path, "amazon.npy"), allow_pickle=True).item()
            elif task=='mnli':
                loaded_dict = np.load(os.path.join(fixed_test_path, "mnli_test.npy"), allow_pickle=True).item()
            elif task=='wic':
                loaded_dict = np.load(os.path.join(fixed_test_path, "super_glue_wic.npy"), allow_pickle=True).item()
            elif task=='multirc':
                loaded_dict = np.load(os.path.join(fixed_test_path, "super_glue_multirc.npy"), allow_pickle=True).item()
            elif task=='sst2':
                loaded_dict = np.load(os.path.join(fixed_test_path, "sst2.npy"), allow_pickle=True).item()
            elif task=='boolq':
                loaded_dict = np.load(os.path.join(fixed_test_path, "super_glue_boolq.npy"), allow_pickle=True).item()
            elif task=='yelp_review_full':
                loaded_dict = np.load(os.path.join(fixed_test_path, "yelp_review_full.npy"), allow_pickle=True).item()
            elif task=='imdb':
                loaded_dict = np.load(os.path.join(fixed_test_path, "imdb.npy"), allow_pickle=True).item()
            elif task=='qqp':
                loaded_dict = np.load(os.path.join(fixed_test_path, "glue_qqp.npy"), allow_pickle=True).item()
            elif task=='dbpedia_14':
                loaded_dict = np.load(os.path.join(fixed_test_path, "dbpedia_14.npy"), allow_pickle=True).item()
            elif task=='ag_news':
                loaded_dict = np.load(os.path.join(fixed_test_path, "ag_news.npy"), allow_pickle=True).item()
            elif task=='yahoo_answers_topics':
                loaded_dict = np.load(os.path.join(fixed_test_path, "yahoo_answers_topics.npy"), allow_pickle=True).item()

            dataset = Dataset.from_dict(loaded_dict)
        
        if self.task == 'wsc': 
            idx = np.where(np.array(dataset['label']) == 1)[0]
            dataset = dataset.select(idx)
        
        if k!=-1:
            dataset = self.select_subset_ds(dataset, k=k)

        if k==-1 and split!='train' and self.task=='multirc':
            self.save_multirc_questions_idx(dataset)
        else:
            dataset = dataset.shuffle(seed=seed)
        
        if return_test==False:
            encoded_dataset = dataset.map(lambda x: self.preprocess_function(x, task,
                                                                            max_length=max_length,
                                                                            max_length_target=target_len,
                                                                            prefix_list=prefix_list),
                                          batched=False)
            encoded_dataset.set_format(type='torch', columns=['source_ids', 'source_mask',
                                                              'target_ids', 'target_mask'])
            dataloader = DataLoader(encoded_dataset, batch_size=batch_size)
            return dataloader
        
        else:
            N = len(dataset)
            dataset_val = dataset.select(np.arange(0, N//2))
            dataset_test = dataset.select(np.arange(N//2, N))

            dataloaders_val_test = []
            for dataset in [dataset_val, dataset_test]:
                encoded_dataset = dataset.map(lambda x: self.preprocess_function(x, task,
                                                                                 max_length=max_length,
                                                                                 max_length_target=target_len,
                                                                                 prefix_list=prefix_list),
                                              batched=False)
                encoded_dataset.set_format(type='torch', columns=['source_ids', 'source_mask',
                                                                  'target_ids', 'target_mask'])
                dataloader = DataLoader(encoded_dataset, batch_size=batch_size)
                dataloaders_val_test.append(dataloader)

            return dataloaders_val_test
