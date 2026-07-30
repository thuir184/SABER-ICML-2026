import json
import math
import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset, DatasetDict
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, default_data_collator

DEVICE = os.environ.get("PD_DEVICE", None)
if DEVICE is None:
    if torch.cuda.is_available():
        DEVICE = "cuda"
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        DEVICE = "mps"
    else:
        DEVICE = "cpu"

_MAPPING_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mapping_dict.json")
with open(_MAPPING_PATH, "r") as _f:
    _M = json.load(_f)
GLUE_DATASETS = _M["GLUE_DATASETS"]
SUPERGLUE_DATASETS = _M["SUPERGLUE_DATASETS"]
TASK_TO_KEYS = {k: tuple(v) for k, v in _M["TASK_TO_KEYS"].items()}
TASK_TO_LABELS = {k: tuple(v) for k, v in _M["TASK_TO_LABELS"].items()}
TEST_FIXED_FILES = _M["TEST_FIXED_FILES"]


def load_task_dataset(
    task_name: str,
    cache_dir: Optional[str] = None,
    data_root: str = "./datasets/src/data",
    superni_dir: str = "./SuperNI",
):
    if task_name.startswith("superni:"):
        import json as _json
        from datasets import Dataset as _HFDataset, DatasetDict as _HFDict, ClassLabel, Features, Value

        fname = task_name.split("superni", 1)[1].lstrip(":")
        dir_path = os.path.join(superni_dir, fname)
        if not os.path.isdir(dir_path):
            raise FileNotFoundError(
                f"SuperNI task '{fname}': expected directory at '{dir_path}' "
                "with train.json and dev/validation/test.json."
            )
        split_files = {
            "train": os.path.join(dir_path, "train.json"),
            "dev": os.path.join(dir_path, "dev.json"),
            "validation": os.path.join(dir_path, "validation.json"),
            "test": os.path.join(dir_path, "test.json"),
        }
        loaded: Dict[str, dict] = {}
        for sk, fp in sorted(split_files.items()):
            if os.path.exists(fp):
                with open(fp, "r") as fh:
                    loaded[sk] = _json.load(fh)
        if "train" not in loaded:
            raise ValueError(f"SuperNI task at '{dir_path}' must include train.json")

        first_out = None
        for sk in ["train", "dev", "validation", "test"]:
            obj = loaded.get(sk)
            if obj and obj.get("Instances"):
                first_out = obj["Instances"][0].get("output")
                break
        is_generation = isinstance(first_out, list)
        label_column = None
        label_to_id: Dict[str, int] = {}
        if is_generation:
            features = Features({
                "input": Value("string"),
                "target": Value("string"),
                "definition": Value("string"),
            })
        else:
            labels_set = set()
            for obj in loaded.values():
                for ex in obj.get("Instances", []):
                    if "output" in ex:
                        labels_set.add(str(ex["output"]))
            label_names = sorted(labels_set)
            label_to_id = {lab: i for i, lab in enumerate(label_names)}
            features = Features({
                "input": Value("string"),
                "label": ClassLabel(names=label_names),
                "definition": Value("string"),
            })
            label_column = "label"
            TASK_TO_LABELS[task_name] = tuple(label_names)

        def _build_split(json_obj: dict):
            definition = ""
            if isinstance(json_obj.get("Definition"), list) and json_obj["Definition"]:
                definition = json_obj["Definition"][0]
            inputs, definitions = [], []
            labels_or_targets = []
            for ex in json_obj.get("Instances", []):
                inputs.append(ex.get("input", ""))
                definitions.append(definition)
                outv = ex.get("output", "")
                if is_generation:
                    t = outv[0] if isinstance(outv, list) and len(outv) > 0 else str(outv)
                    labels_or_targets.append(t)
                else:
                    labels_or_targets.append(label_to_id[str(outv)])
            d = {"input": inputs, "definition": definitions}
            d["target" if is_generation else "label"] = labels_or_targets
            return _HFDataset.from_dict(d, features=features)

        ds_map = {}
        for sk, obj in loaded.items():
            ds = _build_split(obj)
            key = "validation" if sk == "dev" else sk
            ds_map[key] = ds
        dataset = _HFDict(ds_map)
        eval_split = "validation" if "validation" in ds_map else ("test" if "test" in ds_map else "train")
        dataset.info = {"superni_task_mode": "generation" if is_generation else "classification"}
        return dataset, (None if is_generation else "label"), eval_split, "superni"

    if task_name == "amazon":
        def _load(split_name: str) -> Dataset:
            csv_path = os.path.join(data_root, "amazon", f"{split_name}.csv")
            df = pd.read_csv(csv_path, header=None)
            df = df.rename(columns={0: "label", 1: "title", 2: "content"})
            df["label"] = df["label"] - 1
            return Dataset.from_pandas(df)

        dataset = DatasetDict({
            "train": _load("train"),
            "validation": _load("test"),
            "test": _load("test"),
        })
        return dataset, "label", "validation", "amazon"

    if task_name in GLUE_DATASETS:
        if task_name == "ax":
            raise ValueError("GLUE 'ax' has no training labels.")
        if task_name in ["mnli", "mnli_matched", "mnli_mismatched"]:
            train_ds = load_dataset("LysandreJik/glue-mnli-train", split="train", cache_dir=cache_dir)
            glue_mnli = load_dataset("glue", "mnli", cache_dir=cache_dir)
            dataset = DatasetDict({
                "train": train_ds,
                "validation_matched": glue_mnli["validation_matched"],
                "validation_mismatched": glue_mnli["validation_mismatched"],
            })
            eval_split = "validation_mismatched" if task_name == "mnli_mismatched" else "validation_matched"
            return dataset, "label", eval_split, "mnli"
        if task_name == "qnli":
            train_ds = load_dataset("SetFit/qnli", split="train", cache_dir=cache_dir)
            glue_qnli = load_dataset("glue", "qnli", cache_dir=cache_dir)
            dataset = DatasetDict({"train": train_ds, "validation": glue_qnli["validation"]})
            return dataset, "label", "validation", "qnli"
        dataset = load_dataset("glue", task_name, cache_dir=cache_dir)
        return dataset, "label", "validation", task_name

    if task_name in SUPERGLUE_DATASETS:
        subset_map = {"rte_superglue": "rte", "wsc_bool": "wsc"}
        subset = subset_map.get(task_name, task_name)
        if task_name == "stsb":
            ds = load_dataset("stsb_multi_mt", name="en", cache_dir=cache_dir)
            dataset = DatasetDict({
                "train": ds["train"],
                "validation": ds["dev"],
                "test": ds["test"] if "test" in ds else ds["dev"],
            })
            return dataset, "label", "validation", "stsb"
        dataset = load_dataset("super_glue", subset, cache_dir=cache_dir)
        return dataset, "label", "validation", task_name

    dataset = load_dataset(task_name, cache_dir=cache_dir)
    if "label" not in dataset["train"].column_names:
        raise ValueError(f"Cannot infer label column for {task_name}")
    eval_split = "test" if "test" in dataset else "validation"
    return dataset, "label", eval_split, task_name


def balanced_subsample(split: Dataset, label_column: str, k: int, seed: int = 42) -> Dataset:
    random.seed(seed)
    labels = split[label_column]
    unique = sorted(set(labels))
    label_to_idx = {lbl: [] for lbl in unique}
    for idx, lbl in enumerate(labels):
        label_to_idx[lbl].append(idx)
    selected = []
    for lbl, idxs in label_to_idx.items():
        random.shuffle(idxs)
        selected.extend(idxs[:k])
    return split.select(sorted(selected))


def build_prompt_from_example(
    example: Dict,
    task_name: str,
    keys: Tuple[str, ...],
    tokenizer=None,
) -> str:
    prompt_task = "mnli" if task_name in ["mnli_matched", "mnli_mismatched"] else task_name
    keys_for_task = TASK_TO_KEYS.get(prompt_task)
    if keys_for_task is None:
        parts = [f"{k}: {v}" for k, v in example.items() if isinstance(v, str)]
        return "\n".join(parts) + "\nLabel:"

    parts = []
    if prompt_task == "superni":
        for k in keys_for_task:
            if k is None:
                continue
            if k in example:
                parts.append(f"{k}: {example[k]}")
        raw = "\n".join(parts) + "\nOutput:"
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Follow the task definition and produce the correct output."},
            {"role": "user", "content": raw},
        ]
        if tokenizer is None or not hasattr(tokenizer, "apply_chat_template"):
            return raw
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    for k in keys_for_task:
        if k is None:
            continue
        if k in example:
            parts.append(f"{k}: {example[k]}")
    if not parts and "text" in example:
        parts = [f"text: {example['text']}"]
    return "\n".join(parts) + "\nLabel:"


def build_label_token_seqs(label_texts: Tuple[str, ...], tokenizer) -> List[List[int]]:
    return [tokenizer(lbl, add_special_tokens=False)["input_ids"] for lbl in label_texts]


class ProgressiveDecoderPromptTrainer:
    def __init__(
        self,
        base_model_name: str,
        task_list: List[str],
        prefix_len: int = 10,
        max_length: int = 1024,
        lr: float = 3e-2,
        batch_size: int = 8,
        k_per_class: int = 2000,
        num_epochs: int = 10,
        seed: int = 42,
        cache_dir: Optional[str] = None,
        fix_test_data: bool = True,
        test_fixed_dir: str = "./datasets/test_fixed",
        data_root: str = "./datasets/src/data",
        superni_dir: str = "./SuperNI",
        pred_mode: str = "logprob",
        gen_max_new_tokens: int = 5,
    ):
        self.base_model_name = base_model_name
        self.task_list = task_list
        self.prefix_len = prefix_len
        self.max_length = max_length
        self.lr = lr
        self.batch_size = batch_size
        self.k_per_class = k_per_class
        self.num_epochs = num_epochs
        self.seed = seed
        self.cache_dir = cache_dir
        self.fix_test_data = fix_test_data
        self.test_fixed_dir = test_fixed_dir
        self.data_root = data_root
        self.superni_dir = superni_dir
        self.pred_mode = pred_mode
        self.gen_max_new_tokens = gen_max_new_tokens

        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True, use_fast=True)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token if self.tokenizer.eos_token else "<|pad|>"
            if self.tokenizer.pad_token == "<|pad|>":
                self.tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

        self.model = AutoModelForCausalLM.from_pretrained(base_model_name, trust_remote_code=True)
        if len(self.tokenizer) > self.model.get_input_embeddings().num_embeddings:
            self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.config.eos_token_id = self.tokenizer.eos_token_id
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.to(DEVICE)

        self.hidden_size = self.model.get_input_embeddings().embedding_dim
        self.current_prompt: Optional[nn.Parameter] = None
        self.previous_prompts = torch.empty(0, self.hidden_size, device=DEVICE)
        self.task_meta: Dict[str, Dict] = {}
        self._prepare_task_meta()
        self.task_to_acc: Dict[str, float] = {}
        self.per_epoch_acc: Dict = {}
        # Reverse-phase state and similarity caches (parity with t5_decoder.py)
        self.reverse_phase_active: bool = False
        self.previous_prompts_param: Optional[nn.Parameter] = None
        self._reverse_basis_list: List[torch.Tensor] = []
        self._reverse_selected_order: List[str] = []
        self._reverse_selected_indices: List[int] = []
        self.prompt_grad_dirs: Dict[str, np.ndarray] = {}
        self.prev_prompt_grad_dirs: Dict[str, Dict[str, Dict]] = {}
        self.task_to_svd_basis: Dict[str, np.ndarray] = {}
        self.cumulative_restrict_basis: Dict[str, np.ndarray] = {}

    def _maybe_load_fixed_test(self, task_name: str):
        if not self.fix_test_data:
            return None
        fname = TEST_FIXED_FILES.get(task_name)
        if fname is None:
            return None
        path = os.path.join(self.test_fixed_dir, fname)
        if not os.path.exists(path):
            return None
        arr = np.load(path, allow_pickle=True)
        data = arr if isinstance(arr, dict) else arr.item()
        return Dataset.from_dict(data)

    def _prepare_task_meta(self):
        for task_name in self.task_list:
            dataset, label_column, eval_split, canonical = load_task_dataset(
                task_name, cache_dir=self.cache_dir, data_root=self.data_root, superni_dir=self.superni_dir,
            )
            fixed = self._maybe_load_fixed_test(task_name)
            if fixed is not None:
                dataset = DatasetDict({**dataset, "fixed_test": fixed})
                eval_split = "fixed_test"

            labels_key = "mnli" if task_name in ["mnli_matched", "mnli_mismatched"] else task_name
            mode = getattr(dataset, "info", {}).get("superni_task_mode", None)
            is_gen = canonical == "superni" and mode == "generation"
            label_texts = TASK_TO_LABELS.get(labels_key, ())
            if not label_texts and not is_gen:
                raise ValueError(f"No label texts for {task_name} (key={labels_key}).")
            keys = TASK_TO_KEYS.get(canonical, ())
            if canonical in ["mnli", "mnli_matched", "mnli_mismatched"]:
                keys = TASK_TO_KEYS.get("mnli", ())
            label_seqs = build_label_token_seqs(label_texts, self.tokenizer) if not is_gen else []

            self.task_meta[task_name] = {
                "dataset": dataset,
                "label_column": label_column,
                "eval_split": eval_split,
                "canonical_for_prompts": canonical,
                "label_texts": label_texts,
                "label_token_seqs": label_seqs,
                "keys": keys,
                "is_generation": is_gen,
            }

    def _init_prompt_from_text(self, label_texts: Tuple[str, ...]) -> torch.Tensor:
        txt = f"Classify the text into one of: {', '.join(label_texts)}"
        ids = self.tokenizer(txt, add_special_tokens=False)["input_ids"]
        ids_t = torch.tensor(ids, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            emb = self.model.get_input_embeddings()(ids_t)[0]
        if emb.size(0) >= self.prefix_len:
            return emb[: self.prefix_len].detach().clone()
        reps = math.ceil(self.prefix_len / emb.size(0))
        return emb.repeat(reps, 1)[: self.prefix_len].detach().clone()

    def _start_new_task_prompt(self, label_texts: Tuple[str, ...]):
        if self.current_prompt is not None:
            prev = self.current_prompt.detach().to(DEVICE)
            prev.requires_grad = False
            self.previous_prompts = prev if self.previous_prompts.numel() == 0 else torch.cat([prev, self.previous_prompts], dim=0)
        init = self._init_prompt_from_text(label_texts)
        self.current_prompt = nn.Parameter(init.to(DEVICE), requires_grad=True)

    def _get_task_prompt(self) -> torch.Tensor:
        if self.current_prompt is None:
            return self.previous_prompts
        if self.previous_prompts.numel() == 0:
            return self.current_prompt
        return torch.cat([self.previous_prompts, self.current_prompt], dim=0)

    def get_prompts_numpy(self, include_current: bool = True) -> np.ndarray:
        with torch.no_grad():
            prev = self.previous_prompts.detach().cpu()
            if include_current and self.current_prompt is not None:
                curr = self.current_prompt.detach().cpu()
                all_p = curr if prev.numel() == 0 else torch.cat([curr, prev], dim=0)
            else:
                all_p = prev
        return all_p.numpy()

    def _num_previous_prompt_blocks(self) -> int:
        if self.previous_prompts.numel() == 0:
            return 0
        return int(self.previous_prompts.size(0) // self.prefix_len)

    # pruning removed: keep all previous prompts unchanged

    def _build_preprocess_fn(
        self,
        label_texts: Tuple[str, ...],
        label_column: Optional[str],
        canonical: str,
        keys: Tuple[str, ...],
        target_max_length: int,
    ):
        max_len = self.max_length
        tok = self.tokenizer
        input_max = max_len - (target_max_length + 1)
        if input_max <= 0:
            raise ValueError("max_length too small for label set.")

        def fn(examples):
            n = len(examples[label_column]) if label_column else len(examples["target"])
            texts = []
            for i in range(n):
                ex = {k: examples[k][i] for k in examples}
                texts.append(build_prompt_from_example(ex, canonical, keys, tokenizer=tok))
            if label_texts and label_column is not None:
                targets = [label_texts[examples[label_column][i]] for i in range(n)]
            else:
                targets = examples.get("target", [])
            model_in = tok(texts, truncation=True, max_length=input_max, padding=False)
            lab_enc = tok(targets, add_special_tokens=False, truncation=True, max_length=target_max_length, padding=False)
            out_ids, out_attn, out_lab = [], [], []
            for iid, attn, lab in zip(model_in["input_ids"], model_in["attention_mask"], lab_enc["input_ids"]):
                lab = lab + [tok.pad_token_id]
                full_id = iid + lab
                full_attn = attn + [1] * len(lab)
                full_lab = [-100] * len(iid) + lab
                full_id, full_attn, full_lab = full_id[:max_len], full_attn[:max_len], full_lab[:max_len]
                pad_len = max_len - len(full_id)
                if pad_len > 0:
                    full_id += [tok.pad_token_id] * pad_len
                    full_attn += [0] * pad_len
                    full_lab += [-100] * pad_len
                out_ids.append(full_id)
                out_attn.append(full_attn)
                out_lab.append(full_lab)
            return {"input_ids": out_ids, "attention_mask": out_attn, "labels": out_lab}

        return fn

    def _train_step(self, batch) -> torch.Tensor:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        bsz = input_ids.size(0)
        task_p = self._get_task_prompt()
        plen = task_p.size(0)
        with torch.no_grad():
            token_embeds = self.model.get_input_embeddings()(input_ids)
        prompt_b = task_p.unsqueeze(0).expand(bsz, -1, -1)
        inputs_embeds = torch.cat([prompt_b, token_embeds], dim=1)
        prefix_m = torch.ones(bsz, plen, device=DEVICE, dtype=attention_mask.dtype)
        attn_ext = torch.cat([prefix_m, attention_mask], dim=1)
        ignore = torch.full((bsz, plen), -100, device=DEVICE, dtype=labels.dtype)
        labels_ext = torch.cat([ignore, labels], dim=1)
        return self.model(inputs_embeds=inputs_embeds, attention_mask=attn_ext, labels=labels_ext).loss

    def build_superni_chat_prompt(self, example: Dict, keys: Tuple[str, ...]) -> str:
        parts = [f"{k}: {example[k]}" for k in keys if k in example]
        raw = "\n".join(parts) + "\nOutput:"
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Follow the task definition and produce the correct output."},
            {"role": "user", "content": raw},
        ]
        if not hasattr(self.tokenizer, "apply_chat_template"):
            return raw
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    @staticmethod
    def _normalize_text(s: str) -> str:
        return " ".join(s.strip().lower().split())

    @staticmethod
    def _map_generated_to_label(gen_text: str, label_texts: Tuple[str, ...]) -> Optional[str]:
        g = ProgressiveDecoderPromptTrainer._normalize_text(gen_text)
        for lbl in label_texts:
            if ProgressiveDecoderPromptTrainer._normalize_text(lbl) == g:
                return lbl
        return None

    @torch.no_grad()
    def _score_label_for_example(
        self,
        example: Dict,
        label_text: str,
        canonical: str,
        keys: Tuple[str, ...],
    ) -> float:
        self.model.eval()
        if canonical == "superni":
            prompt_text = self.build_superni_chat_prompt(example, keys)
        else:
            prompt_text = build_prompt_from_example(example, canonical, keys, tokenizer=self.tokenizer)
        prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        label_ids = self.tokenizer(label_text, add_special_tokens=False)["input_ids"]
        full = prompt_ids + label_ids
        input_ids = torch.tensor(full, device=DEVICE).unsqueeze(0)
        seq_len = input_ids.size(1)
        task_p = self._get_task_prompt()
        plen = task_p.size(0)
        token_embeds = self.model.get_input_embeddings()(input_ids)
        prompt_b = task_p.unsqueeze(0)
        inputs_embeds = torch.cat([prompt_b, token_embeds], dim=1)
        attn = torch.ones(1, plen + seq_len, device=DEVICE, dtype=torch.long)
        out = self.model(inputs_embeds=inputs_embeds, attention_mask=attn, use_cache=False)
        log_probs = torch.log_softmax(out.logits, dim=-1)
        base = plen + len(prompt_ids)
        logp = sum(log_probs[0, base + j - 1, label_ids[j]].item() for j in range(len(label_ids)))
        return logp / max(len(label_ids), 1)

    @torch.no_grad()
    def _predict_label_for_example(
        self,
        example: Dict,
        label_texts: Tuple[str, ...],
        canonical: str,
        keys: Tuple[str, ...],
    ) -> str:
        if self.pred_mode == "generate":
            return self._predict_label_for_example_generate(example, label_texts, canonical, keys, strict=False)
        scores = [self._score_label_for_example(example, lbl, canonical, keys) for lbl in label_texts]
        return label_texts[int(torch.tensor(scores).argmax().item())]

    @torch.no_grad()
    def _predict_label_for_example_generate(
        self,
        example: Dict,
        label_texts: Tuple[str, ...],
        canonical: str,
        keys: Tuple[str, ...],
        strict: bool = False,
    ) -> str:
        self.model.eval()
        prompt_text = build_prompt_from_example(example, canonical, keys, tokenizer=self.tokenizer)
        if not prompt_text.endswith(" "):
            prompt_text = prompt_text + " "
        prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        task_p = self._get_task_prompt()
        P = task_p.size(0)
        max_pos = int(getattr(self.model.config, "max_position_embeddings", None) or getattr(self.model.config, "n_positions", 10**9))
        max_prompt = max_pos - P - self.gen_max_new_tokens
        if max_prompt <= 0:
            return ""
        if len(prompt_ids) > max_prompt:
            prompt_ids = prompt_ids[-max_prompt:]
        input_ids = torch.tensor(prompt_ids, device=DEVICE).unsqueeze(0)
        token_embeds = self.model.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([task_p.unsqueeze(0), token_embeds], dim=1)
        attn = torch.ones(1, inputs_embeds.size(1), device=DEVICE, dtype=torch.long)
        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        gen_ids = self.model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attn,
            max_new_tokens=self.gen_max_new_tokens,
            do_sample=False,
            pad_token_id=pad_id,
            eos_token_id=self.tokenizer.eos_token_id,
            use_cache=True,
        )[0]
        text_len = input_ids.size(1)
        new_tokens = gen_ids[text_len:]
        gen_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        if strict:
            for lbl in label_texts:
                if self._normalize_text(gen_text) == self._normalize_text(lbl):
                    return lbl
            return gen_text
        mapped = self._map_generated_to_label(gen_text, label_texts)
        return mapped if mapped is not None else gen_text

    @torch.no_grad()
    def evaluate_task(
        self,
        task_name: str,
        split: str = "eval",
        max_samples: Optional[int] = None,
    ):
        meta = self.task_meta[task_name]
        dataset = meta["dataset"]
        label_column = meta["label_column"]
        label_texts = meta["label_texts"]
        canonical = meta["canonical_for_prompts"]
        keys = meta["keys"]
        is_gen = meta.get("is_generation", False)

        if split == "eval":
            split_name = meta["eval_split"]
        elif split == "train":
            split_name = "train"
        else:
            split_name = split
        eval_ds = dataset[split_name]

        if is_gen:
            return self._evaluate_task_generation(eval_ds, canonical, keys, max_samples=max_samples)

        if max_samples is not None:
            n = min(max_samples, len(eval_ds))
            eval_ds = eval_ds.select(range(n))
        correct, total = 0, 0
        for idx in tqdm(range(len(eval_ds)), desc=f"Eval [{task_name}]"):
            ex = eval_ds[idx]
            gold_id = ex[label_column]
            gold = label_texts[gold_id]
            pred = self._predict_label_for_example(ex, label_texts, canonical, keys)
            total += 1
            if pred == gold:
                correct += 1
        return correct / total if total > 0 else 0.0

    @torch.no_grad()
    def _evaluate_task_generation(
        self,
        eval_ds: Dataset,
        canonical: str,
        keys: Tuple[str, ...],
        max_samples: Optional[int] = None,
        max_new_tokens: int = 24,
        print_every: int = 50,
    ) -> Dict[str, float]:
        try:
            from rouge_score import rouge_scorer
            rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        except Exception:
            rouge = None
        try:
            import sacrebleu
        except Exception:
            sacrebleu = None

        if max_samples is not None:
            n = min(max_samples, len(eval_ds))
            eval_ds = eval_ds.select(range(n))

        r1s, r2s, rLs = [], [], []
        preds = []
        refs_list = []
        self.model.eval()
        max_pos = int(getattr(self.model.config, "max_position_embeddings", None) or 1024)
        max_new = getattr(self, "gen_max_new_tokens", max_new_tokens)

        for idx in tqdm(range(len(eval_ds)), desc="Eval [generation]"):
            ex = eval_ds[idx]
            gold = ex.get("target", "")
            if isinstance(gold, list):
                gold_refs = [str(x) for x in gold if x is not None and str(x).strip()]
                gold_refs = gold_refs if gold_refs else [""]
            else:
                gold_refs = [str(gold)]

            if canonical == "superni":
                prompt_text = self.build_superni_chat_prompt(ex, keys)
            else:
                prompt_text = build_prompt_from_example(ex, canonical, keys, tokenizer=self.tokenizer)

            cap = min(self.max_length, max_pos - 1)
            trunc_side = getattr(self.tokenizer, "truncation_side", "right")
            try:
                self.tokenizer.truncation_side = "left"
                enc = self.tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=cap)
            finally:
                self.tokenizer.truncation_side = trunc_side
            enc = {k: v.to(DEVICE) for k, v in enc.items()}
            input_ids = enc["input_ids"]
            attn_mask = enc.get("attention_mask")

            task_p = self._get_task_prompt()
            P = int(task_p.size(0))
            L = int(input_ids.size(1))
            allowed = max_pos - (P + L)
            if allowed <= 0:
                reserve = 1
                max_L = max_pos - P - reserve
                if max_L <= 0:
                    pred_text = ""
                    allowed = reserve
                else:
                    input_ids = input_ids[:, -max_L:]
                    attn_mask = attn_mask[:, -max_L:] if attn_mask is not None else None
                    L = int(input_ids.size(1))
                    allowed = max_pos - (P + L)
            gen_new = min(max_new, max(1, allowed))

            token_embeds = self.model.get_input_embeddings()(input_ids)
            inputs_embeds = torch.cat([task_p.unsqueeze(0), token_embeds], dim=1)
            if attn_mask is None:
                attn_mask = torch.ones_like(input_ids, device=DEVICE)
            prefix_m = torch.ones(1, P, device=DEVICE, dtype=attn_mask.dtype)
            attn_ext = torch.cat([prefix_m, attn_mask], dim=1)

            gen_ids = self.model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attn_ext,
                max_new_tokens=gen_new,
                do_sample=False,
                top_p=1.0,
                num_beams=1,
                repetition_penalty=1.0,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )[0]

            gen_len = len(gen_ids)
            if gen_len <= L:
                cont_ids = gen_ids
            else:
                cont_ids = gen_ids[L:]

            eos_id = self.tokenizer.eos_token_id
            stop_ids = {eos_id}
            try:
                im_end = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
                if im_end != getattr(self.tokenizer, "unk_token_id", -1):
                    stop_ids.add(im_end)
            except Exception:
                pass
            ids_list = cont_ids.tolist() if hasattr(cont_ids, "tolist") else list(cont_ids)
            cut = len(ids_list)
            for i, tid in enumerate(ids_list):
                if tid in stop_ids:
                    cut = i
                    break
            cont_ids = ids_list[:cut] if cut else []
            pred_text = self.tokenizer.decode(cont_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True).strip() if cont_ids else ""
            if "\n" in pred_text:
                pred_text = pred_text.split("\n")[0].strip()

            preds.append(pred_text)
            refs_list.append(gold_refs)
            if rouge is not None:
                best_r1 = best_r2 = best_rL = 0.0
                for ref in gold_refs:
                    s = rouge.score(ref, pred_text)
                    best_r1 = max(best_r1, s["rouge1"].fmeasure)
                    best_r2 = max(best_r2, s["rouge2"].fmeasure)
                    best_rL = max(best_rL, s["rougeL"].fmeasure)
                r1s.append(best_r1)
                r2s.append(best_r2)
                rLs.append(best_rL)

        out = {}
        if rouge and r1s:
            out["rouge1"] = float(np.mean(r1s))
            out["rouge2"] = float(np.mean(r2s))
            out["rougeL"] = float(np.mean(rLs))
        else:
            out["rouge1"] = out["rouge2"] = out["rougeL"] = 0.0
        if sacrebleu and preds:
            max_refs = max(len(r) for r in refs_list)
            ref_streams = [[r[j] if j < len(r) else r[0] for r in refs_list] for j in range(max_refs)]
            out["bleu"] = float(sacrebleu.corpus_bleu(preds, ref_streams).score)
        else:
            out["bleu"] = 0.0
        return out

    def train_single_task(self, task_name: str):
        meta = self.task_meta[task_name]
        dataset = meta["dataset"]
        label_column = meta["label_column"]
        canonical = meta["canonical_for_prompts"]
        label_texts = meta["label_texts"]
        keys = meta["keys"]
        is_gen = meta.get("is_generation", False)

        self._start_new_task_prompt(label_texts)
        train_split = dataset["train"]
        if not is_gen and self.k_per_class > 0:
            train_split = balanced_subsample(train_split, label_column, self.k_per_class, seed=self.seed)

        if not is_gen:
            target_max = max(len(self.tokenizer(lbl, add_special_tokens=False)["input_ids"]) for lbl in label_texts)
        else:
            sample_size = min(len(train_split), 512)
            if sample_size == 0:
                target_max = max(4, self.max_length // 2)
            else:
                targets = [train_split[i].get("target", "") or "" for i in range(sample_size)]
                lens = [len(self.tokenizer(t, add_special_tokens=False)["input_ids"]) for t in targets]
                observed = max(lens) if lens else 0
                target_max = min(max(4, observed), max(4, self.max_length // 2))

        preprocess = self._build_preprocess_fn(label_texts, label_column, canonical, keys, target_max)
        eval_name = meta["eval_split"]
        eval_raw = dataset[eval_name]

        processed_train = train_split.map(
            preprocess,
            batched=True,
            batch_size=16,
            writer_batch_size=16,
            num_proc=1,
            remove_columns=train_split.column_names,
            load_from_cache_file=False,
            keep_in_memory=False,
            desc=f"Tokenize train {task_name}",
        )
        processed_eval = eval_raw.map(
            preprocess,
            batched=True,
            batch_size=16,
            writer_batch_size=16,
            num_proc=1,
            remove_columns=eval_raw.column_names,
            load_from_cache_file=False,
            keep_in_memory=False,
            desc=f"Tokenize eval {task_name}",
        )

        train_loader = DataLoader(
            processed_train,
            shuffle=True,
            collate_fn=default_data_collator,
            batch_size=self.batch_size,
            pin_memory=True,
        )
        eval_loader = DataLoader(
            processed_eval,
            shuffle=False,
            collate_fn=default_data_collator,
            batch_size=self.batch_size,
            pin_memory=True,
        )

        # pruning disabled: do not modify self.previous_prompts

        optimizer = torch.optim.AdamW([self.current_prompt], lr=self.lr)
        self.per_epoch_acc[task_name] = []

        # Two-phase schedule: 10 normal + 2 reverse (if previous prompts exist)
        normal_phase_epochs = 10
        reverse_phase_epochs = 2 if self.previous_prompts.numel() > 0 else 0
        effective_epochs = normal_phase_epochs + reverse_phase_epochs if reverse_phase_epochs > 0 else self.num_epochs

        # Prepare selection signals
        prev_tasks_order_full: List[str] = []
        proj_scores: Dict[str, float] = {}
        cos_scores: Dict[str, float] = {}
        if self.previous_prompts.numel() > 0:
            try:
                curr_idx = self.task_list.index(task_name)
            except ValueError:
                curr_idx = len(self.task_list)
            prev_tasks_order_full = list(reversed(self.task_list[:curr_idx]))
            try:
                chunks_prev = torch.split(self.previous_prompts.detach().to(DEVICE), self.prefix_len, dim=0)
                for idx_prev, t_prev in enumerate(prev_tasks_order_full):
                    if idx_prev >= len(chunks_prev):
                        break
                    g_dir = self._compute_prev_slice_grad_dir_on_current(chunks_prev[idx_prev], eval_loader, max_batches=10)
                    V_np = self.task_to_svd_basis.get(t_prev, None)
                    if g_dir is not None and V_np is not None:
                        V = torch.tensor(V_np, dtype=g_dir.dtype, device=g_dir.device)
                        coeff = V.t() @ g_dir
                        proj_scores[t_prev] = float(torch.norm(coeff).item())
                        v1 = V[:, 0]
                        v1 = v1 / (v1.norm() + 1e-12)
                        cos_scores[t_prev] = float(torch.clamp(torch.dot(v1, g_dir), -1.0, 1.0).item())
                    else:
                        proj_scores[t_prev] = float(0.0)
                        cos_scores[t_prev] = float(-1.0)
            except Exception:
                proj_scores, cos_scores = {}, {}

        for epoch in range(max(effective_epochs, self.num_epochs)):
            self.model.train()
            total_loss = 0.0
            in_reverse = (epoch >= normal_phase_epochs) and (epoch < normal_phase_epochs + reverse_phase_epochs)

            # Enter reverse phase
            if in_reverse and not self.reverse_phase_active and self.previous_prompts.numel() > 0:
                try:
                    self.reverse_phase_active = True
                    if self.current_prompt is not None:
                        self.current_prompt.requires_grad_(False)
                    # Selection by proj_cos thresholds
                    selected_set = set()
                    for t_prev in prev_tasks_order_full:
                        if cos_scores.get(t_prev, -1.0) > 0.0 and proj_scores.get(t_prev, 0.0) > 0.1:
                            selected_set.add(t_prev)
                    if len(selected_set) == 0:
                        selected_set = set(prev_tasks_order_full)
                    chunks_prev = torch.split(self.previous_prompts.detach().to(DEVICE), self.prefix_len, dim=0)
                    sel_indices = [i for i, t in enumerate(prev_tasks_order_full) if t in selected_set]
                    sel_chunks = [chunks_prev[i] for i in sel_indices] if len(sel_indices) > 0 else list(chunks_prev)
                    sel_block = torch.cat(sel_chunks, dim=0) if len(sel_chunks) > 0 else torch.empty(0, self.hidden_size, device=DEVICE)
                    self.previous_prompts_param = nn.Parameter(sel_block.clone(), requires_grad=True)
                    optimizer.add_param_group({"params": [self.previous_prompts_param], "lr": self.lr})
                    self._reverse_selected_order = [prev_tasks_order_full[i] for i in sel_indices] if len(sel_indices) > 0 else list(prev_tasks_order_full)
                    self._reverse_selected_indices = sel_indices if len(sel_indices) > 0 else list(range(len(chunks_prev)))
                    # Build projection bases
                    basis_list = []
                    D = self.prefix_len * self.hidden_size
                    for t_prev in self._reverse_selected_order:
                        parts = []
                        V_np = self.task_to_svd_basis.get(t_prev, None)
                        if V_np is not None:
                            parts.append(torch.tensor(V_np, device=DEVICE, dtype=self.previous_prompts_param.dtype))
                        V_cum = self.cumulative_restrict_basis.get(t_prev, None)
                        if V_cum is not None and getattr(V_cum, "size", 0) > 0:
                            parts.append(torch.tensor(V_cum, device=DEVICE, dtype=self.previous_prompts_param.dtype))
                        if len(parts) == 0:
                            v_self = self.prompt_grad_dirs.get(t_prev, None)
                            if v_self is None:
                                basis_list.append(torch.zeros((D, 0), device=DEVICE, dtype=self.previous_prompts_param.dtype))
                            else:
                                v = torch.tensor(v_self, device=DEVICE, dtype=self.previous_prompts_param.dtype).reshape(-1)
                                v = v / (v.norm() + 1e-12)
                                basis_list.append(v.view(-1, 1))
                        else:
                            Vcat = torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]
                            try:
                                Q, _ = torch.linalg.qr(Vcat, mode="reduced")
                                if Q.shape[1] > 6:
                                    Q = Q[:, :6]
                                basis_list.append(Q)
                            except Exception:
                                basis_list.append(Vcat)
                    self._reverse_basis_list = basis_list
                    print(f"[DecReversePhaseStart] epoch={epoch} task={task_name} entering reverse phase")
                except Exception as e_enter:
                    print("Warning: reverse phase enter failed:", e_enter)
                    self.reverse_phase_active = False
                    self.previous_prompts_param = None

            # Exit reverse phase
            if self.reverse_phase_active and (not in_reverse):
                try:
                    full_chunks = list(torch.split(self.previous_prompts.detach().to(DEVICE), self.prefix_len, dim=0))
                    upd_chunks = list(torch.split(self.previous_prompts_param.detach().to(DEVICE), self.prefix_len, dim=0)) if self.previous_prompts_param is not None else []
                    selected_indices = self._reverse_selected_indices if len(self._reverse_selected_indices) > 0 else list(range(len(full_chunks)))
                    # update cumulative restriction with deltas
                    try:
                        for j, idx_full in enumerate(selected_indices):
                            if j < len(upd_chunks) and idx_full < len(full_chunks):
                                prev_task = self._reverse_selected_order[j] if j < len(self._reverse_selected_order) else None
                                if prev_task is None:
                                    continue
                                delta = (upd_chunks[j] - full_chunks[idx_full]).detach()
                                dflat = delta.reshape(-1)
                                nrm = dflat.norm()
                                if float(nrm.item()) > 0.0:
                                    v = (dflat / nrm).to(torch.float32).cpu().numpy().reshape(-1, 1)
                                    exist = self.cumulative_restrict_basis.get(prev_task, None)
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
                                    self.cumulative_restrict_basis[prev_task] = Vnew
                    except Exception as e_cum:
                        print("Warning: cumulative restrict update failed:", e_cum)
                    # merge back
                    for j, idx_full in enumerate(selected_indices):
                        if j < len(upd_chunks) and idx_full < len(full_chunks):
                            full_chunks[idx_full] = upd_chunks[j]
                    self.previous_prompts = torch.cat(full_chunks, dim=0).detach().to(DEVICE)
                except Exception as e_exit:
                    print("Warning: reverse phase exit failed:", e_exit)
                self.reverse_phase_active = False
                if self.current_prompt is not None:
                    self.current_prompt.requires_grad_(True)
                self.previous_prompts_param = None
                self._reverse_basis_list = []

            for batch in tqdm(train_loader, desc=f"{task_name} ep{epoch}"):
                optimizer.zero_grad()
                loss = self._train_step(batch)
                total_loss += loss.detach().float()
                loss.backward()
                # Project gradients for previous_prompts_param if in reverse phase
                if self.reverse_phase_active and self.previous_prompts_param is not None and self.previous_prompts_param.grad is not None:
                    try:
                        g = self.previous_prompts_param.grad
                        chunks_g = torch.split(g, self.prefix_len, dim=0)
                        for idx_block, gi in enumerate(chunks_g):
                            if idx_block >= len(self._reverse_basis_list):
                                break
                            V = self._reverse_basis_list[idx_block]
                            if V is None or V.numel() == 0:
                                continue
                            gi_flat = gi.reshape(-1)
                            proj = V @ (V.t() @ gi_flat)
                            gi_flat -= proj
                            gi.copy_(gi_flat.view_as(gi))
                    except Exception as e_proj:
                        print("Warning: reverse projection failed:", e_proj)
                optimizer.step()
            avg_loss = (total_loss / len(train_loader)).item()

            train_acc = self.evaluate_task(task_name, split="train", max_samples=100)
            if is_gen and isinstance(train_acc, dict):
                train_scalar = float(train_acc.get("rougeL", 0.0))
            else:
                train_scalar = float(train_acc)
            print(f"[{task_name}] ep{epoch} loss={avg_loss:.4f} train_acc={train_scalar:.4f}")

            self.model.eval()
            if len(eval_loader) > 0 and not is_gen:
                eval_loss = 0.0
                with torch.no_grad():
                    for batch in tqdm(eval_loader, desc=f"{task_name} ep{epoch} eval"):
                        eval_loss += self._train_step(batch).detach().float()
                print(f"[{task_name}] ep{epoch} eval_loss={eval_loss.item() / len(eval_loader):.4f}")

            acc_epoch = self.evaluate_task(task_name, max_samples=100)
            self.per_epoch_acc[task_name].append(acc_epoch)
            if is_gen:
                print(f"[{task_name}] ep{epoch} rouge1={acc_epoch['rouge1']:.4f} rougeL={acc_epoch['rougeL']:.4f} bleu={acc_epoch['bleu']:.2f}")
            else:
                print(f"[{task_name}] ep{epoch} eval_acc={float(acc_epoch):.4f}")

            # Reverse-phase per-previous-task evaluation
            if self.reverse_phase_active and self.previous_prompts_param is not None and self._reverse_selected_order:
                try:
                    chunks_upd = torch.split(self.previous_prompts_param.detach(), self.prefix_len, dim=0)
                    for j, prev_task in enumerate(self._reverse_selected_order):
                        if j >= len(chunks_upd):
                            break
                        prev_prompt_only = chunks_upd[j].to(DEVICE)
                        saved_curr = self.current_prompt
                        saved_prev = self.previous_prompts
                        try:
                            self.current_prompt = None
                            self.previous_prompts = prev_prompt_only
                            acc_prev = self.evaluate_task(prev_task, split="eval", max_samples=100)
                            if isinstance(acc_prev, dict):
                                print(f"[DecReverseEval] epoch={epoch} prev_task={prev_task} rougeL={acc_prev.get('rougeL',0.0):.4f} bleu={acc_prev.get('bleu',0.0):.2f}")
                            else:
                                print(f"[DecReverseEval] epoch={epoch} prev_task={prev_task} acc={float(acc_prev):.4f}")
                        except Exception as e_eval:
                            print("Warning: reverse per-task eval failed:", e_eval)
                        finally:
                            self.current_prompt = saved_curr
                            self.previous_prompts = saved_prev
                except Exception as e_rev:
                    print("Warning: reverse eval loop failed:", e_rev)

        # Store own-task mean dir and SVD basis for future selection/parity
        try:
            mean_dir = self._compute_current_prompt_mean_grad(eval_loader, max_batches=30)
            if mean_dir is not None:
                self.prompt_grad_dirs[task_name] = mean_dir.detach().cpu().numpy()
        except Exception:
            pass
        try:
            Vk = self._compute_task_svd_basis(eval_loader, topk=3, max_batches=30)
            if Vk is not None:
                self.task_to_svd_basis[task_name] = Vk.numpy()
        except Exception:
            pass

        final = self.per_epoch_acc[task_name][-1]
        self.task_to_acc[task_name] = final
        if is_gen and isinstance(final, dict):
            print(f"[{task_name}] final rougeL={final.get('rougeL', 0):.4f} bleu={final.get('bleu', 0):.2f}")
        else:
            print(f"[{task_name}] final acc={float(final):.4f}")
        return final

    def train_sequence(
        self,
        eval_all_tasks: bool = False,
        eval_seen_only: bool = False,
        results_path: Optional[str] = None,
    ):
        history = []
        for idx, task_name in enumerate(self.task_list):
            self.train_single_task(task_name)
            if eval_all_tasks:
                results = {}
                tasks_to_eval = self.task_list[: idx + 1] if eval_seen_only else self.task_list
                for t in tasks_to_eval:
                    meta_t = self.task_meta.get(t, {})
                    ds_t = meta_t.get("dataset")
                    use_test = False
                    if ds_t and "test" in ds_t:
                        if meta_t.get("is_generation"):
                            use_test = True
                        else:
                            try:
                                if meta_t.get("label_column") in list(ds_t["test"].column_names or []):
                                    use_test = True
                            except Exception:
                                pass
                    acc_t = self.evaluate_task(t, split="test" if use_test else "eval")
                    results[t] = acc_t["rougeL"] if meta_t.get("is_generation") else acc_t
                self.task_to_acc = results
                history.append(dict(self.task_to_acc))
            else:
                history.append(dict(self.task_to_acc))
        if results_path:
            with open(results_path, "w") as f:
                json.dump({"final_task_acc": dict(self.task_to_acc), "history": history, "per_epoch_acc": self.per_epoch_acc}, f, indent=2)
        return dict(self.task_to_acc), history
