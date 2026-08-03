import argparse
from pathlib import Path

import numpy as np
from datasets import Dataset, load_dataset


TRAIN_FILES = {
    "yelp_review_full": "yelp_1k.npy",
    "amazon": "amazon_1k.npy",
    "dbpedia_14": "dbpedia_14_1k.npy",
    "yahoo_answers_topics": "yahoo_1k.npy",
    "ag_news": "ag_news_1k.npy",
    "mnli": "mnli_1k.npy",
    "qqp": "qqp_1k.npy",
    "rte": "rte_1k.npy",
    "sst2": "sst2_1k.npy",
    "wic": "wic_1k.npy",
    "cb": "cb_1k.npy",
    "copa": "copa_1k.npy",
    "boolq": "boolq_1k.npy",
    "multirc": "multirc_1k.npy",
    "imdb": "imdb_1k.npy",
}

EVAL_FILES = {
    "yelp_review_full": "yelp_review_full.npy",
    "amazon": "amazon.npy",
    "dbpedia_14": "dbpedia_14.npy",
    "yahoo_answers_topics": "yahoo_answers_topics.npy",
    "ag_news": "ag_news.npy",
    "mnli": "mnli_test.npy",
    "qqp": "glue_qqp.npy",
    "rte": "glue_rte.npy",
    "sst2": "sst2.npy",
    "wic": "super_glue_wic.npy",
    "cb": "super_glue_cb.npy",
    "copa": "super_glue_copa.npy",
    "boolq": "super_glue_boolq.npy",
    "multirc": "super_glue_multirc.npy",
    "imdb": "imdb.npy",
}

TASK_ORDER = list(TRAIN_FILES)


def save_dataset(ds, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {col: list(ds[col]) for col in ds.column_names}
    np.save(path, data, allow_pickle=True)
    print(f"saved {path} rows={len(ds)} cols={ds.column_names}", flush=True)


def balanced_subset(ds, label_col="label", k=1000, seed=0):
    if k < 0 or label_col not in ds.column_names:
        return ds.shuffle(seed=seed)
    labels = np.asarray(ds[label_col])
    rng = np.random.default_rng(seed)
    indices = []
    for label in sorted(set(labels.tolist())):
        label_idx = np.where(labels == label)[0]
        take = min(k, len(label_idx))
        indices.extend(rng.choice(label_idx, size=take, replace=False).tolist())
    rng.shuffle(indices)
    return ds.select(indices)


def normalize_amazon(ds):
    cols = set(ds.column_names)
    if {"review_body", "stars"}.issubset(cols):
        return Dataset.from_dict({
            "label": [int(x) - 1 for x in ds["stars"]],
            "title": list(ds["review_title"]) if "review_title" in cols else [""] * len(ds),
            "content": list(ds["review_body"]),
        })
    if {"text", "label"}.issubset(cols):
        return Dataset.from_dict({
            "label": [int(x) for x in ds["label"]],
            "title": [""] * len(ds),
            "content": list(ds["text"]),
        })
    raise ValueError(f"Unsupported amazon columns: {ds.column_names}")


def load_raw(task, split, cache_dir):
    if task == "amazon":
        try:
            return normalize_amazon(load_dataset("amazon_reviews_multi", "en", split=split, cache_dir=cache_dir))
        except Exception:
            fallback_split = "train" if split == "train" else "test"
            return normalize_amazon(load_dataset("SetFit/amazon_reviews_multi_en", split=fallback_split, cache_dir=cache_dir))
    if task in {"mnli", "qqp", "rte", "sst2"}:
        glue_split = "validation_matched" if task == "mnli" and split != "train" else split
        if task == "mnli" and split == "train":
            return load_dataset("glue", "mnli", split="train", cache_dir=cache_dir)
        return load_dataset("glue", task, split=glue_split, cache_dir=cache_dir)
    if task in {"wic", "cb", "copa", "boolq", "multirc"}:
        return load_dataset("super_glue", task, split=split, cache_dir=cache_dir)
    return load_dataset(task, split=split, cache_dir=cache_dir)


def build_task(task, out_root, k, seed, cache_dir, overwrite):
    train_path = out_root / "train" / TRAIN_FILES[task]
    eval_path = out_root / "test" / EVAL_FILES[task]

    if overwrite or not train_path.exists():
        train_ds = load_raw(task, "train", cache_dir)
        train_ds = balanced_subset(train_ds, k=k, seed=seed)
        save_dataset(train_ds, train_path)
    else:
        print(f"exists {train_path}", flush=True)

    if overwrite or not eval_path.exists():
        eval_split = "validation" if task in {"mnli", "qqp", "rte", "sst2", "wic", "cb", "copa", "boolq", "multirc"} else "test"
        eval_ds = load_raw(task, eval_split, cache_dir)
        save_dataset(eval_ds, eval_path)
    else:
        print(f"exists {eval_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Prepare offline .npy files for the SABER Long Sequence 15-task benchmark.")
    parser.add_argument("--output_dir", default="src/data", help="Directory containing train/ and test/ output folders.")
    parser.add_argument("--cache_dir", default=None, help="Optional HuggingFace datasets cache directory.")
    parser.add_argument("--k_per_class", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tasks", nargs="+", default=TASK_ORDER)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    out_root = Path(args.output_dir)
    for task in args.tasks:
        if task not in TRAIN_FILES:
            raise ValueError(f"Unknown task: {task}")
        print(f"\n=== {task} ===", flush=True)
        build_task(task, out_root, args.k_per_class, args.seed, args.cache_dir, args.overwrite)

    missing = []
    for task in TASK_ORDER:
        for folder, filename in [("train", TRAIN_FILES[task]), ("test", EVAL_FILES[task])]:
            path = out_root / folder / filename
            if not path.exists():
                missing.append(str(path))
    if missing:
        print("\nMissing files:")
        for path in missing:
            print(path)
        raise SystemExit(1)
    print("\nLong Sequence 15-task data is ready.")


if __name__ == "__main__":
    main()
