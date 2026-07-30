# Kaggle Quick Start

Use a GPU notebook, enable Internet if models or HuggingFace datasets need to be downloaded, then run:

```bash
cd /kaggle/working
git clone https://github.com/OptMN-Lab/SABER-ICML-2026.git
cd SABER-ICML-2026
pip install -r requirements-kaggle.txt
```

## Seq2Seq LongSequence Smoke Test

Run from the script directory so local imports resolve exactly as expected:

```bash
cd /kaggle/working/SABER-ICML-2026/src/seq2seq/LongSequence
python train.py \
  --save_dir /kaggle/working/runs \
  --save_name smoke_t5_small_seed1 \
  --task_list sst2 wic \
  --model_name t5-small \
  --cache_dir /kaggle/working/hf_cache \
  --batch_size 4 \
  --num_epochs 1 \
  --prefix_len 10 \
  --freeze_weights 1 \
  --select_k_per_class 20 \
  --seed 1 \
  --selection_method proj_cos
```

## Decoder-Only LongSequence Smoke Test

```bash
cd /kaggle/working/SABER-ICML-2026/src/autoregressive/LongSequence
python main.py \
  --base_model_name gpt2 \
  --tasks "sst2,wic" \
  --prefix_len 10 \
  --max_length 128 \
  --lr 0.03 \
  --batch_size 4 \
  --k_per_class 20 \
  --num_epochs 1 \
  --seed 1 \
  --save_path /kaggle/working/runs/gpt2_smoke_seed1 \
  --eval_all_tasks \
  --eval_seen_only \
  --fix_test_data \
  --test_fixed_dir /kaggle/working/SABER-ICML-2026/src/data/test \
  --data_root /kaggle/working/SABER-ICML-2026/src/data
```

After the smoke test works, increase `--model_name`, `--base_model_name`, `--batch_size`, `--select_k_per_class` or `--k_per_class`, `--num_epochs`, and seeds.

## Checkpoints And Resume

Seq2Seq LongSequence runs save lightweight prompt checkpoints under:

```text
/kaggle/working/runs/<save_name>/checkpoints/
```

Check what has been saved:

```bash
find /kaggle/working/runs -type f | sort
```

Resume from the latest checkpoint:

```bash
cd /kaggle/working/SABER-ICML-2026/src/seq2seq/LongSequence
python train.py \
  --save_dir /kaggle/working/runs \
  --save_name paper_like_t5_large_proj_cos_seed_1_autosave \
  --task_list sst2 wic mnli boolq multirc \
  --model_name t5-large \
  --cache_dir /kaggle/working/hf_cache \
  --batch_size 4 \
  --num_epochs 10 \
  --prefix_len 10 \
  --freeze_weights 1 \
  --select_k_per_class 1000 \
  --pre_processed 1 \
  --seed 1 \
  --selection_method proj_cos \
  --resume_from_checkpoint /kaggle/working/runs/paper_like_t5_large_proj_cos_seed_1_autosave/checkpoints/latest.pt
```

Save a Kaggle version after long runs so `/kaggle/working` outputs and logs are preserved.
