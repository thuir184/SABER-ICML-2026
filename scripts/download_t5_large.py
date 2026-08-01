import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def main():
    parser = argparse.ArgumentParser(description="Download T5-Large for offline Kaggle runs.")
    parser.add_argument("--repo_id", default="t5-large")
    parser.add_argument("--out_dir", default="offline_models/t5-large")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=args.repo_id,
        local_dir=str(out_dir),
        local_dir_use_symlinks=False,
        allow_patterns=[
            "config.json",
            "generation_config.json",
            "model.safetensors",
            "pytorch_model.bin",
            "spiece.model",
            "special_tokens_map.json",
            "tokenizer.json",
            "tokenizer_config.json",
        ],
    )

    print(f"Downloaded {args.repo_id} to {out_dir.resolve()}")
    print("Upload this folder as a Kaggle Dataset, then pass its Kaggle path to --model_name.")


if __name__ == "__main__":
    main()

