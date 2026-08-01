import argparse
from pathlib import Path

import numpy as np


def to_scalar(value):
    if isinstance(value, dict):
        return None
    if isinstance(value, (list, tuple, np.ndarray)):
        arr = np.asarray(value, dtype=float)
        return float(np.mean(arr)) if arr.size else None
    try:
        return float(value)
    except Exception:
        return None


def print_section(title, scores):
    rows = []
    for task, value in scores.items():
        scalar = to_scalar(value)
        if scalar is not None:
            rows.append((task, scalar, value))

    if not rows:
        return

    print(f"\n{title}")
    print("-" * len(title))
    for task, scalar, raw in rows:
        print(f"{task:12s} {scalar:.6f}  raw={raw}")
    print(f"{'AVERAGE':12s} {np.mean([x[1] for x in rows]):.6f}")


def main():
    parser = argparse.ArgumentParser(description="Read SABER results_dict.npy.")
    parser.add_argument("path", help="Path to results_dict.npy")
    args = parser.parse_args()

    path = Path(args.path)
    result = np.load(path, allow_pickle=True).item()

    print(f"Loaded: {path}")
    print(f"Top-level keys: {list(result.keys())}")

    if "test" in result and isinstance(result["test"], dict):
        test = result["test"]
        nested_steps = [k for k, v in test.items() if isinstance(v, dict)]
        if nested_steps:
            last_step = sorted(nested_steps)[-1]
            print_section(f"test step {last_step}", test[last_step])
        else:
            print_section("test", test)

    train_val = {k: v for k, v in result.items() if k != "test"}
    print_section("validation history final epoch", {
        k: (v[-1] if isinstance(v, list) and v else v)
        for k, v in train_val.items()
    })


if __name__ == "__main__":
    main()

