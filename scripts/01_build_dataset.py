from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
from reasoning_eval.dataset.build_demo_dataset import build_demo_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default="data/raw/demo_raw_rules.jsonl")
    parser.add_argument("--save", default="data/processed/demo_benchmark.jsonl")
    args = parser.parse_args()
    rows = build_demo_dataset(args.raw, args.save)
    print(f"Built {len(rows)} benchmark samples -> {args.save}")


if __name__ == "__main__":
    main()

