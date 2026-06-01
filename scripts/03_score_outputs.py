from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
from reasoning_eval.scorer.evaluator import evaluate_files


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", default="data/processed/demo_benchmark.jsonl")
    parser.add_argument("--outputs", default="data/model_outputs/demo_model_outputs.jsonl")
    parser.add_argument("--save", default="outputs/results/demo_results.jsonl")
    args = parser.parse_args()
    results = evaluate_files(args.benchmark, args.outputs, args.save)
    print(f"Scored {len(results)} outputs -> {args.save}")


if __name__ == "__main__":
    main()

