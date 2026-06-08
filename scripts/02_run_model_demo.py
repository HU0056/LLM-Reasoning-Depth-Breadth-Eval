from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
from reasoning_eval.common.io_utils import read_jsonl
from reasoning_eval.model_test.demo_output_loader import load_demo_outputs
from reasoning_eval.model_test.prompt_builder import build_prompt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", default="data/processed/demo_benchmark.jsonl")
    parser.add_argument("--outputs", default="data/model_outputs/demo_model_outputs.jsonl")
    args = parser.parse_args()
    samples = read_jsonl(args.benchmark)
    outputs = load_demo_outputs(args.outputs)
    print(f"Loaded {len(samples)} benchmark samples and {len(outputs)} demo outputs.")
    if samples:
        print("\nExample prompt:\n")
        print(build_prompt(samples[0]))


if __name__ == "__main__":
    main()

