from __future__ import annotations

import argparse
import sys
import time

import _bootstrap  # noqa: F401
from reasoning_eval.common.io_utils import read_jsonl, write_jsonl
from reasoning_eval.model_test.generate_with_api import generate_for_sample


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", default="data/processed/demo_benchmark.jsonl")
    parser.add_argument("--outputs", default="data/model_outputs/api_model_outputs.jsonl")
    parser.add_argument("--model-name", default=None, help="Override model name for the output record")
    parser.add_argument("--n", type=int, default=1, help="Number of responses per sample")
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    samples = read_jsonl(args.benchmark)
    print(f"Loaded {len(samples)} benchmark samples.")

    outputs = []
    for idx, sample in enumerate(samples, start=1):
        sample_id = sample["id"]
        print(f"[{idx}/{len(samples)}] Generating for {sample_id} ...", end=" ", flush=True)
        try:
            responses = generate_for_sample(sample, n=args.n)
            for resp_idx, response in enumerate(responses):
                outputs.append({
                    "sample_id": sample_id,
                    "model_name": args.model_name or "api",
                    "output_type": "api",
                    "response": response,
                    "n": args.n,
                    "resp_idx": resp_idx,
                })
            print(f"OK ({len(responses)} response(s))")
        except Exception as exc:
            print(f"FAILED: {exc}")
            print("Aborting.")
            sys.exit(1)

        # Small delay to avoid rate limiting
        if idx < len(samples):
            time.sleep(0.5)

    write_jsonl(args.outputs, outputs)
    print(f"Saved {len(outputs)} outputs -> {args.outputs}")


if __name__ == "__main__":
    main()
