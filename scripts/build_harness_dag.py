#!/usr/bin/env python3
"""Build gold reasoning DAGs from ground truth using the LLM Harness framework.

Replaces the Jaccard-similarity-based graph_builder with a structured
multi-agent verification pipeline.

Usage::

    # Build DAGs for all GSM8K train samples (demo mode: heuristic, no API)
    python scripts/build_harness_dag.py \\
        --input data/raw/gsm8k/train.jsonl \\
        --output data/processed/gsm8k/train_harness_graphs.jsonl \\
        --limit 5

    # With real API calls
    python scripts/build_harness_dag.py \\
        --input data/raw/gsm8k/train.jsonl \\
        --output data/processed/gsm8k/train_harness_graphs.jsonl \\
        --limit 20
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure the src directory is on sys.path
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
_src = _REPO_ROOT / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from reasoning_eval.harness.builder import batch_build
from reasoning_eval.model_test.llm_client import LLMClient


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build gold DAGs from ground truth using LLM Harness",
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Path to input JSONL file (GSM8K format)",
    )
    parser.add_argument(
        "--output", "-o", required=True,
        help="Path to output JSONL file",
    )
    parser.add_argument(
        "--limit", "-n", type=int, default=None,
        help="Limit to first N samples",
    )
    parser.add_argument(
        "--question-key", default="question",
        help="Key for question text in input",
    )
    parser.add_argument(
        "--answer-key", default="answer",
        help="Key for answer text in input",
    )
    parser.add_argument(
        "--sample-id-key", default="gsm8k_id",
        help="Key for sample ID in input",
    )
    args = parser.parse_args()

    # Read samples
    with open(args.input) as f:
        samples = [json.loads(line) for line in f if line.strip()]

    if args.limit:
        samples = samples[: args.limit]

    print(f"Building gold DAGs for {len(samples)} samples...")

    client = LLMClient()
    if client.demo_mode:
        print("[WARNING] Running in demo mode — using heuristic fallback, not LLM.")

    # Ensure every sample has an id key matching the --sample-id-key
    for i, s in enumerate(samples):
        if args.sample_id_key not in s:
            s[args.sample_id_key] = s.get("id", f"sample_{i:05d}")

    results = batch_build(
        samples,
        client,
        question_key=args.question_key,
        answer_key=args.answer_key,
        sample_id_key=args.sample_id_key,
    )

    # Write output
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Done. Wrote {len(results)} graphs to {args.output}")

    # Print stats
    dag_sizes = [r["harness_gold_dag"]["num_steps"] for r in results]
    edge_counts = [r["harness_gold_dag"]["num_edges"] for r in results]
    print(f"  Avg nodes/sample: {sum(dag_sizes) / len(dag_sizes):.1f}")
    print(f"  Avg edges/sample: {sum(edge_counts) / len(edge_counts):.1f}")


if __name__ == "__main__":
    main()
