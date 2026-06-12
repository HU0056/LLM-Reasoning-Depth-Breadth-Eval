#!/usr/bin/env python
"""Generate LLM responses for benchmark samples.

Usage::

    # Generate CoT outputs for the demo benchmark (rule logic)
    python scripts/run_model_test.py \\
        --benchmark data/processed/demo_benchmark.jsonl \\
        --output data/model_outputs/generated_outputs.jsonl

    # With self-consistency (5 samples per question, for breadth scoring)
    python scripts/run_model_test.py \\
        --benchmark data/processed/demo_benchmark.jsonl \\
        --output data/model_outputs/generated_sc5.jsonl \\
        --n 5 --temperature 0.8

    # Dry-run: print prompts without calling the API
    python scripts/run_model_test.py \\
        --benchmark data/processed/demo_benchmark.jsonl \\
        --dry-run

    # Limit to first N samples (smoke test)
    python scripts/run_model_test.py \\
        --benchmark data/processed/demo_benchmark.jsonl \\
        --output data/model_outputs/smoke_test.jsonl \\
        --limit 3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import _bootstrap  # noqa: F401 — add src to sys.path
from reasoning_eval.common.io_utils import read_jsonl
from reasoning_eval.model_test.generate_with_api import generate_benchmark_outputs
from reasoning_eval.model_test.llm_client import LLMClient
from reasoning_eval.model_test.prompt_builder import build_prompt, get_system_prompt


def dry_run(benchmark_path: str, limit: int | None = None) -> None:
    """Print prompts for inspection without calling any API."""
    samples = read_jsonl(benchmark_path)
    if limit:
        samples = samples[:limit]

    for idx, sample in enumerate(samples, start=1):
        prompt = build_prompt(sample)
        system = get_system_prompt(sample)
        print(f"{'='*70}")
        print(f"[{idx}/{len(samples)}] sample_id={sample.get('id')}  task_type={sample.get('task_type', 'deduction')}")
        print(f"{'='*70}")
        print(f"\n[SYSTEM]\n{system}\n")
        print(f"[USER]\n{prompt}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate LLM responses for benchmark samples")
    parser.add_argument("--benchmark", required=True, help="Path to benchmark JSONL")
    parser.add_argument("--output", help="Path to save generated outputs (JSONL)")
    parser.add_argument("--model", help="Model name override (default: from .env)")
    parser.add_argument("--n", type=int, default=1, help="Responses per sample (1=CoT, >1=Self-Consistency)")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--limit", type=int, help="Only process first N samples")
    parser.add_argument("--delay", type=float, default=0.0, help="Seconds between API calls")
    parser.add_argument("--dry-run", action="store_true", help="Print prompts only, no API calls")
    args = parser.parse_args()

    if not Path(args.benchmark).exists():
        print(f"Error: benchmark file not found: {args.benchmark}", file=sys.stderr)
        sys.exit(1)

    if args.dry_run:
        dry_run(args.benchmark, args.limit)
        return

    if not args.output:
        print("Error: --output is required (unless --dry-run)", file=sys.stderr)
        sys.exit(1)

    client = LLMClient()

    if client.demo_mode:
        print(
            "⚠️  LLMClient is in DEMO MODE (no API_KEY configured).\n"
            "   Copy .env.example → .env and fill in your API key to call a real model.\n"
            "   For now, use the hand-written demo outputs:\n"
            "     data/model_outputs/demo_model_outputs.jsonl\n",
            file=sys.stderr,
        )
        # In demo mode we still print the prompts so the user can inspect them.
        print("--- Prompts that would be sent ---\n")
        dry_run(args.benchmark, args.limit)
        return

    print(f"Model : {client.model_name}")
    print(f"Base  : {client.base_url}")
    print(f"n     : {args.n}  temperature : {args.temperature}")
    print(f"Limit : {args.limit or 'all'}  delay : {args.delay}s\n")

    generate_benchmark_outputs(
        benchmark_path=args.benchmark,
        output_path=args.output,
        client=client,
        model_name=args.model,
        n=args.n,
        temperature=args.temperature,
        limit=args.limit,
        delay=args.delay,
    )


if __name__ == "__main__":
    main()
