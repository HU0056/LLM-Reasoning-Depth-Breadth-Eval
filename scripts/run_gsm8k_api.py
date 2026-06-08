from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import _bootstrap  # noqa: F401

from reasoning_eval.common.io_utils import read_jsonl, write_jsonl
from reasoning_eval.model_test.gsm8k_prompt import build_gsm8k_prompt
from reasoning_eval.model_test.llm_client import LLMClient
from reasoning_eval.scorer.gsm8k_evaluator import evaluate_gsm8k_files
from reasoning_eval.analysis.gsm8k_visualize import draw_gsm8k_dag
from reasoning_eval.analysis.plots import make_summary_plots
from reasoning_eval.analysis.result_analyzer import summarize_results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GSM8K API pipeline: inference → scoring → visualization")
    parser.add_argument(
        "--benchmark",
        default="data/processed/gsm8k/test_graphs.jsonl",
        help="Path to GSM8K benchmark JSONL.",
    )
    parser.add_argument(
        "--outputs",
        default="data/model_outputs/gsm8k_api_outputs.jsonl",
        help="Path for model outputs JSONL.",
    )
    parser.add_argument(
        "--results",
        default="outputs/results/gsm8k_api_results.jsonl",
        help="Path for evaluation results JSONL.",
    )
    parser.add_argument(
        "--report",
        default="outputs/reports/gsm8k_api_summary.csv",
        help="Path for summary CSV report.",
    )
    parser.add_argument(
        "--figures",
        default="outputs/figures/gsm8k",
        help="Directory for DAG figures.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit to first N samples (for quick testing).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="LLM sampling temperature.",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Override model name (reads from .env by default).",
    )
    parser.add_argument(
        "--skip-inference",
        action="store_true",
        help="Skip API inference (use existing outputs file).",
    )
    return parser


def run_inference(args: argparse.Namespace) -> None:
    """Run API inference on GSM8K benchmark samples."""
    samples = read_jsonl(args.benchmark)
    if args.limit:
        samples = samples[: args.limit]

    print(f"Running API inference on {len(samples)} samples...")
    client = LLMClient()
    model_name = args.model_name or client.model_name

    outputs = []
    for idx, sample in enumerate(samples, start=1):
        sample_id = sample["id"]
        prompt = build_gsm8k_prompt(sample)
        print(f"[{idx}/{len(samples)}] {sample_id} ...", end=" ", flush=True)
        try:
            responses = client.generate_cot(prompt, n=1, temperature=args.temperature)
            for resp_idx, response in enumerate(responses):
                outputs.append({
                    "sample_id": sample_id,
                    "model_name": model_name,
                    "output_type": "api",
                    "response": response,
                    "n": 1,
                    "resp_idx": resp_idx,
                })
            print("OK")
        except Exception as exc:
            print(f"FAILED: {exc}")
            sys.exit(1)

        # Rate limiting delay
        if idx < len(samples):
            time.sleep(0.3)

    write_jsonl(args.outputs, outputs)
    print(f"Saved {len(outputs)} outputs → {args.outputs}")


def run_evaluation(args: argparse.Namespace) -> list[dict]:
    """Evaluate model outputs against the benchmark."""
    print("Evaluating model outputs...")
    results = evaluate_gsm8k_files(args.benchmark, args.outputs, args.results)
    correct = sum(1 for r in results if r.get("answer_correct"))
    print(f"Scored {len(results)} outputs (correct: {correct}/{len(results)}) → {args.results}")
    return results


def run_visualization(args: argparse.Namespace, results: list[dict]) -> None:
    """Generate DAG figures and summary report/plots."""
    samples = {s["id"]: s for s in read_jsonl(args.benchmark)}
    print(f"Generating {len(results)} DAG figures...")
    for result in results:
        sample = samples.get(result["sample_id"])
        if sample:
            path = draw_gsm8k_dag(sample, result, args.figures)

    # Summary plots
    make_summary_plots(args.results, args.figures)

    # Summary report
    summarize_results(args.results, args.report)
    print(f"Report → {args.report}")
    print(f"Figures → {args.figures}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Resolve paths relative to project root
    root = Path(__file__).resolve().parents[1]
    args.benchmark = str(root / args.benchmark)
    args.outputs = str(root / args.outputs)
    args.results = str(root / args.results)
    args.report = str(root / args.report)
    args.figures = str(root / args.figures)

    if not args.skip_inference:
        run_inference(args)
    else:
        print(f"Skipping inference, using existing: {args.outputs}")

    results = run_evaluation(args)

    if results:
        run_visualization(args, results)

    # Print quick summary
    correct = sum(1 for r in results if r.get("answer_correct"))
    avg_depth = sum(r.get("score_depth", 0) for r in results) / len(results) if results else 0
    print(f"\n===== Summary =====")
    print(f"Total: {len(results)}")
    print(f"Accuracy: {correct}/{len(results)} ({100*correct/len(results):.1f}%)")
    print(f"Avg Depth: {avg_depth:.1f}")


if __name__ == "__main__":
    main()
