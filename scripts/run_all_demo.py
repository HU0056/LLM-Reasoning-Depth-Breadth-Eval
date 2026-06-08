from __future__ import annotations

import _bootstrap  # noqa: F401
from reasoning_eval.analysis.make_report import make_report
from reasoning_eval.dataset.build_demo_dataset import build_demo_dataset
from reasoning_eval.model_test.demo_output_loader import load_demo_outputs
from reasoning_eval.scorer.evaluator import evaluate_files


def main() -> None:
    raw = "data/raw/demo_raw_rules.jsonl"
    benchmark = "data/processed/demo_benchmark.jsonl"
    outputs = "data/model_outputs/demo_model_outputs.jsonl"
    results = "outputs/results/demo_results.jsonl"
    report = "outputs/reports/summary.csv"
    figures = "outputs/figures"

    samples = build_demo_dataset(raw, benchmark)
    demo_outputs = load_demo_outputs(outputs)
    eval_results = evaluate_files(benchmark, outputs, results)
    make_report(results, benchmark, report, figures)
    print(f"Built {len(samples)} samples")
    print(f"Loaded {len(demo_outputs)} demo outputs")
    print(f"Scored {len(eval_results)} outputs")
    print(f"Results: {results}")
    print(f"Report: {report}")
    print(f"Figures: {figures}")


if __name__ == "__main__":
    main()

