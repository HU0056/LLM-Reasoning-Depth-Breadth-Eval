from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
from reasoning_eval.analysis.make_report import make_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="outputs/results/demo_results.jsonl")
    parser.add_argument("--benchmark", default="data/processed/demo_benchmark.jsonl")
    parser.add_argument("--report", default="outputs/reports/summary.csv")
    parser.add_argument("--figures", default="outputs/figures")
    args = parser.parse_args()
    make_report(args.results, args.benchmark, args.report, args.figures)
    print(f"Analysis report -> {args.report}; figures -> {args.figures}")


if __name__ == "__main__":
    main()

