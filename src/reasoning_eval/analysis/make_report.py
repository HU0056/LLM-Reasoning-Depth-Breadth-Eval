from __future__ import annotations

from reasoning_eval.analysis.plots import make_summary_plots
from reasoning_eval.analysis.result_analyzer import summarize_results
from reasoning_eval.analysis.visualize_dag import draw_lighted_dag
from reasoning_eval.common.io_utils import read_jsonl


def make_report(results_path: str, benchmark_path: str, report_path: str, figures_dir: str) -> None:
    summarize_results(results_path, report_path)
    make_summary_plots(results_path, figures_dir)
    samples = {sample["id"]: sample for sample in read_jsonl(benchmark_path)}
    for result in read_jsonl(results_path):
        sample = samples.get(result["sample_id"])
        if not sample:
            raise ValueError(f"Result references unknown sample_id={result['sample_id']}")
        draw_lighted_dag(sample, result, figures_dir)

