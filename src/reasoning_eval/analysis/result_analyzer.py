from __future__ import annotations

from pathlib import Path

import pandas as pd

from reasoning_eval.common.io_utils import read_jsonl


def summarize_results(results_path: str, report_path: str) -> pd.DataFrame:
    rows = read_jsonl(results_path)
    if not rows:
        raise ValueError(f"No evaluation rows found: {results_path}")
    df = pd.DataFrame(rows)
    summary = (
        df.groupby("output_type", dropna=False)
        .agg(
            answer_accuracy=("answer_correct", "mean"),
            avg_depth=("score_depth", "mean"),
            avg_breadth=("score_breadth", "mean"),
            avg_consistency=("score_consistency", "mean"),
            avg_branch_coverage=("branch_coverage", "mean"),
            missing_premise_rate=("missing_premise_flag", "mean"),
            first_error_rate=("first_error_step", lambda s: s.notna().mean()),
        )
        .reset_index()
    )
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(report_path, index=False, encoding="utf-8-sig")
    return summary

