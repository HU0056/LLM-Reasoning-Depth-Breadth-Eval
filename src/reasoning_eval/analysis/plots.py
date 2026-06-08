from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from reasoning_eval.common.io_utils import read_jsonl


def make_summary_plots(results_path: str, figures_dir: str) -> list[Path]:
    rows = read_jsonl(results_path)
    df = pd.DataFrame(rows)
    out_dir = Path(figures_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    metric_df = df.groupby("output_type")[["score_depth", "score_breadth", "score_consistency"]].mean()
    ax = metric_df.plot(kind="bar", figsize=(9, 5))
    ax.set_ylabel("Score")
    ax.set_ylim(0, 105)
    plt.tight_layout()
    path = out_dir / "scores_by_output_type.png"
    plt.savefig(path, dpi=160)
    plt.close()
    paths.append(path)

    correct_rows = df[df["answer_correct"]]
    if not correct_rows.empty:
        same_answer = correct_rows.groupby("output_type")[["score_depth", "score_consistency"]].mean()
        ax = same_answer.plot(kind="bar", figsize=(8, 4))
        ax.set_ylabel("Score")
        ax.set_ylim(0, 105)
        plt.tight_layout()
        path = out_dir / "correct_answer_process_contrast.png"
        plt.savefig(path, dpi=160)
        plt.close()
        paths.append(path)

    breadth_rows = df[df["output_type"].isin(["broad", "narrow_repeated"])]
    breadth_df = breadth_rows.groupby("output_type")[["score_breadth"]].mean() if not breadth_rows.empty else None
    if breadth_df is not None and not breadth_df.empty and not breadth_df["score_breadth"].isna().all():
        ax = breadth_df.plot(kind="bar", figsize=(5, 4), legend=False)
        ax.set_ylabel("Breadth")
        ax.set_ylim(0, 105)
        plt.tight_layout()
        path = out_dir / "broad_vs_narrow_breadth.png"
        plt.savefig(path, dpi=160)
        plt.close()
        paths.append(path)
    return paths

