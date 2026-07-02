#!/usr/bin/env python3 -u
"""Rebuild DAG scores from cached model responses — zero API calls.

Re-runs evaluator with:
  - Improved mapper (math value uniqueness matching)
  - Fixed dag_lighter (auto-lit entries don't contaminate step indices)
"""

import json
import os
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
SRC = str(PROJECT / "src")
sys.path.insert(0, SRC)

from dotenv import load_dotenv; load_dotenv(PROJECT / ".env")
from openai import OpenAI

from reasoning_eval.scorer.evaluator import evaluate_one
from reasoning_eval.dataset.graph_utils import normalize_graph


class MC:
    """Minimal mapper client wrapper (deepseek-chat, non-reasoning)."""
    demo_mode = False
    def __init__(self, client, model):
        self.c = client
        self.m = model

    def generate(self, prompt, system=None, n=1, temperature=0.1, max_tokens=2048,
                 seed=None):
        msgs = [{"role": "system", "content": system}] if system else []
        msgs.append({"role": "user", "content": prompt})
        kw = {
            "model": self.m, "messages": msgs, "n": n,
            "temperature": temperature, "max_tokens": min(max_tokens, 2048),
            "seed": seed, "timeout": 60,
        }
        r = self.c.chat.completions.create(**kw)
        return [c.message.content or "" for c in r.choices]


def get_mapper_client():
    """Create LLM mapper client — only used for mapping, not model inference."""
    try:
        client = OpenAI(
            api_key=os.getenv("API_KEY"),
            base_url="https://api.deepseek.com",
        )
        return MC(client, "deepseek-chat")
    except Exception:
        return None


def rebuild_file(result_path: str, graph_path: str, mapper_client=None):
    """Re-process all samples in a result file."""
    print(f"\n{'='*60}")
    print(f"Rebuilding: {os.path.basename(result_path)}")
    print(f"Graphs:    {os.path.basename(graph_path)}")

    results = json.load(open(result_path))
    if not isinstance(results, list):
        print("  SKIP: not a list")
        return

    # Load graph lookup
    graphs = {}
    with open(graph_path) as f:
        for line in f:
            g = json.loads(line)
            graphs[g["id"]] = g

    ok = n = 0
    errors = 0
    for r in results:
        sid = r.get("id", "")
        if "error" in r or "response" not in r:
            continue
        if sid not in graphs:
            continue

        g_raw = graphs[sid]
        gold_answer = g_raw["gold_answer"]
        g_raw_graph = g_raw["gold_reasoning_graph"]
        resp_text = r.get("response", "")

        try:
            graph = normalize_graph(g_raw_graph)
            ev = evaluate_one(
                {
                    "id": sid,
                    "gold_reasoning_graph": graph,
                    "gold_answer": gold_answer,
                    "key_branch_nodes": [],
                },
                {
                    "sample_id": sid,
                    "model_name": "rebuild",
                    "output_type": "cot",
                    "response": resp_text,
                },
                mapper_client=mapper_client,  # Math value match fast path, LLM fallback
            )

            lit_nodes = [n for n, st in ev.lighted_graph["nodes"].items() if st == "lit"]
            states = {
                st: sum(1 for v in ev.lighted_graph["nodes"].values() if v == st)
                for st in set(ev.lighted_graph["nodes"].values())
            }

            r["correct"] = ev.answer_correct
            r["depth"] = ev.score_depth
            r["cons"] = ev.score_consistency
            r["lit"] = len(lit_nodes)
            r["states"] = states
            r["lit_nodes"] = lit_nodes
            r["lighted_graph"] = ev.lighted_graph
            r["_rebuilt"] = True  # mark as rebuilt

            n += 1
            if ev.answer_correct:
                ok += 1

        except Exception as e:
            errors += 1
            if errors <= 3:
                print(f"  [{sid}] ERROR: {e}")

    # Save
    json.dump(results, open(result_path, "w"), ensure_ascii=False, indent=2)
    print(f"  {n} rebuilt, errors={errors}")
    if n:
        print(f"  acc={ok/n:.1%} "
              f"avg_d={sum(r.get('depth',0) for r in results if 'error' not in r)/n:.0f} "
              f"avg_c={sum(r.get('cons',0) for r in results if 'error' not in r)/n:.0f}")


def main():
    results_dir = PROJECT / "outputs" / "results"
    gsm8k_graphs = PROJECT / "data" / "processed" / "gsm8k" / "train_graphs_std.jsonl"

    mapper_client = get_mapper_client()
    print(f"Mapper: {'deepseek-chat (LLM fallback)' if mapper_client else 'fast-path only'}")

    # Rebuild GSM8K results
    for fname in sorted(os.listdir(results_dir)):
        if not fname.endswith("_gsm8k_50.json"):
            continue
        path = results_dir / fname
        rebuild_file(str(path), str(gsm8k_graphs), mapper_client)

    # Rebuild Omni-MATH results
    omni_graphs = PROJECT / "data" / "processed" / "omni_math" / "test_graphs_std.jsonl"
    for fname in sorted(os.listdir(results_dir)):
        if not fname.endswith("_omni_50.json"):
            continue
        path = results_dir / fname
        rebuild_file(str(path), str(omni_graphs), mapper_client)


if __name__ == "__main__":
    main()
