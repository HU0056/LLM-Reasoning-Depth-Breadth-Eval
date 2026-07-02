#!/usr/bin/env python3 -u
"""Omni-MATH Unified Benchmark v2: 50 random samples, DeepSeek only, full DAG pipeline.

Usage (agent mode):
    python scripts/bench_omni_v2.py <model_label> <model_id>
"""

import json
import os
import random
import sys
import time

PROJECT = "/home/lz/LLM-Reasoning-Depth-Breadth-Eval"
SRC = f"{PROJECT}/src"
sys.path.insert(0, SRC)

from dotenv import load_dotenv; load_dotenv(f"{PROJECT}/.env")
from openai import OpenAI

from reasoning_eval.scorer.evaluator import evaluate_one
from reasoning_eval.dataset.graph_utils import normalize_graph
from reasoning_eval.model_test.prompt_builder import build_prompt, get_system_prompt

# ── Config ────────────────────────────────────────────────────────────────
N_SAMPLES = 50
TIMEOUT = 120
TEMPERATURE = 0.3
RANDOM_SEED = 42

OMNI_PATH = f"{PROJECT}/data/processed/omni_math/test_graphs_std.jsonl"
SAMPLE_IDS_PATH = f"{PROJECT}/outputs/results/omni_50_sample_ids.json"
RESULTS_DIR = f"{PROJECT}/outputs/results"

# ── MC Wrapper ────────────────────────────────────────────────────────────
class MC:
    demo_mode = False
    def __init__(self, c, m, reasoning=False):
        self.c, self.m, self.r = c, m, reasoning
    def generate(self, prompt, system=None, n=1, temperature=TEMPERATURE,
                 max_tokens=16384, seed=None):
        msgs = [{"role": "system", "content": system}] if system else []
        msgs.append({"role": "user", "content": prompt})
        kw = {"model": self.m, "messages": msgs, "n": n,
              "temperature": temperature,
              "max_tokens": min(max_tokens, 16384),
              "seed": seed, "timeout": TIMEOUT}
        if self.r:
            kw["extra_body"] = {"reasoning_effort": "low"}
        r = self.c.chat.completions.create(**kw)
        return [c.message.content or "" for c in r.choices]


def run(model_label: str, model_id: str):
    """Full pipeline on 50 random Omni-MATH samples."""

    # ── Clients ────────────────────────────────────────────────────────
    gen_client = OpenAI(api_key=os.getenv("API_KEY"),
                        base_url="https://api.deepseek.com")
    reasoning = model_id in ("deepseek-v4-pro", "deepseek-v4-flash")
    target_client = MC(gen_client, model_id, reasoning)

    mapper_mc = MC(OpenAI(api_key=os.getenv("API_KEY"),
                          base_url="https://api.deepseek.com"),
                   "deepseek-chat")

    # ── Load sample IDs ────────────────────────────────────────────────
    if os.path.exists(SAMPLE_IDS_PATH):
        sample_ids = set(json.load(open(SAMPLE_IDS_PATH)))
    else:
        sample_ids = None

    with open(OMNI_PATH) as f:
        all_graphs = [json.loads(line) for line in f]

    if sample_ids:
        samples = [g for g in all_graphs if g["id"] in sample_ids]
    else:
        random.seed(RANDOM_SEED)
        samples = random.sample(all_graphs, min(N_SAMPLES, len(all_graphs)))

    print(f"[{model_label}] {len(samples)} samples (reasoning={reasoning})", flush=True)

    # ── Run ────────────────────────────────────────────────────────────
    ok = n_done = 0
    results = []

    for s in samples:
        sid = s["id"]
        question = s["question"]
        gold_answer = s["gold_answer"]
        g_raw = s["gold_reasoning_graph"]
        t0 = time.time()

        try:
            s["task_type"] = "math"
            graph = normalize_graph(g_raw)
            smp = {"id": sid, "question": question, "gold_answer": gold_answer,
                   "task_type": "math", "gold_reasoning_graph": g_raw}

            resp = target_client.generate(
                prompt=build_prompt(smp), system=get_system_prompt(smp),
                temperature=TEMPERATURE, max_tokens=8192,
            )
            resp_text = resp[0]

            r = evaluate_one(
                {"id": sid, "gold_reasoning_graph": graph,
                 "gold_answer": gold_answer, "key_branch_nodes": []},
                {"sample_id": sid, "model_name": model_label,
                 "output_type": "cot", "response": resp_text},
                mapper_client=mapper_mc,
            )
            elapsed = time.time() - t0

            lit_nodes = [n for n, st in r.lighted_graph["nodes"].items() if st == "lit"]
            states = {st: sum(1 for v in r.lighted_graph["nodes"].values() if v == st)
                      for st in set(r.lighted_graph["nodes"].values())}

            n_done += 1
            if r.answer_correct:
                ok += 1

            results.append({
                "id": sid, "correct": r.answer_correct,
                "depth": r.score_depth, "cons": r.score_consistency,
                "lit": len(lit_nodes), "time": round(elapsed, 1),
                "states": states, "lit_nodes": lit_nodes,
                "response": resp_text, "lighted_graph": r.lighted_graph,
            })

            status = "✓" if r.answer_correct else "✗"
            print(f"  [{n_done:2d}/{len(samples)} {sid}] {status} "
                  f"d={r.score_depth:.0f} c={r.score_consistency:.0f} "
                  f"lit={len(lit_nodes)} ({elapsed:.0f}s)", flush=True)

        except Exception as e:
            print(f"  [{sid}]: FAIL {str(e)[:120]}", flush=True)
            results.append({"id": sid, "error": str(e)[:200]})

    # ── Save ───────────────────────────────────────────────────────────
    tag = model_label.replace(".", "_").replace("-", "_")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = f"{RESULTS_DIR}/bench_{tag}_omni_50.json"
    json.dump(results, open(out_path, "w"), ensure_ascii=False, indent=2)

    if n_done:
        avg_depth = sum(r.get("depth", 0) for r in results if "error" not in r) / n_done
        avg_cons = sum(r.get("cons", 0) for r in results if "error" not in r) / n_done
        print(f"\n[{model_label}] DONE: acc={ok}/{n_done}={ok/n_done:.1%} "
              f"avg_depth={avg_depth:.0f} avg_cons={avg_cons:.0f} → {out_path}", flush=True)


# ── Main ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: bench_omni_v2.py <model_label> <model_id>")
        sys.exit(1)

    # First call: sample and save 50 random IDs (idempotent)
    if not os.path.exists(SAMPLE_IDS_PATH):
        random.seed(RANDOM_SEED)
        with open(OMNI_PATH) as f:
            all_graphs = [json.loads(line) for line in f]
        samples = random.sample(all_graphs, min(N_SAMPLES, len(all_graphs)))
        sample_ids = [g["id"] for g in samples]
        os.makedirs(RESULTS_DIR, exist_ok=True)
        json.dump(sample_ids, open(SAMPLE_IDS_PATH, "w"), indent=2)
        print(f"Sampled {len(sample_ids)} Omni-MATH questions (seed={RANDOM_SEED})",
              flush=True)

    run(sys.argv[1], sys.argv[2])
