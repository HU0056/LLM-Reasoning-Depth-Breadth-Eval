#!/usr/bin/env python3 -u
"""GSM8K Unified Benchmark v2: 50 random samples, multi-model parallel, full DAG pipeline.

Launcher mode:
    python scripts/bench_gsm8k_v2.py
    → Samples 50 random GSM8K questions, spawns one agent per model in parallel.

Agent mode (called internally by launcher):
    python scripts/bench_gsm8k_v2.py --agent <label> <model_id> <api_src>
    → Runs full pipeline (infer → split → map → DAG → score) on all 50 samples.

Single-model direct mode:
    python scripts/bench_gsm8k_v2.py <label> <model_id> <api_src>
    → Same as agent mode but run directly (not via subprocess).
"""

import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

# ── Project root ──────────────────────────────────────────────────────────
PROJECT = Path(__file__).resolve().parent.parent
SRC = str(PROJECT / "src")
sys.path.insert(0, SRC)

# ── Configuration ─────────────────────────────────────────────────────────
N_SAMPLES = 50
TIMEOUT = 90  # seconds per sample
MAX_TOKENS = 4096
TEMPERATURE = 0.3

# All models use the SAME 50 questions (fixed seed for reproducibility)
RANDOM_SEED = 42

# Models to benchmark (skip GLM entirely)
MODELS = [
    # (label, model_id, api_source)
    # Large models — DeepSeek official API
    ("DeepSeek-v4-Pro",   "deepseek-v4-pro",            "DS"),
    ("DeepSeek-v4-Flash", "deepseek-v4-flash",          "DS"),
    # Small models — SiliconFlow
    ("Qwen2.5-7B",        "Qwen/Qwen2.5-7B-Instruct",   "SF"),
    ("Qwen3-8B",          "Qwen/Qwen3-8B",              "SF"),
]

GSM8K_PATH = PROJECT / "data" / "processed" / "gsm8k" / "train_graphs_std.jsonl"
SAMPLE_IDS_PATH = PROJECT / "outputs" / "results" / "gsm8k_50_sample_ids.json"
RESULTS_DIR = PROJECT / "outputs" / "results"
LOG_DIR = PROJECT / "outputs" / "logs"


# ═══════════════════════════════════════════════════════════════════════════
# Agent Mode
# ═══════════════════════════════════════════════════════════════════════════

def run_agent(model_label: str, model_id: str, api_src: str):
    """Full benchmark pipeline for one model: infer → split → map → DAG → score."""
    from dotenv import load_dotenv
    load_dotenv(PROJECT / ".env")

    from openai import OpenAI

    from reasoning_eval.scorer.evaluator import evaluate_one
    from reasoning_eval.dataset.graph_utils import normalize_graph
    from reasoning_eval.model_test.prompt_builder import build_prompt, get_system_prompt

    # ── API clients ──────────────────────────────────────────────────────
    if api_src == "DS":
        gen_client = OpenAI(
            api_key=os.getenv("API_KEY"),
            base_url="https://api.deepseek.com",
        )
        reasoning = model_id in ("deepseek-v4-pro", "deepseek-v4-flash")
    else:
        gen_client = OpenAI(
            api_key=os.getenv("SILICON_FLOW_API_KEY"),
            base_url="https://api.siliconflow.cn/v1",
        )
        reasoning = False

    # Mapper always uses deepseek-chat (non-reasoning, reliable JSON)
    mapper_client = OpenAI(
        api_key=os.getenv("API_KEY"),
        base_url="https://api.deepseek.com",
    )

    # ── MC wrapper ───────────────────────────────────────────────────────
    class MC:
        demo_mode = False

        def __init__(self, client, model, reasoning=False):
            self.c = client
            self.m = model
            self.r = reasoning

        def generate(self, prompt, system=None, n=1, temperature=TEMPERATURE,
                     max_tokens=MAX_TOKENS, seed=None):
            msgs = [{"role": "system", "content": system}] if system else []
            msgs.append({"role": "user", "content": prompt})
            kw = {
                "model": self.m, "messages": msgs, "n": n,
                "temperature": temperature,
                "max_tokens": min(max_tokens, MAX_TOKENS),
                "seed": seed, "timeout": TIMEOUT,
            }
            if self.r:
                kw["extra_body"] = {"reasoning_effort": "low"}
            r = self.c.chat.completions.create(**kw)
            return [c.message.content or "" for c in r.choices]

    target_client = MC(gen_client, model_id, reasoning)
    mapper_mc = MC(mapper_client, "deepseek-chat")

    # ── Load sample IDs ──────────────────────────────────────────────────
    if SAMPLE_IDS_PATH.exists():
        sample_ids = set(json.loads(SAMPLE_IDS_PATH.read_text()))
    else:
        sample_ids = None  # use all

    with open(GSM8K_PATH) as f:
        all_graphs = [json.loads(line) for line in f]

    if sample_ids:
        samples = [g for g in all_graphs if g["id"] in sample_ids]
    else:
        random.seed(RANDOM_SEED)
        samples = random.sample(all_graphs, min(N_SAMPLES, len(all_graphs)))

    print(f"[{model_label}] Loaded {len(samples)} samples (api={api_src}, "
          f"reasoning={reasoning})", flush=True)

    # ── Run ──────────────────────────────────────────────────────────────
    ok = 0
    n_done = 0
    results = []

    for s in samples:
        sid = s["id"]
        question = s["question"]
        gold_answer = s["gold_answer"]
        g_raw = s["gold_reasoning_graph"]
        t0 = time.time()

        try:
            s["task_type"] = "math"
            prompt = build_prompt(s)
            system_prompt = get_system_prompt(s)

            resp = target_client.generate(
                prompt=prompt, system=system_prompt,
                temperature=TEMPERATURE, max_tokens=MAX_TOKENS,
            )
            resp_text = resp[0]

            graph = normalize_graph(g_raw)
            r = evaluate_one(
                {
                    "id": sid,
                    "gold_reasoning_graph": graph,
                    "gold_answer": gold_answer,
                    "key_branch_nodes": [],
                },
                {
                    "sample_id": sid,
                    "model_name": model_label,
                    "output_type": "cot",
                    "response": resp_text,
                },
                mapper_client=mapper_mc,
            )
            elapsed = time.time() - t0

            lit_nodes = [n for n, st in r.lighted_graph["nodes"].items() if st == "lit"]
            states = {
                st: sum(1 for v in r.lighted_graph["nodes"].values() if v == st)
                for st in set(r.lighted_graph["nodes"].values())
            }

            n_done += 1
            if r.answer_correct:
                ok += 1

            results.append({
                "id": sid,
                "correct": r.answer_correct,
                "depth": r.score_depth,
                "cons": r.score_consistency,
                "lit": len(lit_nodes),
                "time": round(elapsed, 1),
                "states": states,
                "lit_nodes": lit_nodes,
                "response": resp_text,
                "lighted_graph": r.lighted_graph,
            })

            status = "✓" if r.answer_correct else "✗"
            print(f"  [{n_done:2d}/{len(samples)} {sid}] {status} "
                  f"d={r.score_depth:.0f} c={r.score_consistency:.0f} "
                  f"lit={len(lit_nodes)} ({elapsed:.0f}s)", flush=True)

        except Exception as e:
            print(f"  [{sid}]: FAIL {str(e)[:120]}", flush=True)
            results.append({"id": sid, "error": str(e)[:200]})

    # ── Save ─────────────────────────────────────────────────────────────
    tag = model_label.replace(".", "_").replace("-", "_").replace("/", "_")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"bench_{tag}_gsm8k_50.json"
    json.dump(results, open(out_path, "w"), ensure_ascii=False, indent=2)

    if n_done:
        avg_depth = sum(r.get("depth", 0) for r in results if "error" not in r) / n_done
        avg_cons = sum(r.get("cons", 0) for r in results if "error" not in r) / n_done
        print(f"\n[{model_label}] DONE: acc={ok}/{n_done}={ok/n_done:.1%} "
              f"avg_depth={avg_depth:.0f} avg_cons={avg_cons:.0f} "
              f"→ {out_path.name}", flush=True)
    else:
        print(f"\n[{model_label}] DONE: 0/{len(samples)} completed → {out_path.name}",
              flush=True)


# ═══════════════════════════════════════════════════════════════════════════
# Launcher Mode
# ═══════════════════════════════════════════════════════════════════════════

def launch_all():
    """Sample 50 random questions, then spawn parallel agents for all models."""
    print("=" * 72)
    print("GSM8K Benchmark v2 — 50 samples, multi-model parallel")
    print("=" * 72)

    # ── Sample questions ─────────────────────────────────────────────────
    with open(GSM8K_PATH) as f:
        all_graphs = [json.loads(line) for line in f]

    random.seed(RANDOM_SEED)
    samples = random.sample(all_graphs, min(N_SAMPLES, len(all_graphs)))
    sample_ids = [g["id"] for g in samples]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLE_IDS_PATH.write_text(json.dumps(sample_ids, indent=2))
    print(f"\nSampled {len(sample_ids)} questions from {len(all_graphs)} total (seed={RANDOM_SEED})")
    print(f"Sample IDs saved to: {SAMPLE_IDS_PATH}")

    # Show sample info
    print(f"\nSample questions:")
    for i, (g, sid) in enumerate(zip(samples[:3], sample_ids[:3])):
        q = g["question"][:80]
        print(f"  [{i+1}] {sid}: {q}... (answer={g['gold_answer']})")
    print(f"  ... and {len(samples) - 3} more")

    # ── Launch agents ────────────────────────────────────────────────────
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    this_script = __file__

    print(f"\nLaunching {len(MODELS)} model agents in parallel...")
    print("-" * 72)

    procs = {}
    for label, mid, src in MODELS:
        tag = label.replace(".", "_").replace("-", "_").replace("/", "_")
        log_path = LOG_DIR / f"gsm8k_{tag}.log"
        p = subprocess.Popen(
            [sys.executable, "-u", this_script, "--agent", label, mid, src],
            stdout=open(str(log_path), "w"),
            stderr=subprocess.STDOUT,
        )
        procs[label] = p
        print(f"  [{label}] pid={p.pid}  log={log_path}")

    print(f"\n{len(procs)} agents running. Monitor with:")
    print(f"  tail -f outputs/logs/gsm8k_*.log")
    print(f"  grep -c DONE outputs/logs/gsm8k_*.log")
    print()

    # ── Wait for all ─────────────────────────────────────────────────────
    for label, p in procs.items():
        rc = p.wait()
        status = "OK" if rc == 0 else f"exit={rc}"
        print(f"  [{label}] finished ({status})")

    print(f"\nAll agents done. Results: outputs/results/bench_*_gsm8k_50.json")
    print("Run: python scripts/summarize.py  (to aggregate)")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--agent":
        # Agent mode: python bench_gsm8k_v2.py --agent <label> <model_id> <api_src>
        if len(sys.argv) != 5:
            print("Usage: bench_gsm8k_v2.py --agent <label> <model_id> <api_src>")
            sys.exit(1)
        run_agent(sys.argv[2], sys.argv[3], sys.argv[4])
    elif len(sys.argv) == 4:
        # Direct single-model mode: python bench_gsm8k_v2.py <label> <model_id> <api_src>
        run_agent(sys.argv[1], sys.argv[2], sys.argv[3])
    else:
        # Launcher mode: python bench_gsm8k_v2.py
        launch_all()
