#!/usr/bin/env python3
"""Qwen model benchmark — GSM8K + Omni-MATH, 10 samples each.

Pre-built DAGs used for both datasets (no harness overhead).
deepseek-chat for mapper (reliable structured JSON).

Usage:
    python scripts/run_qwen_benchmark.py
Output:
    outputs/results/qwen_bench_*.jsonl (per model)
    outputs/results/qwen_bench_summary.json
"""

import sys, json, time, os, random, re
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
sys.path.insert(0, str(_REPO / "src"))

from dotenv import load_dotenv; load_dotenv(_REPO / ".env")
from openai import OpenAI

# ── Config ──
N         = 25
TIMEOUT   = 45
GEN_TOK   = 4096
MAP_TOK   = 512
SEED      = 42
random.seed(SEED)

# Mapper: deepseek-chat (reliable JSON, non-reasoning)
ds_key = os.getenv("API_KEY")
ds_url = os.getenv("BASE_URL", "https://api.deepseek.com")
MAPPER_MODEL = "deepseek-chat"
mapper_client = OpenAI(api_key=ds_key, base_url=ds_url)

# Models under test
MODELS = [
    ("Qwen2.5-7B", "Qwen/Qwen2.5-7B-Instruct"),
    ("Qwen3.5-4B", "Qwen/Qwen3.5-4B"),
    ("Qwen3-8B",   "Qwen/Qwen3-8B"),
]
sf_key  = os.getenv("SILICON_FLOW_API_KEY")
sf_url  = "https://api.siliconflow.cn/v1"
test_ocs = {label: OpenAI(api_key=sf_key, base_url=sf_url) for label, _ in MODELS}


# ── Client wrapper ──
class MC:
    demo_mode = False
    def __init__(self, c, m, t=TIMEOUT): self.c, self.m, self.t = c, m, t
    def generate(self, prompt, system=None, n=1, temperature=0.7,
                 max_tokens=4096, seed=None):
        msgs = [{'role':'system','content':system}] if system else []
        msgs.append({'role':'user','content':prompt})
        r = self.c.chat.completions.create(
            model=self.m, messages=msgs, n=n, temperature=temperature,
            max_tokens=min(max_tokens, 4096), seed=seed, timeout=self.t)
        return [c.message.content or '' for c in r.choices]


# ═══════════════════════════════════
# Load data
# ═══════════════════════════════════
print("=" * 55)
print("LOADING DATA")
print("=" * 55)

# Load all samples, then pick N per dataset
gsm8k_all = []
with open(_REPO / "data/raw/gsm8k/train.jsonl") as f:
    for line in f: gsm8k_all.append(json.loads(line))
gsm8k_samples = random.sample(gsm8k_all, N)
print(f"GSM8K: {len(gsm8k_samples)} samples")

# Map GSM8K samples to pre-built graphs by id
gsm8k_graphs = {}
with open(_REPO / "data/processed/gsm8k/train_graphs.jsonl") as f:
    for line in f:
        g = json.loads(line)
        gsm8k_graphs[g['gsm8k_id']] = g['gold_reasoning_graph']

omni_all = []
with open(_REPO / "data/processed/omni_math/test_graphs_std.jsonl") as f:
    for line in f: omni_all.append(json.loads(line))
for s in omni_all: s['task_type'] = 'math'
omni_samples = random.sample(omni_all, N)
print(f"Omni-MATH: {len(omni_samples)} samples")

# ═══════════════════════════════════
# Imports
# ═══════════════════════════════════
from reasoning_eval.scorer.evaluator import evaluate_one
from reasoning_eval.dataset.graph_utils import normalize_graph
from reasoning_eval.model_test.prompt_builder import build_prompt, get_system_prompt

OUT = _REPO / "outputs" / "results"
OUT.mkdir(parents=True, exist_ok=True)

all_results = {}

for label, model_id in MODELS:
    print(f"\n{'═' * 55}")
    print(f"MODEL: {label} ({model_id})")
    print(f"{'═' * 55}")

    tc = MC(test_ocs[label], model_id)
    mc = MC(mapper_client, MAPPER_MODEL)
    mres = []

    # ── GSM8K ──
    print(f"\n── GSM8K ──")
    for idx, s in enumerate(gsm8k_samples):
        sid = s['gsm8k_id']; ga = s['answer'].split("####")[-1].strip()
        graph = gsm8k_graphs.get(sid)
        e = {'dataset':'GSM8K','id':sid}
        if not graph:
            e['error'] = 'no_prebuilt_graph'; mres.append(e); continue

        t0 = time.time()
        try:
            resp = tc.generate(prompt=build_prompt(s), system=get_system_prompt(s),
                              n=1, temperature=0.3, max_tokens=GEN_TOK)[0]
            gnorm = normalize_graph(graph)
            r = evaluate_one(
                {'id':sid,'gold_reasoning_graph':gnorm,'gold_answer':ga,'key_branch_nodes':[]},
                {'sample_id':sid,'model_name':label,'output_type':'cot','response':resp},
                mapper_client=mc)
            t = time.time()-t0
            lit = [n for n,st in r.lighted_graph['nodes'].items() if st=='lit']
            dims = r.consistency_dimensions.get('dimensions',{})
            e.update(dict(correct=r.answer_correct, depth=r.score_depth,
                consistency=r.score_consistency, breadth=r.score_breadth,
                lit_nodes=lit, lit_count=len(lit), first_error=r.first_error_step,
                contradictions=r.contradiction_count, dimensions=dims,
                response=resp, total_time=round(t,1),
                lighted_graph=r.lighted_graph))
            print(f"  [{idx+1:2d}] {sid}: {'✓' if r.answer_correct else '✗'} "
                  f"d={r.score_depth:.0f} c={r.score_consistency:.0f} lit={len(lit)} ({t:.0f}s)")
        except Exception as ex:
            e['error'] = str(ex)[:200]
            print(f"  [{idx+1:2d}] {sid}: ✗ {str(ex)[:80]}")
        mres.append(e)

    # ── Omni-MATH ──
    print(f"\n── Omni-MATH ──")
    for idx, s in enumerate(omni_samples):
        sid = s['id']; ga = s['gold_answer']
        e = {'dataset':'Omni-MATH','id':sid}
        t0 = time.time()
        try:
            graph = normalize_graph(s['gold_reasoning_graph'])
            resp = tc.generate(prompt=build_prompt(s), system=get_system_prompt(s),
                              n=1, temperature=0.3, max_tokens=GEN_TOK)[0]
            r = evaluate_one(
                {'id':sid,'gold_reasoning_graph':graph,'gold_answer':ga,'key_branch_nodes':[]},
                {'sample_id':sid,'model_name':label,'output_type':'cot','response':resp},
                mapper_client=mc)
            t = time.time()-t0
            lit = [n for n,st in r.lighted_graph['nodes'].items() if st=='lit']
            dims = r.consistency_dimensions.get('dimensions',{})
            e.update(dict(correct=r.answer_correct, depth=r.score_depth,
                consistency=r.score_consistency, breadth=r.score_breadth,
                lit_nodes=lit, lit_count=len(lit), first_error=r.first_error_step,
                contradictions=r.contradiction_count, dimensions=dims,
                response=resp, total_time=round(t,1),
                lighted_graph=r.lighted_graph))
            print(f"  [{idx+1:2d}] {sid}: {'✓' if r.answer_correct else '✗'} "
                  f"d={r.score_depth:.0f} c={r.score_consistency:.0f} lit={len(lit)} ({t:.0f}s)")
        except Exception as ex:
            e['error'] = str(ex)[:200]
            print(f"  [{idx+1:2d}] {sid}: ✗ {str(ex)[:80]}")
        mres.append(e)

    # Save
    fpath = OUT / f"qwen_bench_{label.replace('.','_')}.jsonl"
    with open(fpath, 'w') as f:
        for e in mres: f.write(json.dumps(e, ensure_ascii=False)+'\n')
    all_results[label] = mres
    print(f"  → {fpath}")

# ═══════════════════════════════════
# SUMMARY
# ═══════════════════════════════════
print(f"\n{'═' * 65}")
print("SUMMARY")
print(f"{'═' * 65}")
hdr = f"{'Model':<16} {'DS':>10} {'N':>4} {'Acc%':>7} {'Depth':>7} {'Cons':>7} {'Lit':>5} {'Time':>7}"
print(hdr); print('-'*len(hdr))

summary = {}
for label in all_results:
    summary[label] = {}
    for ds in ['GSM8K','Omni-MATH']:
        rs = [r for r in all_results[label] if r['dataset']==ds and 'error' not in r]
        n = len(rs)
        if n == 0:
            summary[label][ds] = {'n':0}; continue
        s = {'n': n,
             'acc': round(sum(1 for r in rs if r['correct'])/n*100,1),
             'depth': round(sum(r['depth'] for r in rs)/n,1),
             'cons': round(sum(r['consistency'] for r in rs)/n,1),
             'lit': round(sum(r['lit_count'] for r in rs)/n,1),
             'time': round(sum(r['total_time'] for r in rs)/n,1)}
        summary[label][ds] = s
        print(f"{label if ds=='GSM8K' else '':<16} {ds:>10} {s['n']:>4} "
              f"{s['acc']:>6.1f}% {s['depth']:>7.1f} {s['cons']:>7.1f} "
              f"{s['lit']:>5.1f} {s['time']:>6.1f}s")

with open(OUT/"qwen_bench_summary.json",'w') as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print(f"\nSaved: {OUT}/qwen_bench_summary.json")
print("Done.")
