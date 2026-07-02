#!/usr/bin/env python3
"""Collect results from all batch agents into per-model summaries."""
import json, os, sys
from collections import defaultdict

INDIR = '/home/lz/LLM-Reasoning-Depth-Breadth-Eval/outputs/results'

# Merge all batch files
merged = defaultdict(list)
for f in sorted(os.listdir(INDIR)):
    if not f.startswith('bench_') or not f.endswith('.json'): continue
    # Extract model name (before the sample range)
    path = os.path.join(INDIR, f)
    rs = json.load(open(path))
    # Model name is bench_<model>_<range>.json
    name = f.replace('bench_','').rsplit('_0',1)[0].replace('_','-')  # heuristic
    for r in rs:
        r['_model'] = name
        merged[name].append(r)

# Print per-model summary
print(f"{'Model':<25} {'Done':>5} {'Acc%':>7} {'Depth':>7} {'Cons':>7} {'Lit':>5}")
print('-'*60)

full_results = {}
for model in sorted(merged.keys()):
    rs = merged[model]
    ok = sum(1 for r in rs if r.get('correct'))
    errs = sum(1 for r in rs if 'error' in r)
    n = len(rs) - errs
    if n == 0: continue
    depths = [r['depth'] for r in rs if 'error' not in r]
    conss = [r['cons'] for r in rs if 'error' not in r]
    lits = [r.get('lit',0) for r in rs if 'error' not in r]
    print(f"{model:<25} {n:>4}/{len(rs)} {ok/n*100:>6.1f}% {sum(depths)/n:>7.1f} {sum(conss)/n:>7.1f} {sum(lits)/n:>5.1f}")
    full_results[model] = {
        'n': n, 'total': len(rs), 'errors': errs,
        'acc': round(ok/n*100,1), 'depth': round(sum(depths)/n,1),
        'cons': round(sum(conss)/n,1), 'lit': round(sum(lits)/n,1),
    }

# Save merged data
with open(f'{INDIR}/bench_all_merged.json','w') as f:
    json.dump({k: v for k,v in merged.items()}, f, ensure_ascii=False, indent=2)
with open(f'{INDIR}/bench_summary.json','w') as f:
    json.dump(full_results, f, ensure_ascii=False, indent=2)
print(f"\nSaved: {INDIR}/bench_summary.json")
