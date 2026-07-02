#!/usr/bin/env python3
"""Summarize all benchmark results."""
import json, os
from collections import defaultdict

d = defaultdict(list)
for f in sorted(os.listdir('outputs/results/')):
    if not f.startswith('bench_') or not f.endswith('.json'): continue
    if 'merged' in f or 'summary' in f: continue
    rs = json.load(open(f'outputs/results/{f}'))
    for r in rs: d[f].append(r)

# Aggregate by model (merge batch files)
models = defaultdict(lambda: {'ok':0,'n':0,'errs':0,'depth':[],'lit':[],'cons':[]})
for f, rs in d.items():
    # Extract model name: "bench_DeepSeek-v4-Flash-00000.json" → "DeepSeek-v4-Flash"
    name = f.replace('bench_','').replace('.json','')
    # Remove the trailing _NNNNN batch range or _omni suffix
    for suffix in ['_omni','-omni']:
        if name.endswith(suffix): name = name[:-len(suffix)]
    # Remove -NNNNN batch suffix
    import re
    name = re.sub(r'-\d{5}$','',name)
    # Remove _omni at end
    name = name.replace('_omni','')
    # Normalize: replace last _XXXXX with nothing
    name = re.sub(r'_\d{5}$','',name)

    model = name
    for r in rs:
        if 'error' in r: models[model]['errs'] += 1; continue
        models[model]['n'] += 1
        if r.get('correct'): models[model]['ok'] += 1
        models[model]['depth'].append(r['depth'])
        models[model]['lit'].append(r.get('lit',0))
        models[model]['cons'].append(r.get('cons',0))

print(f"{'Model':<25} {'N':>5} {'Errs':>5} {'Acc%':>7} {'Depth':>7} {'Cons':>7} {'Lit':>5}")
print('-'*65)
for m in sorted(models.keys()):
    c = models[m]; n = c['n']
    if n == 0: print(f'{m:<25} {n:>5} {c["errs"]:>5}  (all errored)'); continue
    acc = c['ok']/n*100; d = sum(c['depth'])/n; cons = sum(c['cons'])/n; l = sum(c['lit'])/n
    print(f'{m:<25} {n:>5} {c["errs"]:>5} {acc:>6.1f}% {d:>7.1f} {cons:>7.1f} {l:>5.1f}')
