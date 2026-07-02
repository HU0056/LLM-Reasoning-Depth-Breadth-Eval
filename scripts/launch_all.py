#!/usr/bin/env python3
"""Master launcher: spawns 50 agents for 5 models × 50 Omni-MATH samples."""
import subprocess, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

AGENT = '/tmp/agent.py'
MODELS = [
    ("DeepSeek-v4-Pro",  "deepseek-v4-pro",            "DS"),
    ("DeepSeek-v4-Flash", "deepseek-v4-flash",          "DS"),
    ("GLM-4.5",           "glm-4.5",                    "GLM"),
    ("Qwen2.5-7B",        "Qwen/Qwen2.5-7B-Instruct",   "SF"),
    ("Qwen3.6-27B",       "Qwen/Qwen3.6-27B",           "SF"),
]
BATCH = 5
TOTAL_PER = 50

os.makedirs('/tmp/bench_logs', exist_ok=True)

procs = []
for label, mid, src in MODELS:
    for start in range(0, TOTAL_PER, BATCH):
        end = start + BATCH
        log = f'/tmp/bench_logs/{label.replace(".","_")}_{start}.log'
        p = subprocess.Popen(
            [sys.executable, '-u', AGENT, label, mid, src, str(start), str(end)],
            stdout=open(log, 'w'), stderr=subprocess.STDOUT,
        )
        procs.append(p)

print(f"Launched {len(procs)} agents ({len(MODELS)} models × {TOTAL_PER//BATCH} batches × {BATCH} samples = {len(procs)*BATCH} total)")
print(f"Monitor: grep -l done /tmp/bench_logs/*.log | wc -l")
print(f"Results: python3 scripts/collect_benchmark.py")
