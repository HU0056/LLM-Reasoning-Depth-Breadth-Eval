#!/bin/bash
# Launch 50 parallel Omni-MATH benchmark agents (5 models × 50 samples)
# Usage: bash scripts/launch_benchmark.sh
mkdir -p /tmp/bench_logs

AGENT=/tmp/agent.py

# Model 1: DeepSeek-v4-Pro (DeepSeek official, reasoning_effort=low)
for i in $(seq 0 5 45); do
  python3 -u "$AGENT" "DeepSeek-v4-Pro" "deepseek-v4-pro" "DS" $i $((i+5)) > /tmp/bench_logs/pro_${i}.log 2>&1 &
done

# Model 2: DeepSeek-v4-Flash (DeepSeek official, reasoning_effort=low)
for i in $(seq 0 5 45); do
  python3 -u "$AGENT" "DeepSeek-v4-Flash" "deepseek-v4-flash" "DS" $i $((i+5)) > /tmp/bench_logs/flash_${i}.log 2>&1 &
done

# Model 3: GLM-4.5 (Zhipu official API)
for i in $(seq 0 5 45); do
  python3 -u "$AGENT" "GLM-4.5" "glm-4.5" "GLM" $i $((i+5)) > /tmp/bench_logs/glm45_${i}.log 2>&1 &
done

# Model 4: Qwen2.5-7B (SiliconFlow)
for i in $(seq 0 5 45); do
  python3 -u "$AGENT" "Qwen2.5-7B" "Qwen/Qwen2.5-7B-Instruct" "SF" $i $((i+5)) > /tmp/bench_logs/qwen25_${i}.log 2>&1 &
done

# Model 5: Qwen3.6-27B (SiliconFlow)
for i in $(seq 0 5 45); do
  python3 -u "$AGENT" "Qwen3.6-27B" "Qwen/Qwen3.6-27B" "SF" $i $((i+5)) > /tmp/bench_logs/qwen36_${i}.log 2>&1 &
done

echo "50 agents launched (5 models × 10 batches × 5 samples = 250 total)"
echo "Monitor: tail -f /tmp/bench_logs/*.log"
echo "Check progress: grep -l done /tmp/bench_logs/*.log | wc -l"
echo "Collect results: python3 scripts/collect_benchmark.py"
