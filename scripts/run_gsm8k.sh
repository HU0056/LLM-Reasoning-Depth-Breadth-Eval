#!/bin/bash
# GSM8K benchmark: 5 samples per model, 4 models in parallel
cd /home/lz/LLM-Reasoning-Depth-Breadth-Eval
rm -f /tmp/bench_logs/gsm_*.log

python3 -u scripts/bench_gsm8k.py "DeepSeek-v4-Pro" "deepseek-v4-pro" "DS" 100 > /tmp/bench_logs/gsm_pro.log 2>&1 &
echo "Pro: PID $!"

python3 -u scripts/bench_gsm8k.py "DeepSeek-v4-Flash" "deepseek-v4-flash" "DS" 200 > /tmp/bench_logs/gsm_flash.log 2>&1 &
echo "Flash: PID $!"

python3 -u scripts/bench_gsm8k.py "GLM-5.2" "glm-5.2" "GLM" 300 > /tmp/bench_logs/gsm_glm.log 2>&1 &
echo "GLM: PID $!"

python3 -u scripts/bench_gsm8k.py "Qwen2.5-7B" "Qwen/Qwen2.5-7B-Instruct" "SF" 400 > /tmp/bench_logs/gsm_qwen25.log 2>&1 &
echo "Qwen2.5: PID $!"

echo "4 agents running. Monitor: grep acc= /tmp/bench_logs/gsm_*.log"
