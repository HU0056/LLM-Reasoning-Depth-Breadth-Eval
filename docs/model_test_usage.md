# Model Test 模块使用说明

## 概述

`model_test` 模块负责构造 prompt、调用 LLM API 生成推理输出，为下游 scoring pipeline 提供输入。支持任意 OpenAI-compatible API（OpenAI / DeepSeek / Qwen / 本地 vLLM 等）。

## 架构

```
prompt_builder.py    →  构造 prompt（math / deduction 双模式）
        ↓
generate_with_api.py →  批量调度，单条/多条采样
        ↓
llm_client.py        →  OpenAI 标准客户端，读 .env 配置
        ↓
API (OpenAI-compatible)
        ↓
JSONL outputs        →  喂给 scorer pipeline
```

核心类 `LLMClient` 封装了认证、重试、多路径采样，`generate_with_api.py` 负责 benchmark → outputs 的批量转换。

## 环境配置

### 1. 创建 `.env` 文件

在项目根目录下，复制并编辑：

```bash
cp .env.example .env
```

`.env` 内容：

```ini
API_KEY=sk-your-api-key
BASE_URL=https://api.openai.com/v1
MODEL_NAME=gpt-4o-mini
```

### 2. 各平台参考配置

| 平台 | BASE_URL | MODEL_NAME 示例 |
|---|---|---|
| OpenAI | `https://api.openai.com/v1` | `gpt-4o` / `gpt-4o-mini` |
| DeepSeek | `https://api.deepseek.com` | `deepseek-chat` |
| Qwen (通义千问) | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `qwen-plus` / `qwen-max` |
| 本地 vLLM | `http://localhost:8000/v1` | 部署时指定的模型名 |

### 3. 无 API 时的 demo 模式

当 `API_KEY` 未配置或为 `your_api_key_here` 时，`LLMClient` 自动进入 demo 模式：

- 不发起真实 API 调用
- 会打印所有 prompt 供人工检查
- 提示用户使用 `data/model_outputs/demo_model_outputs.jsonl` 作为替代

## 快速开始

```bash
# 查看 prompt（不调 API）
conda run -n LLMReason python scripts/run_model_test.py \
    --benchmark data/processed/demo_benchmark.jsonl \
    --dry-run

# 小规模测试（2 条）
conda run -n LLMReason python scripts/run_model_test.py \
    --benchmark data/processed/demo_benchmark.jsonl \
    --output data/model_outputs/test.jsonl \
    --limit 2

# 全量生成（CoT，每样本 1 条路径）
conda run -n LLMReason python scripts/run_model_test.py \
    --benchmark data/processed/demo_benchmark.jsonl \
    --output data/model_outputs/cot_outputs.jsonl

# Self-Consistency（每样本 5 条路径，用于 breadth 评分）
conda run -n LLMReason python scripts/run_model_test.py \
    --benchmark data/processed/demo_benchmark.jsonl \
    --output data/model_outputs/sc5_outputs.jsonl \
    --n 5 --temperature 0.8

# 加上 API 调用间隔（避免速率限制）
conda run -n LLMReason python scripts/run_model_test.py \
    --benchmark data/processed/demo_benchmark.jsonl \
    --output data/model_outputs/gsm8k_outputs.jsonl \
    --delay 1.0
```

## 参数说明

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--benchmark` | (必需) | 输入 benchmark JSONL 路径 |
| `--output` | (必需) | 输出 JSONL 路径 |
| `--model` | 来自 `.env` | 覆盖模型名 |
| `--n` | `1` | 每题采样数 (1=CoT, >1=Self-Consistency) |
| `--temperature` | `0.7` | 采样温度 |
| `--limit` | 无 | 仅处理前 N 条 |
| `--delay` | `0.0` | API 调用间隔（秒） |
| `--dry-run` | `False` | 仅打印 prompt，不调 API |

## Prompt 设计

### Math（GSM8K 等数学题）

简洁三行指令，让模型自由发挥思考过程：

```
Solve this math problem step by step.

Question: {题目原文}

Output format:
Step 1: [first reasoning step with calculation]
Step 2: [next step]
...
Final Answer: [the numerical answer, no extra text]
```

多路径模式（`--n > 1`）会额外插入：

```
Solve this problem in {n} different ways. Label each approach with 'Path 1:', 'Path 2:', etc.
```

### Deduction（规则逻辑题）

中文 prompt，同样简洁：

```
请根据以下事实和规则逐步推理：

事实：
- A 成立

规则：
- A -> B
- B -> C

问题：C 是否成立？

输出格式（严格按此格式）：
Step 1: [推理步骤]
Step 2: [推理步骤]
...
Final Answer: [结论]
```

## 输出格式

生成的 JSONL 与 scoring pipeline 完全对齐：

```json
{
  "sample_id": "sample_001",
  "model_name": "gpt-4o-mini",
  "output_type": "cot",
  "response": "Step 1: A 成立。\nStep 2: 由 A -> B，所以 B。\nFinal Answer: C 成立。"
}
```

- `output_type`: `"cot"` (n=1) 或 `"sc_5"` (n=5)
- `response`: 多路径时合并为 `Path 1:\n...\n\nPath 2:\n...` 格式，可被 `step_splitter` 自动拆分

## 生成后评分

```bash
# 评分
conda run -n LLMReason python scripts/03_score_outputs.py \
    --benchmark data/processed/demo_benchmark.jsonl \
    --outputs data/model_outputs/test.jsonl \
    --save outputs/results/test_results.jsonl

# 分析报告 + 图表
conda run -n LLMReason python scripts/04_analyze_results.py \
    --results outputs/results/test_results.jsonl \
    --benchmark data/processed/demo_benchmark.jsonl \
    --report outputs/reports/test_summary.csv \
    --figures outputs/figures

# 或一键跑通
conda run -n LLMReason python scripts/run_all_demo.py
```

## 关键代码路径

| 文件 | 作用 |
|---|---|
| `src/reasoning_eval/model_test/llm_client.py` | OpenAI 客户端，含重试/回退逻辑 |
| `src/reasoning_eval/model_test/prompt_builder.py` | 多任务 prompt 构造 |
| `src/reasoning_eval/model_test/generate_with_api.py` | 批量生成入口 |
| `scripts/run_model_test.py` | CLI 脚本 |
