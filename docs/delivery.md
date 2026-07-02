# 交付说明

## 项目概述

LLM 推理过程深度/广度评估框架。将模型 Chain-of-Thought 拆为原子步骤，映射到黄金推理 DAG，
输出 Depth、Consistency 评分及 DAG Lighting 可视化。

## 架构

```
src/reasoning_eval/
├── harness/              # LLM Agent Framework — 从 ground truth 构造 Gold DAG
│   ├── pipeline.py       # 单次调用 Structurer，直接产出 DAG (~3s/题)
│   ├── agents.py         # Structurer (声明步骤+依赖+数学依据)
│   ├── math_verifier.py  # 数学计算验证（唯一确定性检查）
│   ├── verifiers.py      # 信息性验证（非阻塞）
│   └── schemas.py        # Pydantic DSL: DagNode, DagEdge, Justification
├── scorer/               # 评分管道
│   ├── mapper.py         # 批量 LLM 步骤→节点映射
│   ├── verifier.py       # 图拓扑验证 (reachable-path = valid)
│   ├── depth_scorer.py   # 难度加权深度 (Dijkstra 最短路径)
│   ├── consistency_scorer.py  # 四维一致性
│   ├── breadth_scorer.py # 分叉覆盖率
│   ├── dag_lighter.py    # DAG 点亮 (lit/jump/wrong/redundant/unvisited)
│   ├── step_splitter.py  # 模型输出→步骤列表 (15步上限)
│   └── evaluator.py      # 编排 + 自动点亮相间节点
├── model_test/           # LLM 客户端
│   ├── llm_client.py     # OpenAI-compatible, max_tokens=262144
│   └── prompt_builder.py # math/deduction 双模式
├── dataset/              # 规则逻辑题 DAG (规则引擎前向闭包)
└── analysis/             # 统计报告 + 可视化
```

## 评分指标

| 指标 | 含义 | 区间 |
|---|---|---|
| Depth | `1 − D_remain/D_total`，难度加权进度 | 0-100 |
| Consistency | 四维加权（逻辑非矛盾+依赖完整+目标对齐+结构连贯）× 答案因子 | 0-100 |
| Breadth | 关键分叉覆盖率 | 0-100 或 None |

详见 [docs/scoring.md](scoring.md)。

## 数据集

| 数据集 | 规模 | DAG 来源 |
|---|---|---|
| Demo 规则逻辑 | 5 题 | 规则引擎前向闭包 |
| GSM8K | 7,473 train | LLM 依赖标注 |
| Omni-MATH | 1,600 test | DeepSeek 依赖标注 |

## Benchmark 结果（5 样本/模型/数据集）

### GSM8K

| 模型 | Acc | Depth | Consistency |
|---|---|---|---|
| **DeepSeek-v4-Pro** | **100%** | **65** | 71 |
| DeepSeek-v4-Flash | 100% | 40 | 84 |

### Omni-MATH

| 模型 | Acc | Depth | Consistency |
|---|---|---|---|
| **DeepSeek-v4-Pro** | **40%** | **70** | 53 |
| DeepSeek-v4-Flash | 20% | 20 | 45 |
| Qwen2.5-7B | 0% | 0 | 33 |
| Qwen3.5-4B | 0% | 0 | 33 |

## 测试

41 个单元测试全部通过：

```bash
pytest  # 41 passed
```

## 运行

```bash
pip install -r requirements.txt
bash scripts/run_all_demo.py           # 规则逻辑题 Demo
bash scripts/run_gsm8k.sh              # GSM8K benchmark
python3 scripts/bench_v2.py <model> <id> 0 50  # Omni-MATH benchmark
python3 scripts/summarize.py           # 汇总结果
```

## 环境

`.env`:
```
API_KEY=sk-...              # DeepSeek API
BASE_URL=https://api.deepseek.com
MODEL_NAME=deepseek-chat
SILICON_FLOW_API_KEY=sk-... # SiliconFlow
GLM_API_KEY=...             # 智谱官方
```
