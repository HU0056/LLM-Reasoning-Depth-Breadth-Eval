# 交付说明

## 项目概述

LLM 推理过程深度/广度评估框架。将模型 Chain-of-Thought 拆为原子步骤，映射到黄金推理 DAG，
输出 Depth、Consistency 评分及 DAG Lighting 可视化。

## 架构

```
                     ┌────────────────────────┐
                     │   Gold DAG (预构建)      │
                     │   从 ground truth 标注    │
                     └───────────┬────────────┘
                                 │
┌────────────────────┐          │
│  模型推理            │          │
│  (Step-by-step CoT) │          │
└─────────┬──────────┘          │
          │                      │
          ▼                      ▼
   ┌──────────────┐    ┌──────────────────┐
   │ Step Splitter│    │  Graph Normalize │
   │ (按行拆分)    │    │  (flat→structured)│
   └──────┬───────┘    └────────┬─────────┘
          │                      │
          ▼                      │
   ┌──────────────┐              │
   │   Mapper     │◄─────────────┘
   │ (LLM 批量匹配) │  所有步骤 → gold 节点
   └──────┬───────┘
          │
          ▼
   ┌──────────────┐
   │   Verifier   │  图拓扑验证
   │ (reachable=valid)│  + 自动点亮中间节点
   └──────┬───────┘
          │
          ▼
   ┌──────────────────────────┐
   │  并行评分                  │
   │  Depth / Consistency      │
   │  (+ Breadth 如有标注)      │
   └──────────┬───────────────┘
              │
              ▼
   ┌──────────────┐
   │  DAG Lighter │  节点/边状态标注
   │  + 完整输出保存│  response + lighted_graph
   └──────────────┘
```

### 流水线要点

1. **Step Splitter**: 按行拆分模型输出，识别 `Step N:` / `Path N:` 前缀、`Final Answer:` 和 `\boxed{}`，最多 15 步
2. **Mapper**: 所有步骤在一次 LLM 调用中完成匹配。三层加速：规则文本精确匹配 → 单字母命题快速匹配 → LLM 语义匹配。anti-hallucination: node_id 不在候选列表 → 直接驳回
3. **Verifier v3**: `is_reachable` 判 valid (不再要求直接后继)，自动点亮最短路径中间节点
4. **Depth**: `1 - D_remain/D_total`，Dijkstra 加权最短路径
5. **Consistency**: 四维加权 × 答案因子
6. **DAG Lighter**: 每个节点标 lit/jump/wrong/redundant/contradiction/unvisited，每条边标 used_valid/skipped/wrong/unused

## 模块结构

```
src/reasoning_eval/
├── harness/              # LLM Agent Framework — 从 ground truth 构造 Gold DAG
│   ├── pipeline.py       # 单次调用 Structurer，直接产出 DAG (~3s/题)
│   ├── agents.py         # Structurer (声明步骤+依赖+数学依据)
│   ├── math_verifier.py  # 数学计算验证（唯一确定性检查）
│   ├── verifiers.py      # 信息性验证（非阻塞）
│   ├── schemas.py        # Pydantic DSL: DagNode, DagEdge, Justification
│   └── prompts.py        # LLM 提示模板
├── scorer/               # 评分管道
│   ├── evaluator.py      # 总编排 + 自动点亮相间节点
│   ├── mapper.py         # 批量 LLM 步骤→节点映射（1 次调用）
│   ├── verifier.py       # 图拓扑验证 (reachable-path = valid)
│   ├── depth_scorer.py   # 难度加权深度 (Dijkstra 最短路径)
│   ├── consistency_scorer.py  # 四维一致性
│   ├── breadth_scorer.py # 分叉覆盖率
│   ├── dag_lighter.py    # DAG 点亮 (lit/jump/wrong/redundant/contradiction)
│   └── step_splitter.py  # 模型输出→步骤列表 (15步上限)
├── model_test/           # LLM 客户端
│   ├── llm_client.py     # OpenAI-compatible, max_tokens=262144
│   └── prompt_builder.py # math/deduction 双模式 Prompt 构建
├── dataset/              # 数据处理
│   └── graph_utils.py    # 图规范化 + 拓扑分析
├── common/               # 共用工具
│   ├── schema.py          # EvaluationResult, MappingResult, VerificationResult
│   ├── text_utils.py      # 文本矛盾检测
│   └── io_utils.py        # JSONL 读写
└── analysis/             # 统计报告 + 可视化
```

## Benchmark 运行

### GSM8K (50 题, 4 模型)

```bash
# 全部模型并行 (launcher 模式: 采样 + 分发 agent)
python3 scripts/bench_gsm8k_v2.py

# 单模型
python3 scripts/bench_gsm8k_v2.py "DeepSeek-v4-Pro" "deepseek-v4-pro" "DS"
```

### Omni-MATH (50 题, DeepSeek only)

```bash
python3 scripts/bench_omni_v2.py "DeepSeek-v4-Pro" "deepseek-v4-pro"
```

### 汇总

```bash
python3 scripts/summarize.py
```

## 输出文件位置

```
outputs/
├── results/
│   ├── bench_<model>_gsm8k_50.json    # 每题一条: correct, depth, cons, states,
│   │                                    lit_nodes, response, lighted_graph
│   ├── bench_<model>_omni_50.json     # 同上，Omni-MATH 版本
│   ├── gsm8k_50_sample_ids.json       # 采样题目 ID (seed=42)
│   ├── omni_50_sample_ids.json
│   └── bench_summary.json             # 汇总统计
└── logs/
    ├── gsm8k_<model>.log               # 每个模型的运行日志
    └── omni_<model>.log
```

每条结果记录完整保存：
- `response`: 模型原始 CoT 输出（可重评分）
- `lighted_graph.nodes`: 每个 gold 节点的状态
- `lighted_graph.edges`: 每条依赖边的使用情况
- `lighted_graph.steps`: 每一步到节点的映射 + 原因

## 评分指标

| 指标 | 含义 | 区间 |
|---|---|---|
| Depth | `1 − D_remain/D_total`，难度加权进度 | 0–100 |
| Consistency | 四维加权（逻辑非矛盾+依赖完整+目标对齐+结构连贯）× 答案因子 | 0–100 |
| Breadth | 关键分叉覆盖率 | 0–100 或 None |

详见 [docs/scoring.md](scoring.md)。

## Benchmark 结果

### GSM8K（50 随机样本/模型，seed=42）

| 模型 | API | Acc | Depth | Consistency |
|------|-----|-----|-------|-------------|
| **DeepSeek-v4-Pro** | 官方 | **98%** | **57** | **86** |
| **DeepSeek-v4-Flash** | 官方 | **96%** | **55** | **84** |
| **Qwen3-8B** | SiliconFlow | **96%** | **59** | **82** |
| Qwen2.5-7B | SiliconFlow | 0% | 22 | 29 |

### Omni-MATH（50 随机样本/模型，seed=42，IMO 级别难度）

| 模型 | API | Acc | Depth | Consistency |
|------|-----|-----|-------|-------------|
| **DeepSeek-v4-Flash** | 官方 | **34%** | **31** | **52** |
| DeepSeek-v4-Pro | 官方 | 22% | 28 | 46 |

## 数据集

| 数据集 | 规模 | DAG 来源 | 特点 |
|------|------|--------|------|
| Demo 规则逻辑 | 5 题 | 规则引擎前向闭包 | distractor/反事实分支 |
| GSM8K | 551 题预处理 | LLM 依赖标注 | 四则运算，简单推理 |
| Omni-MATH | 2,200 题 | DeepSeek 依赖标注 | 竞赛数学，IMO 难度 |

数据位置：
- `data/processed/gsm8k/train_graphs_std.jsonl` (140 题), `test_graphs_std.jsonl` (411 题)
- `data/processed/omni_math/test_graphs_std.jsonl` (2200 题)

## 测试

```bash
pytest  # 41 passed
```

## 环境

`.env`:
```bash
API_KEY=sk-...                    # DeepSeek API（必须）
BASE_URL=https://api.deepseek.com
MODEL_NAME=deepseek-chat          # Mapper/Harness 用
SILICON_FLOW_API_KEY=sk-...       # Qwen 系列（可选）
GLM_API_KEY=...                   # 智谱 GLM（可选）
```
