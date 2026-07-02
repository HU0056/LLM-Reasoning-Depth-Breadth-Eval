# LLM Reasoning Depth/Breadth Eval

LLM 推理过程评估框架。将模型 Chain-of-Thought 拆为原子步骤，映射到黄金推理 DAG，
从 Depth、Consistency、Breadth 三维度评估推理质量。

## 技术路线

### 黄金 DAG 构造：LLM Harness Agent Framework（单次调用）

从 ground truth (question + answer) 出发，**纯 LLM 驱动**构造推理 DAG：

```
(question, answer) → Structurer (LLM, 1 次调用) → GoldDag
```

- 零代码启发式匹配，零 Jaccard，零正则
- Harness pipeline v3：单次 LLM 调用直接产出结构化 DAG（~3s）
- 算术 justification 自动标注

### 步骤到节点映射：批量 LLM Mapper

模型输出中**所有步骤在一次 LLM 调用中**完成匹配：

```
所有模型步骤 + 所有 gold 节点 → 1 次 LLM 调用 → 所有匹配结果 + 数学验证
```

- 抗幻觉：输出 node_id 不在候选列表 → 自动驳回
- 数学验证：`math_verifier.py` 检查所有 `=` 和 `<<>>` 表达式

### 评分三维度

| 指标 | 定义 |
|---|---|
| **Depth** | `1 − D_remain/D_total` — 难度加权最短路径进度 |
| **Consistency** | 四维加权：逻辑非矛盾(.30) + 依赖完整(.35) + 目标对齐(.20) + 结构连贯(.15) |
| **Breadth** | 关键分叉覆盖率（需标注 key_branch_nodes） |

### 模型选择

| 用途 | 模型 | API |
|---|---|---|
| 问题求解 | `deepseek-v4-pro` (reasoning_effort=low) | DeepSeek |
| Harness/Mapper | `deepseek-chat` | DeepSeek |
| 备选 | `Qwen/Qwen2.5-7B-Instruct` | SiliconFlow |

> 推理模型必须用 `reasoning_effort=low`，否则隐藏推理消耗可见输出 budget。

## 数据集

| 数据集 | 规模 | DAG 来源 |
|---|---|---|
| **Demo 规则逻辑** | 5 题 | 规则引擎前向闭包 |
| **Omni-MATH** | 1,600 test | DeepSeek 依赖标注 + Harness |

> ~~GSM8K~~ 已删除——预构建图格式与 scorer 不兼容，且数据有根本性问题。

## Benchmark: Omni-MATH（竞赛数学，5 samples/模型）

| 模型 | Acc | Depth | Cons | Lit | 备注 |
|---|---|---|---|---|---|
| **DeepSeek-v4-Pro** | **40%** | **30** | 58 | 1.2 | 最佳，推理模型 |
| DeepSeek-v4-Flash | 20% | 0 | 45 | 0.2 | |
| Qwen2.5-7B-Instruct | 0% | 0 | 33 | 0.2 | SiliconFlow 免费层 |
| Qwen3.5-4B | 0% | 0 | 33 | 0.0 | SiliconFlow 免费层 |
| Qwen3-8B | — | — | — | — | 全部超时 |

> Omni-MATH 是竞赛级数学（IMO 难度），小型模型基本无法解答。depth=0 是模型答错而非 mapper bug。
> 更大模型的 benchmark（GLM-5.2, Qwen3.6-27B）受 SiliconFlow 免费层限流限制未完成。

## 快速开始

```bash
pip install -r requirements.txt
pytest  # 37 tests

# 规则逻辑题 Demo
python scripts/run_all_demo.py

# Harness 构建 DAG
python scripts/build_harness_dag.py --input data/raw/omni_math/test.jsonl --output data/processed/omni_math/harness_graphs.jsonl --limit 10
```

## 环境配置

`.env`:
```
API_KEY=sk-...                    # DeepSeek API key
BASE_URL=https://api.deepseek.com
MODEL_NAME=deepseek-chat
SILICON_FLOW_API_KEY=sk-...       # (可选) SiliconFlow
```

## 模块结构

```
src/reasoning_eval/
├── harness/              # LLM Agent Framework
│   ├── pipeline.py       # 单次调用 DAG 构造
│   ├── agents.py         # Structurer
│   ├── math_verifier.py  # 数学计算验证
│   └── schemas.py        # Pydantic DSL
├── scorer/               # 评分管道
│   ├── mapper.py         # 批量 LLM 步骤→节点映射
│   ├── depth_scorer.py   # 难度加权深度
│   ├── consistency_scorer.py  # 四维一致性
│   └── ...
└── model_test/           # LLM 客户端
```

## 当前局限

- LLM-only mapper 对非推理模型依赖强，推理模型需 `reasoning_effort=low`
- 竞赛数学对小模型完全不适用；需要 32B+ 才能得到有意义的评分
- 仅支持 OpenAI-compatible API

## 后续

- Mapper 批量化（已是单次调用，待验证大模型吞吐）
- MATH / FOLIO / ProofWriter 数据集扩展
- 交互式 DAG 展示 (Streamlit)
