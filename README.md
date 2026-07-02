# LLM Reasoning Depth/Breadth Eval

LLM 推理过程深度/广度评估框架。不只评估 final answer 的正确性，而是将模型输出的 Chain-of-Thought 拆成原子步骤，映射到黄金推理 DAG 上，从多个维度评估推理过程的质量。

## 数据集支持

| 数据集 | 规模 | DAG 构建方式 |
|---|---|---|
| **Demo 规则逻辑题** | 5 题 | 规则引擎前向闭包（确定性） |
| **GSM8K** | 7,473 train + 1,319 test | LLM Harness Agent Framework |
| **Omni-MATH** | 2,000 test | DeepSeek 依赖标注 (omni_math.py) 或 Harness |

### 生产线 A：规则逻辑题

用于 toy 级规则逻辑推理的原型验证。5 道手工构造的逻辑题，包含显式 facts/rules/goal，支持 distractor 规则和反事实分支标注。DAG 由规则引擎确定性构建。

### 生产线 B：数学题（LLM Harness Agent Framework）

从 ground truth 出发，使用多 Agent LLM pipeline 构造带数学依据的推理 DAG。支持 GSM8K 和 Omni-MATH。**这是项目的核心贡献。**

**Harness 六阶段流水线**：

```
(question, answer)
  → Structurer (LLM) → 声明原子步骤 + 依赖 + 数学依据
  → Verification (Code) → 计算/依据/use-def/贡献度/拓扑/类型 六项检查
  → Auditor (LLM) → 语义边验证 + 依据正确性 + 缺失依赖检测
  → Cross-Validate (Code) → 调和 LLM 声明与代码检测
  → Repair Loop (LLM+Code) → 迭代修复直到验证通过或致命退出
  → Verified Gold DAG → 兼容 scorer
```

**Loop Engineering**：修复失败达到阈值时抛出 `HarnessError` 致命错误，不静默回退。

### Omni-MATH 额外支持

`src/reasoning_graph/omni_math.py` 提供独立的 DeepSeek 依赖标注管线（单次调用 + JSON mode），生成的图与 scorer 完全兼容。```bash
python scripts/build_omni_math_graphs.py --mode std --limit 20

## 四个模块

| 模块 | 目录 | 核心能力 |
|---|---|---|
| **Dataset** | `src/reasoning_eval/dataset/` | 规则逻辑题 DAG 构建（规则引擎前向闭包） |
| **Harness** | `src/reasoning_eval/harness/` | LLM Agent Framework：结构化 DAG 声明 → 六项确定性验证 → 语义审计 → 交叉验证 → 修复循环 |
| **Model Test** | `src/reasoning_eval/model_test/` | Prompt 构建（math/deduction）、OpenAI 兼容客户端、批量 API 生成 |
| **Scorer** | `src/reasoning_eval/scorer/` | 步骤拆分 → 防编造节点映射 → 图验证 → 三维评分 → DAG 可视化点亮 |
| **Analysis** | `src/reasoning_eval/analysis/` | 统计报告、DAG 点亮图、总体柱状图 |

额外独立包 `src/reasoning_graph/` 提供 GSM8K 下载与句子级建图（legacy，已被 harness 替代）。

## 核心指标

### Depth（推理深度）— 难度加权进度

```
depth = 1 − D_remain / D_total

D_total  = 从起点到 goal 的最小难度加权路径长度
D_remain = 从模型已点亮的节点集到 goal 的剩余最小难度
```

每条边附有数学依据 (Justification)，不同依据类型有不同基础难度：

```
arithmetic = 1.0    algebra = 1.5    theorem = 3.0
axiom = 1.0         definition = 1.0  substitution = 1.0
simplification = 1.0  equivalence = 2.0  induction = 5.0
```

### Breadth（推理广度）— 关键分叉覆盖率

```
breadth = covered_successors / total_successors
```

只统计样本中标注了 `key_branch_nodes` 的关键分支节点。无标注时返回 None。

### Consistency（推理一致性）— 四维评估

替代旧的减分制，采用加权多维模型：

| 维度 | 权重 | 含义 |
|---|---|---|
| 逻辑非矛盾性 | 0.30 | 推理过程中无相互矛盾的陈述 |
| 依赖完整性 | 0.35 | 每步依赖有效，无缺失前提 |
| 目标对齐性 | 0.20 | 每步缩短到目标的距离 |
| 结构连贯性 | 0.15 | 无冗余/重复/循环模式 |

最终分数 = 加权和 × 答案一致性因子（正确 → 1.0，错误 → 0.5）。

### DAG Lighting — 推理过程可视化

将模型步骤映射到 gold DAG 节点，标记每步状态：

```
lit / jump / redundant / wrong / contradiction / unvisited
```

## 边的定义

```
Edge = (premises: list[节点ID], target: 节点ID, justification: Justification)
```

每条推理边必须：
- 附带数学依据（9 种类型）
- 结论可从前提独立推出
- 满足原子性（单步推理）。豁免：若定理不是题目要证的定理且等价于多步原子推理，可视为原子边。

## 防编造三层保护

Mapper 含三层门控防止模型步骤误匹配到 gold DAG 节点：

1. **阈值门**：confidence < 0.25 直接驳回，< 0.35 不足驳回
2. **方向检查**：gold 节点 token 也必须被模型 step 覆盖（双向重叠），低重叠 → 伪匹配驳回
3. **结构门**：映射到的节点必须从当前验证状态可达

## 技术栈

Python 3.10+, Pydantic, networkx, pandas, matplotlib, pytest, python-dotenv, OpenAI-compatible API

## 快速开始

```bash
pip install -r requirements.txt

# 规则逻辑题 Demo（无需 API）
python scripts/run_all_demo.py

# 使用真实 LLM API 为 GSM8K 构建 DAG
python scripts/build_harness_dag.py \
    --input data/raw/gsm8k/train.jsonl \
    --output data/processed/gsm8k/train_harness_graphs.jsonl \
    --limit 20

# 为 Omni-MATH 构建图（DeepSeek 依赖标注）
python scripts/build_omni_math_graphs.py --mode std --limit 20

# 运行测试 (37 tests)
pytest
```

## 环境配置

`.env` 文件：

```
API_KEY=sk-...
BASE_URL=https://api.deepseek.com
MODEL_NAME=deepseek-v4-flash
```

## Demo 类型

- `correct_full`：答案正确，推理完整
- `correct_jump`：答案正确，但跳过中间前提
- `verbose_redundant`：答案正确，但重复啰嗦
- `wrong`：答案错误或使用错误规则
- `broad`：覆盖多个关键分支，Breadth 高
- `narrow_repeated`：多次采样但反复走同一分支

## 当前局限

- 规则逻辑题仅 5 道 demo，非大规模评测
- Harness 的 DAG 质量依赖 LLM 输出质量（通过验证+审计+修复循环减轻）
- Use-def 链仅追踪数值，不追踪变量名
- GSM8K / Omni-MATH 无 distractor/反事实分支标注
- 仅支持 OpenAI-compatible API（已适配 DeepSeek）
- Omni-MATH 的 DAG 边标注（omni_math.py）较简单，未使用多 Agent 验证

## 后续扩展

- Use-def 链扩展至变量名追踪
- 支持更多数学数据集 (MATH, FOLIO, ProofWriter)
- 接入 SMT solver (Z3) 做形式化依赖验证
- Few-shot example 库扩充，提升 LLM 输出稳定性
- 交互式 DAG 展示 (Streamlit / Gradio)
- 多模型横向评测报告
