# LLM Reasoning Depth/Breadth Eval

LLM 推理过程评估框架。不仅评估 final answer 正确性，更将模型 Chain-of-Thought
拆为原子步骤，映射到黄金推理 DAG，从 Depth、Consistency、Breadth 三维度评估推理质量。

## 技术路线

### 黄金 DAG 构造：LLM Harness Agent Framework

从 ground truth (question + reference answer) 出发，**纯 LLM 驱动**构造推理 DAG，
零代码启发式匹配，零 Jaccard，零正则。

```
(question, answer)
  → Structurer (LLM)    — 分解为原子步骤 + 声明依赖 + 数学依据
  → Verification (Code) — 六项确定性检查：计算/依据/use-def/贡献度/拓扑/类型
  → Auditor (LLM)       — 语义边验证 + 依据正确性检测
  → Cross-Validate      — 调和 LLM 声明与代码检测
  → Repair Loop         — 迭代修复直至通过或致命退出
  → Verified Gold DAG   — 兼容 scorer pipeline
```

**Loop Engineering**：超时/网络错误立即致命。修复轮次用尽 → `HarnessExhaustedError`。不静默回退。

### 步骤到节点映射：LLM-only Mapper

模型输出中的每个推理步骤，通过 **LLM 判断逻辑等价性**来匹配 gold DAG 节点：

```
模型步骤文本 + 所有 gold 节点列表
  → 双判共识（两次独立 LLM 调用，seed 不同，必须一致）
  → 数学验证（math_verifier.py — 唯一的确定性检查：验证所有 = 和 <<>> 表达式）
  → 匹配结果（node_id 或 no_match）
```

- **不做**文本相似度匹配、不做方程解析、不做正则提取
- **只做** LLM 推理 + 数学计算交叉验证
- 幻觉自动驳回：输出 node_id 不在候选列表中 → reject

### 评分三维度

| 指标 | 定义 | 方法 |
|---|---|---|
| **Depth** | `1 − D_remain/D_total` | 难度加权最短路径进度 |
| **Consistency** | 四维加权模型 | 逻辑非矛盾(.30) + 依赖完整(.35) + 目标对齐(.20) + 结构连贯(.15) |
| **Breadth** | `covered/total` 关键分叉 | 仅评估标注了 key_branch_nodes 的样本 |

### 模型选择

| 用途 | 推荐模型 | 原因 |
|---|---|---|
| **问题求解** | `deepseek-v4-flash` (reasoning_effort=low) | 数学推理能力强 |
| **Harness structurer/auditor** | `deepseek-chat` 或 `Qwen2.5-7B-Instruct` | 需要可靠的结构化 JSON 输出 |
| **Mapper 匹配** | `deepseek-chat` 或 `Qwen2.5-7B-Instruct` | 需稳定的 JSON，推理模型隐藏推理会导致输出截断 |
| **Omni-MATH 图构建** | `deepseek-chat` (omni_math.py) | 单调用 JSON mode |

> `deepseek-v4-flash` 是推理模型，隐藏 reasoning_content 与可见 content 共享 max_tokens 预算。
> 可通过 `extra_body={'reasoning_effort': 'low'}` 限制隐藏推理（~480 chars），但不能完全关闭。
> **不要**对结构化 JSON 输出任务使用 `reasoning_effort=high/max/xhigh`。

## 数据集

| 数据集 | 规模 | DAG 来源 |
|---|---|---|
| **Demo 规则逻辑** | 5 题 | 规则引擎前向闭包 |
| **GSM8K** | 7,473 train / 1,319 test | Harness Agent Framework |
| **Omni-MATH** | 1,600 test (已处理) | DeepSeek 依赖标注 (omni_math.py) 或 Harness |

## 冒烟测试结果 (2026-07-02, deepseek-chat)

| 数据集 | 样本 | correct | depth | cons | lit | 耗时 |
|---|---|---|---|---|---|---|
| **GSM8K** | gsm8k_train_00000 | ✅ True | **100** | 61 | 2/4 | 24s |
| **Omni-MATH** | omni_math_test_00000 | ✗ False | 0 | 23 | 1/17 | 80s |

> GSM8K 满分通过全链路（harness 建图 → 模型推理 → mapper 匹配 → scorer 评分）。
> Omni-MATH 因竞赛几何难度过高，模型答错。Mapper 本身功能正常（1/17 节点被正确点亮）。

## 快速开始

```bash
pip install -r requirements.txt

# 运行测试 (37 tests)
pytest

# 规则逻辑题 Demo（无需 API）
python scripts/run_all_demo.py

# Harness 构建 GSM8K DAG
python scripts/build_harness_dag.py \
    --input data/raw/gsm8k/train.jsonl \
    --output data/processed/gsm8k/train_harness_graphs.jsonl \
    --limit 20

# Omni-MATH 图构建
python scripts/build_omni_math_graphs.py --mode std --limit 20
```

## 环境配置

`.env`:
```
API_KEY=sk-...                    # DeepSeek API key
BASE_URL=https://api.deepseek.com
MODEL_NAME=deepseek-chat          # 推荐，非推理模型
SILICON_FLOW_API_KEY=sk-...       # (可选) SiliconFlow Qwen
```

## 模块结构

```
src/reasoning_eval/
├── harness/              # LLM Agent Framework (DAG 构造)
│   ├── agents.py         # Structurer, Auditor, Repairer
│   ├── pipeline.py       # 编排层 + Loop Engineering
│   ├── verifiers.py      # 六项确定性验证
│   ├── math_verifier.py  # 数学计算验证 (唯一保留的确定性检查)
│   ├── schemas.py        # Pydantic DSL
│   └── prompts.py        # Few-shot prompt 模板
├── scorer/               # 评分管道
│   ├── mapper.py         # LLM-only 步骤→节点映射
│   ├── verifier.py       # 图结构验证器
│   ├── depth_scorer.py   # 难度加权深度
│   ├── consistency_scorer.py  # 四维一致性
│   ├── breadth_scorer.py # 分叉覆盖率
│   └── step_splitter.py  # 模型输出→步骤拆分
├── model_test/           # LLM 客户端 + prompt 构建
├── dataset/              # 规则逻辑题 DAG 构建
└── analysis/             # 统计报告 + DAG 可视化
```

## 当前局限

- LLM-only mapper 成本高（每步 2 次 LLM 调用），大样本评测需优化
- Harness structurer/repairer 对 instruct 模型 prompt 长度敏感
- Omni-MATH 代数变量步骤的匹配完全依赖 LLM，数学验证对其无约束力
- Use-def 链仅追踪数值，不追踪变量名
- 仅支持 OpenAI-compatible API

## 后续

- Mapper 批量化（一次 LLM 调用匹配所有步骤）
- 变量感知的数学验证
- MATH / FOLIO / ProofWriter 数据集扩展
- 交互式 DAG 展示 (Streamlit)
