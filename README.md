# LLM Reasoning Depth/Breadth Eval

LLM 推理过程评估框架。将模型 Chain-of-Thought 拆为原子步骤，映射到黄金推理 DAG，
从 **Depth**、**Consistency** 两个维度评估推理质量（Breadth 仅在有 `key_branch_nodes` 标注时启用）。

## 方法

### Gold DAG 构造

从 ground truth (question + solution) 出发，LLM 单次调用直接产出结构化 DAG：

```
(question, solution) → Structurer (LLM, 1 调用, ~3s) → Gold DAG
```

零 Jaccard。零正则。零代码启发式。LLM 声明原子步骤 + 依赖 + 数学依据，代码仅做数学计算交叉验证。

### 步骤到节点映射

模型输出中所有步骤在一次 LLM 调用中完成匹配：

```
模型步骤列表 + gold 节点列表 → Mapper (LLM, 1 调用) → 匹配结果
```

- 规则文本精确匹配（deduction 题）→ confidence 0.95
- Proposition 唯一性匹配（单字母命题）→ confidence 0.80
- LLM 语义匹配（自然语言数学题）→ 批量匹配所有步骤，confidence 0.75
- 抗幻觉：`node_id` 不在候选列表 → 直接驳回

### 步骤验证与 DAG 点亮

每条匹配结果通过图拓扑验证：

| 节点状态 | 含义 | 判定条件 |
|---------|------|---------|
| **lit** | 正确点亮 | mapping 命中 + 从上一节点图拓扑可达 |
| **jump** | 跨节点跳跃 | mapping 命中 + 可达但非直接后继（跨了中间节点） |
| **wrong** | 错误匹配 | mapping 未命中或节点从当前位置不可达 |
| **redundant** | 重复访问 | 节点已被前序步骤点亮 |
| **contradiction** | 逻辑矛盾 | 步骤内容与已点亮命题冲突 |
| **unvisited** | 未访问 | 模型没有覆盖到该节点 |

| 边状态 | 含义 |
|--------|------|
| **used_valid** | 模型沿此边推理，步骤合法 |
| **skipped** | 模型跨过此边（跳跃），未直接使用 |
| **wrong** | 模型尝试使用此边但步骤不合法 |
| **unused** | 边未被模型遍历 |

详细评分标准见 [docs/scoring.md](docs/scoring.md)。

### 评分维度

| 指标 | 定义 | 区间 |
|---|---|---|
| **Depth** | `1 − D_remain/D_total` — 难度加权最短路径进度。100 = 完整路径点亮 | 0–100 |
| **Consistency** | 四维加权模型：逻辑非矛盾(.30) + 依赖完整(.35) + 目标对齐(.20) + 结构连贯(.15) × 答案因子 | 0–100 |
| **Breadth** | 关键分叉节点后继覆盖率（仅标注 `key_branch_nodes` 时启用） | 0–100 或 None |

## 数据集

| 数据集 | 规模 | DAG 来源 | 特点 |
|------|------|--------|------|
| Demo 规则逻辑 | 5 题 | 规则引擎前向闭包 | distractor/反事实分支 |
| **GSM8K** | 551 题 (140 train + 411 test) | LLM 依赖标注 | 四则运算，简单推理 |
| **Omni-MATH** | 2,200 题 | DeepSeek 依赖标注 | 竞赛数学，IMO 难度 |

## Benchmark 结果

### GSM8K（50 随机样本/模型）

| 模型 | API | Acc | Depth | Consistency |
|------|-----|-----|-------|-------------|
| **DeepSeek-v4-Pro** | 官方 | 98% | 54 | 85 |
| **DeepSeek-v4-Flash** | 官方 | 96% | 56 | 84 |
| Qwen2.5-7B | SiliconFlow | 评测中 | — | — |
| Qwen3-8B | SiliconFlow | 评测中 | — | — |

> GSM8K 对现代模型几乎是 solved——DeepSeek 双模型 96-98% accuracy。
> Depth ~55 说明模型通常点亮了约一半的 gold 推理节点。
> 小模型（7B-8B）评测仍在进行中。

### Omni-MATH（50 随机样本/模型，DeepSeek only）

| 模型 | API | Acc | Depth | Consistency |
|------|-----|-----|-------|-------------|
| DeepSeek-v4-Pro | 官方 | 评测中 | — | — |
| DeepSeek-v4-Flash | 官方 | 评测中 | — | — |

> Omni-MATH 是 IMO 级难度，竞赛数学题每题 15-30 个 gold 节点。
> Gold DAG 是句子粒度，模型步骤通过 reachable-path 机制点亮多个节点。

## 运行

### 环境配置

`.env`:
```bash
API_KEY=sk-...                    # DeepSeek API（必须）
BASE_URL=https://api.deepseek.com
MODEL_NAME=deepseek-v4-flash
SILICON_FLOW_API_KEY=sk-...       # Qwen 系列（可选）
GLM_API_KEY=...                   # 智谱 GLM（可选）
```

安装依赖：
```bash
pip install -r requirements.txt
pytest  # 41 tests
```

### GSM8K Benchmark

```bash
# 全部 4 模型并行（采样 50 题 → 4 个 agent 同时跑）
python3 scripts/bench_gsm8k_v2.py

# 单模型直跑
python3 scripts/bench_gsm8k_v2.py "DeepSeek-v4-Pro" "deepseek-v4-pro" "DS"

# 输出：outputs/results/bench_<model>_gsm8k_50.json
```

### Omni-MATH Benchmark

```bash
# 单模型（默认 50 题，seed=42）
python3 scripts/bench_omni_v2.py "DeepSeek-v4-Pro" "deepseek-v4-pro"

# 输出：outputs/results/bench_<model>_omni_50.json
```

### Demo 规则逻辑题

```bash
python3 scripts/run_all_demo.py
```

### 汇总结果

```bash
python3 scripts/summarize.py
```

## 输出文件

### 结果文件格式（每模型一个 JSON）

```
outputs/results/
├── bench_DeepSeek_v4_Pro_gsm8k_50.json    # GSM8K 结果
├── bench_DeepSeek_v4_Pro_omni_50.json     # Omni-MATH 结果
├── gsm8k_50_sample_ids.json               # 采样的题目 ID 列表
├── omni_50_sample_ids.json                # 采样的题目 ID 列表
└── bench_summary.json                      # 汇总统计
```

每条记录包含：
```json
{
  "id": "gsm8k_train_00000",
  "correct": true,
  "depth": 50.0,
  "cons": 100.0,
  "lit": 3,
  "time": 3.2,
  "states": {"lit": 3, "unvisited": 2},
  "lit_nodes": ["0", "2", "3"],
  "response": "Step 1: ...\nStep 2: ...\nFinal Answer: 72",
  "lighted_graph": {
    "nodes": {"0": "lit", "1": "unvisited", "2": "lit", "3": "lit", "4": "unvisited"},
    "edges": {"0->2": "used_valid", "0->3": "unused", "2->3": "used_valid", "3->4": "unused"},
    "steps": [
      {"step_index": 1, "node": "0", "status": "lit", "reason": "first step maps to fact node 0"},
      {"step_index": 2, "node": "2", "status": "lit", "reason": "2 is a direct successor of 0"},
      {"step_index": 3, "node": "3", "status": "lit", "reason": "3 is a direct successor of 2"}
    ]
  }
}
```

### 日志文件

```
outputs/logs/
├── gsm8k_DeepSeek_v4_Pro.log   # 每个模型的运行日志
├── gsm8k_Qwen2_5_7B.log
└── ...
```

## 模块结构

```
src/reasoning_eval/
├── harness/                  # LLM Agent Framework — 从 ground truth 构造 Gold DAG
│   ├── pipeline.py           # 单次调用 Structurer，直接产出 DAG (~3s/题)
│   ├── agents.py             # Structurer (声明步骤+依赖+数学依据)
│   ├── math_verifier.py      # 数学计算验证（唯一确定性检查）
│   ├── verifiers.py          # 信息性验证（非阻塞）
│   ├── schemas.py            # Pydantic DSL: DagNode, DagEdge, Justification
│   └── prompts.py            # LLM 提示模板
├── scorer/                   # 评分管道
│   ├── evaluator.py          # 总编排 + 自动点亮相间节点
│   ├── mapper.py             # 批量 LLM 步骤→节点映射（1 次调用）
│   ├── verifier.py           # 图拓扑验证 (reachable-path = valid)
│   ├── depth_scorer.py       # 难度加权深度 (Dijkstra 最短路径)
│   ├── consistency_scorer.py # 四维一致性
│   ├── breadth_scorer.py     # 分叉覆盖率
│   ├── dag_lighter.py        # DAG 点亮 (lit/jump/wrong/redundant/contradiction)
│   └── step_splitter.py      # 模型输出→步骤列表 (15步上限)
├── model_test/               # LLM 客户端
│   ├── llm_client.py          # OpenAI-compatible, max_tokens=262144
│   └── prompt_builder.py      # math/deduction 双模式 Prompt 构建
├── dataset/                  # 数据处理
│   └── graph_utils.py         # 图规范化 (flat→structured)、拓扑分析
├── common/                   # 共用工具
│   ├── schema.py              # EvaluationResult, MappingResult, VerificationResult
│   ├── text_utils.py          # 文本矛盾检测
│   └── io_utils.py            # JSONL 读写
└── analysis/                 # 统计报告 + 可视化

scripts/
├── bench_gsm8k_v2.py         # GSM8K 统一 benchmark (v2) — launcher + agent
├── bench_omni_v2.py          # Omni-MATH 统一 benchmark (v2)
├── bench_gsm8k.py            # 旧版 GSM8K benchmark (legacy)
├── bench_v2.py               # 旧版 Omni-MATH benchmark (legacy)
├── summarize.py              # 汇总所有 benchmark 结果
└── run_all_demo.py           # Demo 规则逻辑题完整流水线
```

## 设计哲学

1. **LLM 是匹配的权威**：步骤到节点映射完全由 LLM 判断逻辑等价性，代码不做文本相似度
2. **数学计算必须正确**：所有 `=` 表达式必须数值成立（唯一的确定性检查）
3. **粒度自适应**：Gold 句子级 DAG vs 模型步骤级输出，通过 reachable-path 机制容差
4. **诚实评分**：模型用了不同解法 → depth 按实际对齐程度给分，不虚高
5. **全量保存**：每条结果保存模型原始输出 + 完整点灯图，支持人工审计和重评分
