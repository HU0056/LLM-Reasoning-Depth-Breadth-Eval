# LLM Reasoning Depth/Breadth Eval

LLM 推理过程评估框架。将模型 Chain-of-Thought 拆为原子步骤，映射到黄金推理 DAG，
从 Depth、Consistency 两个维度评估推理质量（Breadth 仅在有 key_branch_nodes 标注时启用）。

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
- LLM 语义匹配（自然语言数学题）→ 批量匹配所有步骤
- 抗幻觉：node_id 不在候选列表 → 直接驳回

### 评分维度

| 指标 | 定义 |
|---|---|
| **Depth** | `1 − D_remain/D_total` — 难度加权最短路径进度。depth≈100 表示模型已点亮通往答案的完整路径 |
| **Consistency** | 四维加权模型：逻辑非矛盾(.30) + 依赖完整(.35) + 目标对齐(.20) + 结构连贯(.15) |

详细评分标准见 [docs/scoring.md](docs/scoring.md)。

## 数据集

| 数据集 | 规模 | DAG 来源 | 特点 |
|---|---|---|---|
| Demo 规则逻辑 | 5 题 | 规则引擎前向闭包 | distractor/反事实分支 |
| GSM8K | 7,473 train | LLM 依赖标注 | 四则运算，简单推理 |
| Omni-MATH | 1,600 test | DeepSeek 依赖标注 | 竞赛数学，IMO 难度 |

## Benchmark 结果

### GSM8K（算术推理，5 样本/模型）

| 模型 | Acc | Depth | Consistency |
|---|---|---|---|
| **DeepSeek-v4-Pro** | **100%** | **65** | 71 |
| DeepSeek-v4-Flash | 100% | 40 | 84 |
| GLM-5.2 | — | — | — |
| Qwen2.5-7B | — | — | — |

> GSM8K 对现代模型几乎是 solved——DeepSeek 双模型 100% accuracy。
> Depth 差距揭示了问题：Pro 深度推理更充分（65 vs 40），Flash 倾向于浅层步骤。
> GLM-5.2 / Qwen2.5-7B 受 API 限流影响未完成全量。

### Omni-MATH（竞赛数学，5 样本/模型）

| 模型 | Acc | Depth | Consistency |
|---|---|---|---|
| **DeepSeek-v4-Pro** | **40%** | **70** | 53 |
| DeepSeek-v4-Flash | 20% | 20 | 45 |
| Qwen2.5-7B | 0% | 0 | 33 |
| Qwen3.5-4B | 0% | 0 | 33 |

> Omni-MATH 是 IMO 级难度。4B-8B 模型完全无法解答。
> DeepSeek-v4-Pro 的 depth=70 证明推理模型确实在推演中间步骤。
> 当前 gold DAG 是句子粒度（15-30 节点/题），模型步骤通过 reachable-path 机制点亮多个节点。

## DAG 示例

### Example 1：规则逻辑题 Deduction（correct_full）

Gold DAG：3 节点 (A → B → C)。模型按规则完全推演。
```
lit: A★  B★  C★   depth=100  consistency=100
```

### Example 2：GSM8K 算术题（correct_full）

Gold DAG：8 节点（item costs → subtotals → total → split）。模型逐项计算。
```
lit: 3★ 合计★ 均摊★  depth=75  consistency=73
模型路径与 gold 路径一致（先算各项成本，再求和，最后除以人数）
```

### Example 3：Omni-MATH 几何题（correct）

Gold DAG：17 节点（句子粒度）。模型通过 coordinates 方法推到 HR=1。
```
lit: 2★（起始陈述 + 结论）  depth=60  consistency=54
模型使用了与 gold 不同的推导路径（坐标法 vs 纯几何法），仅起点和终点对齐
```

完整 DAG 数据见 `outputs/results/Ex1-3*.json`。

## 环境

`.env`:
```
API_KEY=sk-...                    # DeepSeek API
BASE_URL=https://api.deepseek.com
MODEL_NAME=deepseek-chat          # Mapper/Harness
SILICON_FLOW_API_KEY=sk-...       # Qwen 系列
GLM_API_KEY=...                   # 智谱 GLM
```

```bash
pip install -r requirements.txt
pytest  # 41 tests
bash scripts/run_all_demo.py       # 规则逻辑题 Demo
```

## 模块结构

```
src/reasoning_eval/
├── harness/              # LLM Agent Framework（DAG 构造）
│   ├── pipeline.py       # 单次调用流水线
│   ├── agents.py         # Structurer Agent
│   ├── math_verifier.py  # 数学计算验证
│   └── schemas.py        # Pydantic DSL
├── scorer/               # 评分管道
│   ├── mapper.py         # 批量 LLM 步骤→节点映射
│   ├── verifier.py       # 图结构验证
│   ├── depth_scorer.py   # 难度加权深度
│   ├── consistency_scorer.py  # 四维一致性
│   └── step_splitter.py  # 步骤拆分
└── model_test/           # LLM 客户端 + Prompt 构建
```

## 设计哲学

1. **LLM 是匹配的权威**：步骤到节点映射完全由 LLM 判断逻辑等价性，代码不做文本相似度
2. **数学计算必须正确**：所有 `=` 表达式必须数值成立（唯一的确定性检查）
3. **粒度自适应**：Gold 句子级 DAG vs 模型步骤级输出，通过 reachable-path 机制容差
4. **诚实评分**：模型用了不同解法 → depth 按实际对齐程度给分，不虚高
