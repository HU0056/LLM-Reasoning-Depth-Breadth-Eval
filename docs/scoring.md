# 评分标准详解

## 总体流程

```
模型原始输出 (Chain-of-Thought 文本)
  │
  ▼
Step Splitter ──→ 拆分文本为 [步骤1, 步骤2, ..., 步骤N]
  │
  ▼
Mapper (LLM) ──→ 每个步骤 → 匹配到 gold DAG 中的一个节点 (或 "无匹配")
  │                 ┌─ 用 deepseek-chat (非推理模型)
  │                 └─ 所有步骤一次 LLM 调用完成
  ▼
Verifier ──→ 检验每个匹配是否合法
  │             ┌─ lit: 匹配到 gold 节点，且从上个节点可达
  │             ├─ jump: 匹配到 gold 节点，可达但中间跳过了句子级节点
  │             ├─ wrong: 不可达 / 无匹配
  │             ├─ redundant: 重复访问
  │             └─ contradiction: 与已点亮的命题矛盾
  ▼
三个维度并行评分 ──→ Depth / Consistency / Breadth
  │
  ▼
DAG Lighter ──→ 可视化：每个节点标记 lit/jump/wrong/unvisited
```

---

## 维度 1: Depth（推理深度）

### 定义

```
depth = 1 − D_remain / D_total

D_total  = 从起点到 goal 的最小加权路径长度（最短路径的边难度之和）
D_remain = 从模型已点亮的节点集中到 goal 的剩余最小加权路径长度
```

### 直觉

- **depth ≈ 100**: 模型步步推进，已点亮了通往答案的完整路径
- **depth ≈ 50**: 模型推了一半，关键中间步骤被点亮但未到终点
- **depth ≈ 0**: 模型没有点亮任何可达 gold 节点

### 每条边的难度

| Justification 类型 | 基础难度 |
|---|---|
| arithmetic | 1.0 |
| algebra | 1.5 |
| theorem | 3.0 |
| axiom | 1.0 |
| definition | 1.0 |
| substitution | 1.0 |
| simplification | 1.0 |
| equivalence | 2.0 |
| induction | 5.0 |

### 计算过程

1. Gold DAG 用 Dijkstra 计算从 start_nodes 到 goal 的最短加权路径 → `D_total`
2. 每处理一个模型步骤，更新 `lit_nodes` 集合（已点亮的 gold 节点）
3. 计算从 `lit_nodes` 到 goal 的最短加权路径 → `D_remain`
4. `depth_at_step = 1 − D_remain / D_total`

**最终 depth = 最后一步完成后的 depth_at_step**

### 粒度修正（v3）

由于 Omni-MATH 的 gold DAG 是句子级（15-30 节点），而模型输出是推理步骤级（3-8 步），**当一个模型步骤跨越多个 gold 句子节点时**：
- Verifier 将该步骤标记为 `valid=True, missing_premise=True`（即 "jump"）
- Evaluator 自动点亮该步骤与上一个 lit 节点之间的**所有中间节点**（通过 Dijkstra 最短路径）
- 这些中间节点的点亮不算入 depth——只是让 depth 计算不因粒度差异而报废

---

## 维度 2: Consistency（推理一致性）

### 定义

**四维加权模型**：每个维度先规约到 [0, 1]，最终得分 = 加权和 × 答案因子。

| 维度 | 权重 | 含义 | 计算方式 |
|---|---|---|---|
| **逻辑非矛盾性** | 0.30 | 推理过程中无相互矛盾 | `1 − 2×(矛盾步骤数/总步骤数)` |
| **依赖完整性** | 0.35 | 依赖有效，无缺失前提 | `1 − (无效步骤+0.5×跳步)/总步骤数` |
| **目标对齐性** | 0.20 | 每步缩短到目标距离 | `有效步骤/总步骤数` |
| **结构连贯性** | 0.15 | 无冗余/重复 | `1 − 2×(冗余步骤/总步骤数)` |

**答案因子**：
- 答案正确 → ×1.0
- 答案错误 → ×0.5

### 各维度详解

**逻辑非矛盾性 (0.30)**：衡量模型是否在推理中自相矛盾。每次检测到前后矛盾的陈述，该维度被扣分。

**依赖完整性 (0.35)**：核心维度。评估步骤之间的逻辑链条是否完整：
- 有效步骤 (valid=True) → 满分贡献
- 跳步 (missing_premise=True) → 半扣分（承认步骤有效但粒度不同）
- 无效步骤 (valid=False) → 全扣分

**目标对齐性 (0.20)**：每个正确的推理步骤都是朝向目标的推进。有效步骤比例越高，对齐性越好。

**结构连贯性 (0.15)**：惩罚冗余和循环。重复访问同一节点或产生循环依赖会被扣分。

### 答案一致性惩罚

答案因子是一种**结构性惩罚**：即使推理看起来内部一致（dimensions 分数高），如果最终答案是错误的，总分直接砍半。这防止了"自信但错误的推理"获得虚高分数。

---

## 维度 3: Breadth（推理广度）

### 定义

```
breadth = covered_successors / total_successors

仅评估样本中标注了 key_branch_nodes 的关键分支节点。
```

### 计算

1. 对每个 `key_branch_node`，获取其所有合法后继（edge status ≠ "distractor"）
2. 在所有采样路径中，遍历该分支节点后的步骤，检查是否覆盖了后继
3. 覆盖率 = 已被覆盖的总后继数 / 所有关键分支节点的总后继数

**如果样本没标注 `key_branch_nodes`（如 Omni-MATH），返回 `None`——不参与广度评分。**

### 用途场景

在多路径采样 (`num_paths > 1`) 时，Breadth 衡量模型是否探索了多种不同的推导路径，而非反复走同一条路。

---

## 节点状态定义

| 状态 | 含义 | 点亮条件 |
|---|---|---|
| **lit** | 模型正确匹配到此节点 | mapping 成功 + verifier 判 valid + 或从上一个节点可达 |
| **jump** | 模型匹配到此节点但跳过了中间节点 | mapping 成功 + verifier 判 missing_premise=True |
| **wrong** | 模型没有正确匹配到此节点 | mapping 失败或无映射 |
| **redundant** | 模型重复访问此节点 | mapping 成功但已由前序步骤点亮 |
| **contradiction** | 模型在此步骤表达了矛盾 | mapping 成功但与已点亮节点矛盾 |
| **unvisited** | 模型未提及此节点 | 无任何对应步骤 |

---

## 答案判定

```
if 模型的 "Final Answer" 中包含 gold_answer 的数字 → 正确
if 模型的 "\boxed{X}" 中 X = gold_answer → 正确
if 模型写出了 "目标成立"（逻辑题）→ 正确
否则 → 错误
```

数学答案支持 `\boxed{72}` 和 `Final Answer: 72` 两种格式。

---

## 防编造三层门

Mapper 防止模型步骤误匹配到 gold 节点：

1. **Schema 门**: LLM 返回的 node_id 必须在 gold DAG 的 `valid_ids` 集合中。不在 → 直接驳回
2. **数学交叉验证**: 步骤文本中包含 "=" 的计算必须数值正确。错误 → 驳回
3. **置信度门**: 规则文本匹配 → 0.95；LLM 匹配 → 0.75。仅高分匹配被接受

---

## 评分的哲学立场

1. **LLM 是匹配的权威，不是代码**：步骤到节点的映射完全由 LLM 判断，代码不做文本相似度、Jaccard、正则匹配
2. **数学计算必须正确**：唯一的确定性检查——所有 "=" 表达式必须成立
3. **Gold DAG 是句子粒度的**：已通过 verifier v3 修正——可达的节点都算点亮，不要求直接后继
4. **竞赛数学对小模型不友好**：Omni-MATH 的 IMO 难度意味着 4B-8B 模型几乎无法得分——这是数据集的特性，非评分缺陷
