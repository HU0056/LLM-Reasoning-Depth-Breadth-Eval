# 交付说明

本文档描述当前仓库已完成的交付内容与项目状态。

## 项目概述

4 个模块完整闭环的大模型推理过程评估原型：

| 模块 | 目录 | 状态 |
|---|---|---|
| Dataset | `src/reasoning_eval/dataset/` | ✅ |
| Model Test | `src/reasoning_eval/model_test/` | ✅ |
| Scorer | `src/reasoning_eval/scorer/` | ✅ |
| Analysis | `src/reasoning_eval/analysis/` | ✅ |

另有独立的 `src/reasoning_graph/` 包提供 GSM8K 数据下载与句子级相似度建图能力。

## 已交付内容

### 1. Dataset — benchmark 构建

- `dag_builder.py`: 从 facts/rules 构建 gold reasoning DAG
- `rule_parser.py`: 规则解析，支持 distractor 标注
- `graph_utils.py`: networkx DAG 构建、最短路径、可达性检测
- `build_demo_dataset.py`: 从 raw JSONL 生成 benchmark

数据：
- `data/raw/demo_raw_rules.jsonl`: 5 道规则逻辑题
- `data/raw/gsm8k/*.jsonl`: GSM8K 7,473 train + 1,319 test
- `data/processed/demo_benchmark.jsonl`: demo benchmark（含 gold DAG）
- `data/processed/gsm8k/*_graphs.jsonl`: GSM8K 句子级推理图

### 2. Model Test — prompt 构造 + API 调用

- `prompt_builder.py`: math / deduction 双模式 prompt，支持多路径 (SC)
- `llm_client.py`: OpenAI-compatible 客户端，retry/backoff，env 驱动
- `generate_with_api.py`: 批量生成，输出与 scorer pipeline 对齐
- `demo_output_loader.py`: 加载手写 demo 输出

### 3. Scorer — 核心评分引擎

- `step_splitter.py`: 模型输出 → 步骤列表，支持多路径 (Path N) 拆分
- `mapper.py`: step → DAG node 映射（规则匹配 + Jaccard 相似度）
- `verifier.py`: RuleBasedVerifier，检测直接后继/跳步/不可达/矛盾
- `depth_scorer.py`: 基于图距离缩短量的深度评分
- `breadth_scorer.py`: 关键分叉节点覆盖率
- `consistency_scorer.py`: 错误步定位 + 矛盾/冗余/答案一致性检测
- `dag_lighter.py`: DAG 节点点亮可视化 (lit/jump/wrong/redundant)
- `evaluator.py`: 总编排，evaluate_one / evaluate_files

### 4. Analysis — 报告与可视化

- `result_analyzer.py`: 按 output_type 聚合统计
- `plots.py`: 总体柱状图 (matplotlib)
- `visualize_dag.py`: 单条样本 DAG 点亮图
- `make_report.py`: 一键出报告 + 图表

### 5. GSM8K 图构建（独立体系）

`src/reasoning_graph/` 包：

- `sentence_parser.py`: 句子拆分 + final answer 提取
- `graph_builder.py`: Jaccard 相似度建图（可替换 similarity_fn）
- `similarity.py`: Jaccard token 重叠
- `pipeline.py` / `cli.py`: 下载 + 建图流水线

## 评分指标

| 指标 | 含义 | 关键文件 |
|---|---|---|
| Depth (0-100) | 推理对目标距离的有效缩短量 | `depth_scorer.py` |
| Breadth (0-100) | 关键分叉节点的有效分支覆盖率 | `breadth_scorer.py` |
| Consistency (0-100) | 错误步/矛盾/冗余/答案综合扣分 | `consistency_scorer.py` |
| First_Error_Step | 最早出错步骤位置 | `consistency_scorer.py` |
| DAG Lighting | 每步节点状态可视化 | `dag_lighter.py` |

## 测试

13 个单元测试全部通过：

```bash
conda run -n LLMReason python -m pytest tests/ -v
```

## 环境

- Python 3.10+
- Conda 环境 `LLMReason`，依赖见 `requirements.txt`
- 安装：`pip install -r requirements.txt -e .`

## 运行

```bash
# 完整 demo 流程（5 道规则题 + 手写输出）
conda run -n LLMReason python scripts/run_all_demo.py

# 分步运行
conda run -n LLMReason python scripts/01_build_dataset.py --raw data/raw/demo_raw_rules.jsonl --save data/processed/demo_benchmark.jsonl
conda run -n LLMReason python scripts/02_run_model_demo.py --benchmark data/processed/demo_benchmark.jsonl --outputs data/model_outputs/demo_model_outputs.jsonl
conda run -n LLMReason python scripts/03_score_outputs.py --benchmark data/processed/demo_benchmark.jsonl --outputs data/model_outputs/demo_model_outputs.jsonl --save outputs/results/demo_results.jsonl
conda run -n LLMReason python scripts/04_analyze_results.py --results outputs/results/demo_results.jsonl --benchmark data/processed/demo_benchmark.jsonl --report outputs/reports/summary.csv --figures outputs/figures

# 调真实 API 生成模型输出
conda run -n LLMReason python scripts/run_model_test.py --benchmark data/processed/demo_benchmark.jsonl --output data/model_outputs/api_outputs.jsonl
```

## 待完成

- scorer pipeline 适配 GSM8K processed graph 格式（当前 scorer 仅支持规则逻辑题的 id/proposition 格式）
- step-node mapper 升级为 semantic embedding
- verifier 升级为训练模型或 LLM-as-judge
- 反事实分支纳入主分数
- 真实 LLM 大规模评测
