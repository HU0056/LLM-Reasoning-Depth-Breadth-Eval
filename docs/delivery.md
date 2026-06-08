# 交付说明

本文档描述当前仓库已经完成并验证通过的 `graph builder` 交付内容。

## 交付范围

本阶段已完成：

- `gsm8k` 数据集下载与本地保存
- `sentence parser`
- `map builder / graph builder`
- 命令行入口与脚本入口
- 基础单元测试
- Windows 本地虚拟环境运行说明

本阶段未完成：

- proof-state 级 canonicalization
- DAG 去重与路径合并
- Breadth / Depth / Consistency 评分器
- 面向多数据集的统一 benchmark pipeline

## 目录结构

```text
LLM-Reasoning-Depth-Breadth-Eval/
├─ data/
│  ├─ raw/gsm8k/
│  └─ processed/gsm8k/
├─ docs/
│  ├─ delivery.md
│  ├─ graph_builder_README.md
│  └─ graph_builder_design.md
├─ scripts/
│  └─ run_gsm8k_pipeline.py
├─ src/reasoning_graph/
│  ├─ cli.py
│  ├─ config.py
│  ├─ dataset.py
│  ├─ graph_builder.py
│  ├─ pipeline.py
│  ├─ schemas.py
│  ├─ sentence_parser.py
│  └─ similarity.py
├─ tests/
│  ├─ test_graph_builder.py
│  └─ test_sentence_parser.py
├─ pyproject.toml
└─ requirements.txt
```

## 核心实现

### 1. 数据下载

- 使用 `datasets.load_dataset(...)` 下载 `gsm8k`
- 优先尝试 `openai/gsm8k`
- 失败时自动设置 `HF_ENDPOINT=https://hf-mirror.com/` 并重试
- 原始数据输出到 `data/raw/gsm8k/*.jsonl`

### 2. 句子切分

- 对 `question` 与 `answer` 按 `.`、`...`、`!`、`?`、换行分割
- 自动忽略空节点，例如 `.\n`
- 要求 `answer` 的最后一个节点匹配 `#### [ans]`

### 3. 图构建

对每个 `answer` 节点：

- 计算其与所有 `question` 句子、以及所有前置 `answer` 句子的相似度
- 默认相似度函数为 `calculate_similarity`，实现为 Jaccard similarity
- 找到最大相似度 `max_sim`
- 必连最大相似度对应的前驱节点
- 额外连接所有满足 `similarity > bound * max_sim` 的候选节点

默认阈值：

- `bound = 1 - 1/e`

### 4. 输出格式

输出文件位于 `data/processed/gsm8k/*_graphs.jsonl`，每条样本结构如下：

```json
{
  "id": "gsm8k_train_00000",
  "gsm8k_id": "gsm8k_train_00000",
  "task_type": "math",
  "question": "...",
  "gold_answer": "72",
  "gold_reasoning_graph": {
    "nodes": [
      "question sentence 1",
      "question sentence 2",
      "answer sentence 1",
      "answer sentence 2",
      "#### 72"
    ],
    "edges": [[0, 2], [2, 3], [3, 4]]
  }
}
```

## 环境与运行

### 依赖

当前项目运行时第三方库仅包含：

- `datasets>=2.20.0`

依赖清单见 [requirements.txt](D:/myprograms/schoolcoursecode/Python/LLM-Reasoning-Depth-Breadth-Eval/requirements.txt:1)。

### 虚拟环境

建议使用仓库内虚拟环境 `.venv`：

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt -e .
```

### 运行命令

完整流程：

```powershell
.venv\Scripts\python -m reasoning_graph.cli run-all --root .
```

仅下载原始数据：

```powershell
.venv\Scripts\python -m reasoning_graph.cli download-gsm8k --root .
```

仅构建图：

```powershell
.venv\Scripts\python -m reasoning_graph.cli build-gsm8k-graphs --root .
```

脚本入口：

```powershell
.venv\Scripts\python scripts\run_gsm8k_pipeline.py
```

## 验证结果

已验证以下命令可运行：

```powershell
.venv\Scripts\python -m unittest discover -s tests -v
.venv\Scripts\python -m reasoning_graph.cli run-all --root .
```

## 备注

- `data*` 当前被 `.gitignore` 忽略，数据文件不会进入版本控制
- `src/reasoning_graph_builder.egg-info/` 为本地安装生成的构建产物，不属于核心源码
- 当前图结构是句子级近似图，更适合做第一阶段工程打底，而不是最终研究版本
