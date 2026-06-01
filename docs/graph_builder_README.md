# Graph Builder README

本文档聚焦当前仓库里 `gsm8k sentence parser + graph builder` 这一部分的使用方式和实现约束。

## 目标

把 `gsm8k` 样本中的：

- `question`
- `answer`

从线性文本切成句子节点，再把这些节点组织成一个句子级有向图，最终输出为项目后续评测流程可消费的 JSONL 数据。

## 当前实现

### 输入

单条 `gsm8k` 原始样本形如：

```text
{
    "question": "...",
    "answer": "step 1.\nstep 2.\n#### ans"
}
```

### 切分规则

- 分隔符：`.`、`...`、`!`、`?`、`\n`
- 忽略空节点
- `answer` 最后一个节点必须匹配 `#### [ans]`

### 建图规则

对于每个 `answer` 节点：

1. 将它与所有 `question` 节点计算相似度
2. 将它与所有前置 `answer` 节点计算相似度
3. 选择最大相似度节点作为必连前驱
4. 对所有满足 `similarity > bound * max_similarity` 的其余节点也连边

默认参数：

- `bound = 1 - 1/e`
- `calculate_similarity = jaccard similarity`

### 输出

输出字段与项目 `input` 约定保持一致：

```json
{
  "id": "gsm8k_test_00000",
  "gsm8k_id": "gsm8k_test_00000",
  "task_type": "math",
  "question": "...",
  "gold_answer": "18",
  "gold_reasoning_graph": {
    "nodes": ["...", "...", "#### 18"],
    "edges": [[0, 2], [2, 3]]
  }
}
```

## 代码位置

- 句子切分：[src/reasoning_graph/sentence_parser.py](D:/myprograms/schoolcoursecode/Python/LLM-Reasoning-Depth-Breadth-Eval/src/reasoning_graph/sentence_parser.py:1)
- 相似度计算：[src/reasoning_graph/similarity.py](D:/myprograms/schoolcoursecode/Python/LLM-Reasoning-Depth-Breadth-Eval/src/reasoning_graph/similarity.py:1)
- 图构建：[src/reasoning_graph/graph_builder.py](D:/myprograms/schoolcoursecode/Python/LLM-Reasoning-Depth-Breadth-Eval/src/reasoning_graph/graph_builder.py:1)
- 数据管道：[src/reasoning_graph/pipeline.py](D:/myprograms/schoolcoursecode/Python/LLM-Reasoning-Depth-Breadth-Eval/src/reasoning_graph/pipeline.py:1)
- CLI：[src/reasoning_graph/cli.py](D:/myprograms/schoolcoursecode/Python/LLM-Reasoning-Depth-Breadth-Eval/src/reasoning_graph/cli.py:1)

## 使用方式

### 环境准备

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt -e .
```

### 下载并构图

```powershell
.venv\Scripts\python -m reasoning_graph.cli run-all --root .
```

### 运行测试

```powershell
.venv\Scripts\python -m unittest discover -s tests -v
```

## 输出文件

- `data/raw/gsm8k/train.jsonl`
- `data/raw/gsm8k/test.jsonl`
- `data/processed/gsm8k/train_graphs.jsonl`
- `data/processed/gsm8k/test_graphs.jsonl`

## 限制

- 这是句子级图，不是语义规范化后的状态图
- Jaccard 对数学表达式和同义改写不够稳，只能作为 baseline
- 当前没有做节点去重、路径合并和全局一致性检查
