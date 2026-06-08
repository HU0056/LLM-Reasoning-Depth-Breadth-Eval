我们现在完成数据集获取，sentence parser，map builder 功能。接下来分条阐述

## 数据集
我们先形式化处理掉 GSM8k 数据集。

### 数据定义
一条典型的样本如下：
```text
{
    'question': 'Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?',
    'answer': 'Natalia sold 48/2 = <<48/2=24>>24 clips in May.\nNatalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n#### 72',
}
```
我们按句子划分（`.`, `...`, `!`, `?`, `\n` 等符号）`sentence` 和 `answer` 中的节点，注意忽略空节点（类似 `.\n`）

最后一个节点必然表现为 `####[ans]` 的形式，提取答案 `[ans]`

### 数据处理
你需要：

+ 从 huggingface 上下载 gsm8k 数据集，保存到 `data` 目录下。如果网络有问题，设置镜像站 `HF_ENDPOINT=https://hf-mirror.com/`

+ 按数据定义的方法，切分节点

+ 生成有向图。对每个 `answer` 中的节点：
  通过固定调用 `calculate_similarity` 计算 `answer` 与所有前置句子的相似度，以及与 `question` 中所有句子的相似度。
  设最大的相似度为 `max_sim`，从对应的节点连有向边向它，另外相似度超过 `bound*max_sim` 的节点也连边

+ 参考 report 中的 input 定义，以 json 格式保存有向图
  保存编号 `gsm8k_id`；
  样本类型 `math`；
  `question` 不变
  `gold_answer` 为最终答案 `[ans]`
  `gold_reasoning_graph` 中，节点为 question 和 answer 句子的拼接列表；连边为有向边列表，有向边为 0-index 的节点列表编号对

#### 补充
`bound` 的默认值为 `1-1/e`

`calculate_similariy` 的默认方法是 jaccard similarity