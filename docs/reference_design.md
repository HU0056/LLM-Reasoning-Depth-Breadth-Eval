📋 技术方案深度分析阶段总结

阶段目标：完成五层架构与核心模块设计

核心结论：确定 Python 技术栈与 LRU+TTL 缓存策略
已完成决策

    五层架构：
        输入层：JSON 样本解析验证
        推理路径处理层：Canonicalization、图构建
        资源管理层：模型与缓存统一管理
        指标计算层：7 个指标计算
        输出层：格式输出与阈值判定

    核心模块：Input Parser、Canonicalizer、Graph Builder、各 Scorer、Aggregator

    技术栈：Python 3.11 + transformers + networkx/igraph + pydantic

    算法复杂度：
        Depth：O(V+E) 优先遍历，支持记忆化
        Breadth：队列层次标记
        Consistency：增量式检查
