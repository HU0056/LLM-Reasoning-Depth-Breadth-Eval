# LLM Reasoning Depth/Breadth Eval 项目交互记录

## 会话概述

本次会话围绕 LLM 推理评估项目的数据分析与可视化展开，主要完成了以下工作：

---

## 1. 项目拉取与同步

### 1.1 初始拉取
- 执行 `git pull` 从远程仓库 `https://github.com/HU0056/LLM-Reasoning-Depth-Breadth-Eval.git` 拉取代码
- 解决合并冲突（README.md、docs/reasoning_depth_breadth_project_report.md）

### 1.2 后续同步
- 处理本地领先远程 15 个提交的情况
- 用户选择放弃本地更改，执行强制重置：`git reset --hard origin/main`
- 成功同步到远程最新版本（HEAD: afa0473）

---

## 2. 项目结构分析

### 核心模块（src/）

| 模块 | 文件 | 职责 |
|------|------|------|
| schema | schema.py | 统一 Pydantic v2 数据结构 |
| loaders | base.py, gsm8k.py | 数据加载器 |
| generators | math_graph.py, fol_graph.py, break_path.py, verbalizer.py | 推理图生成与数据合成 |
| parsers | step_parser.py | CoT 步骤解析 |
| verifier | critic.py, dual_verifier.py | 双轨验证器 |
| scorers | engine.py, depth.py, breadth.py, consistency.py | 三维评分引擎 |

### 评估维度

| 维度 | 分数范围 | 描述 |
|------|----------|------|
| Score_Depth | 0-100 | 逻辑深度，每步是否向目标推进 |
| Score_Breadth | 0-100 | 思维广度，关键决策点是否探索足够分支 |
| Score_Consistency | 0-100 | 推理链路自洽性 |
| Answer_Accuracy | 0-100 | 答案正确率 |

---

## 3. 运行环境问题

### httpx/OpenAI 兼容性问题
- 错误：`TypeError: Client.__init__() got an unexpected keyword argument 'proxies'`
- 解决方案：升级依赖包 `pip install --upgrade openai httpx`

---

## 4. Brief 汇报要点

### 核心价值主张
> 区别于传统"只看答案"的评估（如 GSM8K/MATH），本框架评估**推理过程质量**。

### Brief 结构建议
1. 背景与问题
2. 核心贡献（双轨验证器、ProverGen 范式）
3. 评估维度定义（Depth/Breadth/Consistency + Accuracy）
4. 技术架构（高层）
5. 实验结果
6. 结论与未来工作

---

## 5. 可视化解决方案

### 方案一：交互式 Web 应用（Streamlit）
- 读取 `outputs/results/*.jsonl`、`outputs/reports/summary.csv`、`outputs/figures/*.png`
- 展示雷达图、交互式推理图、样本详情、统计分析

### 方案二：专业 BI 工具（Tableau/Power BI）
- 将 JSONL 转换为 CSV，连接 BI 工具数据源
- 构建企业级报表

### 方案三：知识图谱可视化（Neo4j Bloom）
- 将 DAG 结构转换为节点-关系，导入图数据库
- 交互式知识图谱探索

### 方案四：实时监控仪表盘（Grafana）
- 接入时序数据库，构建持续评估追踪

### 方案五：静态图表美化（Figma）
- 导出基础图表，专业设计美化

---

## 6. 可视化仪表盘实现

### 创建的文件

| 文件 | 功能 |
|------|------|
| [llm_eval_dashboard_v1.html](file:///d:/Projects/LLM-Reasoning-Depth-Breadth-Eval/llm_eval_dashboard_v1.html) | 多标签仪表盘（原有） |
| [llm_reasoning_tree_analyzer_v2.html](file:///d:/Projects/LLM-Reasoning-Depth-Breadth-Eval/llm_reasoning_tree_analyzer_v2.html) | 推理链断裂检测（原有） |
| [llm_tradeoff_scatter_v3.html](file:///d:/Projects/LLM-Reasoning-Depth-Breadth-Eval/llm_tradeoff_scatter_v3.html) | 深度-广度权衡分析（原有） |
| [llm_reasoning_analysis_dashboard.html](file:///d:/Projects/LLM-Reasoning-Depth-Breadth-Eval/llm_reasoning_analysis_dashboard.html) | 新增：动态数据加载仪表盘 |

### 动态仪表盘功能

#### 数据加载
- 从 `outputs/reports/summary.csv` 加载汇总数据
- 从 `outputs/results/demo_results.jsonl` 加载详细结果
- 加载进度条动画

#### 四个标签页

1. **概览**：4个指标卡片（样本数/深度/广度/一致性）、分组柱状图、雷达图、统计表格
2. **数据探索**：多条件筛选（输出类型、答案正确性、评分范围）、搜索、排序
3. **详细分析**：评分详情、推理步骤列表、交互式推理图（vis-network）、验证信息
4. **可视化**：散点图、直方图、图片网格

#### 交互功能
- 散点图点击数据点跳转详情
- 推理图节点点击查看详情
- 图片卡片点击预览大图
- 数据导出为 JSON
- 搜索、筛选、排序
- Toast 通知反馈

#### 访问方式
- 启动本地服务器：`python -m http.server 8080`
- 访问：`http://localhost:8080/llm_reasoning_analysis_dashboard.html`

---

## 7. 数据文件说明

### 输出文件

| 文件路径 | 内容 |
|----------|------|
| [outputs/reports/summary.csv](file:///d:/Projects/LLM-Reasoning-Depth-Breadth-Eval/outputs/reports/summary.csv) | 各类型统计摘要（6行） |
| [outputs/results/demo_results.jsonl](file:///d:/Projects/LLM-Reasoning-Depth-Breadth-Eval/outputs/results/demo_results.jsonl) | 详细结果（9条记录） |
| [outputs/figures/](file:///d:/Projects/LLM-Reasoning-Depth-Breadth-Eval/outputs/figures) | DAG Lighting 可视化图片（12张） |

### 推理类型

| 类型 | 描述 |
|------|------|
| correct_full | 完整推理链 |
| correct_jump | 逻辑跳跃（跳过中间节点） |
| verbose_redundant | 冗余重述 |
| wrong | 错误推理 |
| broad | 广度覆盖（多路径探索） |
| narrow_repeated | 狭窄重复（单路径反复） |

---

## 8. 关键技术点

### 双轨验证器
- Critic Model (LLM+Prompt) 首选方案
- LoRA PRM 降本方案

### ProverGen 范式
- 符号系统生成可验证 proof space
- LLM 自然语言化
- 自动回验

### 数据格式
- JSONL：每行一条记录，便于流式处理
- CSV：结构化汇总数据，便于分析

---

## 附录：命令记录

```bash
# 拉取代码
git pull

# 解决冲突后强制同步
git restore .
git clean -fd
git reset --hard origin/main

# 启动本地服务器
python -m http.server 8080

# 运行评估流程
python scripts/run_pipeline.py --max-records 30

# 升级依赖
pip install --upgrade openai httpx
```

---

*生成时间：2026-07-02*
*项目路径：d:\Projects\LLM-Reasoning-Depth-Breadth-Eval*