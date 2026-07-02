from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401


def _safe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _avg(values: list[float | None]) -> float | None:
    cleaned = [v for v in values if v is not None]
    if not cleaned:
        return None
    return round(sum(cleaned) / len(cleaned), 3)


def _grade(score: float | None) -> str:
    if score is None:
        return "待补充"
    if score >= 85:
        return "优秀"
    if score >= 70:
        return "良好"
    if score >= 55:
        return "一般"
    return "风险"


def _sanitize(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _sanitize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize(v) for v in value]
    return value


def _read_summary(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return [
            {key: _safe_float(value) if key != "output_type" else value for key, value in row.items()}
            for row in csv.DictReader(f)
        ]


def _read_result_file(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if path.suffix == ".jsonl":
        rows = []
        for line_no, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
        return rows
    parsed = json.loads(text)
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict):
        return [parsed]
    raise ValueError(f"Unsupported result shape in {path}")


def _result_paths(path: Path) -> list[Path]:
    if path.is_dir():
        paths = sorted([*path.glob("*.jsonl"), *path.glob("*.json")])
        return [p for p in paths if _is_model_result_path(p)]
    return [path]


def _is_model_result_path(path: Path) -> bool:
    name = path.name
    lowered = name.lower()
    if name.startswith(("Ex", "Example")):
        return False
    if lowered.startswith(("bench_test", "sample_audit")):
        return False
    if lowered in {"bench_summary.json", "bench_all_merged.json"}:
        return False
    return lowered.startswith("bench_") or lowered.endswith(".jsonl")


def _model_name_from_path(path: Path) -> str:
    name = path.stem
    for prefix in ("bench_", "results_", "result_"):
        if name.startswith(prefix):
            name = name[len(prefix) :]
    name = re.sub(r"_(?:\d{5})_(?:\d{5})$", "", name)
    name = re.sub(r"_(?:omni|gsm8k)$", "", name, flags=re.IGNORECASE)
    aliases = {
        "DeepSeek_v4_Flash": "DeepSeek v4 Flash",
        "DeepSeek_v4_Pro": "DeepSeek v4 Pro",
        "GLM_5_2": "GLM 5.2",
        "Qwen2_5_7B": "Qwen2.5-7B",
        "Qwen2_5-7B": "Qwen2.5-7B",
        "Qwen2.5-7B": "Qwen2.5-7B",
        "Qwen3_5-4B": "Qwen3.5-4B",
        "Qwen3-8B": "Qwen3-8B",
    }
    return aliases.get(name, name.replace("_", " "))


def _output_type_from_path_or_id(row: dict[str, Any], path: Path, is_error: bool) -> str:
    if row.get("output_type"):
        return str(row["output_type"])
    if row.get("ds"):
        return str(row["ds"])
    name = path.stem.lower()
    sample_id = str(row.get("sample_id") or row.get("id") or "").lower()
    if "gsm8k" in name or sample_id.startswith("gsm8k"):
        return "GSM8K"
    if "omni" in name or sample_id.startswith("omni"):
        return "Omni-MATH"
    return "error" if is_error else "bench"


def _normalize_row(row: dict[str, Any], path: Path) -> dict[str, Any]:
    model_name = row.get("model_name") or row.get("model") or _model_name_from_path(path)
    sample_id = row.get("sample_id") or row.get("id") or row.get("sample") or "unknown"
    lighted_graph = row.get("lighted_graph") or row.get("lighted") or {}
    nodes = lighted_graph.get("nodes") or {}
    steps = lighted_graph.get("steps") or []
    node_count = len(nodes)
    lit_count = _safe_float(row.get("lit"))
    if lit_count is None:
        lit_count = sum(1 for status in nodes.values() if status in {"lit", "redundant"})
    inferred_breadth = (lit_count / node_count * 100.0) if node_count else None
    first_problem = next(
        (
            step.get("step_index")
            for step in steps
            if step.get("status") in {"wrong", "jump"} or step.get("node") is None
        ),
        None,
    )
    is_error = "error" in row
    output_type = _output_type_from_path_or_id(row, path, is_error)
    response = row.get("resp") or row.get("response") or row.get("model_response") or ""
    return {
        **row,
        "sample_id": sample_id,
        "model_name": str(model_name),
        "output_type": str(output_type),
        "answer_correct": bool(row.get("answer_correct", row.get("correct", False))) if not is_error else False,
        "score_depth": _safe_float(row.get("score_depth", row.get("depth"))),
        "score_breadth": _safe_float(row.get("score_breadth", row.get("breadth"))) or _safe_float(inferred_breadth),
        "score_consistency": _safe_float(row.get("score_consistency", row.get("cons"))),
        "first_error_step": row.get("first_error_step", first_problem),
        "missing_premise_flag": bool(row.get("missing_premise_flag", False)) or any(
            step.get("status") == "jump" for step in steps
        ),
        "branch_coverage": _safe_float(row.get("branch_coverage")) or (
            _safe_float(inferred_breadth) / 100.0 if inferred_breadth is not None else None
        ),
        "contradiction_count": row.get("contradiction_count") or 0,
        "lighted_graph": lighted_graph,
        "response": response,
        "error": row.get("error"),
        "source_file": path.name,
    }


def _load_results(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result_path in _result_paths(path):
        for row in _read_result_file(result_path):
            rows.append(_normalize_row(row, result_path))
    return rows


def _load_benchmark_nodes(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    files = [path] if path.is_file() else sorted(path.rglob("*.jsonl"))
    samples: dict[str, dict[str, Any]] = {}
    for file in files:
        with file.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid benchmark JSON at {file}:{line_no}: {exc}") from exc
                sample_id = row.get("id") or row.get("sample_id") or row.get("gsm8k_id")
                graph = row.get("gold_reasoning_graph") or {}
                nodes = graph.get("nodes") or []
                node_texts: dict[str, str] = {}
                if isinstance(nodes, list):
                    for index, node in enumerate(nodes):
                        if isinstance(node, dict):
                            node_id = str(node.get("id", index))
                            text = node.get("proposition") or node.get("text") or node.get("label") or json.dumps(node, ensure_ascii=False)
                        else:
                            node_id = str(index)
                            text = str(node)
                        node_texts[node_id] = text
                elif isinstance(nodes, dict):
                    node_texts = {str(key): str(value) for key, value in nodes.items()}
                if sample_id:
                    samples[str(sample_id)] = {
                        "question": row.get("question") or "",
                        "gold_answer": row.get("gold_answer") or "",
                        "node_texts": node_texts,
                    }
    return samples


def _summarize_results(rows: list[dict[str, Any]], summary_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for row in rows:
        model = str(row.get("model_name") or "unknown")
        output_type = str(row.get("output_type") or "unknown")
        by_model[model].append(row)
        by_type[output_type].append(row)
        by_pair[f"{model}::{output_type}"].append(row)

    def row_score(row: dict[str, Any]) -> float | None:
        return _avg(
            [
                _safe_float(row.get("score_depth")),
                _safe_float(row.get("score_breadth")),
                _safe_float(row.get("score_consistency")),
            ]
        )

    def summarize_group(name: str, group: list[dict[str, Any]], field_name: str) -> dict[str, Any]:
        scored = [r for r in group if not r.get("error")]
        depth = _avg([_safe_float(r.get("score_depth")) for r in scored])
        breadth = _avg([_safe_float(r.get("score_breadth")) for r in scored])
        consistency = _avg([_safe_float(r.get("score_consistency")) for r in scored])
        composite = _avg([depth, breadth, consistency])
        return {
            field_name: name,
            "count": len(group),
            "scored_count": len(scored),
            "error_count": len(group) - len(scored),
            "answer_accuracy": _avg([1.0 if r.get("answer_correct") else 0.0 for r in scored]),
            "score_depth": depth,
            "score_breadth": breadth,
            "score_consistency": consistency,
            "composite": composite,
            "grade": _grade(composite),
            "missing_premise_rate": _avg([1.0 if r.get("missing_premise_flag") else 0.0 for r in group]),
            "first_error_rate": _avg([1.0 if r.get("first_error_step") is not None else 0.0 for r in group]),
            "avg_branch_coverage": _avg([_safe_float(r.get("branch_coverage")) for r in group]),
        }

    model_summary = [summarize_group(name, group, "model_name") for name, group in sorted(by_model.items())]
    type_summary = [summarize_group(name, group, "output_type") for name, group in sorted(by_type.items())]
    pair_summary = []
    for key, group in sorted(by_pair.items()):
        model, output_type = key.split("::", 1)
        summary = summarize_group(output_type, group, "output_type")
        summary["model_name"] = model
        pair_summary.append(summary)

    detailed_rows = []
    for row in rows:
        detail = row.get("detail") or {}
        lighted_graph = row.get("lighted_graph") or {}
        graph_steps = lighted_graph.get("steps") or []
        benchmark = row.get("benchmark") or {}
        detailed_rows.append(
            {
                "sample_id": row.get("sample_id"),
                "model_name": row.get("model_name"),
                "output_type": row.get("output_type"),
                "answer_correct": bool(row.get("answer_correct")),
                "score_depth": _safe_float(row.get("score_depth")),
                "score_breadth": _safe_float(row.get("score_breadth")),
                "score_consistency": _safe_float(row.get("score_consistency")),
                "composite": row_score(row),
                "error": row.get("error"),
                "source_file": row.get("source_file"),
                "first_error_step": row.get("first_error_step"),
                "missing_premise_flag": bool(row.get("missing_premise_flag")),
                "branch_coverage": _safe_float(row.get("branch_coverage")),
                "contradiction_count": row.get("contradiction_count") or 0,
                "steps": detail.get("steps") or [
                    f"Step {step.get('step_index')}: {step.get('status')} -> {step.get('node') if step.get('node') is not None else 'unmapped'}"
                    for step in graph_steps
                ],
                "mappings": detail.get("mappings") or [],
                "verifications": detail.get("verifications") or [
                    {"reason": step.get("reason") or step.get("status") or ""} for step in graph_steps
                ],
                "depth": _sanitize(detail.get("depth") or []),
                "consistency": _sanitize(detail.get("consistency") or {}),
                "breadth": _sanitize(detail.get("breadth") or {}),
                "final_answer": detail.get("final_answer") or row.get("response") or "",
                "question": benchmark.get("question") or "",
                "gold_answer": benchmark.get("gold_answer") or "",
                "node_texts": benchmark.get("node_texts") or {},
                "lighted_graph": _sanitize(lighted_graph),
            }
        )

    return {
        "generated_from": {
            "results": "outputs/results/demo_results.jsonl",
            "summary": "outputs/reports/summary.csv",
        },
        "model_summary": model_summary,
        "type_summary": type_summary,
        "pair_summary": pair_summary,
        "summary_csv": summary_rows,
        "results": detailed_rows,
    }


HTML_TEMPLATE = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LLM Reasoning Evaluation Outputs</title>
<style>
:root{
  --color-text-primary:#111827;--color-text-secondary:#6b7280;--color-text-tertiary:#9ca3af;
  --color-text-success:#059669;--color-text-danger:#dc2626;--color-text-warning:#d97706;
  --color-background-primary:#ffffff;--color-background-secondary:#f9fafb;
  --color-background-success:#dcfce7;--color-background-danger:#fee2e2;--color-background-warning:#fef3c7;
  --color-border-primary:#d1d5db;--color-border-secondary:#e5e7eb;--color-border-tertiary:#f3f4f6;
  --border-radius-md:8px;
}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;background:var(--color-background-primary);color:var(--color-text-primary)}
.wrap{padding:1.5rem 1rem;display:flex;flex-direction:column;gap:1.5rem;max-width:1180px;margin:0 auto}
.header-row{display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px}
.title{font-size:15px;font-weight:500;color:var(--color-text-primary)}
.sub{font-size:12px;color:var(--color-text-secondary);line-height:1.6}
.tab-row,.model-sel,.toolbar{display:flex;gap:6px;flex-wrap:wrap;align-items:center}
.tab,.model-btn,.ctrl-btn,select{font-size:12px;padding:5px 14px;border:0.5px solid var(--color-border-secondary);border-radius:20px;cursor:pointer;color:var(--color-text-secondary);background:transparent;transition:all .15s}
select{border-radius:8px;background:var(--color-background-primary);color:var(--color-text-primary);padding:5px 9px}
.tab.active,.model-btn.on,.ctrl-btn.act{border-color:var(--color-border-primary);color:var(--color-text-primary);font-weight:500;background:var(--color-background-secondary)}
.section{display:none;animation:fade .2s ease}
.section.show{display:block}
@keyframes fade{from{opacity:.45}to{opacity:1}}
.grid3{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}
.grid2{display:grid;grid-template-columns:minmax(0,1fr) 300px;gap:12px}
@media(max-width:760px){.grid3,.grid2{grid-template-columns:1fr}.wrap{padding:1rem}.chart-wrap{height:230px!important}}
.metric-card,.card{background:var(--color-background-secondary);border-radius:var(--border-radius-md);padding:12px 14px}
.metric-label,.card-title{font-size:11px;color:var(--color-text-tertiary);margin-bottom:4px;text-transform:uppercase;letter-spacing:.04em}
.metric-val{font-size:22px;font-weight:500;color:var(--color-text-primary)}
.metric-delta{font-size:11px;margin-top:2px}.pos{color:var(--color-text-success)}.neg{color:var(--color-text-danger)}.neu{color:var(--color-text-secondary)}
.chart-wrap{position:relative;width:100%;height:280px}.chart-sm{height:210px}
.table-wrap{overflow-x:auto}
table{width:100%;border-collapse:collapse;font-size:13px}
th{font-weight:500;font-size:12px;color:var(--color-text-secondary);text-align:left;padding:6px 10px;border-bottom:0.5px solid var(--color-border-tertiary);white-space:nowrap}
td{padding:7px 10px;border-bottom:0.5px solid var(--color-border-tertiary);color:var(--color-text-primary);vertical-align:middle}
tr:last-child td{border-bottom:none}tbody tr:hover{background:rgba(249,250,251,.85)}
.bar-cell{display:flex;align-items:center;gap:8px;min-width:120px}
.bar-bg{flex:1;height:6px;background:var(--color-border-tertiary);border-radius:3px;overflow:hidden}
.bar-fill{height:100%;border-radius:3px;transition:width .4s}
.badge{display:inline-block;font-size:10px;padding:2px 7px;border-radius:10px;white-space:nowrap}
.badge-hi{background:var(--color-background-success);color:var(--color-text-success)}
.badge-md{background:var(--color-background-warning);color:var(--color-text-warning)}
.badge-lo{background:var(--color-background-danger);color:var(--color-text-danger)}
.badge-gray{background:var(--color-background-primary);color:var(--color-text-secondary)}
.legend{display:flex;gap:14px;flex-wrap:wrap;font-size:11px;color:var(--color-text-secondary)}
.legend span{display:flex;align-items:center;gap:5px}.dot{width:10px;height:10px;border-radius:50%;flex-shrink:0}
.tree-svg-wrap{overflow:auto;background:var(--color-background-secondary);border-radius:var(--border-radius-md);padding:10px;min-height:340px}
.tree-svg-wrap svg{display:block;min-width:640px}
.detail-list{display:flex;flex-direction:column;gap:7px;margin-top:8px}.info-item{font-size:12px;padding:6px 0;border-bottom:0.5px solid var(--color-border-tertiary);display:flex;justify-content:space-between;gap:10px}.info-item:last-child{border-bottom:none}.info-label{color:var(--color-text-secondary);white-space:nowrap}.latex-text{max-width:72%;text-align:right;line-height:1.55;word-break:break-word}#node-detail .info-item:nth-child(3){display:block}#node-detail .info-item:nth-child(3) .latex-text{display:block;max-width:none;text-align:left;margin-top:6px;color:var(--color-text-primary)}
.step-list{font-size:12px;color:var(--color-text-secondary);line-height:1.6;padding-left:16px;margin-top:8px}.step-list li{margin-bottom:3px}
.node{cursor:pointer;transition:transform .15s}.node:hover{transform:scale(1.02)}
</style>
</head>
<body>
<div class="wrap">
  <div class="header-row">
    <div>
      <p class="title">LLM 推理输出评估可视化</p>
      <p class="sub">读取 outputs/results JSONL 自动生成 · 后续追加模型或样例后重新运行脚本即可复现</p>
    </div>
    <div class="tab-row">
      <button class="tab active" onclick="showTab('overview', this)">概览</button>
      <button class="tab" onclick="showTab('compare', this)">输出对比</button>
      <button class="tab" onclick="showTab('tree', this)">推理树</button>
      <button class="tab" onclick="showTab('rank', this)">排行榜</button>
    </div>
  </div>

  <div id="s-overview" class="section show">
    <div class="grid3" style="margin-bottom:14px">
      <div class="metric-card"><div class="metric-label">平均深度指数</div><div class="metric-val" id="mv-depth">—</div><div class="metric-delta pos" id="md-depth">Score_Depth</div></div>
      <div class="metric-card"><div class="metric-label">平均广度指数</div><div class="metric-val" id="mv-breadth">—</div><div class="metric-delta neu" id="md-breadth">Score_Breadth</div></div>
      <div class="metric-card"><div class="metric-label">平均一致性指数</div><div class="metric-val" id="mv-cons">—</div><div class="metric-delta pos" id="md-cons">Score_Consistency</div></div>
    </div>
    <div class="grid2">
      <div class="card">
        <div class="card-title">模型三维雷达</div>
        <div class="chart-wrap"><canvas id="radar-overview"></canvas></div>
        <div class="legend" id="model-legend" style="margin-top:10px"></div>
      </div>
      <div class="card">
        <div class="card-title">数据摘要</div>
        <div class="detail-list" id="overview-facts"></div>
      </div>
    </div>
  </div>

  <div id="s-compare" class="section">
    <div class="header-row" style="margin-bottom:12px">
      <div class="model-sel" id="model-sel"></div>
      <select id="compare-mode" onchange="rebuildCompare()"> <option value="type">按输出类型</option><option value="model">按模型</option></select>
    </div>
    <div class="card">
      <div class="card-title">Depth / Breadth / Consistency 分组柱状图</div>
      <div class="chart-wrap"><canvas id="bar-compare"></canvas></div>
    </div>
    <div class="card" style="margin-top:12px">
      <div class="card-title">输出类型统计</div>
      <div class="table-wrap"><table><thead><tr><th>输出类型</th><th>样本数</th><th>准确率</th><th>平均深度</th><th>平均广度</th><th>平均一致性</th><th>缺失前提率</th><th>首错率</th></tr></thead><tbody id="type-body"></tbody></table></div>
    </div>
  </div>

  <div id="s-tree" class="section">
    <div class="header-row" style="margin-bottom:10px">
      <p class="sub">选择一条输出记录查看 lighted_graph、步骤映射和断裂/冗余状态</p>
      <select id="result-pick" onchange="renderSelectedResult()"></select>
    </div>
    <div class="grid2">
      <div>
        <div class="toolbar" style="margin-bottom:8px">
          <button class="ctrl-btn act" id="toggle-gain" onclick="toggleGain()">显示步骤标注</button>
          <button class="ctrl-btn" id="toggle-break" onclick="toggleBreakPath()">仅聚焦问题节点</button>
        </div>
        <div class="tree-svg-wrap"><svg id="tree-svg" width="100%" viewBox="0 0 720 360" role="img"></svg></div>
        <div class="legend" style="margin-top:10px">
          <span><span class="dot" style="background:#3266ad"></span>lit / used_valid</span>
          <span><span class="dot" style="background:#ef4444"></span>jump / wrong</span>
          <span><span class="dot" style="background:#9ca3af"></span>redundant / unused</span>
          <span><span class="dot" style="background:#10b981"></span>final / high score</span>
        </div>
      </div>
      <div style="display:flex;flex-direction:column;gap:10px">
        <div class="card">
          <div class="card-title">节点详情</div>
          <div class="detail-list" id="node-detail">
            <div class="info-item"><span class="info-label">节点</span><span>点击树节点查看</span></div>
          </div>
          <p class="sub" id="node-hint" style="margin-top:8px">节点颜色和线型沿用 v2：蓝色有效、红色断裂、灰色未访问/冗余。</p>
        </div>
        <div class="card">
          <div class="card-title">记录详情</div>
          <div class="detail-list" id="record-detail"></div>
        </div>
        <div class="card">
          <div class="card-title">推理步骤</div>
          <ol class="step-list" id="step-list"></ol>
        </div>
      </div>
    </div>
  </div>

  <div id="s-rank" class="section">
    <div class="header-row" style="margin-bottom:10px">
      <p class="sub">按综合分排序；当后续出现更多模型时会自动扩展</p>
      <select id="rank-sort" onchange="buildRank()"><option value="composite">按综合分排序</option><option value="score_depth">按深度排序</option><option value="score_breadth">按广度排序</option><option value="score_consistency">按一致性排序</option></select>
    </div>
    <div class="card">
      <div class="table-wrap"><table><thead><tr><th>排名</th><th>模型</th><th>样本数</th><th>Score_Depth</th><th>Score_Breadth</th><th>Score_Consistency</th><th>综合分</th><th>等级</th></tr></thead><tbody id="rank-body"></tbody></table></div>
    </div>
  </div>
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<script>
window.MathJax = {
  tex: { inlineMath: [['\\(', '\\)'], ['$', '$']], displayMath: [['\\[', '\\]'], ['$$', '$$']], processEscapes: true },
  svg: { fontCache: 'global' }
};
</script>
<script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>
<script>
const DATA = __DATA_JSON__;
const COLORS=['#3b82f6','#8b5cf6','#10b981','#f59e0b','#ef4444','#06b6d4','#84cc16','#f97316'];
let radarChart,barChart,showGain=true,focusBreak=false,currentResultIndex=0;
const selectedModels=new Set(DATA.model_summary.map(x=>x.model_name));
function n(v,d='—'){return v===null||v===undefined?d:(Number.isInteger(v)?String(v):Number(v).toFixed(1))}
function pct(v){return v===null||v===undefined?'—':(v*100).toFixed(1)+'%'}
function color(i){return COLORS[i%COLORS.length]}
function hexA(hex,a){const r=parseInt(hex.slice(1,3),16),g=parseInt(hex.slice(3,5),16),b=parseInt(hex.slice(5,7),16);return `rgba(${r},${g},${b},${a})`}
function esc(v){return String(v??'').replace(/[&<>"']/g,ch=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]))}
function typeset(){if(window.MathJax&&MathJax.typesetPromise)MathJax.typesetPromise().catch(()=>{})}
function showTab(t,btn){document.querySelectorAll('.section').forEach(s=>s.classList.remove('show'));document.querySelectorAll('.tab').forEach(b=>b.classList.remove('active'));document.getElementById('s-'+t).classList.add('show');btn.classList.add('active');if(t==='compare'&&!barChart)rebuildCompare();if(t==='tree')renderSelectedResult();if(t==='rank')buildRank();}
function initOverview(){
  const models=DATA.model_summary;
  const avg=k=>{const a=models.filter(m=>m[k]!=null);return a.length?a.reduce((s,m)=>s+m[k],0)/a.length:null};
  document.getElementById('mv-depth').textContent=n(avg('score_depth'));
  document.getElementById('mv-breadth').textContent=n(avg('score_breadth'));
  document.getElementById('mv-cons').textContent=n(avg('score_consistency'));
  document.getElementById('overview-facts').innerHTML=[
    ['模型数',models.length],['输出记录数',DATA.results.length],['正常评分记录',DATA.results.filter(r=>!r.error).length],['错误记录',DATA.results.filter(r=>r.error).length],
    ['答案准确率',pct(DATA.results.filter(r=>!r.error).reduce((s,r)=>s+(r.answer_correct?1:0),0)/Math.max(DATA.results.filter(r=>!r.error).length,1))]
  ].map(([a,b])=>`<div class="info-item"><span class="info-label">${a}</span><span>${b}</span></div>`).join('');
  document.getElementById('model-legend').innerHTML=models.map((m,i)=>`<span><span class="dot" style="background:${color(i)}"></span>${esc(m.model_name)}</span>`).join('');
  radarChart=new Chart(document.getElementById('radar-overview'),{type:'radar',data:{labels:['深度','广度','一致性'],datasets:models.map((m,i)=>({label:m.model_name,data:[m.score_depth||0,m.score_breadth||0,m.score_consistency||0],backgroundColor:hexA(color(i),0.08),borderColor:color(i),pointBackgroundColor:color(i),borderWidth:1.5,pointRadius:3}))},options:{responsive:true,maintainAspectRatio:false,scales:{r:{min:0,max:100,ticks:{font:{size:10},stepSize:25},grid:{color:'rgba(128,128,128,0.15)'},pointLabels:{font:{size:12}}}},plugins:{legend:{display:false},tooltip:{callbacks:{label:c=>`${c.dataset.label}: ${c.raw}`}}}}});
}
function initCompare(){
  document.getElementById('model-sel').innerHTML=DATA.model_summary.map((m,i)=>`<button class="model-btn on" style="border-color:${color(i)}" onclick="toggleModel('${m.model_name}',this)">${m.model_name}</button>`).join('');
  buildTypeTable();
}
function toggleModel(name,btn){if(selectedModels.has(name)&&selectedModels.size>1){selectedModels.delete(name);btn.classList.remove('on');btn.style.borderColor='';}else{selectedModels.add(name);btn.classList.add('on');btn.style.borderColor=color(DATA.model_summary.findIndex(m=>m.model_name===name));}rebuildCompare();}
function rebuildCompare(){
  const mode=document.getElementById('compare-mode').value;
  const rows=(mode==='model'?DATA.model_summary:DATA.pair_summary.filter(r=>selectedModels.has(r.model_name)));
  const labels=rows.map(r=>mode==='model'?r.model_name:r.output_type);
  if(barChart)barChart.destroy();
  barChart=new Chart(document.getElementById('bar-compare'),{type:'bar',data:{labels,datasets:[{label:'深度',data:rows.map(r=>r.score_depth||0),backgroundColor:'#3266ad'},{label:'广度',data:rows.map(r=>r.score_breadth||0),backgroundColor:'#7c3aed'},{label:'一致性',data:rows.map(r=>r.score_consistency||0),backgroundColor:'#059669'}]},options:{responsive:true,maintainAspectRatio:false,scales:{y:{min:0,max:100,grid:{color:'rgba(128,128,128,0.1)'}},x:{grid:{display:false},ticks:{font:{size:10}}}},plugins:{legend:{position:'bottom',labels:{font:{size:11},boxWidth:12,padding:10}}}}});
}
function buildTypeTable(){document.getElementById('type-body').innerHTML=DATA.type_summary.map(r=>`<tr><td style="font-weight:500">${esc(r.output_type)}</td><td>${r.count}</td><td>${pct(r.answer_accuracy)}</td><td>${n(r.score_depth)}</td><td>${n(r.score_breadth)}</td><td>${n(r.score_consistency)}</td><td>${pct(r.missing_premise_rate)}</td><td>${pct(r.first_error_rate)}</td></tr>`).join('');}
function initTree(){document.getElementById('result-pick').innerHTML=DATA.results.map((r,i)=>`<option value="${i}">${esc(r.model_name)} / ${esc(r.output_type)} / ${esc(r.sample_id)}</option>`).join('');renderSelectedResult();}
function toggleGain(){showGain=!showGain;document.getElementById('toggle-gain').classList.toggle('act',showGain);renderSelectedResult();}
function toggleBreakPath(){focusBreak=!focusBreak;document.getElementById('toggle-break').classList.toggle('act',focusBreak);renderSelectedResult();}
function statusColor(s){if(['jump','wrong'].includes(s))return '#dc2626';if(['redundant','unused','unvisited'].includes(s))return '#9ca3af';if(['lit','used_valid'].includes(s))return '#3266ad';return '#64748b'}
function nodeStyle(status,isLeaf){
  if(['jump','wrong'].includes(status))return {fill:'#fef2f2',stroke:'#dc2626',text:'#7f1d1d',w:2,dash:''};
  if(status==='redundant')return {fill:'#f9fafb',stroke:'#9ca3af',text:'#4b5563',w:1,dash:'stroke-dasharray="5 3"'};
  if(status==='lit'&&isLeaf)return {fill:'#f0fdf4',stroke:'#059669',text:'#064e3b',w:1.7,dash:''};
  if(status==='lit')return {fill:'#eff6ff',stroke:'#2563eb',text:'#1e3a5f',w:1,dash:''};
  return {fill:'#f8fafc',stroke:'#64748b',text:'#475569',w:.8,dash:''};
}
function edgeStyle(status,fromStatus,toStatus){
  if(['jump','wrong'].includes(fromStatus)||['jump','wrong'].includes(toStatus))return {stroke:'#dc2626',w:1.5,dash:'stroke-dasharray="6 3"'};
  if(status==='unused'||toStatus==='redundant'||toStatus==='unvisited')return {stroke:'#d1d5db',w:.9,dash:'stroke-dasharray="4 2"'};
  return {stroke:'#cbd5e1',w:1.1,dash:''};
}
function renderSelectedResult(){
  const idx=Number(document.getElementById('result-pick').value||0), r=DATA.results[idx]; currentResultIndex=idx; if(!r)return;
  const nodesObj=r.lighted_graph.nodes||{}, edgesObj=r.lighted_graph.edges||{}, nodeIds=Object.keys(nodesObj);
  const edges=Object.keys(edgesObj).map(e=>e.split('->')).filter(e=>e.length===2);
  if(r.error||!nodeIds.length){document.getElementById('tree-svg').innerHTML='<text x="24" y="42" font-size="13" fill="#dc2626">该记录未生成可视化图：'+(r.error||'无 lighted_graph')+'</text>';updateNodeDetail(null);return;}
  const levels={}; const incoming=new Set(edges.map(e=>e[1])); nodeIds.forEach(id=>{levels[id]=incoming.has(id)?1:0}); for(let pass=0;pass<nodeIds.length;pass++)edges.forEach(([a,b])=>{levels[b]=Math.max(levels[b]||0,(levels[a]||0)+1)});
  const byLevel={}; nodeIds.forEach(id=>{const l=levels[id]||0;(byLevel[l]||(byLevel[l]=[])).push(id)});
  Object.values(byLevel).forEach(arr=>arr.sort((a,b)=>Number(a)-Number(b)||String(a).localeCompare(String(b))));
  const maxLevel=Math.max(...Object.keys(byLevel).map(Number),0), maxWide=Math.max(...Object.values(byLevel).map(a=>a.length),1);
  const width=Math.max(640,maxWide*122+80), height=Math.max(320,(maxLevel+1)*86+56);
  const pos={}; Object.keys(byLevel).forEach(l=>byLevel[l].forEach((id,i,arr)=>{pos[id]={x:(i+1)*width/(arr.length+1),y:34+Number(l)*84}}));
  const problemNodes=new Set(nodeIds.filter(id=>['jump','wrong','redundant'].includes(nodesObj[id]))); edges.forEach(([a,b])=>{if(problemNodes.has(b))problemNodes.add(a)});
  const svg=document.getElementById('tree-svg'); svg.setAttribute('viewBox',`0 0 ${width} ${height}`);
  let html='<defs><marker id="arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M2 1L8 5L2 9" fill="none" stroke="context-stroke" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></marker></defs>';
  edges.forEach(([a,b])=>{const pa=pos[a],pb=pos[b]; if(!pa||!pb)return; const st=edgesObj[`${a}->${b}`]||'unused', es=edgeStyle(st,nodesObj[a],nodesObj[b]), op=focusBreak&&!problemNodes.has(a)&&!problemNodes.has(b)?0.18:1; html+=`<line x1="${pa.x}" y1="${pa.y+20}" x2="${pb.x}" y2="${pb.y-20}" stroke="${es.stroke}" stroke-width="${es.w}" ${es.dash} opacity="${op}" marker-end="url(#arr)"/>`; if(showGain)html+=`<text x="${(pa.x+pb.x)/2}" y="${(pa.y+pb.y)/2-7}" text-anchor="middle" font-size="10" fill="${es.stroke}" opacity="${op}">${st}</text>`});
  nodeIds.forEach(id=>{const p=pos[id], st=nodesObj[id], isLeaf=!edges.some(e=>e[0]===id), ns=nodeStyle(st,isLeaf), op=focusBreak&&!problemNodes.has(id)?0.2:1; html+=`<g class="node" opacity="${op}" onclick="selectGraphNode('${id}')"><rect id="rect-node-${id}" x="${p.x-52}" y="${p.y-20}" width="104" height="40" rx="8" fill="${ns.fill}" stroke="${ns.stroke}" stroke-width="${ns.w}" ${ns.dash}/><text x="${p.x}" y="${p.y-4}" text-anchor="middle" font-size="12" font-weight="500" fill="${ns.text}">节点 ${id}</text><text x="${p.x}" y="${p.y+12}" text-anchor="middle" font-size="10" fill="${ns.text}" opacity=".72">${st}</text></g>`});
  svg.innerHTML=html;
  document.getElementById('record-detail').innerHTML=[['模型',r.model_name],['输出类型',r.output_type],['样例',r.sample_id],['题目',r.question||'—'],['Gold Answer',r.gold_answer||'—'],['最终答案',r.final_answer||'—'],['深度',n(r.score_depth)],['广度',n(r.score_breadth)],['一致性',n(r.score_consistency)],['首个错误步骤',r.first_error_step||'—'],['缺失前提',r.missing_premise_flag?'是':'否']].map(([a,b])=>`<div class="info-item"><span class="info-label">${esc(a)}</span><span class="latex-text">${esc(b)}</span></div>`).join('');
  document.getElementById('step-list').innerHTML=(r.steps||[]).map((s,i)=>`<li><span class="latex-text">${esc(s)}</span><br><span style="color:var(--color-text-tertiary)" class="latex-text">${esc((r.verifications[i]&&r.verifications[i].reason)||'')}</span></li>`).join('');
  updateNodeDetail(nodeIds[0]);
  typeset();
}
function updateNodeDetail(id){
  const r=DATA.results[currentResultIndex]; if(!id||!r){document.getElementById('node-detail').innerHTML='<div class="info-item"><span class="info-label">节点</span><span>—</span></div>';return;}
  const status=(r.lighted_graph.nodes||{})[id], related=(r.lighted_graph.steps||[]).filter(s=>String(s.node)===String(id));
  const nodeText=(r.node_texts||{})[id]||'未在 data/processed 中找到该节点文本';
  document.querySelectorAll('#tree-svg rect').forEach(x=>x.setAttribute('filter',''));
  const rect=document.getElementById('rect-node-'+id); if(rect)rect.setAttribute('filter','drop-shadow(0 0 4px '+statusColor(status)+'66)');
  document.getElementById('node-detail').innerHTML=[
    ['节点ID',id],['状态',status],['节点内容',nodeText],['关联步骤',related.map(s=>s.step_index).join(', ')||'—'],['说明',related.map(s=>s.reason||s.status).filter(Boolean).join(' / ')||'暂无步骤映射']
  ].map(([a,b])=>`<div class="info-item"><span class="info-label">${esc(a)}</span><span class="latex-text">${esc(b)}</span></div>`).join('');
  document.getElementById('node-hint').textContent=status==='jump'?'跳步节点：通常表示模型越过了中间必要推理。':status==='wrong'?'错误节点：当前步骤未能映射到合法后继。':status==='redundant'?'冗余节点：重复访问，不提供新的逻辑增益。':'有效或未访问节点，结合边状态查看推理覆盖情况。';
  typeset();
}
function selectGraphNode(id){updateNodeDetail(id)}
function gradeBadge(g){if(g==='优秀')return 'badge-hi';if(g==='良好'||g==='待补充')return 'badge-md';return 'badge-lo'}
function bar(v,c){return `<div class="bar-cell"><div class="bar-bg"><div class="bar-fill" style="width:${v||0}%;background:${c}"></div></div><span style="font-size:12px;min-width:34px">${n(v)}</span></div>`}
function buildRank(){const k=document.getElementById('rank-sort').value;const rows=[...DATA.model_summary].sort((a,b)=>(b[k]||-1)-(a[k]||-1));document.getElementById('rank-body').innerHTML=rows.map((m,i)=>`<tr style="${i===0?'background:#fffbeb':''}"><td style="font-weight:500;color:${i<3?color(i):'var(--color-text-secondary)'}">${i+1}</td><td style="font-weight:500">${esc(m.model_name)}</td><td>${m.count}</td><td>${bar(m.score_depth,'#3266ad')}</td><td>${bar(m.score_breadth,'#7c3aed')}</td><td>${bar(m.score_consistency,'#059669')}</td><td style="font-weight:500">${n(m.composite)}</td><td><span class="badge ${gradeBadge(m.grade)}">${esc(m.grade)}</span></td></tr>`).join('')}
initOverview();initCompare();initTree();buildRank();
</script>
</body>
</html>
"""


def build_html(results_path: Path, summary_path: Path, benchmark_path: Path, output_path: Path) -> None:
    rows = _load_results(results_path)
    benchmark_nodes = _load_benchmark_nodes(benchmark_path)
    for row in rows:
        row["benchmark"] = benchmark_nodes.get(str(row.get("sample_id")), {})
    summary_rows = _read_summary(summary_path)
    data = _summarize_results(rows, summary_rows)
    data["generated_from"] = {"results": str(results_path), "summary": str(summary_path), "benchmark": str(benchmark_path)}
    html = HTML_TEMPLATE.replace("__DATA_JSON__", json.dumps(data, ensure_ascii=False, allow_nan=False))
    output_path.write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="outputs/results")
    parser.add_argument("--summary", default="outputs/reports/summary.csv")
    parser.add_argument("--benchmark", default="data/processed")
    parser.add_argument("--output", default="llm_reasoning_outputs_visualization.html")
    args = parser.parse_args()
    build_html(Path(args.results), Path(args.summary), Path(args.benchmark), Path(args.output))
    print(f"Visualization HTML -> {args.output}")


if __name__ == "__main__":
    main()
