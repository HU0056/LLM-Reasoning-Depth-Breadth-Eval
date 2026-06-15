from __future__ import annotations

from reasoning_eval.common.schema import MappingResult
from reasoning_eval.common.text_utils import jaccard, normalize_text


def _mentions_rule(step_norm: str, source: str, target: str) -> bool:
    return f"{source.lower()}->{target.lower()}" in step_norm.replace(" ", "")


def map_step_to_node(step_text: str, graph: dict) -> MappingResult:
    step_norm = normalize_text(step_text)
    nodes = graph["nodes"]
    edges = graph["edges"]

    best_node = None
    best_conf = 0.0
    best_reason = "no proposition or rule matched"
    for edge in edges:
        source = edge["source"]
        target = edge["target"]
        rule_text = normalize_text(edge.get("rule_text", ""))
        if rule_text and (_mentions_rule(step_norm, source, target) or rule_text in step_norm):
            return MappingResult(step_text, target, 0.95, f"matched rule text {edge['rule_text']} and maps to target {target}")

    for node in nodes:
        prop = node["proposition"]
        prop_norm = normalize_text(prop)
        if prop_norm and prop_norm in step_norm:
            confidence = 0.8
            if "不成立" in step_norm.replace(" ", ""):
                confidence = 0.7
            if confidence > best_conf:
                best_node = node["id"]
                best_conf = confidence
                best_reason = f"step contains proposition {prop}"
        score = jaccard(step_text, prop)
        if score > best_conf:
            best_node = node["id"]
            best_conf = score
            best_reason = f"jaccard similarity with proposition {prop}: {score:.2f}"

    if best_conf >= 0.2:
        return MappingResult(step_text, best_node, round(best_conf, 3), best_reason)
    return MappingResult(step_text, None, 0.0, best_reason)

