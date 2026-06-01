from __future__ import annotations

from reasoning_eval.common.schema import MappingResult, VerificationResult
from reasoning_eval.dataset.graph_utils import build_nx_graph, shortest_distance_to_goal


def score_depth(graph: dict, mappings: list[MappingResult], verifications: list[VerificationResult]) -> tuple[float, list[dict]]:
    g = build_nx_graph(graph)
    goal = graph["goal_node"]
    start = graph["start_nodes"][0]
    total = shortest_distance_to_goal(g, start, goal)
    if total == float("inf") or total == 0:
        return 0.0, []

    previous_node = None
    detail = []
    progress_sum = 0.0
    for mapping, verification in zip(mappings, verifications):
        current = mapping.matched_node_id
        before_node = previous_node or start
        before = shortest_distance_to_goal(g, before_node, goal)
        after = shortest_distance_to_goal(g, current, goal) if current else float("inf")
        raw_gain = max(0.0, before - after)
        if not verification.valid:
            gain = raw_gain * 0.3 if verification.missing_premise else 0.0
        elif verification.redundant:
            gain = 0.0
        else:
            gain = raw_gain
        progress_sum += gain
        detail.append(
            {
                "previous_node": previous_node,
                "current_node": current,
                "distance_before": before,
                "distance_after": after,
                "progress_gain": gain,
            }
        )
        if verification.valid and not verification.redundant and current:
            previous_node = current

    return round(min(100.0, 100.0 * progress_sum / total), 3), detail

