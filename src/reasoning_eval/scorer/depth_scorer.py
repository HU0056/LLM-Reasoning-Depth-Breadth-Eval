from __future__ import annotations

from reasoning_eval.common.schema import MappingResult, VerificationResult
from reasoning_eval.dataset.graph_utils import build_nx_graph, shortest_distance_to_goal


def _find_longest_path_length(g, start_nodes: list[str], goal: str) -> float:
    """Find the longest simple path length from any start node to the goal."""
    import networkx as nx
    best = 0.0
    for s in start_nodes:
        try:
            for path in nx.all_simple_paths(g, source=s, target=goal):
                best = max(best, float(len(path) - 1))
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue
    return best if best > 0 else 1.0


def score_depth(graph: dict, mappings: list[MappingResult], verifications: list[VerificationResult]) -> tuple[float, list[dict]]:
    g = build_nx_graph(graph)
    goal = graph["goal_node"]
    start_nodes = graph["start_nodes"]

    total = _find_longest_path_length(g, start_nodes, goal)
    if total == float("inf") or total == 0:
        return 0.0, []

    previous_node = None
    detail = []
    visited_valid: set[str] = set()
    progress_sum = 0.0

    for mapping, verification in zip(mappings, verifications):
        current = mapping.matched_node_id

        if verification.valid and not verification.redundant and current and current not in visited_valid:
            visited_valid.add(current)
            # Count each newly-lit unique node as progress
            progress_sum += 1.0

        before_node = previous_node or (start_nodes[0] if start_nodes else "0")
        before = shortest_distance_to_goal(g, before_node, goal)
        after = shortest_distance_to_goal(g, current, goal) if current else float("inf")

        detail.append(
            {
                "previous_node": previous_node,
                "current_node": current,
                "distance_before": before,
                "distance_after": after,
                "progress_gain": 0.0,  # recomputed via unique-node count
            }
        )

        if verification.valid and not verification.redundant and current:
            previous_node = current

    # Normalize: fraction of unique correctly-lit nodes vs longest path length
    depth = round(min(100.0, 100.0 * progress_sum / total), 3)
    return depth, detail

