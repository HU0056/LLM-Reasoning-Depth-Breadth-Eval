"""Depth scorer — reasoning progress measured by difficulty-weighted edge traversal.

Definition (per user requirement):
    For each edge e, assign a reasoning difficulty d(e) based on the mathematical
    justification used.  Let:
      D_total  = min total difficulty from any start node to goal (gold shortest path)
      D_remain = min total difficulty from the set of nodes the model has
                 correctly lit to the goal

    depth = 1 - (D_remain / D_total)

Interpretation:
    depth ≈ 1.0: the model has completed the reasoning — remaining difficulty is 0.
    depth ≈ 0.0: the model has made no progress — remaining difficulty equals total.

This replaces the old "unique-lit-node count / longest path" metric.
"""

from __future__ import annotations

import networkx as nx

from reasoning_eval.common.schema import MappingResult, VerificationResult
from reasoning_eval.dataset.graph_utils import build_nx_graph


def _edge_difficulty(graph: dict, source: str, target: str) -> float:
    """Extract difficulty from an edge's metadata, or default to 1.0."""
    g = build_nx_graph(graph)
    if g.has_edge(source, target):
        data = g.edges[source, target]
        return float(data.get("difficulty", 1.0))
    return 1.0


def _min_difficulty_to_goal(
    g: nx.DiGraph,
    current_nodes: set[str],
    goal: str,
    graph: dict,
) -> float:
    """Minimum total difficulty from any node in current_nodes to goal."""
    if goal in current_nodes:
        return 0.0

    best = float("inf")
    for node in current_nodes:
        if node not in g or goal not in g:
            continue
        try:
            path = nx.shortest_path(g, source=node, target=goal)
            path_difficulty = sum(
                _edge_difficulty(graph, path[i], path[i + 1])
                for i in range(len(path) - 1)
            )
            best = min(best, path_difficulty)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue
    return best if best != float("inf") else float("inf")


def _min_total_difficulty(graph: dict) -> float:
    """Minimum total difficulty from any start to goal (gold baseline)."""
    g = build_nx_graph(graph)
    goal = graph.get("goal_node", "")
    start_nodes = graph.get("start_nodes", [])

    best = float("inf")
    for s in start_nodes:
        if s not in g or goal not in g:
            continue
        try:
            path = nx.shortest_path(g, source=s, target=goal)
            path_difficulty = sum(
                _edge_difficulty(graph, path[i], path[i + 1])
                for i in range(len(path) - 1)
            )
            best = min(best, path_difficulty)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue
    return best if best != float("inf") else 1.0


def score_depth(
    graph: dict,
    mappings: list[MappingResult],
    verifications: list[VerificationResult],
) -> tuple[float, list[dict]]:
    """Compute difficulty-weighted depth score.

    Returns (score_0_to_100, detail_list).
    """
    g = build_nx_graph(graph)
    goal = graph.get("goal_node", "")
    start_nodes = graph.get("start_nodes", [])
    D_total = _min_total_difficulty(graph)

    if D_total == float("inf") or D_total == 0:
        return 0.0, [{"error": "invalid total difficulty", "D_total": D_total}]

    # Collect correctly lit nodes
    lit_nodes: set[str] = set(start_nodes)  # start nodes are "given" by the problem
    detail: list[dict] = []

    for idx, (mapping, verification) in enumerate(zip(mappings, verifications)):
        current = mapping.matched_node_id

        if verification.valid and not verification.redundant and current:
            lit_nodes.add(current)

        D_remain = _min_difficulty_to_goal(g, lit_nodes, goal, graph)
        if D_remain == float("inf"):
            depth_at_step = 0.0
        else:
            depth_at_step = 1.0 - (D_remain / D_total)

        detail.append({
            "step_index": idx,
            "node": current,
            "valid": verification.valid,
            "lit_nodes_count": len(lit_nodes),
            "D_total": D_total,
            "D_remain": D_remain,
            "depth_at_step": round(depth_at_step, 4),
            "difficulty_reduced": round(D_total - D_remain, 3),
        })

    # Final depth: from the final set of lit nodes
    D_remain = _min_difficulty_to_goal(g, lit_nodes, goal, graph)
    if D_remain == float("inf"):
        depth = 0.0
    else:
        depth = max(0.0, min(1.0, 1.0 - (D_remain / D_total)))

    return round(depth * 100.0, 3), detail
