from __future__ import annotations

from reasoning_eval.dataset.graph_utils import build_nx_graph, get_valid_branches
from reasoning_eval.scorer.mapper import map_step_to_node


def score_breadth(graph: dict, sampled_paths: list[list[str]], key_branch_nodes: list[str]) -> tuple[float | None, dict]:
    if not key_branch_nodes:
        return None, {"reason": "sample has no key_branch_nodes", "branch_coverage": None}
    g = build_nx_graph(graph)
    covered: dict[str, set[str]] = {node: set() for node in key_branch_nodes}
    totals: dict[str, list[str]] = {}
    for node in key_branch_nodes:
        successors = get_valid_branches(g, node)
        totals[node] = successors
        for path in sampled_paths:
            mapped_nodes = [map_step_to_node(step, graph).matched_node_id for step in path]
            for succ in successors:
                if succ in mapped_nodes:
                    covered[node].add(succ)

    total_successors = sum(len(v) for v in totals.values())
    covered_successors = sum(len(v) for v in covered.values())
    coverage = covered_successors / total_successors if total_successors else 0.0
    return round(coverage * 100, 3), {
        "branch_coverage": round(coverage, 3),
        "covered_successors": {k: sorted(v) for k, v in covered.items()},
        "total_successors": totals,
    }

