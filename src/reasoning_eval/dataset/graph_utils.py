from __future__ import annotations

import networkx as nx


def _is_flat_graph(graph: dict) -> bool:
    """Detect GSM8K-style flat graphs: nodes are strings, edges are [int, int] pairs."""
    nodes = graph.get("nodes", [])
    if nodes and isinstance(nodes[0], str):
        return True
    return False


def normalize_graph(graph: dict) -> dict:
    """Convert a flat (GSM8K-style) graph into the structured format expected by scorers.

    Flat format::

        {"nodes": ["step text", ...], "edges": [[0, 2], [1, 3], ...]}

    Structured format::

        {"nodes": [{"id": "0", "proposition": "step text"}, ...],
         "edges": [{"source": "0", "target": "2", "rule_text": ""}, ...],
         "goal_node": "4", "start_nodes": ["0", "1"]}
    """
    if not _is_flat_graph(graph):
        return graph

    raw_nodes = graph["nodes"]
    raw_edges = graph.get("edges", [])

    # Convert nodes
    nodes = []
    for i, text in enumerate(raw_nodes):
        nid = str(i)
        nodes.append({"id": nid, "proposition": text})

    # Convert edges
    edges = []
    for e in raw_edges:
        edges.append({"source": str(e[0]), "target": str(e[1]), "rule_text": ""})

    # Determine start nodes (no incoming edges) and goal node (no outgoing edges)
    all_sources = {str(e[0]) for e in raw_edges}
    all_targets = {str(e[1]) for e in raw_edges}
    start_nodes = sorted(all_sources - all_targets)
    goal_node = sorted(all_targets - all_sources)

    # Fallback: use node 0 as start and last node as goal
    if not start_nodes:
        start_nodes = ["0"]
    goal = goal_node[0] if goal_node else str(len(raw_nodes) - 1)

    return {
        "nodes": nodes,
        "edges": edges,
        "goal_node": goal,
        "start_nodes": start_nodes,
    }


def build_nx_graph(graph: dict | object) -> nx.DiGraph:
    g = nx.DiGraph()
    # Normalize flat graphs first
    if isinstance(graph, dict) and _is_flat_graph(graph):
        graph = normalize_graph(graph)

    nodes = graph["nodes"] if isinstance(graph, dict) else graph.nodes
    edges = graph["edges"] if isinstance(graph, dict) else graph.edges
    for node in nodes:
        data = node if isinstance(node, dict) else node.__dict__
        g.add_node(data["id"], **data)
    for edge in edges:
        data = edge if isinstance(edge, dict) else edge.__dict__
        src = data.get("source") or (data.get("premises", [""])[0] if data.get("premises") else "")
        tgt = data.get("target", "")
        g.add_edge(src, tgt, **data)
    if not nx.is_directed_acyclic_graph(g):
        raise ValueError("Reasoning graph must be a DAG")
    return g


def shortest_distance_to_goal(g: nx.DiGraph, node: str, goal: str) -> float:
    if node not in g or goal not in g:
        return float("inf")
    try:
        return float(nx.shortest_path_length(g, source=node, target=goal))
    except nx.NetworkXNoPath:
        return float("inf")


def get_successors(g: nx.DiGraph, node: str) -> list[str]:
    return list(g.successors(node)) if node in g else []


def get_predecessors(g: nx.DiGraph, node: str) -> list[str]:
    return list(g.predecessors(node)) if node in g else []


def get_valid_branches(g: nx.DiGraph, node: str) -> list[str]:
    return [succ for succ in get_successors(g, node) if g.edges[node, succ].get("status") != "distractor"]


def is_reachable(g: nx.DiGraph, source: str, target: str) -> bool:
    return source in g and target in g and nx.has_path(g, source, target)


def is_direct_successor(g: nx.DiGraph, source: str, target: str) -> bool:
    return source in g and target in g and g.has_edge(source, target)

