from __future__ import annotations

import networkx as nx


def build_nx_graph(graph: dict | object) -> nx.DiGraph:
    g = nx.DiGraph()
    nodes = graph["nodes"] if isinstance(graph, dict) else graph.nodes
    edges = graph["edges"] if isinstance(graph, dict) else graph.edges
    for node in nodes:
        data = node if isinstance(node, dict) else node.__dict__
        g.add_node(data["id"], **data)
    for edge in edges:
        data = edge if isinstance(edge, dict) else edge.__dict__
        g.add_edge(data["source"], data["target"], **data)
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

