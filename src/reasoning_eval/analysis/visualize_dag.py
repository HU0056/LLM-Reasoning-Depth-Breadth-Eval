from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx


NODE_COLORS = {
    "unvisited": "#bdbdbd",
    "lit": "#2ca25f",
    "redundant": "#ffd92f",
    "jump": "#fdae61",
    "wrong": "#de2d26",
    "contradiction": "#756bb1",
    "counterfactual": "#3182bd",
}
EDGE_COLORS = {"unused": "#bdbdbd", "used_valid": "#2ca25f", "skipped": "#fdae61", "wrong": "#de2d26"}


def draw_lighted_dag(sample: dict, result: dict, figures_dir: str) -> Path:
    graph = sample["gold_reasoning_graph"]
    g = nx.DiGraph()
    for node in graph["nodes"]:
        g.add_node(node["id"])
    for edge in graph["edges"]:
        g.add_edge(edge["source"], edge["target"])
    pos = nx.spring_layout(g, seed=7)
    node_status = result["lighted_graph"]["nodes"]
    edge_status = result["lighted_graph"]["edges"]
    node_colors = [NODE_COLORS.get(node_status.get(node, "unvisited"), "#bdbdbd") for node in g.nodes]
    edge_colors = [EDGE_COLORS.get(edge_status.get(f"{u}->{v}", "unused"), "#bdbdbd") for u, v in g.edges]

    plt.figure(figsize=(6, 4))
    nx.draw_networkx_nodes(g, pos, node_color=node_colors, node_size=1200, edgecolors="#333333")
    nx.draw_networkx_edges(g, pos, edge_color=edge_colors, arrows=True, arrowsize=18, width=2)
    nx.draw_networkx_labels(g, pos, font_size=10)
    plt.title(f"{result['sample_id']} / {result['output_type']}")
    plt.axis("off")
    out = Path(figures_dir) / f"dag_{result['sample_id']}_{result['output_type']}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()
    return out

