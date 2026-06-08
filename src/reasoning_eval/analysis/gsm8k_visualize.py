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
}

EDGE_COLORS = {
    "unused": "#bdbdbd",
    "used_valid": "#2ca25f",
    "skipped": "#fdae61",
    "wrong": "#de2d26",
}


def _truncate(text: str, max_len: int = 40) -> str:
    """Truncate text for node labels."""
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _short_label(text: str, idx: int, max_len: int = 32) -> str:
    """Create a readable node label with index prefix."""
    truncated = _truncate(text, max_len)
    return f"[{idx}] {truncated}"


def draw_gsm8k_dag(sample: dict, result: dict, figures_dir: str) -> Path:
    """Draw a GSM8K reasoning DAG with node coverage highlighting.

    Args:
        sample: GSM8K benchmark sample with gold_reasoning_graph.
        result: Evaluation result dict with lighted_graph.
        figures_dir: Output directory for figures.

    Returns:
        Path to the saved PNG.
    """
    graph = sample["gold_reasoning_graph"]
    nodes: list[str] = graph["nodes"]
    edges_list: list[list[int]] = graph.get("edges", [])

    # Build networkx graph
    g = nx.DiGraph()
    for idx in range(len(nodes)):
        g.add_node(idx)
    for src, tgt in edges_list:
        if src < len(nodes) and tgt < len(nodes):
            g.add_edge(src, tgt)

    # Determine layout
    if len(nodes) <= 10:
        pos = nx.spring_layout(g, seed=42, k=1.5, iterations=50)
    else:
        pos = nx.spring_layout(g, seed=42, k=0.8, iterations=100)

    # Node colors from lighted graph
    lighted = result.get("lighted_graph", {})
    node_status_map = lighted.get("nodes", {})
    node_colors = [
        NODE_COLORS.get(node_status_map.get(str(idx), "unvisited"), "#bdbdbd")
        for idx in range(len(nodes))
    ]

    # Edge colors
    edge_status_map = lighted.get("edges", {})
    edge_colors = [
        EDGE_COLORS.get(edge_status_map.get(f"{src}->{tgt}", "unused"), "#bdbdbd")
        for src, tgt in g.edges()
    ]

    # Create labels
    labels = {idx: _short_label(nodes[idx], idx) for idx in range(len(nodes))}

    # Draw
    fig, ax = plt.subplots(figsize=(14, 8))
    nx.draw_networkx_nodes(
        g, pos, node_color=node_colors, node_size=1400, edgecolors="#333333", linewidths=1.2, ax=ax
    )
    nx.draw_networkx_edges(
        g, pos, edge_color=edge_colors, arrows=True, arrowsize=16, width=2,
        connectionstyle="arc3,rad=0.08", ax=ax,
    )
    nx.draw_networkx_labels(g, pos, labels=labels, font_size=8, ax=ax)

    correct_str = "CORRECT" if result.get("answer_correct") else "WRONG"
    plt.title(
        f"{result.get('sample_id', '?')} | {result.get('output_type', '?')} | {correct_str} "
        f"(depth={result.get('score_depth', 0):.1f})",
        fontsize=12,
    )
    plt.axis("off")
    plt.tight_layout()

    out_dir = Path(figures_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"dag_{result['sample_id']}_api.png"
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path
