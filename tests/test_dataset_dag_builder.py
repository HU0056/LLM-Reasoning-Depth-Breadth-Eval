from reasoning_eval.common.schema import Rule
from reasoning_eval.dataset.dag_builder import build_reasoning_graph
from reasoning_eval.dataset.graph_utils import build_nx_graph, is_direct_successor


def test_dag_builder_builds_rule_edges():
    graph = build_reasoning_graph([ "A" ], [Rule("A", "B", "A -> B"), Rule("B", "C", "B -> C")], "C")
    g = build_nx_graph(graph)
    assert set(g.nodes) == {"A", "B", "C"}
    assert is_direct_successor(g, "A", "B")
    assert is_direct_successor(g, "B", "C")

