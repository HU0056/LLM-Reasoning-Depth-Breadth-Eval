from dataclasses import asdict

from reasoning_eval.common.schema import Rule
from reasoning_eval.dataset.dag_builder import build_reasoning_graph
from reasoning_eval.scorer.mapper import map_step_to_node


def test_mapper_maps_rule_step_to_target_node():
    graph = asdict(build_reasoning_graph(["A"], [Rule("A", "B", "A -> B")], "B"))
    mapping = map_step_to_node("由 A -> B 推出 B", graph)
    assert mapping.matched_node_id == "B"
    assert "rule→" in mapping.reason

