from dataclasses import asdict

from reasoning_eval.common.schema import Rule
from reasoning_eval.dataset.dag_builder import build_reasoning_graph
from reasoning_eval.scorer.depth_scorer import score_depth
from reasoning_eval.scorer.mapper import map_step_to_node
from reasoning_eval.scorer.verifier import RuleBasedVerifier


def _score_steps(steps):
    graph = asdict(build_reasoning_graph(["A"], [Rule("A", "B", "A -> B"), Rule("B", "C", "B -> C")], "C"))
    mappings = [map_step_to_node(step, graph) for step in steps]
    verifier = RuleBasedVerifier(graph)
    history = set()
    previous = None
    verifications = []
    for mapping in mappings:
        result = verifier.verify(mapping, previous, history)
        verifications.append(result)
        if result.valid and mapping.matched_node_id:
            history.add(mapping.matched_node_id)
            if not result.redundant:
                previous = mapping.matched_node_id
    return score_depth(graph, mappings, verifications)[0]


def test_depth_scorer_full_path_high_score():
    assert _score_steps(["A 成立", "由 A -> B 推出 B", "由 B -> C 推出 C"]) == 100


def test_depth_scorer_redundant_steps_do_not_inflate_score():
    normal = _score_steps(["A 成立", "由 A -> B 推出 B", "由 B -> C 推出 C"])
    redundant = _score_steps(["A 成立", "A 是已知事实", "A 真的成立", "由 A -> B 推出 B", "由 B -> C 推出 C"])
    assert redundant == normal

