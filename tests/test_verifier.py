from dataclasses import asdict

from reasoning_eval.common.schema import MappingResult, Rule
from reasoning_eval.dataset.dag_builder import build_reasoning_graph
from reasoning_eval.scorer.verifier import RuleBasedVerifier


def test_verifier_detects_jump_from_a_to_c():
    graph = asdict(build_reasoning_graph(["A"], [Rule("A", "B", "A -> B"), Rule("B", "C", "B -> C")], "C"))
    verifier = RuleBasedVerifier(graph)
    first = verifier.verify(MappingResult("A 成立", "A", 0.8, "fact"), None, set())
    assert first.valid
    jump = verifier.verify(MappingResult("所以 C 成立", "C", 0.8, "contains C"), "A", {"A"})
    assert jump.missing_premise
    # v3: reachable-through-path is valid (model steps coarser than gold granularity)
    assert jump.valid

