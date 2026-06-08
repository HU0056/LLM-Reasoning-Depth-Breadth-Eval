from dataclasses import asdict

from reasoning_eval.common.schema import Rule
from reasoning_eval.dataset.dag_builder import build_reasoning_graph
from reasoning_eval.scorer.breadth_scorer import score_breadth


def _branch_graph():
    rules = [Rule("A", "B", "A -> B"), Rule("A", "C", "A -> C"), Rule("B", "D", "B -> D"), Rule("C", "D", "C -> D")]
    return asdict(build_reasoning_graph(["A"], rules, "D"))


def test_breadth_scorer_broad_high_score():
    score, detail = score_breadth(_branch_graph(), [["A 成立", "A -> B 所以 B"], ["A 成立", "A -> C 所以 C"]], ["A"])
    assert score == 100
    assert detail["branch_coverage"] == 1


def test_breadth_scorer_narrow_repeated_low_score():
    score, detail = score_breadth(_branch_graph(), [["A 成立", "A -> B 所以 B"], ["A 成立", "A -> B 所以 B"]], ["A"])
    assert score == 50
    assert detail["branch_coverage"] == 0.5

