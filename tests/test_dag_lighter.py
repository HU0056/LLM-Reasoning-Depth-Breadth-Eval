from reasoning_eval.common.schema import MappingResult, VerificationResult
from reasoning_eval.scorer.dag_lighter import light_dag


def test_dag_lighter_marks_lit_jump_wrong():
    graph = {
        "nodes": [{"id": "A"}, {"id": "B"}, {"id": "C"}],
        "edges": [{"source": "A", "target": "B"}, {"source": "B", "target": "C"}],
    }
    mappings = [
        MappingResult("A", "A", 0.8, "fact"),
        MappingResult("C", "C", 0.8, "jump"),
        MappingResult("?", None, 0.0, "none"),
    ]
    verifications = [
        VerificationResult(True, False, False, False, "ok"),
        VerificationResult(False, False, True, False, "jump"),
        VerificationResult(False, False, False, False, "wrong"),
    ]
    lighted = light_dag(graph, mappings, verifications)
    assert lighted["nodes"]["A"] == "lit"
    assert lighted["nodes"]["C"] == "jump"
    assert lighted["steps"][2]["status"] == "wrong"

