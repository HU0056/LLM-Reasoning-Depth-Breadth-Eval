from reasoning_eval.common.schema import VerificationResult
from reasoning_eval.scorer.consistency_scorer import score_consistency


def test_consistency_scorer_reports_first_error_step():
    verifications = [
        VerificationResult(True, False, False, False, "ok"),
        VerificationResult(False, False, True, False, "jump"),
    ]
    score, detail = score_consistency(verifications, answer_correct=True)
    assert detail["first_error_step"] == 2
    assert detail["missing_premise_flag"]
    # New multi-dimensional model: 2 steps, 1 missing_premise, 0 contradictions
    # dim1=1.0, dim2=0.25, dim3=0.5, dim4=1.0
    # raw = 0.30*1.0 + 0.35*0.25 + 0.20*0.5 + 0.15*1.0 = 0.6375
    # score = 63.75
    assert score == 63.75

