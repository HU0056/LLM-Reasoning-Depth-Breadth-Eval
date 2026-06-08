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
    assert score == 50

