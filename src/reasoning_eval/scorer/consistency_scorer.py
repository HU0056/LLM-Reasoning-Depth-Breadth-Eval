from __future__ import annotations

from reasoning_eval.common.schema import VerificationResult


def score_consistency(
    verifications: list[VerificationResult],
    answer_correct: bool,
) -> tuple[float, dict]:
    first_error_step = None
    missing_premise_flag = False
    contradiction_count = 0
    redundant_count = 0
    deductions: list[str] = []

    for idx, verification in enumerate(verifications, start=1):
        if first_error_step is None and (not verification.valid or verification.contradiction):
            first_error_step = idx
        if verification.missing_premise:
            missing_premise_flag = True
        if verification.contradiction:
            contradiction_count += 1
        if verification.redundant:
            redundant_count += 1

    score = 100.0
    if first_error_step is not None:
        score -= 30
        deductions.append(f"first_error_step={first_error_step}: -30")
    if missing_premise_flag:
        score -= 20
        deductions.append("missing_premise: -20")
    if contradiction_count:
        penalty = 20 * contradiction_count
        score -= penalty
        deductions.append(f"contradictions={contradiction_count}: -{penalty}")
    redundancy_ratio = redundant_count / len(verifications) if verifications else 0.0
    if redundancy_ratio:
        penalty = 20 * min(1.0, redundancy_ratio)
        score -= penalty
        deductions.append(f"redundancy_ratio={redundancy_ratio:.3f}: -{penalty:.3f}")
    if not answer_correct:
        score -= 20
        deductions.append("answer_inconsistency: -20")

    return round(max(0.0, score), 3), {
        "first_error_step": first_error_step,
        "missing_premise_flag": missing_premise_flag,
        "contradiction_count": contradiction_count,
        "redundancy_ratio": round(redundancy_ratio, 3),
        "deductions": deductions,
    }

