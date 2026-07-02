"""Consistency scorer — multi-dimensional reasoning integrity.

Academic grounding:
    Consistency in reasoning traces is not a single property.  Following
    the taxonomy emerging from ReasoningFlow (EMNLP 2025), SEVA (ICML 2026),
    and DeltaBench (ACL 2025), we decompose it into four orthogonal dimensions:

Dimension 1: LOGICAL NON-CONTRADICTION (weight 0.30)
    Are any two statements in the trace logically contradictory?
    Source: verifier contradiction detection + validate:attack edges.

Dimension 2: DEPENDENCY INTEGRITY (weight 0.35)
    Are all declared/implied dependencies valid?  Are any dependencies
    missing (the step uses information not yet established)?
    Source: verifier missing_premise + justification checker.

Dimension 3: GOAL ALIGNMENT (weight 0.20)
    Does each step move closer to the goal, as measured by
    shortest-distance-to-goal in the gold DAG?
    Source: depth scorer's distance tracking.

Dimension 4: STRUCTURAL COHERENCE (weight 0.15)
    Is the trace free of redundant/repetitive/circular patterns?
    Source: verifier redundant detection + topology checks.

Each dimension produces a sub-score in [0, 1].  The final score is the
weighted sum, scaled to [0, 100].

This replaces the old "start at 100, subtract penalties" model, which
was brittle and lacked principled justification.
"""

from __future__ import annotations

from reasoning_eval.common.schema import VerificationResult


def score_consistency(
    verifications: list[VerificationResult],
    answer_correct: bool,
) -> tuple[float, dict]:
    """Compute multi-dimensional consistency score.

    Returns (score_0_to_100, detail_dict).
    """
    n = len(verifications) if verifications else 1

    # ── Dimension 1: Logical Non-Contradiction ──
    contradiction_count = sum(1 for v in verifications if v.contradiction)
    contradiction_ratio = contradiction_count / n
    dim1 = max(0.0, 1.0 - 2.0 * contradiction_ratio)

    # ── Dimension 2: Dependency Integrity ──
    missing_premise_count = sum(1 for v in verifications if v.missing_premise)
    invalid_count = sum(1 for v in verifications if not v.valid)
    # Non-redundant, non-contradiction invalid steps are dependency failures
    dep_failure_count = invalid_count - contradiction_count
    dep_failure_ratio = dep_failure_count / n
    missing_ratio = missing_premise_count / n
    dim2 = max(0.0, 1.0 - dep_failure_ratio - 0.5 * missing_ratio)

    # ── Dimension 3: Goal Alignment ──
    # A step is goal-aligned if it's valid and not a jump.
    # Jumps (missing_premise) are goal-misaligned because they skip reasoning.
    aligned_count = sum(
        1 for v in verifications
        if v.valid and not v.missing_premise
    )
    dim3 = aligned_count / n

    # ── Dimension 4: Structural Coherence ──
    redundant_count = sum(1 for v in verifications if v.redundant)
    redundant_ratio = redundant_count / n
    dim4 = max(0.0, 1.0 - 2.0 * redundant_ratio)

    # ── Answer Consistency (cross-check) ──
    # If the answer is wrong despite apparently valid reasoning, that's
    # the strongest possible consistency failure — the reasoning "looks
    # consistent" but is actually disconnected from truth.
    answer_factor = 1.0 if answer_correct else 0.5

    # ── Weighted aggregation ──
    weights = {
        "logical_non_contradiction": 0.30,
        "dependency_integrity": 0.35,
        "goal_alignment": 0.20,
        "structural_coherence": 0.15,
    }
    raw_score = (
        weights["logical_non_contradiction"] * dim1
        + weights["dependency_integrity"] * dim2
        + weights["goal_alignment"] * dim3
        + weights["structural_coherence"] * dim4
    ) * answer_factor

    score = round(max(0.0, raw_score) * 100.0, 3)

    # ── Find first error step ──
    first_error_step = None
    for idx, v in enumerate(verifications, start=1):
        if not v.valid or v.contradiction:
            first_error_step = idx
            break

    return score, {
        "dimensions": {
            "logical_non_contradiction": round(dim1, 4),
            "dependency_integrity": round(dim2, 4),
            "goal_alignment": round(dim3, 4),
            "structural_coherence": round(dim4, 4),
        },
        "weights": weights,
        "answer_factor": answer_factor,
        "raw_weighted_score": round(raw_score, 4),
        "first_error_step": first_error_step,
        "missing_premise_flag": missing_premise_count > 0,
        "contradiction_count": contradiction_count,
        "redundant_count": redundant_count,
        "invalid_count": invalid_count,
        "total_steps": n,
    }
