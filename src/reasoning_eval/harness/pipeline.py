"""Harness Pipeline — single-pass LLM-driven DAG construction.

Design principle:
    The LLM is the authority on step decomposition and dependency
    declaration.  Code should not second-guess LLM output — that creates
    an expensive, slow repair loop that adds no value.

Pipeline:
    (question, answer) → Structurer (LLM, 1 call) → GoldDag

Math verification is informational only, not blocking.
"""

from __future__ import annotations

from reasoning_eval.harness.agents import run_structurer
from reasoning_eval.harness.schemas import (
    DagEdge,
    DagNode,
    EdgeType,
    GoldDag,
    HarnessError,
    Justification,
)
from reasoning_eval.harness.verifiers import run_all_checks


class HarnessPipeline:
    """Single-pass DAG construction.  No repair, no audit, no cross-validation."""

    def __init__(self, client) -> None:
        self._client = client

    def build(self, question: str, answer: str) -> GoldDag:
        """Run structurer (1 LLM call), build GoldDag directly."""
        solution = _safe_phase("structurer", lambda: run_structurer(
            question, answer, self._client,
        ))

        # Informational verification only (not blocking)
        verification = _safe_phase("verifier", lambda: run_all_checks(solution))

        return _assemble(solution, verification)


def _safe_phase(name, fn):
    try:
        return fn()
    except Exception as e:
        raise HarnessError(f"Phase '{name}' failed: {e}", phase=name) from e


def _assemble(solution, verification=None):
    nodes = [
        DagNode(id=f"step_{s.index}", type=s.node_type, text=s.text,
                expression=s.expression)
        for s in solution.steps
    ]
    edges = [
        DagEdge(
            premises=[f"step_{dep}"], target=f"step_{s.index}",
            edge_type=EdgeType.INFER,
            justification=(s.justifications[i] if i < len(s.justifications)
                           else Justification.arithmetic()),
            rationale="",
        )
        for s in solution.steps
        for i, dep in enumerate(s.depends_on)
    ]
    return GoldDag(
        nodes=nodes, edges=edges,
        num_steps=len(nodes), num_edges=len(edges),
        verification_report=verification,
    )
