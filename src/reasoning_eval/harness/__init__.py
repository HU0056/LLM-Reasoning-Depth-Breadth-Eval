"""LLM Harness Agent Framework for Gold DAG Construction.

Architecture (v3 — single-pass):
    (question, answer) → Structurer (1 LLM call) → GoldDag
"""

from reasoning_eval.harness.schemas import (
    DagEdge, DagNode, GoldDag, HarnessError,
    Justification, JustificationType,
    StepDeclaration, StructuredSolution,
    VerificationReport,
)
from reasoning_eval.harness.pipeline import HarnessPipeline
from reasoning_eval.harness.builder import build_gold_dag_from_ground_truth, batch_build

__all__ = [
    "DagEdge", "DagNode", "GoldDag", "HarnessError",
    "Justification", "JustificationType",
    "StepDeclaration", "StructuredSolution",
    "VerificationReport",
    "HarnessPipeline",
    "build_gold_dag_from_ground_truth",
    "batch_build",
]
