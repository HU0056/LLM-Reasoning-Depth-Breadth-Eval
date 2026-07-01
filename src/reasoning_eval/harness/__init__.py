"""LLM Harness Agent Framework for Gold DAG Construction.

Design patterns from ReasoningFlow, GoV, VeryTrace, VPRM, and SEVA.

Architecture
------------
::

    Raw (question, ground-truth answer)
      │
      ▼
    Structurer Agent    ← LLM declares steps + dependencies + JUSTIFICATIONS
      │
      ▼
    Deterministic Checks ← computation, justification, use-def, topology,
      │                     contribution, type consistency (Python, 100% reliable)
      ▼
    Auditor Agent       ← LLM verifies edge validity + justification correctness
      │
      ▼
    Cross-Validator     ← reconcile LLM vs code discrepancies
      │
      ▼
    Repair Loop         ← LLM fixes issues; FATAL ERROR on exhaustion (no fallback)
      │
      ▼
    Verified Gold DAG   ← feeds into existing scorer pipeline
"""

from reasoning_eval.harness.schemas import (
    CrossValidationResult,
    DagEdge,
    DagNode,
    FabricationGate,
    GoldDag,
    HarnessError,
    HarnessExhaustedError,
    HarnessParseError,
    HarnessVerificationError,
    Justification,
    JustificationType,
    LoopState,
    StepDeclaration,
    StructuredSolution,
    VerificationReport,
)
from reasoning_eval.harness.pipeline import HarnessPipeline
from reasoning_eval.harness.builder import build_gold_dag_from_ground_truth, batch_build

__all__ = [
    # Core types
    "DagEdge",
    "DagNode",
    "StepDeclaration",
    "StructuredSolution",
    "VerificationReport",
    "CrossValidationResult",
    "GoldDag",
    # Justification
    "Justification",
    "JustificationType",
    # Loop engineering
    "LoopState",
    "HarnessError",
    "HarnessExhaustedError",
    "HarnessParseError",
    "HarnessVerificationError",
    # Anti-fabrication
    "FabricationGate",
    # Pipeline
    "HarnessPipeline",
    "build_gold_dag_from_ground_truth",
    "batch_build",
]
