from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass(frozen=True)
class Rule:
    source: str
    target: str
    text: str
    distractor: bool = False


@dataclass
class ReasoningNode:
    id: str
    proposition: str
    type: str


@dataclass
class ReasoningEdge:
    source: str
    target: str
    rule_text: str
    status: str = "normal"


@dataclass
class ReasoningGraph:
    nodes: list[ReasoningNode]
    edges: list[ReasoningEdge]
    start_nodes: list[str]
    goal_node: str


@dataclass
class SplitResult:
    steps: list[str]
    final_answer: Optional[str]
    sampled_paths: list[list[str]] = field(default_factory=list)


@dataclass
class MappingResult:
    step_text: str
    matched_node_id: Optional[str]
    confidence: float
    reason: str


@dataclass
class VerificationResult:
    valid: bool
    redundant: bool
    missing_premise: bool
    contradiction: bool
    reason: str


@dataclass
class EvaluationResult:
    sample_id: str
    model_name: str
    output_type: str
    answer_correct: bool
    score_depth: float
    score_breadth: Optional[float]
    score_consistency: float
    first_error_step: Optional[int]
    missing_premise_flag: bool
    branch_coverage: Optional[float]
    contradiction_count: int
    lighted_graph: dict[str, Any]
    detail: dict[str, Any]

