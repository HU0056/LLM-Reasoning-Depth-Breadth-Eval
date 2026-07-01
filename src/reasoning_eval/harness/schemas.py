"""DSL schemas for structured reasoning declaration and verification.

Edge model (per user requirement):
    Edge = (premise_set: list[str], conclusion: str, justification: Justification)
    Each edge must be independently derivable from its premises using only the
    stated justification (mathematical axiom, theorem, or algebraic rule).

Atomicity exemption:
    If a theorem is NOT the theorem the problem asks to prove, AND the theorem
    is equivalent to multi-step atomic reasoning, it may be treated as atomic
    (i.e., a single edge with that theorem as justification — not a jump).

Borrows from ReasoningFlow (8+14 types), VeryTrace (DSL+deterministic+LLM),
VPRM (rule-based step checks), GoV (DAG verification).
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ═══════════════════════════════════════════════
# Node types
# ═══════════════════════════════════════════════

class NodeType(str, Enum):
    GIVEN = "given"
    OPERATION = "operation"
    FACT = "fact"
    CONCLUSION = "conclusion"
    VERIFICATION = "verification"


# ═══════════════════════════════════════════════
# Edge types
# ═══════════════════════════════════════════════

class EdgeType(str, Enum):
    INFER = "infer"
    EXECUTE = "execute"
    SUPPORT = "support"
    CORRECT = "correct"
    REFERENCE = "reference"


# ═══════════════════════════════════════════════
# Justification — the mathematical basis of an edge
# ═══════════════════════════════════════════════

class JustificationType(str, Enum):
    """Taxonomy of reasoning justifications."""
    ARITHMETIC = "arithmetic"          # + - × ÷ on known values
    ALGEBRA = "algebra"               # substitution, equation solving, factoring
    THEOREM = "theorem"               # application of a named mathematical theorem
    AXIOM = "axiom"                   # fundamental axiom (reflexivity, transitivity, etc.)
    DEFINITION = "definition"         # unfolding or applying a definition
    SIMPLIFICATION = "simplification"  # reducing an expression to canonical form
    SUBSTITUTION = "substitution"      # replacing equals with equals
    EQUIVALENCE = "equivalence"        # logical equivalence transformation
    INDUCTION = "induction"            # mathematical induction step


_JUSTIFICATION_ATOMICITY: dict[str, tuple[bool, float]] = {
    # (is_atomic_by_default, base_difficulty)
    "arithmetic":     (True,  1.0),
    "algebra":        (True,  1.5),
    "theorem":        (False, 3.0),   # may be treated as atomic per exemption rule
    "axiom":          (True,  1.0),
    "definition":     (True,  1.0),
    "simplification": (True,  1.0),
    "substitution":   (True,  1.0),
    "equivalence":    (True,  2.0),
    "induction":      (False, 5.0),
}


class Justification(BaseModel):
    """The mathematical warrant that licenses an inference step."""
    type: JustificationType = Field(
        description="Category of mathematical reasoning used"
    )
    reference: str = Field(
        default="",
        description="Specific rule/theorem name, e.g. '分配律', '勾股定理', '等式性质'",
    )
    is_atomic: bool = Field(
        default=True,
        description="Whether this is a single-step inference (True) or a "
                    "compound inference that should be decomposed (False)",
    )
    exemption: bool = Field(
        default=False,
        description="True if this theorem is exempt from the atomicity requirement "
                    "(not the theorem being proved, equivalent to multi-step atomic reasoning)",
    )

    @property
    def base_difficulty(self) -> float:
        """Base difficulty score for this justification type."""
        return _JUSTIFICATION_ATOMICITY.get(self.type.value, (True, 1.0))[1]

    @classmethod
    def arithmetic(cls, ref: str = "") -> "Justification":
        return cls(type=JustificationType.ARITHMETIC, reference=ref or "arithmetic")

    @classmethod
    def algebra(cls, ref: str = "") -> "Justification":
        return cls(type=JustificationType.ALGEBRA, reference=ref or "algebraic manipulation")

    @classmethod
    def theorem(cls, ref: str, exempt: bool = False) -> "Justification":
        return cls(
            type=JustificationType.THEOREM, reference=ref,
            is_atomic=exempt, exemption=exempt,
        )


# ═══════════════════════════════════════════════
# DAG elements
# ═══════════════════════════════════════════════

class DagNode(BaseModel):
    """One node in the reasoning DAG."""
    id: str
    type: NodeType
    text: str
    expression: Optional[str] = None
    variables_defined: list[str] = Field(default_factory=list)
    variables_used: list[str] = Field(default_factory=list)


class DagEdge(BaseModel):
    """One directed inference edge with mathematical justification.

    Edge = ({premises}, conclusion, justification)
    The conclusion is derivable from the premise set using ONLY the justification.
    """
    premises: list[str] = Field(
        default_factory=list,
        description="IDs of premise nodes (multi-source supported)",
    )
    target: str = Field(description="Conclusion node ID")
    edge_type: EdgeType = Field(default=EdgeType.INFER)
    justification: Justification = Field(
        default_factory=Justification.arithmetic,
        description="Mathematical warrant for this inference",
    )
    rationale: str = Field(default="")

    # Backward compatibility
    @property
    def source(self) -> str:
        return self.premises[0] if self.premises else ""


class StepDeclaration(BaseModel):
    """One step as declared by the Structurer Agent."""
    index: int
    text: str
    depends_on: list[int] = Field(default_factory=list)
    node_type: NodeType = Field(default=NodeType.OPERATION)
    expression: Optional[str] = None
    # NEW: justification for each dependency
    justifications: list["Justification"] = Field(
        default_factory=list,
        description="One justification per depends_on entry (same order)",
    )


class StructuredSolution(BaseModel):
    """Complete structured declaration of a solution's reasoning DAG."""
    steps: list[StepDeclaration]
    final_answer: str


# ═══════════════════════════════════════════════
# Verification types
# ═══════════════════════════════════════════════

class ComputationCheck(BaseModel):
    step_index: int
    expression: str
    computed_value: Optional[float] = None
    declared_value: Optional[float] = None
    matches: bool = False
    error: Optional[str] = None


class JustificationCheck(BaseModel):
    """Checks whether a declared justification is valid for the operation."""
    step_index: int
    dep_index: int
    justification_type: str
    is_plausible: bool = True
    error: str = ""


class UseDefCheck(BaseModel):
    step_index: int
    variable: str
    defined_in_step: Optional[int] = None
    declared_dep: bool = False
    consistent: bool = True


class ContributionCheck(BaseModel):
    """Checks whether a node contributes to reaching the conclusion."""
    node_id: str
    on_critical_path: bool = True
    reachable_from_start: bool = True
    goal_reachable: bool = True
    contributes: bool = True


class TopologyCheck(BaseModel):
    has_cycles: bool = False
    cycle_nodes: list[str] = Field(default_factory=list)
    dangling_nodes: list[str] = Field(default_factory=list)
    unreachable_conclusion: bool = False
    non_contributing_nodes: list[str] = Field(default_factory=list)
    is_valid_dag: bool = True


class TypeConsistencyCheck(BaseModel):
    inconsistencies: list[str] = Field(default_factory=list)
    is_consistent: bool = True


class VerificationReport(BaseModel):
    computation: list[ComputationCheck] = Field(default_factory=list)
    justification: list[JustificationCheck] = Field(default_factory=list)
    use_def: list[UseDefCheck] = Field(default_factory=list)
    topology: Optional[TopologyCheck] = None
    type_consistency: Optional[TypeConsistencyCheck] = None
    contribution: list[ContributionCheck] = Field(default_factory=list)
    all_passed: bool = False
    summary: str = ""


# ═══════════════════════════════════════════════
# Fabrication detection (mapper anti-fabrication)
# ═══════════════════════════════════════════════

class FabricationGate(BaseModel):
    """Controls whether a mapper match is genuine or fabricated."""
    min_confidence: float = Field(
        default=0.35,
        description="Mapping confidence below this → treated as fabricated",
    )
    require_bi_directional: bool = Field(
        default=False,
        description="If True, model step text AND gold node text must match each other",
    )
    require_structural: bool = Field(
        default=True,
        description="If True, the mapped node must be reachable from the current "
                    "verification state",
    )
    strict_mode: bool = Field(
        default=False,
        description="If True, ANY unmatched step is treated as fabrication and "
                    "penalized. Default False (allows unmapped steps without penalty).",
    )


# ═══════════════════════════════════════════════
# Audit types
# ═══════════════════════════════════════════════

class AuditVerdict(BaseModel):
    edge_source: str
    edge_target: str
    valid: bool
    confidence: float = Field(ge=0.0, le=1.0)
    error_category: str = "none"
    justification_ok: bool = True
    suggestion: str = ""


class AuditReport(BaseModel):
    verdicts: list[AuditVerdict] = Field(default_factory=list)
    valid_edge_count: int = 0
    invalid_edge_count: int = 0
    missing_edges: list[tuple[int, int]] = Field(default_factory=list)
    overall_quality: float = Field(default=1.0, ge=0.0, le=1.0)


# ═══════════════════════════════════════════════
# Cross-validation
# ═══════════════════════════════════════════════

class CrossValidationResult(BaseModel):
    llm_edges: int = 0
    code_edges: int = 0
    agreed_edges: list[tuple[int, int]] = Field(default_factory=list)
    llm_only_edges: list[tuple[int, int]] = Field(default_factory=list)
    code_only_edges: list[tuple[int, int]] = Field(default_factory=list)
    resolved_edges: list[DagEdge] = Field(default_factory=list)
    unresolved_conflicts: list[str] = Field(default_factory=list)


# ═══════════════════════════════════════════════
# Loop engineering
# ═══════════════════════════════════════════════

class HarnessError(RuntimeError):
    """Fatal harness error — pipeline cannot produce a valid DAG."""
    def __init__(self, message: str, phase: str = "", detail: dict | None = None):
        super().__init__(message)
        self.phase = phase
        self.detail = detail or {}


class HarnessExhaustedError(HarnessError):
    """Raised when repair loop exhausts max rounds without reaching validity."""


class HarnessParseError(HarnessError):
    """Raised when JSON extraction fails repeatedly — LLM output is unusable."""


class HarnessVerificationError(HarnessError):
    """Raised when deterministic verification finds fatal structural errors."""


class LoopState(BaseModel):
    """Tracks repair loop health."""
    round_number: int = 0
    max_rounds: int = 2
    consecutive_json_failures: int = 0
    max_json_failures: int = 3
    consecutive_verification_failures: int = 0
    max_verification_failures: int = 2
    fatal: bool = False
    fatal_reason: str = ""

    def record_json_failure(self) -> None:
        self.consecutive_json_failures += 1
        if self.consecutive_json_failures >= self.max_json_failures:
            self.fatal = True
            self.fatal_reason = (
                f"JSON extraction failed {self.consecutive_json_failures} times "
                f"consecutively — LLM output format is unreliable"
            )

    def record_verification_failure(self) -> None:
        self.consecutive_verification_failures += 1

    def record_success(self) -> None:
        self.consecutive_json_failures = 0

    def check_round_exhaustion(self) -> None:
        if self.round_number >= self.max_rounds:
            self.fatal = True
            self.fatal_reason = (
                f"Repair loop exhausted: {self.round_number} rounds, "
                f"verification still failing after {self.consecutive_verification_failures} "
                f"consecutive failures"
            )

    def raise_if_fatal(self) -> None:
        if not self.fatal:
            return
        if "JSON" in self.fatal_reason:
            raise HarnessParseError(self.fatal_reason, phase="repair")
        if "exhausted" in self.fatal_reason:
            raise HarnessExhaustedError(self.fatal_reason, phase="repair")
        raise HarnessError(self.fatal_reason, phase="repair")


# ═══════════════════════════════════════════════
# Gold DAG
# ═══════════════════════════════════════════════

class GoldDag(BaseModel):
    nodes: list[DagNode] = Field(default_factory=list)
    edges: list[DagEdge] = Field(default_factory=list)
    num_steps: int = 0
    num_edges: int = 0
    verification_report: Optional[VerificationReport] = None
    audit_report: Optional[AuditReport] = None
    cross_validation: Optional[CrossValidationResult] = None

    def to_legacy_graph(self) -> dict:
        """Convert to legacy dict format for existing scorers."""
        return {
            "nodes": [
                {"id": n.id, "proposition": n.text, "type": n.type.value}
                for n in self.nodes
            ],
            "edges": [
                {
                    "source": e.premises[0] if e.premises else "",
                    "target": e.target,
                    "rule_text": e.rationale,
                    "status": "normal",
                    "justification": e.justification.model_dump(),
                    "difficulty": e.justification.base_difficulty,
                }
                for e in self.edges
            ],
            "goal_node": self.nodes[-1].id if self.nodes else "0",
            "start_nodes": [
                n.id for n in self.nodes if n.type == NodeType.GIVEN
            ] or ([self.nodes[0].id] if self.nodes else ["0"]),
        }
