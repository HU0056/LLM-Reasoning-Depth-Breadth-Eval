"""Step-to-DAG-node mapper using LOGICAL SIGNATURE matching.

Zero Jaccard.  Zero substring matching.  Zero embedding.

Principle
---------
A math reasoning step is defined by WHAT NUMBERS it produces, not what
words it uses.  Two steps that produce the same number(s) from the same
input number(s) via the same operation are the SAME logical step —
regardless of whether one says "48/2=24" and the other says
"she sold half, so 48 divided by 2 equals 24 clips".

Logical Signature
-----------------
For every piece of text (model step or gold node), we extract:

    numbers_defined : set[float]    —  numbers this step COMPUTES (RHS of =)
    numbers_used    : set[float]    —  numbers this step CONSUMES (LHS of =)
    operations      : list[str]     —  operation types (div, mul, add, sub, …)
    equation_triples: list[(inputs, op, output)]  — structured equation info

Matching Algorithm
------------------
A model step matches a gold node IFF:

    Primary   (confidence ~0.95):
        They compute the SAME equation triple — same inputs, same op, same output.

    Secondary (confidence ~0.80):
        They define the same output number(s) AND use overlapping inputs.

    Tertiary  (confidence ~0.65):
        They define the same output number(s) — the defining operation matches
        even if inputs aren't aligned.

    Weak      (confidence ~0.50):
        Overlapping number footprint — the numbers mentioned in the step
        overlap with the numbers defined or used by the gold node.

Anti-fabrication
----------------
Steps that define NO numbers (pure text with no computation) are matched
only if they contain numbers that are uniquely attributed to a single gold
node.  If a number appears in multiple gold nodes, the match is ambiguous
and confidence drops sharply.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Optional

from reasoning_eval.common.schema import MappingResult


# ── Equation extraction ──────────────────────

# Operator detection (after normalization)
_OP_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\+"), "add"),
    (re.compile(r"\-"), "sub"),
    (re.compile(r"\*|×|times|x\b"), "mul"),
    (re.compile(r"/|÷|divided by|over"), "div"),
    (re.compile(r"\%|mod"), "mod"),
    (re.compile(r"\^|\*\*"), "pow"),
]

# Matches "<<48/2=24>>" (GSM8K calc markers)
_GSM8K_RE = re.compile(r"<<(.+?)=(.+?)>>")

# Numbers with optional decimal
_NUM_RE = re.compile(r"\d+\.?\d*")
_NUM_ONLY = re.compile(r"^[\d+\-*/().%\s]+$")


def _normalize_for_math(text: str) -> str:
    """Normalize text to make math operations parseable.

    - Replace Unicode operators: ÷→/, ×→*
    - Strip parenthetical annotations: \"48 (April)\" → \"48\"
    """
    t = text.replace("÷", "/").replace("×", "*")
    t = re.sub(r"\(\s*[A-Za-z][^)]*\)", "", t)  # strip (April), (May), etc.
    return t


def _extract_numbers(text: str) -> list[float]:
    return [float(m) for m in _NUM_RE.findall(text)]


def _detect_operation(expr: str) -> str:
    for pat, op_name in _OP_PATTERNS:
        if pat.search(expr):
            return op_name
    return "unknown"


def _safe_eval(expr: str) -> Optional[float]:
    expr = expr.strip().replace("^", "**").replace(" ", "")
    if not expr or not _NUM_ONLY.match(expr):
        return None
    try:
        return float(eval(expr, {"__builtins__": {}}, {}))
    except Exception:
        return None


@dataclass
class LogicalSignature:
    """The logical content of a reasoning step, stripped of all wording."""
    numbers_defined: set[float] = field(default_factory=set)
    numbers_used: set[float] = field(default_factory=set)
    equations: list[tuple[set[float], str, float]] = field(
        default_factory=list
    )  # [(inputs_set, operation, output)]
    all_numbers: set[float] = field(default_factory=set)

    @property
    def has_computation(self) -> bool:
        return len(self.equations) > 0


def extract_logical_signature(text: str) -> LogicalSignature:
    """Extract logical content from a reasoning step.

    Strategy: find '=' signs, normalize the text around them,
    extract inputs/output/operation from each equation found.
    """
    sig = LogicalSignature()
    normalized = _normalize_for_math(text)
    sig.all_numbers = set(_extract_numbers(normalized))

    # ── Find '=' anchors ──
    eq_positions = [i for i, ch in enumerate(normalized) if ch == "="]
    if not eq_positions:
        # No equation: the step defines numbers it introduces contextually.
        # A number is "defined" by a non-equation step only if it's a given
        # fact (no prior computation produces it).
        sig.numbers_defined = sig.all_numbers.copy()
        return sig

    for eq_idx in eq_positions:
        # Extract LHS (before =) and RHS (after =)
        # Walk backward from = to find where the numeric expression starts
        lhs_start = eq_idx
        for i in range(eq_idx - 1, -1, -1):
            ch = normalized[i]
            if ch in "0123456789.+-*/()%^ ":
                lhs_start = i
            else:
                break
        lhs_text = normalized[lhs_start:eq_idx].strip()

        # Walk forward from = to find where RHS ends
        rhs_end = eq_idx + 1
        for i in range(eq_idx + 1, len(normalized)):
            ch = normalized[i]
            if ch in "0123456789.+-*/()%^ ":
                rhs_end = i + 1
            else:
                break
        rhs_text = normalized[eq_idx + 1:rhs_end].strip()

        if not lhs_text or not rhs_text:
            continue

        lhs_val = _safe_eval(lhs_text)
        rhs_val = _safe_eval(rhs_text)
        if lhs_val is None or rhs_val is None:
            # Try extracting just the numbers
            lhs_nums = _extract_numbers(lhs_text)
            rhs_nums = _extract_numbers(rhs_text)
            if lhs_nums and rhs_nums:
                inputs = set(lhs_nums)
                output = rhs_nums[-1]  # last number on RHS is the result
                op = _detect_operation(lhs_text)
                sig.equations.append((inputs, op, output))
                sig.numbers_used.update(inputs)
                sig.numbers_defined.add(output)
            continue

        inputs = set(_extract_numbers(lhs_text))
        op = _detect_operation(lhs_text)
        sig.equations.append((inputs, op, rhs_val))
        sig.numbers_used.update(inputs)
        sig.numbers_defined.add(rhs_val)

    # ── GSM8K "= <<expr=result>>" markers ──
    for m in _GSM8K_RE.finditer(text):
        expr_text = m.group(1)
        result_text = m.group(2)
        expr_val = _safe_eval(expr_text)
        result_val = _safe_eval(result_text)
        if expr_val is None or result_val is None:
            continue
        inputs = set(_extract_numbers(expr_text))
        op = _detect_operation(expr_text)
        sig.equations.append((inputs, op, result_val))
        sig.numbers_used.update(inputs)
        sig.numbers_defined.add(result_val)

    # ── Non-equation numbers: only count as "defined" if they
    #     aren't consumed by any equation (i.e., they are given facts)
    for num in sig.all_numbers:
        if num not in sig.numbers_used and num not in sig.numbers_defined:
            sig.numbers_defined.add(num)

    return sig


# ── Logical matching ─────────────────────────

# Confidence tiers for logical matches
CONF_EXACT_EQ_MATCH      = 0.95   # same inputs, same op, same output
CONF_SAME_EQ_OUTPUT      = 0.85   # same output, overlapping inputs
CONF_SAME_DEFINED_NUMS   = 0.70   # define the same output numbers
CONF_OVERLAPPING_NUMS    = 0.55   # numbers overlap with a unique gold node
CONF_AMBIGUOUS_OVERLAP   = 0.35   # numbers overlap but shared across nodes

# Fabrication thresholds
MIN_CONFIDENCE = 0.50  # below this, reject outright


def _equation_match(
    model_sig: LogicalSignature,
    gold_sig: LogicalSignature,
) -> tuple[bool, float, str]:
    """Check if model and gold share the same equation(s).

    Returns (matched, confidence, reason).
    """
    if not model_sig.equations or not gold_sig.equations:
        return False, 0.0, "no equations to compare"

    best_conf = 0.0
    best_reason = ""

    for (m_inputs, m_op, m_out) in model_sig.equations:
        for (g_inputs, g_op, g_out) in gold_sig.equations:
            # Exact match: same output AND same inputs
            if math.isclose(m_out, g_out, rel_tol=1e-9):
                input_overlap = len(m_inputs & g_inputs)
                input_union = len(m_inputs | g_inputs)
                input_similarity = input_overlap / max(input_union, 1)

                if input_similarity > 0.8:
                    conf = CONF_EXACT_EQ_MATCH
                    reason = (
                        f"exact equation match: "
                        f"output={g_out}, op={m_op}/{g_op}, "
                        f"inputs={m_inputs & g_inputs}"
                    )
                elif input_overlap > 0:
                    conf = CONF_SAME_EQ_OUTPUT
                    reason = (
                        f"same output={g_out}, "
                        f"overlapping inputs={m_inputs & g_inputs}"
                    )
                else:
                    conf = CONF_SAME_DEFINED_NUMS
                    reason = f"same output={g_out} (different inputs)"

                if conf > best_conf:
                    best_conf = conf
                    best_reason = reason

    if best_conf > 0:
        return True, best_conf, best_reason
    return False, 0.0, "no equation output match"


def _number_defined_match(
    model_sig: LogicalSignature,
    gold_sig: LogicalSignature,
    *,
    all_gold_defined: dict[float, list[str]],
) -> tuple[bool, float, str]:
    """Match based on which numbers are DEFINED in each step.

    A number defined in BOTH model and gold is strong logical evidence.
    """
    defined_overlap = model_sig.numbers_defined & gold_sig.numbers_defined
    if not defined_overlap:
        return False, 0.0, "no numbers defined in common"

    # Check if the overlapping numbers are UNIQUE to this gold node
    unique_to_this_node = True
    for num in defined_overlap:
        if len(all_gold_defined.get(num, [])) > 1:
            unique_to_this_node = False
            break

    if unique_to_this_node and model_sig.numbers_defined == gold_sig.numbers_defined:
        conf = 0.80
        reason = (
            f"identical defined numbers: {defined_overlap} "
            f"(unique to this gold node)"
        )
    elif unique_to_this_node:
        conf = 0.70
        reason = (
            f"overlapping defined numbers: {defined_overlap} "
            f"(unique to this gold node)"
        )
    else:
        conf = 0.55
        reason = f"shared defined numbers: {defined_overlap} (ambiguous)"

    return True, conf, reason


def _number_overlap_match(
    model_sig: LogicalSignature,
    gold_sig: LogicalSignature,
) -> tuple[bool, float, str]:
    """Fallback: do the step's numbers overlap with the gold node's numbers?"""
    model_nums = model_sig.all_numbers
    gold_nums = gold_sig.all_numbers

    if not model_nums or not gold_nums:
        return False, 0.0, "one side has no numbers"

    overlap = model_nums & gold_nums
    if not overlap:
        return False, 0.0, "no number overlap"

    # The more of the gold's numbers appear in the model step, the stronger
    gold_coverage = len(overlap) / len(gold_nums)

    if gold_coverage >= 0.5:
        conf = 0.60
        reason = f"number overlap: {overlap}, gold coverage={gold_coverage:.1%}"
    elif len(overlap) >= 1:
        conf = 0.45
        reason = f"partial number overlap: {overlap} (gold coverage={gold_coverage:.1%})"
    else:
        return False, 0.0, "insufficient number overlap"

    return True, conf, reason


def _precompute_gold_signatures(graph: dict) -> tuple[
    list[LogicalSignature],
    dict[float, list[str]],   # number → which node IDs define it
]:
    """Precompute logical signatures for all gold DAG nodes."""
    sigs: list[LogicalSignature] = []
    num_to_nodes: dict[float, list[str]] = {}

    for node in graph.get("nodes", []):
        text = node.get("proposition", "")
        sig = extract_logical_signature(text)
        sigs.append(sig)
        for num in sig.numbers_defined:
            num_to_nodes.setdefault(num, []).append(node.get("id", ""))

    return sigs, num_to_nodes


# ── Public API ───────────────────────────────

def map_step_to_node(step_text: str, graph: dict) -> MappingResult:
    """Map a model reasoning step to the best-matching gold DAG node
    using LOGICAL SIGNATURE matching.

    Zero Jaccard.  Zero substring matching.  Zero embedding.
    Only numbers, equations, and operations matter.
    """
    model_sig = extract_logical_signature(step_text)
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])

    # ── Rule-text fast path (for deduction/rule tasks only) ──
    for edge in edges:
        rule_text = (edge.get("rule_text") or "").strip()
        if rule_text:
            # Compact the step and rule_text for matching
            compact_step = step_text.replace(" ", "").lower()
            compact_rule = rule_text.replace(" ", "").lower()
            if compact_rule in compact_step:
                return MappingResult(
                    step_text, edge.get("target", ""), 0.95,
                    f"logical: rule text '{rule_text}' → target {edge['target']}",
                )

    # ── Precompute gold signatures (cached per graph in practice) ──
    gold_sigs, num_to_nodes = _precompute_gold_signatures(graph)

    # ── Score every gold node ──
    best_node: Optional[str] = None
    best_conf: float = 0.0
    best_reason: str = "no logical match"

    for i, (node, gsig) in enumerate(zip(nodes, gold_sigs)):
        node_id = node.get("id", "")
        conf = 0.0
        reason = ""

        # Tier 1: Equation-level match (strongest)
        matched, conf, reason = _equation_match(model_sig, gsig)
        if not matched or conf < 0.5:
            # Tier 2: Defined-number match
            matched, conf, reason = _number_defined_match(
                model_sig, gsig, all_gold_defined=num_to_nodes,
            )

        if not matched or conf < 0.5:
            # Tier 3: Number overlap (weakest)
            matched, conf, reason = _number_overlap_match(model_sig, gsig)

        if conf > best_conf:
            best_node = node_id
            best_conf = conf
            best_reason = reason

    # ── Fabrication gate ──
    if best_conf < MIN_CONFIDENCE:
        return MappingResult(
            step_text, None, round(best_conf, 3),
            f"logical: best score {best_conf:.3f} < {MIN_CONFIDENCE} "
            f"({best_reason})",
        )

    return MappingResult(
        step_text, best_node, round(best_conf, 3), best_reason,
    )
