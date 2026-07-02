"""Step-to-DAG-node mapper using LOGICAL SIGNATURE matching.

Zero Jaccard.  Zero substring matching.  Zero embedding.

Principle
---------
A math reasoning step is defined by WHAT NUMBERS it produces, not what
words it uses.  For non-numeric logic (deduction problems), a step is
defined by WHAT PROPOSITIONS it establishes.

Logical Signature
-----------------
For every piece of text (model step or gold node), we extract:

    numbers_defined   —  numbers this step COMPUTES (RHS of =)
    numbers_used      —  numbers this step CONSUMES (LHS of =)
    equations         —  [(inputs_set, operation, output)]
    propositions      —  {symbol} for non-numeric logic (A, B, C, ...)

Matching Algorithm
------------------
Numeric steps (has equations or numbers):
    Tier 1 (0.95):  same equation triple — inputs + op + output
    Tier 2 (0.85):  same output + overlapping inputs
    Tier 3 (0.70):  same defined numbers (unique to this gold node)
    Tier 4 (0.55):  number footprint overlaps (unique to this gold node)
    Tier 5 (0.45):  partial number overlap

Non-numeric steps (no numbers, no equations):
    Tier P1 (0.90): rule-text match → edge target
    Tier P2 (0.80): proposition symbol match (unique to this gold node)
    Tier P3 (0.60): proposition symbol overlap (shared by multiple nodes)

Anti-fabrication
----------------
- Step numbers ("Step 1:", "Step 2:") are stripped before extraction
- A number mentioned in many gold nodes is AMBIGUOUS — match is weak
- Pure-text steps with no logical content are rejected
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Optional

from reasoning_eval.common.schema import MappingResult


# ── Pre-processing ────────────────────────────

_STEP_PREFIX_RE = re.compile(
    r"^(?:Step\s*\d+|第[一二三四五六七八九十0-9]+步|步骤\s*\d+)\s*[:：]?\s*",
    re.IGNORECASE,
)
_FINAL_ANSWER_RE = re.compile(
    r"^(?:Final Answer|最终答案)\s*[:：]\s*",
    re.IGNORECASE,
)
_NEGATION_RE = re.compile(r"(不成立|不正确|not\s+true|false)", re.IGNORECASE)
_AFFIRM_RE  = re.compile(r"(成立|正确|true)", re.IGNORECASE)


def _strip_step_prefix(text: str) -> str:
    """Remove 'Step N:' and 'Final Answer:' prefixes that leak step numbers."""
    t = _STEP_PREFIX_RE.sub("", text).strip()
    t = _FINAL_ANSWER_RE.sub("", t).strip()
    return t


def _is_negation(text: str) -> bool:
    return bool(_NEGATION_RE.search(text))


# ── Equation extraction ──────────────────────

# Operator detection
_OP_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\+"), "add"),
    (re.compile(r"\-"), "sub"),
    (re.compile(r"\*|×|times|x\b"), "mul"),
    (re.compile(r"/|÷|divided\s+by|over"), "div"),
    (re.compile(r"\%|mod"), "mod"),
    (re.compile(r"\^|\*\*"), "pow"),
]

# Natural-language equation: keyword templates that signal computation
_RESULT_KW = re.compile(
    r"\b(?:gives?\s+(?:us\s+)?|yields?|produces?|results?\s+in|equals?|is\s+equal\s+to|"
    r"comes?\s+out\s+to|we\s+get|we\s+have|that\s+gives?|makes?|would\s+be|"
    r"is\s+the\s+(?:result|outcome|product|sum|quotient|difference))\b",
    re.IGNORECASE,
)
_OP_KW_FRACTION = re.compile(r"\b(?:half|a\s+third|a\s+quarter|twice|double|triple)\b", re.IGNORECASE)
_OP_KW_SUM = re.compile(r"\b(?:sum|total|altogether)\b", re.IGNORECASE)
_OP_KW_DIFF = re.compile(r"\b(?:difference\s+between|less\b(?!\s+than)|minus|take\s+away)\b", re.IGNORECASE)
_OP_KW_MORE = re.compile(r"\bmore\s+than\b", re.IGNORECASE)
_OP_KW_PROD = re.compile(r"\b(?:product\s+of|times|multiplied\s+by)\b", re.IGNORECASE)

# Matches "<<48/2=24>>" (GSM8K calc markers)
_GSM8K_RE = re.compile(r"<<(.+?)=(.+?)>>")
_NUM_RE = re.compile(r"\d+\.?\d*")
_NUM_ONLY = re.compile(r"^[\d+\-*/().%\s]+$")


def _normalize_for_math(text: str) -> str:
    t = text.replace("÷", "/").replace("×", "*")
    t = re.sub(r"\(\s*[A-Za-z][^)]*\)", "", t)
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


def _keyword_to_op(label: str) -> str:
    """Map natural-language template label to operation type."""
    mapping = {
        "eq_kw": "unknown", "fraction": "div", "result_is": "unknown",
        "sum_of": "add", "diff_between": "sub", "subtract": "sub",
        "more_than": "add", "product_of": "mul",
    }
    return mapping.get(label, "unknown")


def _extract_nl_equations(text: str) -> list[tuple[set[float], str, float]]:
    """Extract equations from natural-language phrasing without '=' signs.

    Strategy:
    1. Find ALL numbers in the text with their positions
    2. Find result keywords ("gives", "yields", "equals", ...)
    3. Find operation keywords ("half", "sum", "difference", ...)
    4. Numbers BEFORE the result keyword are inputs; the number AFTER is output
    5. If no result keyword, use operation keyword to infer structure
    """
    results: list[tuple[set[float], str, float]] = []

    # Find all numbers with positions
    num_positions: list[tuple[float, int, int]] = []
    for m in _NUM_RE.finditer(text):
        num_positions.append((float(m.group()), m.start(), m.end()))

    if len(num_positions) < 2:
        return results

    # Find result keyword positions
    result_spans: list[tuple[int, int]] = [
        (m.start(), m.end()) for m in _RESULT_KW.finditer(text)
    ]

    # Find operation keyword positions
    op_signals: list[tuple[int, int, str]] = []
    for pat, op_name in [
        (_OP_KW_FRACTION, "div"), (_OP_KW_SUM, "add"),
        (_OP_KW_DIFF, "sub"), (_OP_KW_MORE, "add"),
        (_OP_KW_PROD, "mul"),
    ]:
        for m in pat.finditer(text):
            op_signals.append((m.start(), m.end(), op_name))

    # Strategy A: result keyword splits numbers into inputs (before) and output (after)
    for r_start, r_end in result_spans:
        inputs: list[float] = []
        outputs: list[float] = []
        for num, ns, ne in num_positions:
            if ne <= r_start:
                inputs.append(num)
            elif ns >= r_end:
                outputs.append(num)
        if inputs and outputs:
            op = _detect_operation(text[max(0, r_start - 40):r_end + 20])
            # Use operation keyword if available near the result keyword
            for os_start, os_end, os_op in op_signals:
                if abs(os_start - r_start) < 30:
                    op = os_op
                    break
            results.append((set(inputs), op, outputs[0]))
            # Only use one result-keyword-based equation per text segment
            break

    # Strategy B: no result keyword — use operation keyword to find
    # the relation among numbers
    if not results:
        for os_start, os_end, op in op_signals:
            nearby_nums = [
                num for num, ns, ne in num_positions
                if abs(ns - os_start) < 60
            ]
            if len(nearby_nums) >= 2:
                # Heuristic: last nearby number is the output
                results.append((set(nearby_nums[:-1]), op, nearby_nums[-1]))
                break

    return results


# ── Proposition extraction (non-numeric logic) ─

_PROP_RE = re.compile(r"\b([A-Z])\b")


def _extract_propositions(text: str) -> set[str]:
    """Extract single-char proposition symbols from deduction text.

    Only captures uppercase letters.  Stopwords are filtered case-sensitively:
    UPPERCASE 'A' is a proposition; lowercase 'a' is an article.
    """
    stopwords = {
        'Step','Final','Answer','Path','The','In','Find','Let','If',
        'So','We','For','To','Is','On','At','Be','It','Or','By',
        'Not','Are','Can','Do','Go','He','Hi','Me','My','No','Of',
        'Oh','Ok','Us','Am','As','An',
    }
    cleaned = re.sub(
        r'\b(' + '|'.join(stopwords) + r')\b',
        ' ', text,
    )
    return set(_PROP_RE.findall(cleaned))


@dataclass
class LogicalSignature:
    """The logical content of a reasoning step, stripped of all wording."""
    numbers_defined: set[float] = field(default_factory=set)
    numbers_used: set[float] = field(default_factory=set)
    equations: list[tuple[set[float], str, float]] = field(
        default_factory=list
    )  # [(inputs_set, operation, output)]
    all_numbers: set[float] = field(default_factory=set)
    propositions: set[str] = field(default_factory=set)
    is_negation: bool = False

    @property
    def has_computation(self) -> bool:
        return len(self.equations) > 0

    @property
    def is_numeric(self) -> bool:
        return len(self.all_numbers) > 0

    @property
    def is_propositional(self) -> bool:
        return not self.is_numeric and len(self.propositions) > 0

    @property
    def is_empty(self) -> bool:
        return not self.is_numeric and not self.is_propositional


def extract_logical_signature(text: str) -> LogicalSignature:
    """Extract logical content from a reasoning step.

    1. Strip step prefixes to avoid leaking step numbers.
    2. For numeric content: find '=' anchors, extract equations.
    3. For non-numeric content: extract proposition symbols.
    """
    clean = _strip_step_prefix(text)
    normalized = _normalize_for_math(clean)

    sig = LogicalSignature()
    sig.all_numbers = set(_extract_numbers(normalized))
    sig.propositions = _extract_propositions(clean)
    sig.is_negation = _is_negation(clean)

    # ── Try natural-language equation templates FIRST ──
    # These catch "48 divided by 2 gives us 24", "half of 48 is 24", etc.
    nl_eqs = _extract_nl_equations(normalized)
    if nl_eqs:
        for inputs, op, output in nl_eqs:
            sig.equations.append((inputs, op, output))
            sig.numbers_used.update(inputs)
            sig.numbers_defined.add(output)
        # Don't return — also try '=' anchors if present

    # ── Find '=' anchors (numeric equations) ──
    eq_positions = [i for i, ch in enumerate(normalized) if ch == "="]
    if not eq_positions and not nl_eqs:
        # No equation at all: numbers in this step are "given" facts
        sig.numbers_defined = sig.all_numbers.copy()
        return sig

    for eq_idx in eq_positions:
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
            lhs_nums = _extract_numbers(lhs_text)
            rhs_nums = _extract_numbers(rhs_text)
            if lhs_nums and rhs_nums:
                inputs = set(lhs_nums)
                output = rhs_nums[-1]
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

    # ── GSM8K "<<expr=result>>" markers ──
    for m in _GSM8K_RE.finditer(text):
        expr_text, result_text = m.group(1), m.group(2)
        expr_val = _safe_eval(expr_text)
        result_val = _safe_eval(result_text)
        if expr_val is None or result_val is None:
            continue
        inputs = set(_extract_numbers(expr_text))
        op = _detect_operation(expr_text)
        sig.equations.append((inputs, op, result_val))
        sig.numbers_used.update(inputs)
        sig.numbers_defined.add(result_val)

    # Non-equation numbers: given facts
    for num in sig.all_numbers:
        if num not in sig.numbers_used and num not in sig.numbers_defined:
            sig.numbers_defined.add(num)

    return sig


# ── Logical matching ─────────────────────────

CONF_EXACT_EQ_MATCH    = 0.95
CONF_SAME_EQ_OUTPUT    = 0.85
CONF_SAME_DEFINED_NUMS = 0.70
CONF_OVERLAPPING_NUMS  = 0.55
CONF_AMBIGUOUS_OVERLAP = 0.45
CONF_PROP_EXACT        = 0.80
CONF_PROP_SHARED       = 0.60

MIN_CONFIDENCE = 0.50


def _equation_match(
    model_sig: LogicalSignature,
    gold_sig: LogicalSignature,
) -> tuple[bool, float, str]:
    if not model_sig.equations or not gold_sig.equations:
        return False, 0.0, "no equations to compare"

    best_conf = 0.0
    best_reason = ""

    for (m_inputs, m_op, m_out) in model_sig.equations:
        for (g_inputs, g_op, g_out) in gold_sig.equations:
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
    defined_overlap = model_sig.numbers_defined & gold_sig.numbers_defined
    if not defined_overlap:
        return False, 0.0, "no numbers defined in common"

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
    model_nums = model_sig.all_numbers
    gold_nums = gold_sig.all_numbers

    if not model_nums or not gold_nums:
        return False, 0.0, "one side has no numbers"

    overlap = model_nums & gold_nums
    if not overlap:
        return False, 0.0, "no number overlap"

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


# ── Proposition matching (non-numeric logic) ──

def _proposition_match(
    model_sig: LogicalSignature,
    gold_sig: LogicalSignature,
    *,
    all_gold_props: dict[str, list[str]],
) -> tuple[bool, float, str]:
    """Match based on proposition symbols (A, B, C, ...).

    Used for deduction problems with no numbers.
    """
    if not model_sig.propositions or not gold_sig.propositions:
        return False, 0.0, "no propositions to compare"

    overlap = model_sig.propositions & gold_sig.propositions
    if not overlap:
        return False, 0.0, "no proposition overlap"

    # Check uniqueness: is this the only gold node containing this symbol?
    unique = True
    for prop in overlap:
        if len(all_gold_props.get(prop, [])) > 1:
            unique = False
            break

    if unique and model_sig.propositions == gold_sig.propositions:
        return True, CONF_PROP_EXACT, (
            f"exact proposition match: {overlap} "
            f"(unique to this gold node)"
        )
    elif unique:
        return True, 0.70, (
            f"proposition overlap: {overlap} "
            f"(unique to this gold node)"
        )
    else:
        return True, CONF_PROP_SHARED, (
            f"proposition overlap: {overlap} (shared across nodes)"
        )


def _precompute_gold_signatures(graph: dict) -> tuple[
    list[LogicalSignature],
    dict[float, list[str]],
    dict[str, list[str]],
]:
    """Precompute logical signatures for all gold DAG nodes."""
    sigs: list[LogicalSignature] = []
    num_to_nodes: dict[float, list[str]] = {}
    prop_to_nodes: dict[str, list[str]] = {}

    for node in graph.get("nodes", []):
        text = node.get("proposition", "")
        sig = extract_logical_signature(text)
        sigs.append(sig)
        nid = node.get("id", "")
        for num in sig.numbers_defined:
            num_to_nodes.setdefault(num, []).append(nid)
        for prop in sig.propositions:
            prop_to_nodes.setdefault(prop, []).append(nid)

    return sigs, num_to_nodes, prop_to_nodes


# ── Public API ───────────────────────────────

# Module-level cache for gold signatures (per graph id)
_graph_cache: dict[int, tuple] = {}
_graph_cache_id = 0


def map_step_to_node(
    step_text: str,
    graph: dict,
    *,
    client=None,
) -> MappingResult:
    """Map a model reasoning step to the best-matching gold DAG node.

    Tiered matching:
    Tier 1 (0.95): Rule-text match → edge target
    Tier 2 (0.95): Exact equation match — same inputs + op + output
    Tier 3 (0.85): Same equation output + overlapping inputs
    Tier 4 (0.70): Same defined numbers (unique to this gold node)
    Tier 5 (0.55): Number/proposition overlap
    Tier 6 (0.75): LLM-assisted match (only if client provided + signatures fail)

    Parameters
    ----------
    step_text : str
        Model reasoning step.
    graph : dict
        Gold DAG.
    client :
        Optional LLMClient for Tier 6 fallback.  If None, LLM tier is skipped.
    """
    model_sig = extract_logical_signature(step_text)
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])

    # ── Tier 1: Rule-text fast path (deduction tasks) ──
    for edge in edges:
        rule_text = (edge.get("rule_text") or "").strip()
        if rule_text:
            compact_step = step_text.replace(" ", "").lower()
            compact_rule = rule_text.replace(" ", "").lower()
            if compact_rule in compact_step:
                return MappingResult(
                    step_text, edge.get("target", ""), 0.95,
                    f"logical: rule text '{rule_text}' → target {edge['target']}",
                )

    # ── Precompute gold signatures ──
    gold_sigs, num_to_nodes, prop_to_nodes = _precompute_gold_signatures(graph)

    # ── Tiers 2-5: Score every gold node by logical signature ──
    best_node: Optional[str] = None
    best_conf: float = 0.0
    best_reason: str = "no logical match"

    for i, (node, gsig) in enumerate(zip(nodes, gold_sigs)):
        node_id = node.get("id", "")
        matched, conf, reason = False, 0.0, ""

        if model_sig.is_numeric and gsig.is_numeric:
            matched, conf, reason = _equation_match(model_sig, gsig)
            if not matched or conf < 0.5:
                matched, conf, reason = _number_defined_match(
                    model_sig, gsig, all_gold_defined=num_to_nodes,
                )
            if not matched or conf < 0.5:
                matched, conf, reason = _number_overlap_match(model_sig, gsig)

        elif model_sig.is_propositional and gsig.is_propositional:
            matched, conf, reason = _proposition_match(
                model_sig, gsig, all_gold_props=prop_to_nodes,
            )

        elif model_sig.is_propositional and gsig.is_numeric:
            matched, conf, reason = _number_overlap_match(model_sig, gsig)

        elif model_sig.is_numeric and gsig.is_propositional:
            matched, conf, reason = _number_overlap_match(model_sig, gsig)

        if conf > best_conf:
            best_node = node_id
            best_conf = conf
            best_reason = reason

    # ── Tier 6: LLM-assisted fallback ──
    if best_conf < MIN_CONFIDENCE and client is not None:
        try:
            from reasoning_eval.scorer.mapper_llm import llm_assisted_match
            llm_result = llm_assisted_match(
                step_text, graph, model_sig, gold_sigs, nodes, client,
            )
            if llm_result is not None and llm_result.matched_node_id:
                return llm_result
        except ImportError:
            pass  # mapper_llm not available

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
