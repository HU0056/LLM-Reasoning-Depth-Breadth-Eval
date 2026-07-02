"""Math computation verifier — the ONLY deterministic check that stays.

Checks every = expression by comparing computed vs declared values.
Pure Python, 100% reliable, no LLM involved.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass


_NUM_ONLY = re.compile(r"^[\d+\-*/().%\s]+$")
_EQ_RE = re.compile(r"(?P<lhs>[\d\s+\-*/().%^]+?)\s*=\s*(?P<rhs>[\d\s+\-*/().%^]+)")
_GSM8K_RE = re.compile(r"<<(.+?)=(.+?)>>")


def _safe_eval(expr: str) -> float | None:
    expr = expr.strip().replace("^", "**").replace(" ", "")
    if not expr or not _NUM_ONLY.match(expr):
        return None
    try:
        return float(eval(expr, {"__builtins__": {}}, {}))
    except Exception:
        return None


@dataclass
class ComputationResult:
    expression: str
    lhs: float | None = None
    rhs: float | None = None
    matches: bool = False
    error: str = ""


def verify_computations(text: str) -> list[ComputationResult]:
    """Find ALL = or <<...>> expressions in text and verify them."""
    results: list[ComputationResult] = []

    # Standard equations
    for m in _EQ_RE.finditer(text):
        lhs = _safe_eval(m.group("lhs"))
        rhs = _safe_eval(m.group("rhs"))
        if lhs is None or rhs is None:
            results.append(ComputationResult(m.group(0), error="unparseable"))
            continue
        matches = math.isclose(lhs, rhs, rel_tol=1e-9)
        results.append(ComputationResult(m.group(0), lhs, rhs, matches,
            "" if matches else f"computed {lhs} ≠ declared {rhs}"))

    # GSM8K <<expr=result>>
    for m in _GSM8K_RE.finditer(text):
        expr_val = _safe_eval(m.group(1))
        result_val = _safe_eval(m.group(2))
        if expr_val is None or result_val is None:
            continue
        matches = math.isclose(expr_val, result_val, rel_tol=1e-9)
        results.append(ComputationResult(
            m.group(0), expr_val, result_val, matches,
            "" if matches else f"computed {expr_val} ≠ declared {result_val}",
        ))

    return results


def all_computations_valid(text: str) -> bool:
    """True iff every computation in the text is correct."""
    results = verify_computations(text)
    if not results:
        return True  # no computations → vacuously correct
    return all(r.matches for r in results)
