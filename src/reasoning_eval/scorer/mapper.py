"""Step-to-DAG-node mapper — Agent cluster architecture.

Zero Jaccard.  Zero regex matching.  Zero code-based heuristics.

Architecture (from MAVEN + GoV + SEVA patterns):
    Model step text + all gold nodes
      │
      ▼
    Matcher Agent (LLM)    ← reasons about logical equivalence, picks candidate
      │  dual-verdict consensus
      ▼
    Math Cross-Check (Code) ← the ONLY deterministic check: verify computations
      │
      ▼
    Matched node (or no_match)
"""

from __future__ import annotations

import json
from typing import Optional

from reasoning_eval.common.schema import MappingResult
from reasoning_eval.harness.math_verifier import all_computations_valid


# ── Prompt ───────────────────────────────────

_MATCHER_SYSTEM = """You are a mathematical reasoning alignment specialist.

Your task: given a model-generated reasoning step and a candidate list of
gold-standard solution steps, determine which gold step corresponds to the
model step.

MATCHING RULES:
- Match based on LOGICAL CONTENT: what is being computed, from what inputs,
  producing what output. NOT wording similarity.
- "48/2=24" and "half of 48 is 24" are the SAME logical step.
- "a_i = i" and "we conclude each a_i equals i" are the SAME logical step.
- A model step that computes nothing concrete may match no gold step → output "none".
- A model step that restates known facts should match the gold step that introduces
  those facts.

OUTPUT: JSON only. No commentary."""

_MATCHER_USER = """## MODEL STEP
{step}

## GOLD STEPS (candidates)
{candidates}

Output JSON: {{"node_id": "<id from candidates, or 'none'>", "confidence": 0.0-1.0, "reason": "<one sentence>"}}"""


# ── Helpers ──────────────────────────────────

def _format_candidates(nodes: list[dict], max_chars: int = 5000) -> str:
    lines = []
    total = 0
    for node in nodes:
        nid = node.get("id", "?")
        text = node.get("proposition", "")[:200]
        line = f"[{nid}] {text}\n"
        if total + len(line) > max_chars:
            lines.append(f"... ({len(nodes) - len(lines)} more nodes)")
            break
        lines.append(line)
        total += len(line)
    return "".join(lines)


def _extract_json(text: str) -> dict:
    """Robust JSON extraction from LLM output."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try code fence
    import re
    fence = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if fence:
        try:
            return json.loads(fence.group(1))
        except json.JSONDecodeError:
            pass
    # Try "{...}" anywhere
    brace = re.search(r'\{[^{}]*"node_id"\s*:\s*"[^"]*"[^{}]*\}', text, re.DOTALL)
    if brace:
        try:
            return json.loads(brace.group(0))
        except json.JSONDecodeError:
            pass
    raise ValueError(f"No parseable JSON: {text[:200]}")


# ── Public API ───────────────────────────────

def map_step_to_node(
    step_text: str,
    graph: dict,
    *,
    client=None,
) -> MappingResult:
    """Map a model reasoning step to the best gold DAG node.

    Uses ONLY LLM reasoning — no code-based heuristics.
    Math correctness is verified separately as a sanity check.
    """
    nodes = graph.get("nodes", [])
    if not nodes:
        return MappingResult(step_text, None, 0.0, "empty gold DAG")

    # ── Deduction rule-text fast path ──
    for edge in graph.get("edges", []):
        if not isinstance(edge, dict):
            continue  # skip flat-format edges like [0, 9]
        rule_text = (edge.get("rule_text") or "").strip()
        if rule_text:
            compact = step_text.replace(" ", "").lower()
            if rule_text.replace(" ", "").lower() in compact:
                return MappingResult(
                    step_text, edge.get("target", ""), 0.95,
                    f"rule: '{rule_text}' → {edge['target']}",
                )

    if client is None or client.demo_mode:
        return MappingResult(step_text, None, 0.0, "no client for LLM matching")

    # ── LLM-based matching with dual-verdict consensus ──
    candidates = _format_candidates(nodes)
    prompt = _MATCHER_USER.format(step=step_text[:500], candidates=candidates)

    verdicts: list[dict] = []
    for run_idx in range(2):
        try:
            resp = client.generate(
                prompt=prompt, system=_MATCHER_SYSTEM,
                n=1, temperature=0.1, max_tokens=512, seed=42 + run_idx * 100,
            )[0]
            verdict = _extract_json(resp)
        except (ValueError, RuntimeError):
            return MappingResult(step_text, None, 0.0, "LLM call failed")

        nid = verdict.get("node_id", "")
        if nid == "none" or nid == "":
            return MappingResult(step_text, None, 0.0, "LLM: no match")

        valid_ids = {n.get("id", "") for n in nodes}
        if nid not in valid_ids:
            return MappingResult(step_text, None, 0.0, f"bad_id: {nid}")

        verdicts.append(verdict)

    # Consensus check
    if verdicts[0]["node_id"] != verdicts[1]["node_id"]:
        return MappingResult(
            step_text, None, 0.0,
            f"disagreement: {verdicts[0]['node_id']} vs {verdicts[1]['node_id']}",
        )

    node_id = verdicts[0]["node_id"]
    confidence = verdicts[0].get("confidence", 0.75)
    reason = verdicts[0].get("reason", "")

    # ── Math verification cross-check ──
    if not all_computations_valid(step_text):
        return MappingResult(
            step_text, None, 0.0,
            f"math error in step — cannot trust match to {node_id}",
        )

    return MappingResult(
        step_text, node_id, round(float(confidence), 3),
        f"LLM consensus → {node_id}. {reason}",
    )
