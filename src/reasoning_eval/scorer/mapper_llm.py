"""LLM-assisted step-to-node matching — Tier 4 fallback.

Only invoked when logical signatures fail to match (confidence < 0.50).
The LLM is NOT trusted — it is AUDITED.  Every claim the LLM makes is
cross-validated against deterministic checks before acceptance.

Anti-hallucination constraints
------------------------------
1. Schema gate: output must be a valid node_id from the candidate list, or "none"
2. Cross-validation: LLM-chosen node must share at least one number or proposition with the model step
3. Dual-verdict consensus: two independent calls must agree on the same node_id
4. Audit logging: every LLM verdict records the evidence (or lack thereof)
"""

from __future__ import annotations

import json
import re
from typing import Optional

from reasoning_eval.common.schema import MappingResult

# ── Prompt ───────────────────────────────────

_MATCHER_SYSTEM = """You are a mathematical reasoning alignment specialist.

Your ONLY task: given a model-generated reasoning step and a list of gold
(ground-truth) solution steps, identify which gold step the model step
corresponds to.

CRITICAL RULES:
1. You MUST choose from the provided candidate node IDs ONLY.
2. Match based on MATHEMATICAL CONTENT, not wording similarity.
   - Two steps match if they compute the same thing from the same inputs.
   - "48/2=24" and "half of 48 is 24" are the same step.
   - "a_i = i" and "we conclude a_i equals i" are the same step.
3. If NO gold step matches, output "none".
4. Output ONLY valid JSON. No commentary.
5. DO NOT fabricate node IDs. Use EXACTLY the IDs provided."""

_MATCHER_USER = """## MODEL STEP
{step_text}

## CANDIDATE GOLD STEPS
{candidates}

Output JSON with exactly these fields:
{{"matched_node_id": "<node_id from candidates or 'none'>", "reason": "<one sentence>"}}"""

# ── JSON extraction ───────────────────────────

_JSON_RE = re.compile(r'\{[^{}]*"matched_node_id"\s*:\s*"[^"]*"[^{}]*\}', re.DOTALL)


def _extract_llm_json(text: str) -> dict:
    """Extract the JSON verdict from LLM output. Handles truncation.

    Raises ValueError if no valid JSON found.
    """
    text = text.strip()
    if not text:
        raise ValueError("LLM output is empty")

    # Try naive parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try markdown code fence
    fence = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if fence:
        try:
            return json.loads(fence.group(1))
        except json.JSONDecodeError:
            pass

    # Try regex extraction
    m = _JSON_RE.search(text)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass

    # ── Handle truncated JSON (max_tokens cutoff) ──
    # If we see '{"matched_node_id": "X"', complete the JSON
    trunc = re.search(
        r'\{\s*"matched_node_id"\s*:\s*"([^"]*)"',
        text,
    )
    if trunc:
        node_id = trunc.group(1)
        # Verify it's in the valid set (caller will validate again anyway)
        return {"matched_node_id": node_id, "reason": "(truncated)"}

    raise ValueError(f"LLM output contains no parseable JSON verdict: {text[:200]}")


# ── Candidate formatting ──────────────────────

_MAX_STEP_CHARS = 300   # truncate model step text in prompt
_MAX_PROMPT_CHARS = 6000  # total prompt ceiling


def _format_candidates(
    nodes: list[dict],
    gold_sigs: list,
    max_candidates: int = 30,
) -> str:
    """Format gold nodes as a candidate list for the LLM prompt.

    Each candidate shows: node_id | type | text
    """
    lines = []
    for i, (node, sig) in enumerate(zip(nodes, gold_sigs)):
        nid = node.get("id", f"node_{i}")
        ntype = node.get("type", "unknown")
        text = node.get("proposition", "")
        # Truncate very long texts
        if len(text) > 200:
            text = text[:197] + "..."
        # Add logical signature hints for the LLM
        sig_hint = ""
        if sig.numbers_defined:
            sig_hint += f" [defines: {sorted(sig.numbers_defined)}]"
        if sig.equations:
            eq_strs = [f"({' '.join(map(str, sorted(i)))})_{o}={round(r,3)}" for i, o, r in sig.equations[:2]]
            sig_hint += f" [eqs: {'; '.join(eq_strs)}]"
        lines.append(f"  {nid} [{ntype}]: \"{text}\"{sig_hint}")

    if len(lines) > max_candidates:
        lines = lines[:max_candidates]
        lines.append(f"  ... ({len(nodes) - max_candidates} more nodes omitted)")

    return "\n".join(lines)


# ── Cross-validation ──────────────────────────

def _cross_validate(
    matched_node_id: str,
    model_sig,
    nodes: list[dict],
    gold_sigs: list,
) -> bool:
    """Verify LLM's claim: does the chosen node share logical content with the model step?

    Returns True if the match is plausible.
    """
    # Find the chosen node
    for node, gsig in zip(nodes, gold_sigs):
        if node.get("id") == matched_node_id:
            # Check: any number overlap?
            num_overlap = model_sig.all_numbers & gsig.all_numbers
            # Check: any proposition overlap?
            prop_overlap = model_sig.propositions & gsig.propositions

            if num_overlap or prop_overlap:
                return True
            # If both are empty: allow (both might be pure verbal with context-dependent matching)
            if not model_sig.all_numbers and not gsig.all_numbers and not model_sig.propositions and not gsig.propositions:
                return True

            # LLM picked a node with zero logical overlap — suspicious
            return False

    # Node not found in candidates (shouldn't happen due to schema validation)
    return False


# ── Cache ──────────────────────────────────

# Per-graph cache: avoids redundant LLM calls for the same (step_text, graph) pair
_llm_match_cache: dict[tuple[int, str], Optional[MappingResult]] = {}
_MAX_CACHE_SIZE = 1024


def _cache_key(graph_hash: int, step_text: str) -> tuple[int, str]:
    return (graph_hash, step_text)


# ── Public API ────────────────────────────────

LLM_MATCH_CONFIDENCE = 0.75  # LLM matches are inherently less certain than equation matches


def llm_assisted_match(
    step_text: str,
    graph: dict,
    model_sig,
    gold_sigs: list,
    nodes: list[dict],
    client,
) -> Optional[MappingResult]:
    """Attempt LLM-assisted matching when logical signatures fail.

    Parameters
    ----------
    step_text : str
        The model reasoning step text (original, not stripped).
    graph : dict
        The gold DAG.
    model_sig : LogicalSignature
        Pre-computed logical signature of the model step.
    gold_sigs : list[LogicalSignature]
        Pre-computed signatures for all gold nodes.
    nodes : list[dict]
        Gold DAG nodes.
    client : LLMClient
        Must NOT be in demo mode.

    Returns
    -------
    MappingResult or None
        None if LLM matching fails, rejects, or is unavailable.
    """
    if client.demo_mode:
        return None

    candidates_text = _format_candidates(nodes, gold_sigs)
    step_short = step_text[:_MAX_STEP_CHARS]

    # ── Dual verdict: two independent calls ──
    verdicts: list[dict] = []
    for run_idx in range(2):
        seed = 42 + run_idx * 100
        try:
            responses = client.generate(
                prompt=_MATCHER_USER.format(
                    step_text=step_short,
                    candidates=candidates_text,
                ),
                system=_MATCHER_SYSTEM,
                n=1,
                temperature=0.1,  # very low temperature for consistency
                max_tokens=2048,
                seed=seed,
            )
            text = responses[0]
            verdict = _extract_llm_json(text)
        except (ValueError, RuntimeError, json.JSONDecodeError) as e:
            # LLM call failed or produced unparseable output — LLM tier is unavailable
            import sys
            print(f"[mapper_llm] LLM call {run_idx} failed: {e}", file=sys.stderr)
            return None

        # ── Schema gate: validate node_id ──
        node_id = verdict.get("matched_node_id", "")
        if node_id == "none" or node_id == "":
            return None  # LLM says no match

        valid_ids = {n.get("id", "") for n in nodes}
        if node_id not in valid_ids:
            import sys
            print(
                f"[mapper_llm] LLM hallucinated node_id '{node_id}' "
                f"(not in valid set of {len(valid_ids)} IDs) — rejected",
                file=sys.stderr,
            )
            return None  # Hallucination rejected

        verdicts.append(verdict)

    # ── Dual-verdict consensus ──
    if verdicts[0]["matched_node_id"] != verdicts[1]["matched_node_id"]:
        import sys
        print(
            f"[mapper_llm] Dual verdicts DISAGREE: "
            f"{verdicts[0]['matched_node_id']} vs {verdicts[1]['matched_node_id']} — rejected",
            file=sys.stderr,
        )
        return None

    matched_id = verdicts[0]["matched_node_id"]
    reason = verdicts[0].get("reason", "")

    # ── Cross-validation gate ──
    if not _cross_validate(matched_id, model_sig, nodes, gold_sigs):
        import sys
        print(
            f"[mapper_llm] Cross-validation FAILED: LLM picked {matched_id} "
            f"but no logical content overlap with model step — rejected",
            file=sys.stderr,
        )
        return None

    return MappingResult(
        step_text, matched_id, LLM_MATCH_CONFIDENCE,
        f"LLM-assisted: dual consensus → {matched_id}. {reason}",
    )
