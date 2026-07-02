"""Step-to-DAG-node mapper — single LLM call per sample.

Architecture:
    All model steps + all gold nodes → 1 LLM call → all mappings.

Zero Jaccard. Zero regex. Zero code heuristics.
Only math cross-check stays.
"""

from __future__ import annotations

import json
import re
from typing import Optional

from reasoning_eval.common.schema import MappingResult

_PROMPT_SYSTEM = """You are a math reasoning alignment specialist. Match each model
step to the gold step it logically corresponds to.

RULES:
- Match based on LOGICAL CONTENT (what is computed, from what inputs, to what output).
- NOT wording similarity.
- "48/2=24" == "half of 48 is 24"
- If a step has no match, use node_id "none".

OUTPUT: valid JSON only. No commentary."""

_PROMPT_USER = """## MODEL STEPS
{steps}

## GOLD STEPS
{candidates}

Output JSON: {{"mappings":[{{"step_index":0,"node_id":"<id or none>","reason":"..."}},...]}}"""


def _format_candidates(nodes, max_chars=5000):
    parts, n = [], 0
    for nd in nodes:
        line = f"[{nd.get('id','?')}] {nd.get('proposition','')[:200]}\n"
        if n + len(line) > max_chars: break
        parts.append(line); n += len(line)
    return "".join(parts)


def _extract_json(text):
    text = text.strip()
    try: return json.loads(text)
    except: pass
    m = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if m:
        try: return json.loads(m.group(1))
        except: pass
    m = re.search(r'\{.*"mappings".*\}', text, re.DOTALL)
    if m:
        try: return json.loads(m.group(0))
        except: pass
    raise ValueError(f"No parseable JSON: {text[:200]}")


def _extract_computed_value(text: str) -> str | None:
    """Extract the final computed numeric result from a step or gold node.

    GSM8K gold nodes: "80/100 * 10 = <<80/100*10=8>>8 more" → "8"
    Model steps: "80% of 10 is 8, so 10 + 8 = 18." → "18" (last = value)
    """
    # Try <<calc=result>> format first (GSM8K gold nodes)
    m = re.findall(r'<<[^>]*?=\s*([0-9]+(?:\.[0-9]+)?)\s*>>', text)
    if m:
        return m[-1]  # Last <<>> result is the node's output
    # Try last = <integer> pattern
    m = re.findall(r'=\s*([0-9]+(?:\.[0-9]+)?)', text)
    if m:
        return m[-1]  # Last = value is the computed result
    return None


def _math_value_match(steps: list[str],
                      nodes: list[dict]) -> dict[int, tuple[str, str]]:
    """Match steps to gold nodes where the computed numeric value is unique.

    Each gold reasoning step computes a specific value (e.g. "10+8=18" → 18).
    If a model step computes the same value AND that value is unique across
    all gold nodes, it's a deterministic match — no LLM needed.

    Returns {step_index: (node_id, matched_value)}.
    """
    # Build {value: [node_ids]} from gold nodes
    val_to_nodes: dict[str, list[str]] = {}
    for n in nodes:
        prop = n.get('proposition', '') or ''
        val = _extract_computed_value(prop)
        if val is not None:
            val_to_nodes.setdefault(val, []).append(n.get('id', ''))

    # Only uniquely-valued nodes can be matched deterministically
    unique_vals = {v: ids[0] for v, ids in val_to_nodes.items()
                   if len(ids) == 1}

    matches: dict[int, tuple[str, str]] = {}
    for i, step in enumerate(steps):
        val = _extract_computed_value(step)
        if val is not None and val in unique_vals:
            matches[i] = (unique_vals[val], val)

    return matches


def _build_proposition_map(nodes: list[dict]) -> dict[str, list[str]]:
    """Build {proposition: [node_id, ...]} for single-letter props."""
    import re
    pm = {}
    for n in nodes:
        prop = (n.get("proposition") or "").strip()
        if re.fullmatch(r'[A-Z]', prop):
            pm.setdefault(prop, []).append(n.get("id", ""))
    return pm


def _proposition_fast_match(step: str, prop_map: dict[str, list[str]],
                             nodes: list[dict]) -> str | None:
    """If step mentions a unique single-letter proposition, return its node_id."""
    import re
    # Remove English stopwords (case-insensitive)
    stopwords = '(Step|Final|Answer|Path|The|In|Find|Let|If|So|We|For|To|Is|On|At|Be|It|Or|By)'
    cleaned = re.sub(rf'\b{stopwords}\b', ' ', step, flags=re.IGNORECASE)
    # Match standalone uppercase letters — use non-word-boundary aware matching
    # to handle Chinese text where \b doesn't apply
    props = set(re.findall(r'(?:^|[^A-Za-z])([A-Z])(?:[^A-Za-z]|$)', cleaned))
    for p in props:
        if p in prop_map and len(prop_map[p]) == 1:
            return prop_map[p][0]
    return None


def map_all_steps(steps: list[str], graph: dict, *, client=None) -> list[MappingResult]:
    """Match all model steps to gold nodes in ONE LLM call."""
    nodes = graph.get("nodes", [])
    if not nodes:
        return [MappingResult(s, None, 0.0, "empty graph") for s in steps]

    # Fast path: rule-text matches (works without LLM client)
    edge_matches: dict[int, MappingResult] = {}
    for i, step in enumerate(steps):
        for edge in graph.get("edges", []):
            if not isinstance(edge, dict): continue
            rt = (edge.get("rule_text") or "").strip()
            if rt and rt.replace(" ","").lower() in step.replace(" ","").lower():
                edge_matches[i] = MappingResult(
                    step, edge.get("target",""), 0.95, f"rule→{edge['target']}")
                break

    # Fast path 2: single-proposition match (deduction: "A 成立" → node A)
    prop_map = _build_proposition_map(nodes)
    for i, step in enumerate(steps):
        if i in edge_matches: continue
        nid = _proposition_fast_match(step, prop_map, nodes)
        if nid:
            edge_matches[i] = MappingResult(step, nid, 0.80, f"prop→{nid}")

    # Fast path 3: math value uniqueness match
    #   "80% of 10 is 8, so 10+8=18" → last computed value = "18"
    #   If "18" is uniquely computed by gold node 6 → deterministic match
    math_matches = _math_value_match(steps, nodes)
    for i, step in enumerate(steps):
        if i in edge_matches:
            continue
        if i in math_matches:
            nid, val = math_matches[i]
            edge_matches[i] = MappingResult(
                step, nid, 0.85,
                f"math: computes {val}→{nid}")

    # No LLM client → return fast-path matches only
    if client is None or client.demo_mode:
        return [edge_matches.get(i, MappingResult(s, None, 0.0, "no mapper"))
                for i, s in enumerate(steps)]

    # Remaining steps need LLM
    remaining = [(i, s) for i, s in enumerate(steps) if i not in edge_matches]
    if not remaining:
        return [edge_matches.get(i, MappingResult(s, None, 0.0, "?"))
                for i, s in enumerate(steps)]

    step_list = "\n".join(f"[{i}] {s[:300]}" for i, s in remaining)
    candidates = _format_candidates(nodes)
    prompt = _PROMPT_USER.format(steps=step_list, candidates=candidates)

    try:
        resp = client.generate(prompt=prompt, system=_PROMPT_SYSTEM,
                               n=1, temperature=0.1, max_tokens=2048)[0]
        data = _extract_json(resp)
    except Exception:
        data = {"mappings": []}

    valid_ids = {n.get("id", "") for n in nodes}
    for m in data.get("mappings", []):
        idx = m.get("step_index", -1)
        nid = m.get("node_id", "none")
        reason = m.get("reason", "")
        if nid != "none" and nid in valid_ids:
            edge_matches[idx] = MappingResult(
                steps[idx] if idx < len(steps) else "",
                nid, 0.75, f"LLM: {reason}")

    return [edge_matches.get(i, MappingResult(steps[i], None, 0.0, "no mapping"))
            for i in range(len(steps))]


# Legacy wrapper for evaluator compatibility
def map_step_to_node(step_text: str, graph: dict, *, client=None) -> MappingResult:
    results = map_all_steps([step_text], graph, client=client)
    return results[0]
