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

    # No LLM client → return rule-text matches only
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
