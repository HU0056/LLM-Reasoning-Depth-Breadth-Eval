"""LLM Agent wrappers with justification support and fatal error propagation.

Key changes from v1:
- Structurer now requires justifications per dependency
- Auditor validates justification correctness (not just edge validity)
- All agents propagate HarnessError on fatal failures
- Loop engineering: no silent fallback to heuristic mode
"""

from __future__ import annotations

import json
import re
from typing import Optional, Type, TypeVar

from pydantic import BaseModel

from reasoning_eval.harness.prompts import (
    AUDITOR_SYSTEM,
    AUDITOR_USER,
    REPAIRER_SYSTEM,
    REPAIRER_USER,
    STRUCTURER_SYSTEM,
    STRUCTURER_USER,
)
from reasoning_eval.harness.schemas import (
    AuditReport,
    AuditVerdict,
    CrossValidationResult,
    HarnessParseError,
    Justification,
    JustificationType,
    NodeType,
    StepDeclaration,
    StructuredSolution,
    VerificationReport,
)

T = TypeVar("T", bound=BaseModel)

# ═══════════════════════════════════════════════
# JSON extraction — hardened
# ═══════════════════════════════════════════════

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _try_parse_json(text: str) -> Optional[str]:
    stripped = text.strip()
    try:
        json.loads(stripped)
        return stripped
    except (json.JSONDecodeError, ValueError):
        return None


def extract_json(text: str) -> str:
    """Hardened JSON extraction.  Raises HarnessParseError on failure."""
    # Strategy 1: whole text is JSON
    result = _try_parse_json(text)
    if result is not None:
        return result

    # Strategy 2: markdown code fence
    m = _JSON_BLOCK_RE.search(text)
    if m:
        inner = m.group(1).strip()
        result = _try_parse_json(inner)
        if result is not None:
            return result
        text = inner

    # Strategy 3: balanced brace/bracket scan (string-aware)
    for start_char, end_char in [("{", "}"), ("[", "]")]:
        start = text.find(start_char)
        if start == -1:
            continue
        depth, in_string, escape = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"' and not in_string:
                in_string = True
                continue
            if ch == '"' and in_string:
                in_string = False
                continue
            if in_string:
                continue
            if ch == start_char:
                depth += 1
            elif ch == end_char:
                depth -= 1
                if depth == 0:
                    extracted = text[start : i + 1]
                    result = _try_parse_json(extracted)
                    if result is not None:
                        return result
                    raise HarnessParseError(
                        f"Found balanced braces but content is not valid JSON. "
                        f"First 300 chars: {extracted[:300]}",
                        phase="json_extract",
                    )

    raise HarnessParseError(
        f"No valid JSON found. First 200 chars: {text[:200]}...",
        phase="json_extract",
    )


def _safe_node_type(raw: str) -> NodeType:
    """Parse node_type with fallback to 'operation'."""
    try:
        return NodeType(raw)
    except ValueError:
        return NodeType.OPERATION

def _parse_justification(j: dict) -> Justification:
    """Parse a justification dict from LLM output, with validation."""
    jtype_str = j.get("type", "arithmetic")
    try:
        jtype = JustificationType(jtype_str)
    except ValueError:
        jtype = JustificationType.ARITHMETIC
    return Justification(
        type=jtype,
        reference=j.get("reference", ""),
        is_atomic=j.get("is_atomic", True),
        exemption=j.get("exemption", False),
    )


def _format_steps(steps: list[StepDeclaration]) -> str:
    lines = []
    for s in steps:
        dep_str = f" depends_on=[{', '.join(map(str, s.depends_on))}]" if s.depends_on else ""
        expr_str = f" | expr={s.expression}" if s.expression else ""
        just_str = ""
        if s.justifications:
            j_parts = [f"{j.type.value}:{j.reference}" for j in s.justifications]
            just_str = f" | just=[{', '.join(j_parts)}]"
        lines.append(
            f"  step_{s.index}: [{s.node_type.value}]{expr_str}{just_str}\n"
            f"    \"{s.text}\"{dep_str}"
        )
    return "\n".join(lines)


# ═══════════════════════════════════════════════
# Agent 1: Structurer
# ═══════════════════════════════════════════════

def run_structurer(question: str, answer: str, client) -> StructuredSolution:
    """Decompose reference solution into atomic steps with justifications."""
    prompt = STRUCTURER_USER.format(question=question, answer=answer)
    responses = client.generate(
        prompt=prompt, system=STRUCTURER_SYSTEM,
        n=1, temperature=0.3, max_tokens=262144,
    )
    text = responses[0]
    json_str = extract_json(text)
    data = json.loads(json_str)

    steps = []
    for s in data["steps"]:
        justs = [
            _parse_justification(j)
            for j in s.get("justifications", [])
        ]
        steps.append(StepDeclaration(
            index=s["index"],
            text=s["text"],
            depends_on=s.get("depends_on", []),
            node_type=_safe_node_type(s.get("node_type", "operation")),
            expression=s.get("expression"),
            justifications=justs,
        ))

    # Validate: justifications count must match depends_on count
    for s in steps:
        if len(s.justifications) != len(s.depends_on):
            raise HarnessParseError(
                f"step_{s.index}: {len(s.depends_on)} dependencies but "
                f"{len(s.justifications)} justifications — counts must match. "
                f"deps={s.depends_on}",
                phase="structurer",
                detail={"step_index": s.index, "deps": s.depends_on},
            )

    return StructuredSolution(
        steps=steps,
        final_answer=str(data.get("final_answer", "")),
    )


# ═══════════════════════════════════════════════
# Agent 2: Auditor
# ═══════════════════════════════════════════════

def run_auditor(
    question: str,
    solution: StructuredSolution,
    client,
) -> AuditReport:
    """Semantically audit edges AND their justifications."""
    steps_text = _format_steps(solution.steps)

    edge_lines = []
    for s in solution.steps:
        for dep in s.depends_on:
            edge_lines.append(f"  step_{dep} → step_{s.index}")
    edges_text = "\n".join(edge_lines) if edge_lines else "(no edges declared)"

    prompt = AUDITOR_USER.format(
        question=question, steps_text=steps_text, edges_text=edges_text,
    )

    responses = client.generate(
        prompt=prompt, system=AUDITOR_SYSTEM,
        n=1, temperature=0.3, max_tokens=262144,
    )
    text = responses[0]
    json_str = extract_json(text)
    data = json.loads(json_str)

    verdicts = [
        AuditVerdict(
            edge_source=v.get("edge_source", ""),
            edge_target=v.get("edge_target", ""),
            valid=v.get("valid", True),
            confidence=v.get("confidence", 0.5),
            error_category=v.get("error_category", "none"),
            justification_ok=v.get("justification_ok", True),
            suggestion=v.get("suggestion", ""),
        )
        for v in data.get("verdicts", [])
    ]
    missing = [
        tuple(e) for e in data.get("missing_edges", [])
    ]
    valid_count = sum(1 for v in verdicts if v.valid)
    invalid_count = len(verdicts) - valid_count

    return AuditReport(
        verdicts=verdicts,
        valid_edge_count=valid_count,
        invalid_edge_count=invalid_count,
        missing_edges=missing,
        overall_quality=data.get("overall_quality", 0.5),
    )


# ═══════════════════════════════════════════════
# Agent 3: Repairer
# ═══════════════════════════════════════════════

def run_repairer(
    question: str,
    solution: StructuredSolution,
    verification: VerificationReport,
    audit: AuditReport,
    cross_validation: CrossValidationResult,
    client,
) -> StructuredSolution:
    """Repair a DAG based on all detected issues."""
    prompt = REPAIRER_USER.format(
        question=question,
        current_solution=json.dumps(solution.model_dump(), ensure_ascii=False, indent=2),
        verification_report=json.dumps(verification.model_dump(), ensure_ascii=False, indent=2),
        audit_report=json.dumps(audit.model_dump(), ensure_ascii=False, indent=2),
        cross_validation=json.dumps(cross_validation.model_dump(), ensure_ascii=False, indent=2),
    )

    responses = client.generate(
        prompt=prompt, system=REPAIRER_SYSTEM,
        n=1, temperature=0.3, max_tokens=262144,
    )
    text = responses[0]
    json_str = extract_json(text)
    data = json.loads(json_str)

    steps = []
    for s in data["steps"]:
        justs = [_parse_justification(j) for j in s.get("justifications", [])]
        steps.append(StepDeclaration(
            index=s["index"],
            text=s["text"],
            depends_on=s.get("depends_on", []),
            node_type=_safe_node_type(s.get("node_type", "operation")),
            expression=s.get("expression"),
            justifications=justs,
        ))

    return StructuredSolution(
        steps=steps,
        final_answer=str(data.get("final_answer", solution.final_answer)),
    )
