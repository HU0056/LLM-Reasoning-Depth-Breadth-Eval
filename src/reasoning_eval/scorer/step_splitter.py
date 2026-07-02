"""Step splitter — extracts reasoning steps and final answer from model output."""

from __future__ import annotations

import re

from reasoning_eval.common.schema import SplitResult


FINAL_RE = re.compile(r"(?:Final Answer|最终答案)\s*[:：]\s*(.+)", re.IGNORECASE)
BOXED_RE = re.compile(r"\\boxed\{([^}]+)\}", re.IGNORECASE)
STEP_PREFIX_RE = re.compile(
    r"^(?:Step\s*\d+|第[一二三四五六七八九十0-9]+步|步骤\s*\d+)\s*[:：]?\s*",
    re.IGNORECASE,
)
PATH_RE = re.compile(r"^\s*(?:Path|路径)\s*\d+\s*[:：]?\s*$", re.IGNORECASE)


def _extract_final_answer(text: str) -> str | None:
    """Extract the final answer from a line or full text.

    Priority:
    1. Explicit "Final Answer: X" marker
    2. LaTeX \\boxed{X} marker
    3. Last line as-is (fallback)
    """
    m = FINAL_RE.search(text)
    if m:
        return m.group(1).strip()

    m = BOXED_RE.search(text)
    if m:
        return m.group(1).strip()

    return None


def _split_single_path(text: str) -> SplitResult:
    steps: list[str] = []
    final_answer: str | None = None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        ans = _extract_final_answer(line)
        if ans is not None:
            final_answer = ans

        # Lines that are PURE final-answer markers should not become steps
        if FINAL_RE.search(line) and len(FINAL_RE.sub("", line).strip()) == 0:
            continue

        cleaned = STEP_PREFIX_RE.sub("", line).strip()
        if cleaned:
            steps.append(cleaned)

    # If no explicit answer found, try extracting from full text
    if final_answer is None:
        final_answer = _extract_final_answer(text)

    return SplitResult(steps=steps, final_answer=final_answer)


def _is_conclusion_only(text: str) -> bool:
    """Check if a line is ONLY a boxed answer with no reasoning content."""
    stripped = text.strip()
    return bool(BOXED_RE.fullmatch(stripped))


def split_steps(response: str) -> SplitResult:
    path_chunks: list[list[str]] = []
    current: list[str] = []
    saw_path = False
    for line in response.splitlines():
        if PATH_RE.match(line):
            saw_path = True
            if current:
                path_chunks.append(current)
                current = []
            continue
        current.append(line)
    if current:
        path_chunks.append(current)

    if saw_path:
        sampled_paths = [_split_single_path("\n".join(chunk)).steps for chunk in path_chunks if chunk]
        all_steps = [step for path in sampled_paths for step in path]
        final_answer = _split_single_path(response).final_answer
        return SplitResult(steps=all_steps, final_answer=final_answer, sampled_paths=sampled_paths)
    result = _split_single_path(response)
    result.sampled_paths = [result.steps]
    return result

