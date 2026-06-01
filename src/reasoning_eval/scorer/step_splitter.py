from __future__ import annotations

import re

from reasoning_eval.common.schema import SplitResult


FINAL_RE = re.compile(r"(?:Final Answer|最终答案)\s*[:：]\s*(.+)", re.IGNORECASE)
STEP_PREFIX_RE = re.compile(r"^(?:Step\s*\d+|第[一二三四五六七八九十0-9]+步|步骤\s*\d+)\s*[:：]?\s*", re.IGNORECASE)
PATH_RE = re.compile(r"^\s*(?:Path|路径)\s*\d+\s*[:：]?\s*$", re.IGNORECASE)


def _split_single_path(text: str) -> SplitResult:
    steps: list[str] = []
    final_answer = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        final_match = FINAL_RE.search(line)
        if final_match:
            final_answer = final_match.group(1).strip()
            continue
        cleaned = STEP_PREFIX_RE.sub("", line).strip()
        if cleaned:
            steps.append(cleaned)
    return SplitResult(steps=steps, final_answer=final_answer)


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

