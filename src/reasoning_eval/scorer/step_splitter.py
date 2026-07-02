"""Step splitter v3 — line-based + compact + 15-step cap."""
from __future__ import annotations
import re
from reasoning_eval.common.schema import SplitResult

FINAL_RE = re.compile(r"(?:Final Answer|最终答案)\s*[:：]\s*(.+)", re.IGNORECASE)
BOXED_RE = re.compile(r"\\boxed\{([^}]+)\}", re.IGNORECASE)
STEP_PREFIX_RE = re.compile(r"^(?:Step\s*\d+|第[一二三四五六七八九十0-9]+步|步骤\s*\d+)\s*[:：]?\s*", re.IGNORECASE)
PATH_RE = re.compile(r"^\s*(?:Path|路径)\s*\d+\s*[:：]?\s*$", re.IGNORECASE)
MAX_STEPS = 15

def _extract_final_answer(text):
    m = FINAL_RE.search(text) or BOXED_RE.search(text)
    return m.group(1).strip() if m else None

def _compact(steps):
    out, buf = [], ""
    for s in steps:
        if len(s) < 15 and not re.search(r'[.!?。！？]\s*$', s): buf = (buf + " " + s).strip() if buf else s
        else:
            if buf:
                if out: out[-1] = out[-1] + " " + buf
                else: out.append(buf)
                buf = ""
            out.append(s)
    if buf:
        if out: out[-1] = out[-1] + " " + buf
        else: out.append(buf)
    return out

def _split_single_path(text):
    steps, final_answer = [], None
    for line in text.splitlines():
        line = line.strip()
        if not line: continue
        ans = _extract_final_answer(line)
        if ans is not None: final_answer = ans
        if FINAL_RE.search(line) and len(FINAL_RE.sub("", line).strip()) == 0: continue
        cleaned = STEP_PREFIX_RE.sub("", line).strip()
        if cleaned: steps.append(cleaned)
    steps = _compact(steps)
    if len(steps) > MAX_STEPS:
        steps = steps[:3] + steps[-(MAX_STEPS-3):]
    if final_answer is None: final_answer = _extract_final_answer(text)
    return SplitResult(steps=steps, final_answer=final_answer)

def split_steps(response):
    path_chunks, current, saw_path = [], [], False
    for line in response.splitlines():
        if PATH_RE.match(line):
            saw_path = True
            if current: path_chunks.append(current); current = []
            continue
        current.append(line)
    if current: path_chunks.append(current)
    if saw_path:
        sp = [_split_single_path("\n".join(c)).steps for c in path_chunks if c]
        return SplitResult(steps=[s for p in sp for s in p], final_answer=_split_single_path(response).final_answer, sampled_paths=sp)
    r = _split_single_path(response)
    r.sampled_paths = [r.steps]
    return r
