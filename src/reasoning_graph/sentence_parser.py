from __future__ import annotations

import re


SENTENCE_SPLIT_PATTERN = re.compile(r"(?:\.\.\.|[.!?](?=\s|$)|\r\n|[\r\n])+")
FINAL_ANSWER_PATTERN = re.compile(r"^\s*####\s*(.+?)\s*$")
LIST_MARKER_PATTERN = re.compile(r"^(?:\d+|[A-Za-z])[\.)]?$")
INLINE_LIST_MARKER_PATTERN = re.compile(r"^(\d+)\.\s+")
MATH_SPAN_PATTERNS = [
    re.compile(r"\\\[(.*?)(\\\])", re.DOTALL),
    re.compile(r"\\\((.*?)(\\\))", re.DOTALL),
    re.compile(r"\$\$(.*?)(\$\$)", re.DOTALL),
]
MATH_PUNCTUATION_PLACEHOLDERS = {
    ".": "\x00DOT\x00",
    "!": "\x00BANG\x00",
    "?": "\x00QMARK\x00",
}


def split_sentences(text: str) -> list[str]:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        return []

    normalized = _merge_layout_lines(normalized)
    protected = _protect_math_punctuation(normalized)
    parts = SENTENCE_SPLIT_PATTERN.split(protected)
    return [
        _restore_math_punctuation(part.strip())
        for part in parts
        if part and part.strip()
    ]


def _merge_layout_lines(text: str) -> str:
    logical_lines = _merge_display_math_blocks(text.split("\n"))
    logical_lines = _merge_list_marker_lines(logical_lines)
    return "\n".join(logical_lines)


def _merge_display_math_blocks(lines: list[str]) -> list[str]:
    logical_lines: list[str] = []
    math_lines: list[str] = []
    end_token: str | None = None

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue

        if end_token is not None:
            math_lines.append(line)
            if _line_closes_display_math(line, end_token):
                _append_display_math(logical_lines, math_lines)
                math_lines = []
                end_token = None
            continue

        end_token = _display_math_end_token(line)
        if end_token is not None:
            math_lines = [line]
            if line != "$$" and _line_closes_display_math(line, end_token):
                _append_display_math(logical_lines, math_lines)
                math_lines = []
                end_token = None
            continue

        logical_lines.append(line)

    if math_lines:
        _append_display_math(logical_lines, math_lines)
    return logical_lines


def _display_math_end_token(line: str) -> str | None:
    if line.startswith(r"\["):
        return r"\]"
    if line.startswith("$$"):
        return "$$"
    if line.startswith(r"\begin{"):
        match = re.match(r"\\begin\{([^}]+)\}", line)
        if match:
            return rf"\end{{{match.group(1)}}}"
    return None


def _line_closes_display_math(line: str, end_token: str) -> bool:
    if end_token == "$$":
        return line.endswith("$$")
    return end_token in line


def _append_display_math(logical_lines: list[str], math_lines: list[str]) -> None:
    math_text = " ".join(math_lines)
    if logical_lines:
        logical_lines[-1] = f"{logical_lines[-1]} {math_text}".strip()
    else:
        logical_lines.append(math_text)


def _merge_list_marker_lines(lines: list[str]) -> list[str]:
    merged: list[str] = []
    index = 0
    while index < len(lines):
        line = INLINE_LIST_MARKER_PATTERN.sub(r"\1) ", lines[index])
        if (
            LIST_MARKER_PATTERN.match(line)
            and index + 1 < len(lines)
            and not LIST_MARKER_PATTERN.match(lines[index + 1])
        ):
            merged.append(f"{line.rstrip('.)')}) {lines[index + 1]}")
            index += 2
            continue
        merged.append(line)
        index += 1
    return merged


def _protect_math_punctuation(text: str) -> str:
    protected = text
    for pattern in MATH_SPAN_PATTERNS:
        protected = pattern.sub(_protect_math_match, protected)
    return protected


def _protect_math_match(match: re.Match[str]) -> str:
    content = match.group(1)
    closing = match.group(2)
    for punctuation, placeholder in MATH_PUNCTUATION_PLACEHOLDERS.items():
        content = content.replace(punctuation, placeholder)
    return f"{match.group(0)[: match.start(1) - match.start(0)]}{content}{closing}"


def _restore_math_punctuation(text: str) -> str:
    restored = text
    for punctuation, placeholder in MATH_PUNCTUATION_PLACEHOLDERS.items():
        restored = restored.replace(placeholder, punctuation)
    return restored


def extract_final_answer(answer_nodes: list[str]) -> str:
    if not answer_nodes:
        raise ValueError("answer_nodes is empty")

    match = FINAL_ANSWER_PATTERN.match(answer_nodes[-1])
    if not match:
        raise ValueError(f"last answer node must match '#### [ans]', but got: {answer_nodes[-1]}")
    return match.group(1).strip()
