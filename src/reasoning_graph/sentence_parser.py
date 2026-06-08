from __future__ import annotations

import re


SENTENCE_SPLIT_PATTERN = re.compile(r"(?:\.\.\.|[.!?](?=\s|$)|\r\n|[\r\n])+")
FINAL_ANSWER_PATTERN = re.compile(r"^\s*####\s*(.+?)\s*$")


def split_sentences(text: str) -> list[str]:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        return []

    parts = SENTENCE_SPLIT_PATTERN.split(normalized)
    return [part.strip() for part in parts if part and part.strip()]


def extract_final_answer(answer_nodes: list[str]) -> str:
    if not answer_nodes:
        raise ValueError("answer_nodes is empty")

    match = FINAL_ANSWER_PATTERN.match(answer_nodes[-1])
    if not match:
        raise ValueError(f"last answer node must match '#### [ans]', but got: {answer_nodes[-1]}")
    return match.group(1).strip()
