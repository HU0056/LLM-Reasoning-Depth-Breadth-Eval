from __future__ import annotations

import re


INFERENCE_WORDS = {"所以", "因此", "推出", "得到", "可得", "therefore", "then", "so"}


def normalize_text(text: str) -> str:
    normalized = text.strip().lower()
    replacements = {
        "=>": "->",
        "→": "->",
        "推出": "->",
        "得到": "->",
        "可以推出": "->",
        "，": " ",
        "。": " ",
        ",": " ",
        ".": " ",
        "：": ":",
    }
    for old, new in replacements.items():
        normalized = normalized.replace(old, new)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def tokenize(text: str) -> set[str]:
    normalized = normalize_text(text)
    return set(re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]+|->", normalized))


def jaccard(a: str, b: str) -> float:
    ta = tokenize(a)
    tb = tokenize(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def contains_inference_word(text: str) -> bool:
    lowered = text.lower()
    return any(word in lowered for word in INFERENCE_WORDS)


def contradicts_known(step: str, propositions: set[str]) -> bool:
    compact = normalize_text(step).replace(" ", "")
    for prop in propositions:
        p = prop.lower()
        if f"{p}不成立" in compact or f"not{p}" in compact:
            return True
    return False

