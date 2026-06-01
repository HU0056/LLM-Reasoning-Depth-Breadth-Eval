from __future__ import annotations

from reasoning_eval.common.schema import Rule


def parse_rules(raw_rules: list[dict]) -> list[Rule]:
    rules: list[Rule] = []
    for raw in raw_rules:
        source = raw.get("source")
        target = raw.get("target")
        text = raw.get("text")
        if not source or not target or not text:
            raise ValueError(f"Rule must include source, target, and text: {raw}")
        rules.append(Rule(source=source, target=target, text=text, distractor=bool(raw.get("distractor", False))))
    return rules


def parse_facts(raw_facts: list[str]) -> list[str]:
    if not isinstance(raw_facts, list) or not all(isinstance(item, str) and item for item in raw_facts):
        raise ValueError("facts must be a non-empty list of proposition strings")
    return raw_facts

