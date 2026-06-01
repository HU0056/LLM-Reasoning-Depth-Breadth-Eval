from __future__ import annotations

from dataclasses import asdict

from reasoning_eval.common.io_utils import read_jsonl, write_jsonl
from reasoning_eval.dataset.dag_builder import build_reasoning_graph
from reasoning_eval.dataset.rule_parser import parse_facts, parse_rules


def build_demo_dataset(raw_path: str, save_path: str) -> list[dict]:
    processed: list[dict] = []
    for raw in read_jsonl(raw_path):
        facts = parse_facts(raw["facts"])
        rules = parse_rules(raw["rules"])
        graph = build_reasoning_graph(facts, rules, raw["goal"])
        processed.append(
            {
                "id": raw["id"],
                "question": raw["question"],
                "facts": facts,
                "rules": [asdict(rule) for rule in rules],
                "goal": raw["goal"],
                "gold_answer": f"{raw['goal']} 成立",
                "gold_reasoning_graph": asdict(graph),
                "key_branch_nodes": raw.get("key_branch_propositions", []),
                "counterfactual_branches": raw.get("counterfactual_branches", []),
            }
        )
    write_jsonl(save_path, processed)
    return processed

