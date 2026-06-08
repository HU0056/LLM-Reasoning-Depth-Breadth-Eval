from __future__ import annotations


def build_prompt(sample: dict) -> str:
    facts = "\n".join(f"{fact} 成立" for fact in sample["facts"])
    rules = "\n".join(rule["text"] for rule in sample["rules"])
    return (
        "请根据以下事实和规则一步一步推理。\n"
        f"事实：\n{facts}\n"
        f"规则：\n{rules}\n"
        f"问题：\n{sample['goal']} 是否成立？\n\n"
        "请按格式输出：\nStep 1: ...\nStep 2: ...\nFinal Answer: ..."
    )

