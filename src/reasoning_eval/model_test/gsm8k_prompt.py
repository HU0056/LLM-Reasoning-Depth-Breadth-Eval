from __future__ import annotations


def build_gsm8k_prompt(sample: dict) -> str:
    question = sample["question"]
    return (
        "请逐步推理求解以下数学问题。每一步计算写在 << >> 中。\n"
        "请严格按照格式输出：\n"
        f"问题：{question}\n\n"
        "输出格式：\n"
        "Step 1: ...\n"
        "Step 2: ...\n"
        "Final Answer: #### <数值>\n"
        "注意：最终答案必须单独一行，以 #### 开头后跟具体数值。"
    )
