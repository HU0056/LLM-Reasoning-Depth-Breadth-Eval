from __future__ import annotations


def build_prompt(sample: dict, num_paths: int = 1) -> str:
    """Build a concise prompt that elicits step-by-step reasoning.

    Detects task_type and dispatches to the appropriate builder.
    Falls back to auto-detection: samples with 'facts' + 'rules' keys
    are deduction tasks; everything else is math.
    """
    task_type = sample.get("task_type")
    if task_type is None:
        # Auto-detect: deduction samples have explicit facts/rules
        if "facts" in sample and "rules" in sample:
            task_type = "deduction"
        else:
            task_type = "math"
    if task_type == "deduction":
        return _build_deduction_prompt(sample, num_paths)
    return _build_math_prompt(sample, num_paths)


# ---------------------------------------------------------------------------
# Math (GSM8K / MATH) prompt
# ---------------------------------------------------------------------------

_MATH_SYSTEM = (
    "You are a careful mathematician. "
    "Solve problems step by step, showing every calculation. "
    "Think before each step — verify your arithmetic."
)

_MATH_FORMAT_HINT = (
    "Output format:\n"
    "Step 1: [first reasoning step with calculation]\n"
    "Step 2: [next step]\n"
    "...\n"
    "Final Answer: [the numerical answer, no extra text]"
)


def _build_math_prompt(sample: dict, num_paths: int) -> str:
    question = sample.get("question", "").strip()

    if num_paths > 1:
        path_instruction = (
            f"Solve this problem in {num_paths} different ways. "
            "Label each approach with 'Path 1:', 'Path 2:', etc. "
            "Within each path, use the step format below.\n\n"
        )
    else:
        path_instruction = ""

    return (
        f"Solve this math problem step by step.\n\n"
        f"Question: {question}\n\n"
        f"{path_instruction}"
        f"{_MATH_FORMAT_HINT}"
    )


def get_math_system_prompt() -> str:
    return _MATH_SYSTEM


# ---------------------------------------------------------------------------
# Deduction / rule-logic prompt
# ---------------------------------------------------------------------------

_DEDUCTION_SYSTEM = (
    "你是一个严谨的逻辑推理助手。请根据已知事实和规则，逐步推导直到得出结论。"
    "每一步只应用一条规则，不要跳步，不要重复已推导的命题。"
)

_DEDUCTION_FORMAT_HINT = (
    "输出格式（严格按此格式）：\n"
    "Step 1: [推理步骤]\n"
    "Step 2: [推理步骤]\n"
    "...\n"
    "Final Answer: [结论 — 例如 'C 成立' 或 'C 不成立']"
)


def _build_deduction_prompt(sample: dict, num_paths: int) -> str:
    facts = sample.get("facts", [])
    rules = sample.get("rules", [])
    goal = sample.get("goal", "")

    facts_block = "\n".join(f"- {f} 成立" for f in facts)
    rules_block = "\n".join(
        f"- {r['text'] if isinstance(r, dict) else r}"
        for r in rules
    )

    if num_paths > 1:
        path_instruction = (
            f"\n请用 {num_paths} 种不同的推理路径来推导，每条路径用 'Path 1:' / 'Path 2:' 等标注。"
            "每条路径内部仍使用 Step 格式。\n"
        )
    else:
        path_instruction = ""

    return (
        f"请根据以下事实和规则逐步推理：\n\n"
        f"事实：\n{facts_block}\n\n"
        f"规则：\n{rules_block}\n\n"
        f"问题：{goal} 是否成立？\n"
        f"{path_instruction}\n"
        f"{_DEDUCTION_FORMAT_HINT}"
    )


def get_deduction_system_prompt() -> str:
    return _DEDUCTION_SYSTEM


def get_system_prompt(sample: dict) -> str:
    """Return the appropriate system prompt for a sample."""
    task_type = sample.get("task_type", "deduction")
    if task_type == "math":
        return _MATH_SYSTEM
    return _DEDUCTION_SYSTEM
