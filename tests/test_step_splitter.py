from reasoning_eval.scorer.step_splitter import split_steps


def test_step_splitter_handles_step_and_final_answer():
    result = split_steps("Step 1: A 成立。\nStep 2: 由 A -> B 推出 B。\nFinal Answer: B 成立。")
    assert result.steps == ["A 成立。", "由 A -> B 推出 B。"]
    assert result.final_answer == "B 成立。"

