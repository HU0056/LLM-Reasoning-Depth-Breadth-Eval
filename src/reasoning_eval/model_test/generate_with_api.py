from __future__ import annotations

from reasoning_eval.model_test.llm_client import LLMClient
from reasoning_eval.model_test.prompt_builder import build_prompt


def generate_for_sample(sample: dict, n: int = 1) -> list[str]:
    client = LLMClient()
    return client.generate_cot(build_prompt(sample), n=n)

