from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from openai import OpenAI


class LLMClient:
    def __init__(self) -> None:
        load_dotenv()
        self.api_key = os.getenv("API_KEY")
        self.base_url = os.getenv("BASE_URL", "https://api.openai.com/v1")
        self.model_name = os.getenv("MODEL_NAME", "gpt-4o-mini")

    def _single_call(self, prompt: str, temperature: float) -> str:
        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        response = client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "你是一个逻辑推理助手，请严格按照 Step 1, Step 2, ... 格式逐步推理。"},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=2048,
        )
        content = response.choices[0].message.content
        return content if content else ""

    def generate_cot(self, prompt: str, n: int = 1, temperature: float = 0.7) -> list[str]:
        if not self.api_key or self.api_key == "your_api_key_here":
            raise RuntimeError("API_KEY is not configured. Demo mode uses data/model_outputs/demo_model_outputs.jsonl.")

        if n == 1:
            return [self._single_call(prompt, temperature)]

        with ThreadPoolExecutor(max_workers=min(n, 8)) as executor:
            futures = [executor.submit(self._single_call, prompt, temperature) for _ in range(n)]
            return [f.result() for f in as_completed(futures)]

