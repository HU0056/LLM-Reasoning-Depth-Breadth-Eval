from __future__ import annotations

import os

from dotenv import load_dotenv


class LLMClient:
    def __init__(self) -> None:
        load_dotenv()
        self.api_key = os.getenv("API_KEY")
        self.base_url = os.getenv("BASE_URL", "https://api.openai.com/v1")
        self.model_name = os.getenv("MODEL_NAME", "gpt-4o-mini")

    def generate_cot(self, prompt: str, n: int = 1, temperature: float = 0.7) -> list[str]:
        if not self.api_key or self.api_key == "your_api_key_here":
            raise RuntimeError("API_KEY is not configured. Demo mode uses data/model_outputs/demo_model_outputs.jsonl.")
        raise NotImplementedError(
            "Real API calls are reserved for future integration. "
            f"Configured endpoint={self.base_url}, model={self.model_name}, n={n}, temperature={temperature}."
        )

