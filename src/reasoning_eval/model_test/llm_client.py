from __future__ import annotations

import os
import time
from typing import Any

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Optional OpenAI import — graceful fallback when not installed
# ---------------------------------------------------------------------------
try:
    from openai import (
        APIError,
        APITimeoutError,
        OpenAI,
        RateLimitError,
    )
    _OPENAI_AVAILABLE = True
except ImportError:  # pragma: no cover
    _OPENAI_AVAILABLE = False


# ---------------------------------------------------------------------------
# Retry / backoff
# ---------------------------------------------------------------------------

_RETRYABLE = (RateLimitError, APITimeoutError) if _OPENAI_AVAILABLE else ()
_MAX_RETRIES = 3
_BACKOFF_FACTOR = 2.0  # seconds multiplier per retry


def _warn_missing_dependency() -> None:
    import sys

    print(
        "[llm_client] 'openai' package not found. "
        "Install it with: pip install openai\n"
        "Falling back to demo mode — real API calls disabled.",
        file=sys.stderr,
    )


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class LLMClient:
    """OpenAI-compatible LLM client.

    Reads configuration from environment variables (via .env)::

        API_KEY=sk-...
        BASE_URL=https://api.openai.com/v1
        MODEL_NAME=gpt-4o-mini

    When ``API_KEY`` is missing or ``openai`` is not installed the client
    runs in *demo mode* — calls raise ``RuntimeError`` with a clear message,
    so the caller can fall back to loading hand-written demo outputs.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model_name: str | None = None,
    ) -> None:
        load_dotenv()

        self.api_key = api_key or os.getenv("API_KEY")
        self.base_url = base_url or os.getenv("BASE_URL", "https://api.openai.com/v1")
        self.model_name = model_name or os.getenv("MODEL_NAME", "gpt-4o-mini")

        self._client: Any = None
        self._demo_mode = False

        if not _OPENAI_AVAILABLE:
            _warn_missing_dependency()
            self._demo_mode = True
        elif not self.api_key or self.api_key == "your_api_key_here":
            self._demo_mode = True
        else:
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def demo_mode(self) -> bool:
        """True when no real API key is configured."""
        return self._demo_mode

    def generate(
        self,
        prompt: str,
        system: str | None = None,
        n: int = 1,
        temperature: float = 0.7,
        max_tokens: int = 262144,
        seed: int | None = None,
    ) -> list[str]:
        """Send a chat completion request, return *n* response texts.

        Raises ``RuntimeError`` in demo mode.
        """
        if self._demo_mode:
            raise RuntimeError(
                "LLMClient is in demo mode — no API key configured. "
                "Set API_KEY in .env or use data/model_outputs/demo_model_outputs.jsonl."
            )

        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        kwargs: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "n": n,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if seed is not None:
            kwargs["seed"] = seed

        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                response = self._client.chat.completions.create(**kwargs)
                return [choice.message.content or "" for choice in response.choices]
            except _RETRYABLE:
                if attempt == _MAX_RETRIES:
                    raise
                time.sleep(_BACKOFF_FACTOR * attempt)
            except APIError:
                if attempt == _MAX_RETRIES:
                    raise
                time.sleep(_BACKOFF_FACTOR * attempt)

        # Should not be reachable — safety net
        raise RuntimeError("LLMClient.generate exceeded retries (unexpected).")

    def generate_cot(
        self,
        prompt: str,
        system: str | None = None,
        n: int = 1,
        temperature: float = 0.7,
    ) -> list[str]:
        """Generate with built-in CoT system prompt when *system* is None."""
        if system is None:
            system = (
                "You are a helpful assistant that solves problems step by step. "
                "Always show your reasoning in numbered steps."
            )
        return self.generate(
            prompt=prompt,
            system=system,
            n=n,
            temperature=temperature,
        )
