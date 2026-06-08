from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from datasets import load_dataset


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_gsm8k_dataset():
    dataset_names = ["openai/gsm8k", "gsm8k"]

    for dataset_name in dataset_names:
        try:
            return load_dataset(dataset_name, "main")
        except Exception:
            os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com/")
            try:
                return load_dataset(dataset_name, "main")
            except Exception:
                continue

    raise RuntimeError("failed to load GSM8K from Hugging Face and mirror")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_directory(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
