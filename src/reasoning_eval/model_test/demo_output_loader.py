from __future__ import annotations

from reasoning_eval.common.io_utils import read_jsonl


def load_demo_outputs(path: str) -> list[dict]:
    outputs = read_jsonl(path)
    required = {"sample_id", "model_name", "output_type", "response"}
    for row in outputs:
        missing = required - row.keys()
        if missing:
            raise ValueError(f"Model output missing fields {sorted(missing)}: {row}")
    return outputs

