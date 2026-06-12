from __future__ import annotations

import time
from typing import Optional

from reasoning_eval.common.io_utils import read_jsonl, write_jsonl
from reasoning_eval.model_test.llm_client import LLMClient
from reasoning_eval.model_test.prompt_builder import build_prompt, get_system_prompt


def generate_one(
    sample: dict,
    client: LLMClient,
    model_name: str | None = None,
    n: int = 1,
    temperature: float = 0.7,
) -> dict:
    """Generate model output for a single benchmark sample.

    Returns a dict ready for the scoring pipeline::

        {
            "sample_id": "...",
            "model_name": "...",
            "output_type": "cot",       # or "sc_N" for n>1
            "response": "Step 1: ...\\nFinal Answer: ...",
        }
    """
    prompt = build_prompt(sample, num_paths=1)
    system = get_system_prompt(sample)

    responses = client.generate(
        prompt=prompt,
        system=system,
        n=n,
        temperature=temperature,
    )

    if n == 1:
        response_text = responses[0]
    else:
        # Combine multiple sampled paths into one text block so
        # step_splitter can separate them via Path N markers.
        parts: list[str] = []
        for idx, resp in enumerate(responses, start=1):
            parts.append(f"Path {idx}:\n{resp}\n")
        response_text = "\n".join(parts)

    output_type = "cot" if n == 1 else f"sc_{n}"

    return {
        "sample_id": sample["id"],
        "model_name": model_name or client.model_name,
        "output_type": output_type,
        "response": response_text,
    }


def generate_benchmark_outputs(
    benchmark_path: str,
    output_path: str,
    client: LLMClient | None = None,
    model_name: str | None = None,
    n: int = 1,
    temperature: float = 0.7,
    limit: int | None = None,
    sample_ids: list[str] | None = None,
    delay: float = 0.0,
) -> list[dict]:
    """Generate model outputs for every sample in a benchmark JSONL file.

    Parameters
    ----------
    benchmark_path:
        Path to a benchmark JSONL (e.g. ``data/processed/demo_benchmark.jsonl``).
    output_path:
        Where to write the generated outputs (JSONL).
    client:
        Pre-configured ``LLMClient``.  Created automatically when ``None``.
    model_name:
        Override the model name recorded in output metadata.
    n:
        Number of responses per sample (1 = CoT, >1 = Self-Consistency).
    temperature:
        Sampling temperature.
    limit:
        Only process the first *limit* samples (useful for smoke tests).
    sample_ids:
        Only process samples whose ``id`` is in this list.
    delay:
        Seconds to sleep between API calls (rate-limit friendliness).

    Returns
    -------
    list[dict]
        The output payloads written to *output_path*.
    """
    if client is None:
        client = LLMClient()

    samples = read_jsonl(benchmark_path)

    if sample_ids is not None:
        id_set = set(sample_ids)
        samples = [s for s in samples if s.get("id") in id_set]

    if limit is not None:
        samples = samples[:limit]

    outputs: list[dict] = []
    total = len(samples)

    for idx, sample in enumerate(samples, start=1):
        sid = sample.get("id", "?")
        print(f"[{idx}/{total}] generating for {sid} ...", flush=True)

        result = generate_one(
            sample,
            client=client,
            model_name=model_name,
            n=n,
            temperature=temperature,
        )
        outputs.append(result)

        if delay > 0 and idx < total:
            time.sleep(delay)

    write_jsonl(output_path, outputs)
    print(f"Saved {len(outputs)} outputs to {output_path}", flush=True)
    return outputs
