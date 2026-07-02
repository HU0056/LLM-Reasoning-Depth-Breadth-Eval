"""High-level DAG builder — integrates harness pipeline into existing codebase.

Usage::

    from reasoning_eval.model_test.llm_client import LLMClient
    from reasoning_eval.harness.builder import build_gold_dag_from_ground_truth

    client = LLMClient()
    gold_dag = build_gold_dag_from_ground_truth(
        question="Natalia sold clips to 48 of her friends...",
        answer="Natalia sold 48/2 = <<48/2=24>>24 clips in May...",
        client=client,
    )

    # Convert to legacy format for existing scorers:
    legacy_graph = gold_dag.to_legacy_graph()
"""

from __future__ import annotations

from reasoning_eval.harness.pipeline import HarnessPipeline
from reasoning_eval.harness.schemas import GoldDag


def build_gold_dag_from_ground_truth(
    question: str,
    answer: str,
    client,
) -> GoldDag:
    """Build a verified gold DAG from a (question, ground-truth answer) pair.

    This is the main entry point.  It delegates to ``HarnessPipeline``
    which orchestrates the Structurer → Verify → Audit → Cross-Validate →
    Repair → Re-Verify loop.

    Parameters
    ----------
    question : str
        The problem statement (e.g. GSM8K "question" field).
    answer : str
        The ground-truth reference solution (e.g. GSM8K "answer" field).
    client : LLMClient
        OpenAI-compatible client (may be in demo mode).

    Returns
    -------
    GoldDag
        A verified gold DAG with nodes, edges, and full verification trail.
    """
    pipeline = HarnessPipeline(client)
    return pipeline.build(question, answer)


def batch_build(
    samples: list[dict],
    client,
    question_key: str = "question",
    answer_key: str = "answer",
    sample_id_key: str = "gsm8k_id",
) -> list[dict]:
    """Build gold DAGs for a batch of samples.

    Parameters
    ----------
    samples :
        List of dicts with question/answer fields.
    client :
        LLM client instance.
    question_key :
        Dict key for the question text.
    answer_key :
        Dict key for the reference answer text.
    sample_id_key :
        Dict key for sample identifier.

    Returns
    -------
    list[dict]
        Samples augmented with ``"gold_reasoning_graph"`` (legacy format)
        and ``"harness_gold_dag"`` (full GoldDag dict).
    """
    pipeline = HarnessPipeline(client)
    results = []
    for sample in samples:
        question = sample[question_key]
        answer = sample[answer_key]
        gold_dag = pipeline.build(question, answer)
        sample_augmented = dict(sample)
        sample_augmented["gold_reasoning_graph"] = gold_dag.to_legacy_graph()
        sample_augmented["harness_gold_dag"] = gold_dag.model_dump()
        sample_augmented["goal"] = gold_dag.nodes[-1].text if gold_dag.nodes else ""
        if "gold_answer" not in sample_augmented:
            sample_augmented["gold_answer"] = gold_dag.nodes[-1].text if gold_dag.nodes else ""
        results.append(sample_augmented)
    return results
