"""Evaluation orchestrator — integrates all scorers with the new multi-dimensional model.

Wire-up:
    structurer → DAG construction  (harness, offline)
    mapper → step→node mapping      (with anti-fabrication)
    verifier → step validity        (graph-based)
    depth → difficulty-weighted     (new definition)
    breadth → branch coverage       (unchanged)
    consistency → multi-dimensional (new: 4 dimensions)
    dag_lighter → visualization     (unchanged)
"""

from __future__ import annotations

from dataclasses import asdict

from reasoning_eval.common.io_utils import read_jsonl, write_jsonl
from reasoning_eval.common.schema import EvaluationResult
from reasoning_eval.dataset.graph_utils import normalize_graph
from reasoning_eval.scorer.breadth_scorer import score_breadth
from reasoning_eval.scorer.consistency_scorer import score_consistency
from reasoning_eval.scorer.dag_lighter import light_dag
from reasoning_eval.scorer.depth_scorer import score_depth
from reasoning_eval.scorer.mapper import map_step_to_node
from reasoning_eval.scorer.step_splitter import split_steps
from reasoning_eval.scorer.verifier import RuleBasedVerifier


def answer_is_correct(
    final_answer: str | None,
    goal: str | None,
    gold_answer: str,
) -> bool:
    if not final_answer:
        return False
    compact = final_answer.replace(" ", "")
    gold_compact = gold_answer.replace(" ", "")
    if gold_compact in compact:
        return True
    if goal:
        goal_yes = f"{goal}成立" in compact
        goal_no = f"{goal}不成立" in compact
        return goal_yes and not goal_no
    return False


def evaluate_one(
    sample: dict,
    output: dict,
    *,
    mapper_client=None,
) -> EvaluationResult:
    """Evaluate one model output against its gold DAG.

    Parameters
    ----------
    mapper_client :
        Optional LLMClient for LLM-assisted step-to-node matching (Tier 6 fallback).
        When None, only logical signature matching is used.
    """
    graph = normalize_graph(sample["gold_reasoning_graph"])
    split = split_steps(output["response"])
    verifier = RuleBasedVerifier(graph)

    mappings = []
    verifications = []
    fabrication_count = 0

    paths = split.sampled_paths if split.sampled_paths else [split.steps]
    for path in paths:
        history: set[str] = set()
        previous_node: str | None = None
        for step in path:
            mapping = map_step_to_node(step, graph, client=mapper_client)
            verification = verifier.verify(mapping, previous_node, history)
            mappings.append(mapping)
            verifications.append(verification)

            # Track fabrications
            if mapping.is_fabricated:
                fabrication_count += 1

            if verification.valid and mapping.matched_node_id:
                history.add(mapping.matched_node_id)
                if not verification.redundant:
                    previous_node = mapping.matched_node_id

    correct = answer_is_correct(
        split.final_answer, sample.get("goal"), sample["gold_answer"],
    )

    # New multi-dimensional scorers
    depth, depth_detail = score_depth(graph, mappings, verifications)
    breadth, breadth_detail = score_breadth(
        graph, split.sampled_paths, sample.get("key_branch_nodes", []),
    )
    consistency, consistency_detail = score_consistency(verifications, correct)
    lighted = light_dag(graph, mappings, verifications)

    # Extract depth summary for top-level fields
    depth_summary = {}
    if depth_detail:
        last = depth_detail[-1]
        depth_summary = {
            "D_total": last.get("D_total"),
            "D_remain": last.get("D_remain"),
            "final_progress": last.get("depth_at_step"),
        }

    return EvaluationResult(
        sample_id=sample["id"],
        model_name=output["model_name"],
        output_type=output["output_type"],
        answer_correct=correct,
        score_depth=depth,
        score_breadth=breadth,
        score_consistency=consistency,
        first_error_step=consistency_detail.get("first_error_step"),
        missing_premise_flag=consistency_detail.get("missing_premise_flag", False),
        branch_coverage=breadth_detail.get("branch_coverage"),
        contradiction_count=consistency_detail.get("contradiction_count", 0),
        lighted_graph=lighted,
        consistency_dimensions=consistency_detail,
        depth_detail_summary=depth_summary,
        detail={
            "steps": split.steps,
            "final_answer": split.final_answer,
            "mappings": [asdict(item) for item in mappings],
            "verifications": [asdict(item) for item in verifications],
            "depth": depth_detail,
            "breadth": breadth_detail,
            "consistency": consistency_detail,
            "fabrication_count": fabrication_count,
        },
    )


def evaluate_files(
    benchmark_path: str,
    outputs_path: str,
    save_path: str,
) -> list[EvaluationResult]:
    samples = {s["id"]: s for s in read_jsonl(benchmark_path)}
    outputs = read_jsonl(outputs_path)
    results = []
    for output in outputs:
        sample_id = output["sample_id"]
        if sample_id not in samples:
            raise ValueError(
                f"Model output references unknown sample_id={sample_id}"
            )
        results.append(evaluate_one(samples[sample_id], output))
    write_jsonl(save_path, results)
    return results
