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


def answer_is_correct(final_answer: str | None, goal: str | None, gold_answer: str) -> bool:
    if not final_answer:
        return False
    compact = final_answer.replace(" ", "")
    gold_compact = gold_answer.replace(" ", "")
    # Math tasks: check gold answer is present
    if gold_compact in compact:
        return True
    # Deduction tasks: check "X成立" / "X不成立"
    if goal:
        goal_yes = f"{goal}成立" in compact
        goal_no = f"{goal}不成立" in compact
        return goal_yes and not goal_no
    return False


def evaluate_one(sample: dict, output: dict) -> EvaluationResult:
    graph = normalize_graph(sample["gold_reasoning_graph"])
    split = split_steps(output["response"])
    verifier = RuleBasedVerifier(graph)
    mappings = []
    verifications = []
    paths = split.sampled_paths if split.sampled_paths else [split.steps]
    for path in paths:
        history: set[str] = set()
        previous_node = None
        for step in path:
            mapping = map_step_to_node(step, graph)
            verification = verifier.verify(mapping, previous_node, history)
            mappings.append(mapping)
            verifications.append(verification)
            if verification.valid and mapping.matched_node_id:
                history.add(mapping.matched_node_id)
                if not verification.redundant:
                    previous_node = mapping.matched_node_id

    correct = answer_is_correct(split.final_answer, sample.get("goal"), sample["gold_answer"])
    depth, depth_detail = score_depth(graph, mappings, verifications)
    breadth, breadth_detail = score_breadth(graph, split.sampled_paths, sample.get("key_branch_nodes", []))
    consistency, consistency_detail = score_consistency(verifications, correct)
    lighted = light_dag(graph, mappings, verifications)

    return EvaluationResult(
        sample_id=sample["id"],
        model_name=output["model_name"],
        output_type=output["output_type"],
        answer_correct=correct,
        score_depth=depth,
        score_breadth=breadth,
        score_consistency=consistency,
        first_error_step=consistency_detail["first_error_step"],
        missing_premise_flag=consistency_detail["missing_premise_flag"],
        branch_coverage=breadth_detail.get("branch_coverage"),
        contradiction_count=consistency_detail["contradiction_count"],
        lighted_graph=lighted,
        detail={
            "steps": split.steps,
            "final_answer": split.final_answer,
            "mappings": [asdict(item) for item in mappings],
            "verifications": [asdict(item) for item in verifications],
            "depth": depth_detail,
            "breadth": breadth_detail,
            "consistency": consistency_detail,
        },
    )


def evaluate_files(benchmark_path: str, outputs_path: str, save_path: str) -> list[EvaluationResult]:
    samples = {sample["id"]: sample for sample in read_jsonl(benchmark_path)}
    outputs = read_jsonl(outputs_path)
    results = []
    for output in outputs:
        sample_id = output["sample_id"]
        if sample_id not in samples:
            raise ValueError(f"Model output references unknown sample_id={sample_id}")
        results.append(evaluate_one(samples[sample_id], output))
    write_jsonl(save_path, results)
    return results
