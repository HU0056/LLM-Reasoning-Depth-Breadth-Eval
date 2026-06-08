from __future__ import annotations

import re
from dataclasses import asdict

from reasoning_eval.common.io_utils import read_jsonl, write_jsonl
from reasoning_eval.common.schema import MappingResult, VerificationResult
from reasoning_eval.common.text_utils import jaccard, normalize_text, tokenize
from reasoning_eval.scorer.step_splitter import split_steps


_FINAL_PREFIX_RE = re.compile(r"^#{1,4}\s*")


def _clean_number(text: str | None) -> str:
    """Normalize a numeric answer string for comparison."""
    if text is None:
        return ""
    cleaned = text.strip().replace(",", "").replace(" ", "")
    cleaned = _FINAL_PREFIX_RE.sub("", cleaned)
    cleaned = cleaned.lstrip("$€£¥")
    cleaned = cleaned.rstrip(".。")
    return cleaned


def gsm8k_answer_is_correct(final_answer: str | None, gold_answer: str) -> bool:
    """Check if the extracted final answer matches the gold answer."""
    if not final_answer:
        return False
    return _clean_number(final_answer) == _clean_number(gold_answer)


def gsm8k_map_step_to_node(step_text: str, nodes: list[str]) -> MappingResult:
    """Map a model reasoning step to the most similar gold DAG node via Jaccard similarity."""
    step_norm = normalize_text(step_text)
    step_tokens = tokenize(step_text)

    best_idx: int | None = None
    best_conf = 0.0
    best_reason = "no sufficiently similar node"

    for idx, node_text in enumerate(nodes):
        node_norm = normalize_text(node_text)
        if not node_norm:
            continue

        # Exact substring match gets high confidence
        if node_norm in step_norm:
            confidence = 0.85
            if confidence > best_conf:
                best_idx = idx
                best_conf = confidence
                best_reason = f"step contains sentence substring"
                continue

        # Jaccard similarity
        score = jaccard(step_text, node_text)
        if score > best_conf:
            best_idx = idx
            best_conf = score
            best_reason = f"jaccard similarity: {score:.3f}"

    if best_conf >= 0.10 and best_idx is not None:
        return MappingResult(
            step_text=step_text,
            matched_node_id=str(best_idx),
            confidence=round(best_conf, 3),
            reason=best_reason,
        )
    return MappingResult(step_text=step_text, matched_node_id=None, confidence=0.0, reason=best_reason)


def gsm8k_verify(
    mapping: MappingResult,
    step_index: int,
    visited: set[int],
    graph: dict,
) -> VerificationResult:
    """Simplified GSM8K verification: check non-redundancy and forward progress."""
    node_id_str = mapping.matched_node_id
    if node_id_str is None:
        return VerificationResult(
            valid=False, redundant=False, missing_premise=False, contradiction=False,
            reason="no mapped DAG node",
        )

    node_idx = int(node_id_str)

    if node_idx in visited:
        return VerificationResult(
            valid=True, redundant=True, missing_premise=False, contradiction=False,
            reason=f"node {node_idx} was already covered",
        )

    # Check if this node has any incoming edges from previously visited nodes
    # If so, the step follows the reasoning chain
    edges = graph.get("edges", [])
    has_predecessor = any(
        src in visited and tgt == node_idx for src, tgt in edges
    )

    if not visited or has_predecessor:
        return VerificationResult(
            valid=True, redundant=False, missing_premise=False, contradiction=False,
            reason=f"node {node_idx} is a new reasoning step",
        )

    # No predecessor visited — this is a "jump" (skipped intermediate reasoning)
    return VerificationResult(
        valid=True, redundant=False, missing_premise=True, contradiction=False,
        reason=f"node {node_idx} skips intermediate nodes",
    )


def gsm8k_light_dag(
    graph: dict,
    mappings: list[MappingResult],
    verifications: list[VerificationResult],
) -> dict:
    """Light up the GSM8K DAG based on model step coverage."""
    nodes = graph["nodes"]
    edges_list = graph.get("edges", [])

    # Initialize all nodes and edges as unvisited/unused
    node_status: dict[str, str] = {}
    for idx in range(len(nodes)):
        node_status[str(idx)] = "unvisited"

    edge_status: dict[str, str] = {}
    for src, tgt in edges_list:
        edge_status[f"{src}->{tgt}"] = "unused"

    visited_in_order: list[int] = []
    step_states: list[dict] = []

    for idx, (mapping, verification) in enumerate(zip(mappings, verifications), start=1):
        node_id_str = mapping.matched_node_id
        if node_id_str is None:
            step_states.append({"step_index": idx, "node": None, "status": "wrong", "reason": verification.reason})
            continue

        node_idx = int(node_id_str)

        if verification.contradiction:
            status = "contradiction"
        elif verification.redundant:
            status = "redundant"
        elif verification.missing_premise:
            status = "jump"
        elif verification.valid:
            status = "lit"
        else:
            status = "wrong"

        node_status[node_id_str] = status

        # Mark edges between consecutive valid nodes
        if verification.valid and not verification.redundant:
            if visited_in_order:
                prev = visited_in_order[-1]
                edge_key = f"{prev}->{node_idx}"
                if edge_key in edge_status:
                    edge_status[edge_key] = "used_valid"
                elif node_idx not in visited_in_order:
                    edge_key_rev = f"{node_idx}->{prev}"
                    if edge_key_rev in edge_status:
                        # Edge direction is reversed — just mark as used
                        edge_status[edge_key_rev] = "used_valid"
            visited_in_order.append(node_idx)

        step_states.append({
            "step_index": idx,
            "node": node_id_str,
            "status": status,
            "reason": verification.reason,
        })

    return {"nodes": node_status, "edges": edge_status, "steps": step_states}


def gsm8k_score_depth(
    graph: dict,
    mappings: list[MappingResult],
) -> tuple[float, list[dict]]:
    """Depth score: ratio of gold answer-nodes covered by the model."""
    nodes = graph["nodes"]

    # Determine which nodes are "answer" nodes (reasoning chain).
    # We split based on: question nodes are first N, answer nodes follow.
    # The last node is always `####[ans]`.
    # Since we don't have an explicit split marker, we treat all nodes after
    # the first as potential answer/reasoning nodes.
    # The depth is: how many unique nodes did the model cover?

    all_indices = list(range(len(nodes)))
    if not all_indices:
        return 0.0, []

    covered: set[int] = set()
    detail: list[dict] = []

    for mapping in mappings:
        node_id_str = mapping.matched_node_id
        if node_id_str is not None:
            node_idx = int(node_id_str)
            is_new = node_idx not in covered
            covered.add(node_idx)
            detail.append({
                "node_index": node_idx,
                "node_text": nodes[node_idx][:80] if node_idx < len(nodes) else "",
                "newly_covered": is_new,
                "confidence": mapping.confidence,
            })

    # Score as percentage of nodes covered
    total_nodes = len(nodes)
    coverage_ratio = len(covered) / total_nodes if total_nodes > 0 else 0.0
    score = round(min(100.0, 100.0 * coverage_ratio), 3)

    return score, detail


def evaluate_gsm8k_sample(sample: dict, output: dict) -> dict:
    """Evaluate one GSM8K sample against its gold reasoning graph."""
    graph = sample["gold_reasoning_graph"]
    nodes: list[str] = graph["nodes"]
    gold_answer = sample["gold_answer"]

    # 1. Extract steps from model response
    split = split_steps(output["response"])

    # 2. Check answer correctness
    correct = gsm8k_answer_is_correct(split.final_answer, gold_answer)

    # 3. Map each step to a graph node
    visited: set[int] = set()
    mappings: list[MappingResult] = []
    verifications: list[VerificationResult] = []

    for step_idx, step in enumerate(split.steps):
        mapping = gsm8k_map_step_to_node(step, nodes)
        mappings.append(mapping)

        # Update visited set for verification context
        verification = gsm8k_verify(mapping, step_idx, visited, graph)
        verifications.append(verification)

        if mapping.matched_node_id is not None and not verification.redundant:
            visited.add(int(mapping.matched_node_id))

    # 4. Score depth
    depth, depth_detail = gsm8k_score_depth(graph, mappings)

    # 5. Light up the DAG
    lighted = gsm8k_light_dag(graph, mappings, verifications)

    # 6. Consistency: simple scoring based on correctness and step validity
    valid_steps = sum(1 for v in verifications if v.valid)
    total_steps = len(verifications)
    consistency = round(100.0 * valid_steps / total_steps, 3) if total_steps > 0 else 0.0

    first_error = None
    missing_premise = False
    contradiction_count = 0
    for idx, v in enumerate(verifications, start=1):
        if not v.valid and first_error is None:
            first_error = idx
        if v.missing_premise:
            missing_premise = True
        if v.contradiction:
            contradiction_count += 1

    return {
        "sample_id": sample["id"],
        "model_name": output.get("model_name", "api"),
        "output_type": output.get("output_type", "api"),
        "answer_correct": correct,
        "score_depth": depth,
        "score_breadth": None,
        "score_consistency": consistency,
        "first_error_step": first_error,
        "missing_premise_flag": missing_premise,
        "branch_coverage": None,
        "contradiction_count": contradiction_count,
        "lighted_graph": lighted,
        "detail": {
            "steps": split.steps,
            "final_answer": split.final_answer,
            "mappings": [asdict(m) for m in mappings],
            "verifications": [asdict(v) for v in verifications],
            "depth": depth_detail,
            "breadth": {},
            "consistency": {
                "valid_steps": valid_steps,
                "total_steps": total_steps,
            },
        },
    }


def evaluate_gsm8k_files(
    benchmark_path: str,
    outputs_path: str,
    save_path: str,
) -> list[dict]:
    """Batch-evaluate GSM8K model outputs against the benchmark."""
    samples = {s["id"]: s for s in read_jsonl(benchmark_path)}
    outputs = read_jsonl(outputs_path)
    results = []
    for output in outputs:
        sample_id = output["sample_id"]
        if sample_id not in samples:
            raise ValueError(f"Model output references unknown sample_id={sample_id}")
        results.append(evaluate_gsm8k_sample(samples[sample_id], output))
    write_jsonl(save_path, results)
    return results
