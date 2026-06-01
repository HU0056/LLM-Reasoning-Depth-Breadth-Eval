from __future__ import annotations

from collections.abc import Callable


SimilarityFn = Callable[[str, str], float]


def build_reasoning_edges(
    question_nodes: list[str],
    answer_nodes: list[str],
    similarity_fn: SimilarityFn,
    bound: float,
) -> list[list[int]]:
    if not answer_nodes:
        return []

    nodes = question_nodes + answer_nodes
    question_count = len(question_nodes)
    edges: set[tuple[int, int]] = set()

    for answer_idx, answer_node in enumerate(answer_nodes):
        target_index = question_count + answer_idx
        candidate_indexes = list(range(question_count)) + list(
            range(question_count, target_index)
        )

        if not candidate_indexes:
            continue

        scored_candidates = [
            (candidate_index, similarity_fn(nodes[candidate_index], answer_node))
            for candidate_index in candidate_indexes
        ]
        scored_candidates.sort(key=lambda item: (-item[1], item[0]))

        best_index, max_similarity = scored_candidates[0]
        edges.add((best_index, target_index))

        threshold = bound * max_similarity
        for candidate_index, similarity in scored_candidates[1:]:
            if similarity > threshold:
                edges.add((candidate_index, target_index))

    return [list(edge) for edge in sorted(edges)]
