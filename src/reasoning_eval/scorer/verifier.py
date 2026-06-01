from __future__ import annotations

from reasoning_eval.common.schema import MappingResult, VerificationResult
from reasoning_eval.common.text_utils import contains_inference_word, contradicts_known
import networkx as nx

from reasoning_eval.dataset.graph_utils import build_nx_graph, is_direct_successor, is_reachable


class RuleBasedVerifier:
    def __init__(self, graph: dict):
        self.graph = graph
        self.nx_graph = build_nx_graph(graph)
        self.goal = graph["goal_node"]
        self.start_nodes = set(graph["start_nodes"])

    def verify(
        self,
        mapping: MappingResult,
        previous_node: str | None,
        history: set[str],
    ) -> VerificationResult:
        node = mapping.matched_node_id
        if node is None:
            return VerificationResult(False, False, False, False, "no mapped DAG node")

        contradiction = contradicts_known(mapping.step_text, history)
        if contradiction:
            return VerificationResult(False, False, False, True, "step contradicts an already lit proposition")

        if node in history:
            return VerificationResult(True, True, False, False, f"node {node} was already visited")

        if previous_node is None:
            if node in self.start_nodes:
                return VerificationResult(True, False, False, False, f"first step maps to fact node {node}")
            return VerificationResult(False, False, False, False, f"first step must start from facts {sorted(self.start_nodes)}")

        if is_direct_successor(self.nx_graph, previous_node, node):
            edge = self.nx_graph.edges[previous_node, node]
            if edge.get("status") == "distractor":
                return VerificationResult(False, False, False, False, f"edge {previous_node}->{node} is a distractor rule")
            return VerificationResult(True, False, False, False, f"{node} is a direct successor of {previous_node}")

        if is_reachable(self.nx_graph, previous_node, node):
            dist = nx.shortest_path_length(self.nx_graph, source=previous_node, target=node)
            missing = dist > 1 or contains_inference_word(mapping.step_text)
            return VerificationResult(False, False, missing, False, f"{node} is reachable from {previous_node} but skipped intermediate nodes")

        return VerificationResult(False, False, False, False, f"{node} is not reachable from current node {previous_node}")
