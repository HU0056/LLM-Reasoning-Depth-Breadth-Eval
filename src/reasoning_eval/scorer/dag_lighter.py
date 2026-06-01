from __future__ import annotations

from reasoning_eval.common.schema import MappingResult, VerificationResult


def light_dag(graph: dict, mappings: list[MappingResult], verifications: list[VerificationResult]) -> dict:
    node_status = {node["id"]: "unvisited" for node in graph["nodes"]}
    edge_status = {f"{edge['source']}->{edge['target']}": "unused" for edge in graph["edges"]}
    previous_lit = None
    step_states = []

    for idx, (mapping, verification) in enumerate(zip(mappings, verifications), start=1):
        node = mapping.matched_node_id
        if node is None:
            step_states.append({"step_index": idx, "node": None, "status": "wrong"})
            continue
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
        node_status[node] = status
        if previous_lit and verification.valid and not verification.redundant:
            key = f"{previous_lit}->{node}"
            if key in edge_status:
                edge_status[key] = "used_valid"
        elif previous_lit and verification.missing_premise:
            key = f"{previous_lit}->{node}"
            if key in edge_status:
                edge_status[key] = "skipped"
        elif previous_lit and not verification.valid:
            key = f"{previous_lit}->{node}"
            if key in edge_status:
                edge_status[key] = "wrong"
        if verification.valid and not verification.redundant:
            previous_lit = node
        step_states.append({"step_index": idx, "node": node, "status": status, "reason": verification.reason})

    return {"nodes": node_status, "edges": edge_status, "steps": step_states}

