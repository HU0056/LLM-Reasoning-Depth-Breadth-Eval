from __future__ import annotations

from reasoning_eval.common.schema import ReasoningEdge, ReasoningGraph, ReasoningNode, Rule


def _node_type(prop: str, facts: list[str], goal: str, reachable: set[str]) -> str:
    if prop in facts:
        return "fact"
    if prop == goal:
        return "goal"
    if prop not in reachable:
        return "distractor"
    return "derived"


def build_reasoning_graph(facts: list[str], rules: list[Rule], goal: str) -> ReasoningGraph:
    propositions = set(facts) | {goal}
    for rule in rules:
        propositions.add(rule.source)
        propositions.add(rule.target)

    reachable = set(facts)
    changed = True
    while changed:
        changed = False
        for rule in rules:
            if rule.source in reachable and rule.target not in reachable and not rule.distractor:
                reachable.add(rule.target)
                changed = True

    nodes = [
        ReasoningNode(id=prop, proposition=prop, type=_node_type(prop, facts, goal, reachable))
        for prop in sorted(propositions)
    ]
    edges = [
        ReasoningEdge(
            source=rule.source,
            target=rule.target,
            rule_text=rule.text,
            status="distractor" if rule.distractor or rule.source not in reachable else "normal",
        )
        for rule in rules
    ]
    return ReasoningGraph(nodes=nodes, edges=edges, start_nodes=list(facts), goal_node=goal)

