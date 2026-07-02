"""Deterministic Verifiers — code-level checks with 100% reliability.

Checks:
1. Computation correctness
2. Justification plausibility (type matches operation)
3. Use-def chain consistency
4. Node contribution (every node on a start→goal path)
5. DAG topology (acyclic, connected)
6. Type consistency (edge types match node types)
7. Reasoning difficulty per edge
"""

from __future__ import annotations

import math
import re
from typing import Optional

import networkx as nx

from reasoning_eval.harness.schemas import (
    ComputationCheck,
    ContributionCheck,
    DagEdge,
    DagNode,
    Justification,
    JustificationCheck,
    JustificationType,
    NodeType,
    StepDeclaration,
    StructuredSolution,
    TopologyCheck,
    TypeConsistencyCheck,
    UseDefCheck,
    VerificationReport,
)

# ═══════════════════════════════════════════════
# 1. Computation
# ═══════════════════════════════════════════════

_EXPR_RE = re.compile(
    r"(?P<lhs>[\d\s+\-*/().%^]+)\s*=\s*(?P<rhs>[\d\s+\-*/().%^]+)"
)
_GSM8K_CALC_RE = re.compile(r"<<(.+?)=(.+?)>>")


def _safe_eval(expr: str) -> Optional[float]:
    expr = expr.strip().replace("^", "**").replace(" ", "")
    if not expr or not re.fullmatch(r"[\d+\-*/().%]+", expr):
        return None
    try:
        return float(eval(expr, {"__builtins__": {}}, {}))
    except Exception:
        return None


def verify_computation(steps: list[StepDeclaration]) -> list[ComputationCheck]:
    checks: list[ComputationCheck] = []
    for step in steps:
        if not step.expression:
            continue
        check = ComputationCheck(step_index=step.index, expression=step.expression)
        m = _EXPR_RE.search(step.expression)
        if m:
            lhs_val = _safe_eval(m.group("lhs"))
            rhs_val = _safe_eval(m.group("rhs"))
            if lhs_val is not None and rhs_val is not None:
                check.computed_value = lhs_val
                check.declared_value = rhs_val
                check.matches = math.isclose(lhs_val, rhs_val, rel_tol=1e-9)
                if not check.matches:
                    check.error = (
                        f"Computed {m.group('lhs')} = {lhs_val}, "
                        f"but declared = {rhs_val}"
                    )
                checks.append(check)
                continue
        for m_gsm in _GSM8K_CALC_RE.finditer(step.expression):
            expr_val = _safe_eval(m_gsm.group(1))
            declared = _safe_eval(m_gsm.group(2))
            if expr_val is not None and declared is not None:
                check.computed_value = expr_val
                check.declared_value = declared
                check.matches = math.isclose(expr_val, declared, rel_tol=1e-9)
                if not check.matches:
                    check.error = (
                        f"GSM8K calc: {m_gsm.group(1)} = {expr_val}, "
                        f"declared = {declared}"
                    )
                break
        else:
            check.error = "Could not parse expression"
        checks.append(check)
    return checks


# ═══════════════════════════════════════════════
# 2. Justification plausibility
# ═══════════════════════════════════════════════

# Map operation patterns → expected justification
_OPERATION_JUSTIFICATION_MAP: list[tuple[re.Pattern, JustificationType, str]] = [
    (re.compile(r"[+\-]\s*\d"), JustificationType.ARITHMETIC, "addition/subtraction"),
    (re.compile(r"[*/÷]\s*\d"), JustificationType.ARITHMETIC, "multiplication/division"),
    (re.compile(r"="), JustificationType.ALGEBRA, "equation"),
    (re.compile(r"substitut|代入|replace|替换"), JustificationType.SUBSTITUTION, "substitution"),
    (re.compile(r"simplif|化简|约分|通分"), JustificationType.SIMPLIFICATION, "simplification"),
    (re.compile(r"factor|因式|分解"), JustificationType.ALGEBRA, "factoring"),
    (re.compile(r"induct|归纳"), JustificationType.INDUCTION, "induction"),
    (re.compile(r"define|定义|means?|is a"), JustificationType.DEFINITION, "definition"),
    (re.compile(r"theorem|定理|law|律"), JustificationType.THEOREM, "theorem"),
]


def _guess_expected_justification(step_text: str) -> tuple[JustificationType, str]:
    """Heuristically guess the expected justification type from step text."""
    for pattern, jtype, desc in _OPERATION_JUSTIFICATION_MAP:
        if pattern.search(step_text.lower()):
            return jtype, desc
    return JustificationType.ALGEBRA, "general reasoning"


def verify_justifications(
    steps: list[StepDeclaration],
) -> list[JustificationCheck]:
    """Check whether each declared justification is plausible for its operation."""
    checks: list[JustificationCheck] = []
    for step in steps:
        for j_idx, (dep, just) in enumerate(
            zip(step.depends_on, step.justifications)
        ):
            expected_type, expected_desc = _guess_expected_justification(step.text)
            is_plausible = True
            error = ""

            # Theorem claims need a reference
            if just.type == JustificationType.THEOREM and not just.reference:
                is_plausible = False
                error = "Theorem justification requires a reference (theorem name)"

            # Induction must have "induction" keyword in text
            if just.type == JustificationType.INDUCTION:
                if not re.search(r"induct|归纳|n\s*=\s*k|base case", step.text.lower()):
                    is_plausible = False
                    error = "Induction justification but no induction pattern in step text"

            # Arithmetic should match simple arithmetic operations
            if just.type == JustificationType.ARITHMETIC:
                if not re.search(r"[\d]+\s*[\+\-\*/]\s*[\d]+", step.text):
                    is_plausible = False
                    error = (
                        f"Arithmetic justification but no arithmetic operation "
                        f"found in: '{step.text[:80]}'"
                    )

            # Type mismatch flag (not fatal, just a warning)
            if just.type != expected_type and is_plausible:
                error = (
                    f"Declared '{just.type.value}' but step text suggests "
                    f"'{expected_type.value}' ({expected_desc})"
                )

            checks.append(JustificationCheck(
                step_index=step.index,
                dep_index=dep,
                justification_type=just.type.value,
                is_plausible=is_plausible,
                error=error,
            ))
    return checks


# ═══════════════════════════════════════════════
# 3. Use-def
# ═══════════════════════════════════════════════

def _extract_numbers(text: str) -> set[float]:
    return {float(m) for m in re.findall(r"\d+\.?\d*", text)}


def verify_use_def(steps: list[StepDeclaration]) -> list[UseDefCheck]:
    checks: list[UseDefCheck] = []
    number_first_seen: dict[float, int] = {}
    for step in steps:
        for num in _extract_numbers(step.text):
            if num not in number_first_seen:
                number_first_seen[num] = step.index
            else:
                defining_step = number_first_seen[num]
                declared_dep = defining_step in step.depends_on
                checks.append(UseDefCheck(
                    step_index=step.index,
                    variable=str(int(num) if num == int(num) else num),
                    defined_in_step=defining_step,
                    declared_dep=declared_dep,
                    consistent=declared_dep,
                ))
    return checks


# ═══════════════════════════════════════════════
# 4. Node contribution — every node must be on a path from start to goal
# ═══════════════════════════════════════════════

def _build_nx_dag(
    steps: list[StepDeclaration],
) -> tuple[nx.DiGraph, list[str], list[str], str]:
    g = nx.DiGraph()
    node_ids = [f"step_{s.index}" for s in steps]
    given_ids = [
        f"step_{s.index}" for s in steps
        if s.node_type == NodeType.GIVEN
    ]
    for nid in node_ids:
        g.add_node(nid)
    for step in steps:
        target = f"step_{step.index}"
        for dep_idx in step.depends_on:
            source = f"step_{dep_idx}"
            if source in g:
                g.add_edge(source, target)
    goal = node_ids[-1]
    return g, node_ids, given_ids, goal


def verify_contribution(steps: list[StepDeclaration]) -> list[ContributionCheck]:
    """Check every node contributes to the conclusion."""
    g, node_ids, given_ids, goal = _build_nx_dag(steps)
    checks: list[ContributionCheck] = []

    # Find nodes reachable from ANY given node
    reachable_from_start: set[str] = set()
    for gid in given_ids:
        if gid in g:
            reachable_from_start.update(nx.descendants(g, gid))
            reachable_from_start.add(gid)

    # Find nodes from which the goal is reachable
    goal_reachable_from: set[str] = set()
    for nid in node_ids:
        if nid != goal and nid in g and nx.has_path(g, nid, goal):
            goal_reachable_from.add(nid)
    goal_reachable_from.add(goal)

    # A node contributes iff it is on at least one path from a given to the goal
    all_critical: set[str] = set()
    for gid in given_ids:
        if gid not in g:
            continue
        for nid in node_ids:
            if nid in reachable_from_start and nid in goal_reachable_from:
                try:
                    if nx.has_path(g, gid, nid) and nx.has_path(g, nid, goal):
                        all_critical.add(nid)
                except (nx.NodeNotFound, nx.NetworkXNoPath):
                    pass

    for nid in sorted(node_ids):
        on_path = nid in all_critical
        checks.append(ContributionCheck(
            node_id=nid,
            on_critical_path=on_path,
            reachable_from_start=nid in reachable_from_start,
            goal_reachable=nid in goal_reachable_from,
            contributes=on_path,
        ))
    return checks


# ═══════════════════════════════════════════════
# 5. Topology (extended with contribution)
# ═══════════════════════════════════════════════

def verify_topology(steps: list[StepDeclaration]) -> TopologyCheck:
    g, node_ids, given_ids, goal = _build_nx_dag(steps)

    has_cycles = not nx.is_directed_acyclic_graph(g)
    cycle_nodes: list[str] = []
    if has_cycles:
        try:
            cycle = nx.find_cycle(g)
            cycle_nodes = [n for edge in cycle for n in edge]
        except nx.NetworkXNoCycle:
            pass

    dangling = [
        n for n in node_ids
        if n != node_ids[0]
        and g.in_degree(n) == 0
        and n not in given_ids
    ]

    unreachable = True
    starts = [
        n for n in node_ids
        if g.in_degree(n) == 0 or n in given_ids
    ]
    for s in starts:
        if s != goal and nx.has_path(g, s, goal):
            unreachable = False
            break

    # Non-contributing nodes
    contrib_checks = verify_contribution(steps)
    non_contrib = [c.node_id for c in contrib_checks if not c.contributes]

    is_valid = not has_cycles and not unreachable and len(non_contrib) == 0

    return TopologyCheck(
        has_cycles=has_cycles,
        cycle_nodes=cycle_nodes,
        dangling_nodes=dangling,
        unreachable_conclusion=unreachable,
        non_contributing_nodes=non_contrib,
        is_valid_dag=is_valid,
    )


# ═══════════════════════════════════════════════
# 6. Type consistency
# ═══════════════════════════════════════════════

_VALID_EDGE_TYPE_PAIRS: dict[str, set[tuple[NodeType, NodeType]]] = {
    "infer": {
        (NodeType.GIVEN, NodeType.OPERATION),
        (NodeType.OPERATION, NodeType.OPERATION),
        (NodeType.OPERATION, NodeType.CONCLUSION),
        (NodeType.GIVEN, NodeType.CONCLUSION),
        (NodeType.FACT, NodeType.OPERATION),
        (NodeType.FACT, NodeType.CONCLUSION),
    },
    "execute": {
        (NodeType.GIVEN, NodeType.OPERATION),
    },
    "support": {
        (NodeType.OPERATION, NodeType.VERIFICATION),
        (NodeType.CONCLUSION, NodeType.VERIFICATION),
    },
    "correct": {
        (NodeType.VERIFICATION, NodeType.OPERATION),
    },
    "reference": {
        (NodeType.FACT, NodeType.OPERATION),
    },
}


def verify_type_consistency(
    nodes: list[DagNode],
    edges: list[DagEdge],
) -> TypeConsistencyCheck:
    node_map = {n.id: n.type for n in nodes}
    inconsistencies: list[str] = []
    for edge in edges:
        src_type = node_map.get(edge.source)
        tgt_type = node_map.get(edge.target)
        if src_type is None or tgt_type is None:
            inconsistencies.append(
                f"Edge {edge.source}→{edge.target}: node not found"
            )
            continue
        valid_pairs = _VALID_EDGE_TYPE_PAIRS.get(edge.edge_type.value, set())
        if valid_pairs and (src_type, tgt_type) not in valid_pairs:
            inconsistencies.append(
                f"Edge {edge.source}({src_type.value})"
                f"→{edge.target}({tgt_type.value}): "
                f"type '{edge.edge_type.value}' not valid"
            )
    return TypeConsistencyCheck(
        inconsistencies=inconsistencies,
        is_consistent=len(inconsistencies) == 0,
    )


# ═══════════════════════════════════════════════
# 7. Reasoning difficulty per edge
# ═══════════════════════════════════════════════

def edge_difficulty(justification: Justification) -> float:
    """Compute difficulty of a single edge from its justification.

    Base difficulty × premise_count_factor × atomicity_factor.
    """
    base = justification.base_difficulty
    # Theorem exemption: if theorem is exempt, treat as atomic
    if justification.type == JustificationType.THEOREM and justification.exemption:
        base = 1.5  # treat multi-step theorem as single atomic step
    return base


def compute_min_total_difficulty(
    g: nx.DiGraph,
    edges: list[DagEdge],
    start_nodes: list[str],
    goal: str,
) -> float:
    """Minimum total difficulty from any start node to goal (Dijkstra)."""
    edge_map: dict[tuple[str, str], float] = {}
    for e in edges:
        src = e.premises[0] if e.premises else e.source
        tgt = e.target
        edge_map[(src, tgt)] = edge_difficulty(e.justification)

    best = float("inf")
    for s in start_nodes:
        if s not in g or goal not in g:
            continue
        try:
            path = nx.shortest_path(g, source=s, target=goal)
            path_difficulty = sum(
                edge_map.get((path[i], path[i + 1]), 1.0)
                for i in range(len(path) - 1)
            )
            best = min(best, path_difficulty)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue
    return best if best != float("inf") else 1.0


# ═══════════════════════════════════════════════
# Aggregate
# ═══════════════════════════════════════════════

def run_all_checks(solution: StructuredSolution) -> VerificationReport:
    comp_checks = verify_computation(solution.steps)
    just_checks = verify_justifications(solution.steps)
    use_def_checks = verify_use_def(solution.steps)
    topo_check = verify_topology(solution.steps)

    nodes = [
        DagNode(id=f"step_{s.index}", type=s.node_type, text=s.text,
                expression=s.expression)
        for s in solution.steps
    ]
    edges = [
        DagEdge(
            premises=[f"step_{dep}"],
            target=f"step_{s.index}",
            edge_type="infer",
            justification=(
                s.justifications[i] if i < len(s.justifications)
                else Justification.arithmetic()
            ),
            rationale="",
        )
        for s in solution.steps
        for i, dep in enumerate(s.depends_on)
    ]
    type_check = verify_type_consistency(nodes, edges)
    contrib_checks = verify_contribution(solution.steps)

    comp_ok = all(c.matches for c in comp_checks) if comp_checks else True
    just_ok = all(j.is_plausible for j in just_checks) if just_checks else True
    use_def_ok = all(c.consistent for c in use_def_checks) if use_def_checks else True
    contrib_ok = all(c.contributes for c in contrib_checks) if contrib_checks else True

    all_passed = (
        comp_ok and just_ok and use_def_ok
        and topo_check.is_valid_dag and type_check.is_consistent
        and contrib_ok
    )

    lines = []
    if comp_checks:
        n_fail = sum(1 for c in comp_checks if not c.matches)
        lines.append(f"Computation: {len(comp_checks)} checked, {n_fail} errors")
    if just_checks:
        n_fail = sum(1 for j in just_checks if not j.is_plausible)
        lines.append(f"Justification: {len(just_checks)} checked, {n_fail} implausible")
    if use_def_checks:
        n_fail = sum(1 for c in use_def_checks if not c.consistent)
        lines.append(f"Use-def: {len(use_def_checks)} checked, {n_fail} inconsistencies")
    lines.append(
        f"Topology: {'valid' if topo_check.is_valid_dag else 'INVALID'}"
        + (f", {len(topo_check.non_contributing_nodes)} non-contributing"
           if topo_check.non_contributing_nodes else "")
    )
    lines.append(f"Type: {'OK' if type_check.is_consistent else 'ISSUES'}")

    return VerificationReport(
        computation=comp_checks,
        justification=just_checks,
        use_def=use_def_checks,
        topology=topo_check,
        type_consistency=type_check,
        contribution=contrib_checks,
        all_passed=all_passed,
        summary=" | ".join(lines),
    )
