"""Tests for the Harness Agent Framework."""

from reasoning_eval.harness.schemas import (
    DagEdge,
    DagNode,
    EdgeType,
    GoldDag,
    NodeType,
    StepDeclaration,
    StructuredSolution,
)
from reasoning_eval.harness.verifiers import (
    verify_computation,
    verify_topology,
    verify_type_consistency,
    verify_use_def,
)


class TestVerifiers:
    """Deterministic verifier tests."""

    def test_computation_correct_expr(self):
        steps = [
            StepDeclaration(
                index=0, text="48/2 = 24",
                depends_on=[], node_type=NodeType.OPERATION,
                expression="48 / 2 = 24",
            ),
        ]
        checks = verify_computation(steps)
        assert len(checks) == 1
        assert checks[0].matches is True

    def test_computation_wrong_expr(self):
        steps = [
            StepDeclaration(
                index=0, text="48/2 = 25",
                depends_on=[], node_type=NodeType.OPERATION,
                expression="48 / 2 = 25",
            ),
        ]
        checks = verify_computation(steps)
        assert len(checks) == 1
        assert checks[0].matches is False

    def test_use_def_consistent(self):
        steps = [
            StepDeclaration(
                index=0, text="April clips = 48",
                depends_on=[], node_type=NodeType.GIVEN,
            ),
            StepDeclaration(
                index=1, text="May clips = 48/2 = 24",
                depends_on=[0], node_type=NodeType.OPERATION,
                expression="48/2=24",
            ),
        ]
        checks = verify_use_def(steps)
        # 48 appears in step 1 but was first seen in step 0
        assert any(c.variable == "48" and c.consistent for c in checks)

    def test_topology_valid_dag(self):
        steps = [
            StepDeclaration(
                index=0, text="fact A", depends_on=[],
                node_type=NodeType.GIVEN,
            ),
            StepDeclaration(
                index=1, text="derived B", depends_on=[0],
                node_type=NodeType.OPERATION,
            ),
            StepDeclaration(
                index=2, text="answer C", depends_on=[1],
                node_type=NodeType.CONCLUSION,
            ),
        ]
        topo = verify_topology(steps)
        assert topo.is_valid_dag is True
        assert topo.has_cycles is False

    def test_topology_cycle(self):
        steps = [
            StepDeclaration(
                index=0, text="A", depends_on=[2],
                node_type=NodeType.GIVEN,
            ),
            StepDeclaration(
                index=1, text="B", depends_on=[0],
                node_type=NodeType.OPERATION,
            ),
            StepDeclaration(
                index=2, text="C", depends_on=[1],
                node_type=NodeType.CONCLUSION,
            ),
        ]
        topo = verify_topology(steps)
        assert topo.has_cycles is True

    def test_type_consistency_valid(self):
        nodes = [
            DagNode(id="step_0", type=NodeType.GIVEN, text="given"),
            DagNode(id="step_1", type=NodeType.OPERATION, text="op"),
        ]
        edges = [
            DagEdge(premises=["step_0"], target="step_1",
                    edge_type=EdgeType.INFER, rationale=""),
        ]
        check = verify_type_consistency(nodes, edges)
        assert check.is_consistent is True

    def test_type_consistency_invalid(self):
        nodes = [
            DagNode(id="step_0", type=NodeType.CONCLUSION, text="c"),
            DagNode(id="step_1", type=NodeType.GIVEN, text="g"),
        ]
        edges = [
            DagEdge(premises=["step_0"], target="step_1",
                    edge_type=EdgeType.INFER, rationale=""),
        ]
        check = verify_type_consistency(nodes, edges)
        # CONCLUSION→GIVEN is not a valid INFER pair
        assert check.is_consistent is False


class TestGoldDag:
    """GoldDag serialization tests."""

    def test_to_legacy_graph(self):
        dag = GoldDag(
            nodes=[
                DagNode(id="step_0", type=NodeType.GIVEN, text="Natalia sold 48 clips"),
                DagNode(id="step_1", type=NodeType.OPERATION, text="48/2 = 24",
                        expression="48/2=24"),
                DagNode(id="step_2", type=NodeType.CONCLUSION, text="48+24 = 72"),
            ],
            edges=[
                DagEdge(premises=["step_0"], target="step_1",
                        edge_type=EdgeType.INFER, rationale="uses 48"),
                DagEdge(premises=["step_0"], target="step_2",
                        edge_type=EdgeType.INFER, rationale="uses 48"),
                DagEdge(premises=["step_1"], target="step_2",
                        edge_type=EdgeType.INFER, rationale="uses 24"),
            ],
            num_steps=3,
            num_edges=3,
        )
        legacy = dag.to_legacy_graph()
        assert len(legacy["nodes"]) == 3
        assert len(legacy["edges"]) == 3
        assert legacy["goal_node"] == "step_2"
        assert legacy["start_nodes"] == ["step_0"]
        assert legacy["nodes"][0]["type"] == "given"
