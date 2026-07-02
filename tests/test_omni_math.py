import unittest

from reasoning_graph.omni_math import (
    DependencyValidationError,
    build_dependency_prompt,
    build_omni_math_payload,
    dependencies_to_edges,
    parse_dependency_response,
)


class OmniMathPayloadTest(unittest.TestCase):
    def test_build_payload_maps_raw_sample_to_project_schema(self) -> None:
        payload = build_omni_math_payload(
            {
                "problem": "What is 1+1?",
                "solution": "Compute 1+1=2.",
                "answer": "2",
            },
            7,
        )

        self.assertEqual(payload["id"], "omni_math_test_00007")
        self.assertEqual(payload["gsm8k_id"], "omni_math_test_00007")
        self.assertEqual(payload["task_type"], "math")
        self.assertEqual(payload["question"], "What is 1+1?")
        self.assertEqual(payload["gold_answer"], "2")
        self.assertEqual(
            payload["gold_reasoning_graph"],
            {
                "nodes": ["What is 1+1", "Compute 1+1=2", "#### 2"],
                "edges": [],
            },
        )


class OmniMathDependencyParsingTest(unittest.TestCase):
    def test_parse_plain_json(self) -> None:
        parsed = parse_dependency_response(
            '{"dependencies":[{"target":1,"predecessors":[0]}]}'
        )
        self.assertEqual(parsed["dependencies"][0]["target"], 1)

    def test_parse_markdown_json_fence(self) -> None:
        parsed = parse_dependency_response(
            '```json\n{"dependencies":[{"target":1,"predecessors":[0]}]}\n```'
        )
        self.assertEqual(parsed["dependencies"][0]["predecessors"], [0])

    def test_reject_invalid_json(self) -> None:
        with self.assertRaises(DependencyValidationError):
            parse_dependency_response("not json")

    def test_deduplicates_and_sorts_edges(self) -> None:
        edges = dependencies_to_edges(
            {"dependencies": [{"target": 3, "predecessors": [2, 0, 2]}]},
            first_reasoning_index=3,
            node_count=4,
            mode="std",
        )
        self.assertEqual(edges, [[0, 3], [2, 3]])

    def test_merges_duplicate_targets(self) -> None:
        edges = dependencies_to_edges(
            {
                "dependencies": [
                    {"target": 3, "predecessors": [0]},
                    {"target": 3, "predecessors": [2]},
                ]
            },
            first_reasoning_index=3,
            node_count=4,
            mode="std",
        )
        self.assertEqual(edges, [[0, 3], [2, 3]])

    def test_accepts_integer_strings(self) -> None:
        edges = dependencies_to_edges(
            {"dependencies": [{"target": "3", "predecessors": ["0", "2"]}]},
            first_reasoning_index=3,
            node_count=4,
            mode="std",
        )
        self.assertEqual(edges, [[0, 3], [2, 3]])

    def test_accepts_later_predecessor_when_graph_is_acyclic(self) -> None:
        edges = dependencies_to_edges(
            {"dependencies": [{"target": 3, "predecessors": [4]}]},
            first_reasoning_index=3,
            node_count=5,
            mode="test",
        )
        self.assertEqual(edges, [[4, 3]])

    def test_rejects_self_dependency(self) -> None:
        with self.assertRaises(DependencyValidationError):
            dependencies_to_edges(
                {"dependencies": [{"target": 3, "predecessors": [3]}]},
                first_reasoning_index=3,
                node_count=5,
                mode="test",
            )

    def test_rejects_cycles(self) -> None:
        with self.assertRaises(DependencyValidationError) as context:
            dependencies_to_edges(
                {
                    "dependencies": [
                        {"target": 3, "predecessors": [4]},
                        {"target": 4, "predecessors": [3]},
                    ]
                },
                first_reasoning_index=3,
                node_count=5,
                mode="test",
            )
        message = str(context.exception)
        self.assertIn("Remove or redirect", message)
        self.assertIn("3->4", message)
        self.assertIn("4->3", message)
        self.assertNotIn("3->3", message)
        self.assertNotIn("4->4", message)

    def test_std_rejects_empty_predecessors(self) -> None:
        with self.assertRaises(DependencyValidationError):
            dependencies_to_edges(
                {"dependencies": [{"target": 2, "predecessors": []}]},
                first_reasoning_index=2,
                node_count=3,
                mode="std",
            )

    def test_test_mode_allows_empty_predecessors(self) -> None:
        edges = dependencies_to_edges(
            {"dependencies": [{"target": 2, "predecessors": []}]},
            first_reasoning_index=2,
            node_count=3,
            mode="test",
        )
        self.assertEqual(edges, [])

    def test_retry_prompt_includes_validation_feedback(self) -> None:
        messages = build_dependency_prompt(
            ["question", "reason"],
            1,
            "std",
            previous_response='{"dependencies":[]}',
            validation_error="std mode requires dependencies for: 1",
        )
        self.assertIn("failed validation", messages[1]["content"])
        self.assertIn("std mode requires dependencies for: 1", messages[1]["content"])

    def test_prompt_explains_how_to_fix_cycles(self) -> None:
        messages = build_dependency_prompt(["question", "claim", "proof"], 1, "std")
        prompt = messages[1]["content"]
        self.assertIn("must not also depend on the claim", prompt)
        self.assertIn("remove or redirect the weakest edge", prompt)


if __name__ == "__main__":
    unittest.main()
