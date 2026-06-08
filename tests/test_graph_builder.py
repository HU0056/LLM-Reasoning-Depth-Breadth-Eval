import unittest

from reasoning_graph.graph_builder import build_reasoning_edges
from reasoning_graph.similarity import calculate_similarity


class GraphBuilderTest(unittest.TestCase):
    def test_build_reasoning_edges(self) -> None:
        question_nodes = ["Natalia sold clips to 48 friends in April"]
        answer_nodes = [
            "Natalia sold 24 clips in May",
            "Natalia sold 72 clips altogether in April and May",
            "#### 72",
        ]

        edges = build_reasoning_edges(
            question_nodes=question_nodes,
            answer_nodes=answer_nodes,
            similarity_fn=calculate_similarity,
            bound=1 - 1 / 2.718281828459045,
        )

        self.assertIn([0, 1], edges)
        self.assertIn([1, 2], edges)
        self.assertIn([2, 3], edges)


if __name__ == "__main__":
    unittest.main()
