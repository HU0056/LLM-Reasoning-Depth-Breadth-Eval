import unittest

from reasoning_graph.sentence_parser import extract_final_answer, split_sentences


class SentenceParserTest(unittest.TestCase):
    def test_split_sentences_ignores_empty_nodes(self) -> None:
        text = "Step one.\n\nStep two... Step three! #### 42"
        self.assertEqual(
            split_sentences(text),
            ["Step one", "Step two", "Step three", "#### 42"],
        )

    def test_split_sentences_keeps_display_math_with_context(self) -> None:
        text = (
            "Set x=0:\n"
            "\\[\n"
            "f(0)=g(0)\n"
            "\\]\n"
            "Therefore,\n"
            "$$\n"
            "f(x)=g(x)\n"
            "$$\n"
            "Done."
        )

        self.assertEqual(
            split_sentences(text),
            [
                "Set x=0: \\[ f(0)=g(0) \\]",
                "Therefore, $$ f(x)=g(x) $$",
                "Done",
            ],
        )

    def test_split_sentences_merges_layout_list_markers(self) -> None:
        text = "The solutions are:\n1\n\\( f(x)=0 \\)\n2\n\\( g(x)=x \\)"

        self.assertEqual(
            split_sentences(text),
            ["The solutions are:", "1) \\( f(x)=0 \\)", "2) \\( g(x)=x \\)"],
        )

    def test_extract_final_answer(self) -> None:
        self.assertEqual(extract_final_answer(["reason", "#### 72"]), "72")


if __name__ == "__main__":
    unittest.main()
