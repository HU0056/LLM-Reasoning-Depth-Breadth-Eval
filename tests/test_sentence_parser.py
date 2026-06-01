import unittest

from reasoning_graph.sentence_parser import extract_final_answer, split_sentences


class SentenceParserTest(unittest.TestCase):
    def test_split_sentences_ignores_empty_nodes(self) -> None:
        text = "Step one.\n\nStep two... Step three! #### 42"
        self.assertEqual(
            split_sentences(text),
            ["Step one", "Step two", "Step three", "#### 42"],
        )

    def test_extract_final_answer(self) -> None:
        self.assertEqual(extract_final_answer(["reason", "#### 72"]), "72")


if __name__ == "__main__":
    unittest.main()
