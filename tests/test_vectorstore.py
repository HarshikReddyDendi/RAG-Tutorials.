import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.rag.vectorstore import FaissVectorStore


class FaissVectorStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        self.store = object.__new__(FaissVectorStore)
        self.store.persist_dir = Path(temporary_directory.name)
        self.store.index = None
        self.store.metadata = []

    def test_search_returns_cosine_ranked_results_and_clamps_top_k(self) -> None:
        self.store.add_embeddings(
            np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
            [{"text": "first"}, {"text": "second"}],
        )

        results = self.store.search(
            np.array([[0.9, 0.1]], dtype=np.float32),
            top_k=10,
        )

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["metadata"]["text"], "first")
        self.assertGreater(results[0]["score"], results[1]["score"])

    def test_search_rejects_uninitialized_index(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "not loaded or built"):
            self.store.search(np.array([[1.0, 0.0]], dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
