import tempfile
import unittest
from pathlib import Path

from src.rag.data_loader import load_all_documents


class LoadAllDocumentsTests(unittest.TestCase):
    def test_loads_text_files_recursively_with_source_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            data_dir = Path(temporary_directory)
            nested_dir = data_dir / "nested"
            nested_dir.mkdir()
            text_file = nested_dir / "notes.txt"
            text_file.write_text("RAG retrieves relevant context.", encoding="utf-8")

            documents = load_all_documents(data_dir)

        self.assertEqual(len(documents), 1)
        self.assertEqual(documents[0].page_content, "RAG retrieves relevant context.")
        self.assertEqual(documents[0].metadata["source_file"], "notes.txt")
        self.assertEqual(documents[0].metadata["file_type"], "txt")

    def test_raises_for_missing_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            missing_directory = Path(temporary_directory) / "missing"

            with self.assertRaises(FileNotFoundError):
                load_all_documents(missing_directory)


if __name__ == "__main__":
    unittest.main()
