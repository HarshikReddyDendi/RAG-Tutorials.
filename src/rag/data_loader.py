from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document


SUPPORTED_EXTENSIONS = {".pdf", ".txt"}


def load_all_documents(data_dir: str | Path) -> list[Document]:
    """Load PDF and text documents recursively from a directory."""
    directory = Path(data_dir).expanduser()
    if not directory.is_absolute():
        directory = directory.resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"Document directory does not exist: {directory}")

    files = sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not files:
        raise FileNotFoundError(
            f"No PDF or TXT documents found in {directory}"
        )

    documents: list[Document] = []
    for path in files:
        if path.suffix.lower() == ".pdf":
            loaded = PyPDFLoader(str(path)).load()
        else:
            loaded = TextLoader(str(path), encoding="utf-8").load()

        for document in loaded:
            document.metadata["source_file"] = path.name
            document.metadata["file_type"] = path.suffix.lower().lstrip(".")
        documents.extend(loaded)

    if not documents:
        raise ValueError(f"No readable document content found in {directory}")
    return documents
