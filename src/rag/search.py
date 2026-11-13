import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_groq import ChatGroq

from .data_loader import load_all_documents
from .vectorstore import FaissVectorStore


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LLM_MODEL = "openai/gpt-oss-20b"
_PLACEHOLDER_KEYS = {"your_key_here", "your_api_key_here"}


def _project_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


class RAGSearch:
    def __init__(
        self,
        persist_dir: str | Path = "faiss_store",
        data_dir: str | Path = "data",
        embedding_model: str = "all-MiniLM-L6-v2",
        llm_model: str = DEFAULT_LLM_MODEL,
        rebuild: bool = False,
        api_key: str | None = None,
        use_llm: bool = True,
    ) -> None:
        load_dotenv(PROJECT_ROOT / ".env")
        self.vectorstore = FaissVectorStore(
            _project_path(persist_dir), embedding_model=embedding_model
        )
        if rebuild or not (
            self.vectorstore.index_path.is_file()
            and self.vectorstore.metadata_path.is_file()
        ):
            documents = load_all_documents(_project_path(data_dir))
            self.vectorstore.build_from_documents(documents)
        else:
            self.vectorstore.load()

        configured_key = api_key or os.getenv("GROQ_API_KEY", "")
        normalized_key = configured_key.strip()
        self.llm = None
        if use_llm and normalized_key.lower() not in _PLACEHOLDER_KEYS and normalized_key:
            self.llm = ChatGroq(
                api_key=normalized_key,
                model=llm_model,
                temperature=0.1,
                max_tokens=1024,
            )
            print(f"[INFO] Groq LLM initialized: {llm_model}")
        elif use_llm:
            print(
                "[INFO] GROQ_API_KEY is not configured; running in retrieval-only mode."
            )

    def retrieve(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        return self.vectorstore.query(query, top_k=top_k)

    def search_and_summarize(self, query: str, top_k: int = 5) -> str:
        return self.summarize_results(query, self.retrieve(query, top_k=top_k))

    def summarize_results(
        self, query: str, results: list[dict[str, Any]]
    ) -> str:
        if self.llm is None:
            raise RuntimeError(
                "LLM is unavailable. Set a valid GROQ_API_KEY in the project .env file."
            )

        context = "\n\n".join(
            result["metadata"].get("text", "")
            for result in results
            if result["metadata"].get("text")
        )
        if not context:
            return "No relevant documents found."

        prompt = (
            "Answer the question using only the supplied context. "
            "If the context does not contain the answer, say so.\n\n"
            f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        )
        response = self.llm.invoke(prompt)
        return str(response.content)
