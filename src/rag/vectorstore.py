import pickle
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from langchain_core.documents import Document

from .embedding import EmbeddingPipeline


class FaissVectorStore:
    def __init__(
        self,
        persist_dir: str | Path = "faiss_store",
        embedding_model: str = "all-MiniLM-L6-v2",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> None:
        self.persist_dir = Path(persist_dir).expanduser()
        if not self.persist_dir.is_absolute():
            self.persist_dir = self.persist_dir.resolve()
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.index: faiss.Index | None = None
        self.metadata: list[dict[str, Any]] = []
        self.embedding_pipeline = EmbeddingPipeline(
            model_name=embedding_model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    @property
    def index_path(self) -> Path:
        return self.persist_dir / "faiss.index"

    @property
    def metadata_path(self) -> Path:
        return self.persist_dir / "metadata.pkl"

    def build_from_documents(self, documents: list[Document]) -> None:
        if not documents:
            raise ValueError("Cannot build a vector store without documents")

        print(f"[INFO] Building vector store from {len(documents)} documents...")
        chunks = self.embedding_pipeline.chunk_documents(documents)
        if not chunks:
            raise ValueError("Document splitting produced no searchable chunks")

        embeddings = self.embedding_pipeline.embed_chunks(chunks)
        metadata = [
            {**chunk.metadata, "text": chunk.page_content}
            for chunk in chunks
        ]
        self.index = None
        self.metadata = []
        self.add_embeddings(embeddings, metadata)
        self.save()
        print(f"[INFO] Vector store built and saved to {self.persist_dir}")

    def add_embeddings(
        self,
        embeddings: np.ndarray,
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        vectors = np.ascontiguousarray(embeddings, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[0] == 0:
            raise ValueError("embeddings must be a non-empty 2D array")
        if metadatas is not None and len(metadatas) != vectors.shape[0]:
            raise ValueError("The number of metadata entries must match embeddings")

        faiss.normalize_L2(vectors)
        if self.index is None:
            self.index = faiss.IndexFlatIP(vectors.shape[1])
        elif self.index.d != vectors.shape[1]:
            raise ValueError("Embedding dimensions do not match the existing index")

        self.index.add(vectors)
        if metadatas is not None:
            self.metadata.extend(metadatas)
        else:
            self.metadata.extend({} for _ in range(vectors.shape[0]))
        print(f"[INFO] Added {vectors.shape[0]} vectors to the FAISS index.")

    def save(self) -> None:
        if self.index is None or self.index.ntotal == 0:
            raise ValueError("Cannot save an empty FAISS index")
        if self.index.ntotal != len(self.metadata):
            raise ValueError("FAISS index and metadata counts do not match")

        self.persist_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path))
        with self.metadata_path.open("wb") as file:
            pickle.dump(self.metadata, file)
        print(f"[INFO] Saved FAISS index and metadata to {self.persist_dir}")

    def load(self) -> None:
        missing = [
            path for path in (self.index_path, self.metadata_path) if not path.is_file()
        ]
        if missing:
            raise FileNotFoundError(
                "Vector store files are missing: " + ", ".join(map(str, missing))
            )

        index = faiss.read_index(str(self.index_path))
        with self.metadata_path.open("rb") as file:
            metadata = pickle.load(file)
        if not isinstance(metadata, list) or len(metadata) != index.ntotal:
            raise ValueError("Saved FAISS index and metadata are inconsistent")
        self.index = index
        self.metadata = metadata
        print(f"[INFO] Loaded {index.ntotal} vectors from {self.persist_dir}")

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[dict[str, Any]]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        if self.index is None:
            raise RuntimeError("The vector store is not loaded or built")
        if self.index.ntotal == 0:
            return []

        query = np.ascontiguousarray(query_embedding, dtype=np.float32)
        if query.ndim == 1:
            query = query.reshape(1, -1)
        if query.ndim != 2 or query.shape[1] != self.index.d:
            raise ValueError("Query embedding dimensions do not match the FAISS index")
        faiss.normalize_L2(query)
        scores, indices = self.index.search(query, min(top_k, self.index.ntotal))

        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < 0:
                continue
            results.append(
                {
                    "index": int(idx),
                    "score": float(score),
                    "metadata": self.metadata[idx],
                }
            )
        return results

    def query(self, query_text: str, top_k: int = 5) -> list[dict[str, Any]]:
        if not query_text.strip():
            raise ValueError("query_text must not be empty")
        print(f"[INFO] Searching for: {query_text!r}")
        query_embedding = self.embedding_pipeline.embed_texts([query_text])
        return self.search(query_embedding, top_k=top_k)
