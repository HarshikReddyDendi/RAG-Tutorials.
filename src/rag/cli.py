import argparse
import sys
from pathlib import Path

from .search import PROJECT_ROOT, RAGSearch


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="Search local PDF and text documents with RAG."
    )
    parser.add_argument(
        "query",
        nargs="?",
        default="What is my experience at Siritek?",
        help="Question to search for",
    )
    parser.add_argument("--top-k", type=int, default=3, help="Results to retrieve")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data",
        help="Directory containing PDF and TXT documents",
    )
    parser.add_argument(
        "--store-dir",
        type=Path,
        default=PROJECT_ROOT / "faiss_store",
        help="Directory for the persisted FAISS index",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Recreate the vector index from source documents",
    )
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Retrieve and show passages without calling Groq",
    )
    args = parser.parse_args(argv)
    if args.top_k <= 0:
        parser.error("--top-k must be greater than zero")

    rag = RAGSearch(
        persist_dir=args.store_dir,
        data_dir=args.data_dir,
        rebuild=args.rebuild,
        use_llm=not args.retrieval_only,
    )
    results = rag.retrieve(args.query, top_k=args.top_k)
    if not results:
        print("No relevant documents found.")
        return 0

    print(f"\nTop {len(results)} matching passages:")
    for rank, result in enumerate(results, start=1):
        metadata = result["metadata"]
        source = metadata.get("source_file", metadata.get("source", "unknown"))
        page = metadata.get("page")
        page_text = f", page {page + 1}" if isinstance(page, int) else ""
        print(
            f"\n[{rank}] {source}{page_text} "
            f"(cosine score: {result['score']:.3f})"
        )
        print(metadata.get("text", ""))

    if not args.retrieval_only:
        if rag.llm is None:
            print(
                "\n[INFO] Add a valid GROQ_API_KEY to the project .env file "
                "to generate an answer."
            )
        else:
            print("\nAnswer:")
            print(rag.summarize_results(args.query, results))
    return 0
