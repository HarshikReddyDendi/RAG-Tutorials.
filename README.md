# Local document RAG

This project indexes PDF and TXT files from `data/`, retrieves relevant passages
with FAISS, and can optionally use Groq to generate an answer. The local
`data/` and `notebook/` directories are excluded from Git because they may
contain private documents and notebook output. Add your own PDF or TXT files to
`data/` after cloning.

## Setup

Install the project dependencies with `uv sync` (or install `requirements.txt`).
Copy `.env.example` to `.env` and set a valid Groq API key to enable answer
generation. The actual `.env` file is ignored by Git.

## Run

From the project root:

```powershell
uv run rag "What is my experience at Siritek?"
```

Or, with the active project environment:

```powershell
python -m src.app "What is my experience at Siritek?"
```

The first run loads documents and builds `faiss_store/`. Later runs reuse that
index. To rebuild after changing source documents:

```powershell
uv run rag --rebuild "What is my experience at Siritek?"
```

To run retrieval without calling Groq:

```powershell
uv run rag --retrieval-only "What skills are listed?"
```

Only PDF and TXT files are loaded recursively from the selected data directory.
