<div align="center">

# Document RAG

<p><strong>Ask questions over your PDFs and text files.</strong><br>
Retrieve relevant passages locally with FAISS and optionally generate grounded answers with Groq.</p>

[![Animated project highlights](https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=20&duration=2800&pause=900&color=46D9A6&center=true&vCenter=true&width=760&lines=Load+PDF+and+TXT+documents;Find+relevant+context+with+FAISS;Generate+grounded+answers+with+Groq)](https://github.com/DenverCoder1/readme-typing-svg)

![Python 3.13+](https://img.shields.io/badge/Python-3.13%2B-3776AB?logo=python&logoColor=white)
![FAISS](https://img.shields.io/badge/Vector%20Search-FAISS-4B8BBE)
![Groq](https://img.shields.io/badge/LLM-Groq-F55036)

</div>

## Overview

This project is a small, runnable Retrieval-Augmented Generation (RAG) pipeline.
It reads PDF and TXT files, splits them into chunks, embeds them with
`all-MiniLM-L6-v2`, and searches a persistent FAISS index. When a valid Groq API
key is configured, retrieved passages are sent to an LLM to produce a
context-grounded answer.

## How it works

```text
PDF / TXT files
      │
      ▼
Document loading ──► Text chunking ──► Sentence Transformers
                                           │
Question ──► Query embedding ──────────────┤
                                           ▼
                                      FAISS top-k
                                           │
                            Retrieved context + question
                                           │
                                           ▼
                                  Groq answer (optional)
```

## Quick start

### 1. Install dependencies

From the project root:

```powershell
uv sync
```

Alternatively, install from `requirements.txt` into your active environment.

### 2. Add documents

The repository intentionally excludes `data/` and `notebook/` from Git because
local documents and notebook outputs can contain private information. Create
`data/` and put your own `.pdf` and `.txt` files there; subdirectories are
searched recursively.

### 3. Configure Groq (optional)

Copy the example environment file and add your API key:

```powershell
Copy-Item .env.example .env
```

Edit `.env`:

```dotenv
GROQ_API_KEY=your_actual_groq_api_key
```

The `.env` file is ignored by Git. Without a valid key, retrieval still works
and the CLI prints matching passages without generating an answer.

### 4. Ask a question

```powershell
uv run rag "What skills are listed?"
```

The first run builds `faiss_store/`; subsequent runs reuse the saved index.
Rebuild after changing your documents:

```powershell
uv run rag --rebuild "What experience is described?"
```

Run retrieval without calling Groq:

```powershell
uv run rag --retrieval-only "What skills are listed?"
```

You can also run the app module directly:

```powershell
python -m src.app "What skills are listed?"
```

## Project structure

```text
src/rag/
├── cli.py          # Command-line interface
├── data_loader.py  # PDF and TXT ingestion
├── embedding.py    # Chunking and sentence embeddings
├── search.py       # Retrieval and optional Groq generation
└── vectorstore.py  # FAISS indexing and persistence
tests/              # Focused loader and vector-store tests
```

## Tests

```powershell
python -m unittest discover -s tests -v
```

## Notes

- Supported source files: `.pdf` and `.txt`.
- The default embedding model is `all-MiniLM-L6-v2`.
- The default Groq chat model is `openai/gpt-oss-20b`.
- `data/`, `notebook/`, `.env`, and the generated `faiss_store/` are excluded
  from version control.
