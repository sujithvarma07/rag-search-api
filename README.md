# rag-search-api

A semantic search API built on retrieval-augmented generation (RAG). Given a set of documents, the system indexes them using vector embeddings and allows natural language queries to retrieve the most relevant results, optionally generating a summarized answer using an LLM.

## Motivation

Full-text search works well for keyword matching but falls apart when the query and the document use different words to describe the same concept. This project explores building a search layer that understands meaning, not just tokens.

## Planned stack

- **FastAPI** — REST API layer
- **OpenAI** — text embeddings (`text-embedding-3-small`) and chat completions
- **ChromaDB** — local vector store for development
- **Pydantic** — request/response validation
- **pytest** — test suite

## Status

Work in progress. Setting up project structure.

## Running locally

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```
