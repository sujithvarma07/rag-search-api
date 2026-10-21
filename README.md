# rag-search-api

A semantic search API built on retrieval-augmented generation (RAG). Documents are chunked, embedded with OpenAI, and stored in ChromaDB. Natural language queries retrieve the most relevant chunks, optionally with an LLM-generated answer.

## Stack

- **FastAPI** for the REST layer
- **OpenAI** for embeddings (`text-embedding-3-small`) and chat completions
- **ChromaDB** as the local vector store
- **Pydantic** for validation and settings
- **slowapi** for rate limiting
- **pytest** for tests

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
# create a .env file with the variables listed below
uvicorn app.main:app --reload
```

The interactive API docs are served at `http://localhost:8000/docs`.

### Docker

```bash
docker build -t rag-search-api .
docker run --env-file .env -p 8000:8000 rag-search-api
```

## Environment variables

| Variable | Required | Default | Description |
| --- | --- | --- | --- |
| `OPENAI_API_KEY` | yes | | OpenAI API key used for embeddings and answer generation |
| `API_KEY` | yes | | Value clients must send in the `X-API-Key` header |
| `CHROMA_PATH` | no | `./chroma_db` | Directory where ChromaDB persists data |
| `EMBEDDING_MODEL` | no | `text-embedding-3-small` | OpenAI embedding model |
| `TOP_K` | no | `5` | Default number of results to retrieve |

## Endpoints

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/health` | Service status, ChromaDB status, and API version |
| `POST` | `/documents` | Chunk, embed, and store documents |
| `POST` | `/documents/batch` | Ingest several requests concurrently |
| `DELETE` | `/documents/{doc_id}` | Delete a document from the store |
| `POST` | `/search` | Semantic search with optional `min_score`, pagination, and `generate_answer` |

Ingest and search endpoints require the `X-API-Key` header and are rate limited to 10 requests per minute.

## Testing

```bash
pytest
```
