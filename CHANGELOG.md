# Changelog

All notable changes to this project are documented here.

## [Unreleased]

### Added
- Docker support: `Dockerfile` (Python 3.11 slim) and `.dockerignore`.
- Health check now reports ChromaDB reachability and API version.
- Concurrent batch ingestion via `POST /documents/batch`.
- API key authentication (`X-API-Key`) on ingest and search routes.
- Rate limiting (10 requests/minute) on search and ingest endpoints using slowapi.
- Pagination for search results (`page`, `page_size`).
- Optional LLM answer generation on search (`generate_answer`).
- Minimum score filtering for search results (`min_score`).
- Document deletion via `DELETE /documents/{doc_id}`.
- Exponential backoff retry for OpenAI calls using tenacity.
- Unit tests for data models and the embedding service.

## Initial development

### Added
- Project scaffold with FastAPI, OpenAI embeddings, and ChromaDB.
- Settings module based on pydantic-settings.
- Core Pydantic models for documents and search.
- Async embedding service and ChromaDB vector store wrapper.
- Document ingestion and semantic search endpoints.
- Structured error handling, request logging middleware, and text chunking utility.
