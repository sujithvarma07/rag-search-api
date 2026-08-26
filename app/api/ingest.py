import asyncio

from fastapi import APIRouter, Depends, Request

from app.config import settings
from app.core.embeddings import EmbeddingService
from app.db.vector_store import VectorStore
from app.models import IngestRequest
from app.utils.auth import verify_api_key
from app.utils.chunking import chunk_text
from app.utils.errors import DocumentNotFoundError
from app.utils.limiter import limiter

router = APIRouter(dependencies=[Depends(verify_api_key)])

embedding_service = EmbeddingService(settings)
vector_store = VectorStore(settings)


async def _ingest_request(payload: IngestRequest) -> dict[str, int]:
    chunk_ids: list[str] = []
    chunk_texts: list[str] = []
    chunk_metadatas: list[dict] = []

    for document in payload.documents:
        chunks = chunk_text(document.content)

        for index, chunk in enumerate(chunks):
            chunk_ids.append(f"{document.id}_chunk_{index}")
            chunk_texts.append(chunk)
            chunk_metadatas.append(
                {
                    **document.metadata,
                    "parent_id": document.id,
                    "chunk_index": index,
                }
            )

    embeddings = await embedding_service.embed(chunk_texts)

    vector_store.add(
        documents=chunk_texts,
        embeddings=embeddings,
        ids=chunk_ids,
        metadatas=chunk_metadatas,
    )

    return {"ingested": len(payload.documents), "chunks": len(chunk_texts)}


@router.post("/documents")
@limiter.limit("10/minute")
async def ingest_documents(request: Request, payload: IngestRequest) -> dict[str, int]:
    return await _ingest_request(payload)


@router.post("/documents/batch")
@limiter.limit("10/minute")
async def ingest_documents_batch(
    request: Request, payload: list[IngestRequest]
) -> dict[str, int]:
    results = await asyncio.gather(*(_ingest_request(item) for item in payload))

    return {
        "ingested": sum(result["ingested"] for result in results),
        "chunks": sum(result["chunks"] for result in results),
    }


@router.delete("/documents/{doc_id}")
async def delete_document(doc_id: str) -> dict[str, int]:
    deleted_chunks = vector_store.delete(doc_id)

    if deleted_chunks == 0:
        raise DocumentNotFoundError(doc_id)

    return {"deleted_chunks": deleted_chunks}
