from fastapi import APIRouter

from app.config import settings
from app.core.embeddings import EmbeddingService
from app.db.vector_store import VectorStore
from app.models import IngestRequest
from app.utils.chunking import chunk_text
from app.utils.errors import DocumentNotFoundError

router = APIRouter()

embedding_service = EmbeddingService(settings)
vector_store = VectorStore(settings)


@router.post("/documents")
async def ingest_documents(request: IngestRequest) -> dict[str, int]:
    chunk_ids: list[str] = []
    chunk_texts: list[str] = []
    chunk_metadatas: list[dict] = []

    for document in request.documents:
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

    return {"ingested": len(request.documents), "chunks": len(chunk_texts)}


@router.delete("/documents/{doc_id}")
async def delete_document(doc_id: str) -> dict[str, int]:
    deleted_chunks = vector_store.delete(doc_id)

    if deleted_chunks == 0:
        raise DocumentNotFoundError(doc_id)

    return {"deleted_chunks": deleted_chunks}
