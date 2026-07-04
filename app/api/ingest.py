from fastapi import APIRouter

from app.config import settings
from app.core.embeddings import EmbeddingService
from app.db.vector_store import VectorStore
from app.models import IngestRequest

router = APIRouter()

embedding_service = EmbeddingService(settings)
vector_store = VectorStore(settings)


@router.post("/documents")
async def ingest_documents(request: IngestRequest) -> dict[str, int]:
    texts = [document.content for document in request.documents]
    embeddings = await embedding_service.embed(texts)

    vector_store.add(
        documents=texts,
        embeddings=embeddings,
        ids=[document.id for document in request.documents],
        metadatas=[document.metadata for document in request.documents],
    )

    return {"ingested": len(request.documents)}
