from fastapi import APIRouter

from app.config import settings
from app.core.embeddings import EmbeddingService
from app.db.vector_store import VectorStore
from app.models import Document, SearchQuery, SearchResponse, SearchResult

router = APIRouter()

embedding_service = EmbeddingService(settings)
vector_store = VectorStore(settings)


@router.post("/search")
async def search(query: SearchQuery) -> SearchResponse:
    embeddings = await embedding_service.embed([query.query])
    matches = vector_store.query(embedding=embeddings[0], n_results=query.top_k)

    results = [
        SearchResult(
            document=Document(
                id=match["id"],
                content=match["document"],
                metadata=match["metadata"] or {},
            ),
            score=match["distance"],
        )
        for match in matches
    ]

    return SearchResponse(results=results, query=query.query)
