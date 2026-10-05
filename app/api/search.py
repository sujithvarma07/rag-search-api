from fastapi import APIRouter, Depends, Request

from app.config import settings
from app.core.embeddings import EmbeddingService
from app.core.llm import LLMService
from app.db.vector_store import VectorStore
from app.models import Document, SearchQuery, SearchResponse, SearchResult
from app.utils.auth import verify_api_key
from app.utils.limiter import limiter

router = APIRouter(tags=["search"], dependencies=[Depends(verify_api_key)])

embedding_service = EmbeddingService(settings)
vector_store = VectorStore(settings)
llm_service = LLMService(settings)


@router.post(
    "/search",
    summary="Semantic search",
    description=(
        "Embeds the query and retrieves the most similar chunks from ChromaDB. "
        "Supports metadata filtering, a minimum score threshold, offset-based "
        "pagination, and optional LLM answer generation from the returned chunks. "
        "Requires the `X-API-Key` header. Rate limited to 10 requests per minute."
    ),
    response_description="Paginated search results with optional generated answer",
)
@limiter.limit("10/minute")
async def search(request: Request, query: SearchQuery) -> SearchResponse:
    embeddings = await embedding_service.embed([query.query])

    # fetch enough candidates from the vector store to cover the requested page,
    # falling back to top_k when it already covers the requested window
    fetch_count = max(query.top_k, query.page * query.page_size)
    matches = vector_store.query(
        embedding=embeddings[0],
        n_results=fetch_count,
        where=query.metadata_filter,
    )

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

    if query.min_score is not None:
        results = [result for result in results if result.score >= query.min_score]

    total = len(results)
    offset = (query.page - 1) * query.page_size
    page_results = results[offset:offset + query.page_size]

    answer = None
    if query.generate_answer and page_results:
        context_chunks = [result.document.content for result in page_results]
        answer = await llm_service.generate_answer(query.query, context_chunks)

    return SearchResponse(
        results=page_results,
        query=query.query,
        answer=answer,
        page=query.page,
        page_size=query.page_size,
        total=total,
    )
