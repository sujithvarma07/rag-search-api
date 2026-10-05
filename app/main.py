from fastapi import FastAPI
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app.api import ingest, search
from app.config import Settings
from app.utils.errors import register_exception_handlers
from app.utils.limiter import limiter
from app.utils.logging import RequestLoggingMiddleware

settings = Settings()

app = FastAPI(
    title="RAG Search API",
    description="Semantic document search using retrieval-augmented generation",
    version="0.1.0",
    contact={
        "name": "Sujith Kakarlapudi",
        "email": "sujithvarma07@gmail.com",
        "url": "https://github.com/sujithvarma07/rag-search-api",
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    },
    openapi_tags=[
        {
            "name": "documents",
            "description": "Ingest, chunk, embed, and delete documents in the vector store.",
        },
        {
            "name": "search",
            "description": "Semantic search over ingested documents with optional answer generation.",
        },
        {
            "name": "health",
            "description": "Service and dependency health reporting.",
        },
    ],
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(RequestLoggingMiddleware)

register_exception_handlers(app)

app.include_router(ingest.router)
app.include_router(search.router)


@app.get(
    "/health",
    tags=["health"],
    summary="Check service health",
    description=(
        "Reports overall API status, whether the ChromaDB collection is reachable, "
        "and the running API version."
    ),
    response_description="Service status, chroma status, and api version",
)
def health_check() -> dict[str, str]:
    try:
        ingest.vector_store.collection.count()
        chroma_status = "ok"
    except Exception:
        chroma_status = "unavailable"

    return {
        "status": "ok",
        "chroma": chroma_status,
        "version": app.version,
    }
