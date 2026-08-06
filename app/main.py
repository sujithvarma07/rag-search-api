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
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(RequestLoggingMiddleware)

register_exception_handlers(app)

app.include_router(ingest.router)
app.include_router(search.router)


@app.get("/health")
def health_check():
    return {"status": "ok"}
