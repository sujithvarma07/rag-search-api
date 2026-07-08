from fastapi import FastAPI

from app.api import ingest, search
from app.config import Settings
from app.utils.errors import register_exception_handlers

settings = Settings()

app = FastAPI(
    title="RAG Search API",
    description="Semantic document search using retrieval-augmented generation",
    version="0.1.0",
)

register_exception_handlers(app)

app.include_router(ingest.router)
app.include_router(search.router)


@app.get("/health")
def health_check():
    return {"status": "ok"}
