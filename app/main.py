from fastapi import FastAPI

from app.api import ingest
from app.config import Settings

settings = Settings()

app = FastAPI(
    title="RAG Search API",
    description="Semantic document search using retrieval-augmented generation",
    version="0.1.0",
)

app.include_router(ingest.router)


@app.get("/health")
def health_check():
    return {"status": "ok"}
