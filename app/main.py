from fastapi import FastAPI

app = FastAPI(
    title="RAG Search API",
    description="Semantic document search using retrieval-augmented generation",
    version="0.1.0",
)


@app.get("/health")
def health_check():
    return {"status": "ok"}
