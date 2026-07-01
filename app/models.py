from pydantic import BaseModel, Field


class Document(BaseModel):
    id: str
    content: str
    metadata: dict = Field(default_factory=dict)


class IngestRequest(BaseModel):
    documents: list[Document]


class SearchQuery(BaseModel):
    query: str
    top_k: int = 5


class SearchResult(BaseModel):
    document: Document
    score: float


class SearchResponse(BaseModel):
    results: list[SearchResult]
    query: str
