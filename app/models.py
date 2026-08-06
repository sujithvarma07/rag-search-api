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
    min_score: float | None = None
    generate_answer: bool = False
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=10, ge=1, le=100)


class SearchResult(BaseModel):
    document: Document
    score: float


class SearchResponse(BaseModel):
    results: list[SearchResult]
    query: str
    answer: str | None = None
    page: int = 1
    page_size: int = 10
    total: int = 0
