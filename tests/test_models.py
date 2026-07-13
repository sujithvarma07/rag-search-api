import pytest
from pydantic import ValidationError

from app.models import (
    Document,
    IngestRequest,
    SearchQuery,
    SearchResponse,
    SearchResult,
)


def test_document_requires_id_and_content() -> None:
    document = Document(id="doc-1", content="hello world")

    assert document.id == "doc-1"
    assert document.content == "hello world"
    assert document.metadata == {}


def test_document_metadata_defaults_are_independent() -> None:
    first = Document(id="doc-1", content="a")
    second = Document(id="doc-2", content="b")

    first.metadata["source"] = "test"

    assert first.metadata == {"source": "test"}
    assert second.metadata == {}


def test_document_missing_required_fields_raises() -> None:
    with pytest.raises(ValidationError):
        Document(content="missing id")

    with pytest.raises(ValidationError):
        Document(id="doc-1")


def test_ingest_request_accepts_empty_document_list() -> None:
    request = IngestRequest(documents=[])

    assert request.documents == []


def test_ingest_request_holds_multiple_documents() -> None:
    request = IngestRequest(
        documents=[
            Document(id="doc-1", content="first"),
            Document(id="doc-2", content="second", metadata={"lang": "en"}),
        ]
    )

    assert len(request.documents) == 2
    assert request.documents[1].metadata == {"lang": "en"}


def test_search_query_default_top_k() -> None:
    query = SearchQuery(query="what is rag?")

    assert query.top_k == 5


def test_search_query_custom_top_k() -> None:
    query = SearchQuery(query="what is rag?", top_k=10)

    assert query.top_k == 10


def test_search_query_requires_query_field() -> None:
    with pytest.raises(ValidationError):
        SearchQuery()


def test_search_result_holds_document_and_score() -> None:
    document = Document(id="doc-1", content="hello")
    result = SearchResult(document=document, score=0.87)

    assert result.document.id == "doc-1"
    assert result.score == 0.87


def test_search_result_score_must_be_numeric() -> None:
    document = Document(id="doc-1", content="hello")

    with pytest.raises(ValidationError):
        SearchResult(document=document, score="not-a-number")


def test_search_response_defaults_and_structure() -> None:
    document = Document(id="doc-1", content="hello")
    result = SearchResult(document=document, score=0.5)
    response = SearchResponse(results=[result], query="hello")

    assert response.query == "hello"
    assert len(response.results) == 1
    assert response.results[0].score == 0.5


def test_search_response_accepts_no_results() -> None:
    response = SearchResponse(results=[], query="no matches")

    assert response.results == []
