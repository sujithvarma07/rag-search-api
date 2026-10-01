from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from app.api import ingest, search
from app.main import app

API_HEADERS = {"X-API-Key": "test-x-api-key"}


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_health_check_returns_ok(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "chroma" in body
    assert body["version"] == "0.1.0"


def test_ingest_documents_requires_api_key(client: TestClient) -> None:
    response = client.post("/documents", json={"documents": []})

    assert response.status_code == 422


def test_ingest_documents_embeds_and_stores_chunks(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ingest.embedding_service, "embed", AsyncMock(return_value=[[0.1, 0.2]]))
    monkeypatch.setattr(ingest.vector_store, "add", MagicMock())

    payload = {"documents": [{"id": "doc-1", "content": "hello world"}]}
    response = client.post("/documents", json=payload, headers=API_HEADERS)

    assert response.status_code == 200
    body = response.json()
    assert body["ingested"] == 1
    assert body["chunks"] == 1
    ingest.vector_store.add.assert_called_once()


def test_search_returns_ranked_results(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(search.embedding_service, "embed", AsyncMock(return_value=[[0.1, 0.2]]))
    monkeypatch.setattr(
        search.vector_store,
        "query",
        MagicMock(
            return_value=[
                {
                    "id": "doc-1_chunk_0",
                    "document": "hello world",
                    "metadata": {},
                    "distance": 0.92,
                }
            ]
        ),
    )

    response = client.post("/search", json={"query": "hello"}, headers=API_HEADERS)

    assert response.status_code == 200
    body = response.json()
    assert body["query"] == "hello"
    assert len(body["results"]) == 1
    assert body["results"][0]["document"]["id"] == "doc-1_chunk_0"


def test_search_filters_results_below_min_score(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(search.embedding_service, "embed", AsyncMock(return_value=[[0.1, 0.2]]))
    monkeypatch.setattr(
        search.vector_store,
        "query",
        MagicMock(
            return_value=[
                {"id": "doc-1", "document": "a", "metadata": {}, "distance": 0.9},
                {"id": "doc-2", "document": "b", "metadata": {}, "distance": 0.2},
            ]
        ),
    )

    response = client.post(
        "/search",
        json={"query": "hello", "min_score": 0.5},
        headers=API_HEADERS,
    )

    assert response.status_code == 200
    body = response.json()
    assert len(body["results"]) == 1
    assert body["results"][0]["document"]["id"] == "doc-1"
