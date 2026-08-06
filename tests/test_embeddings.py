from unittest.mock import AsyncMock, MagicMock

import pytest
from openai import RateLimitError

from app.config import Settings
from app.core.embeddings import EmbeddingService


def _make_embedding_response(vectors: list[list[float]]) -> MagicMock:
    response = MagicMock()
    response.data = [MagicMock(embedding=vector) for vector in vectors]
    return response


def _rate_limit_error() -> RateLimitError:
    response = MagicMock()
    response.status_code = 429
    return RateLimitError("rate limited", response=response, body=None)


@pytest.mark.asyncio
async def test_embed_returns_embedding_vectors_for_each_text(mock_settings: Settings) -> None:
    service = EmbeddingService(mock_settings)
    service.client = MagicMock()
    service.client.embeddings.create = AsyncMock(
        return_value=_make_embedding_response([[0.1, 0.2], [0.3, 0.4]])
    )

    result = await service.embed(["hello", "world"])

    assert result == [[0.1, 0.2], [0.3, 0.4]]
    service.client.embeddings.create.assert_awaited_once_with(
        model=mock_settings.embedding_model,
        input=["hello", "world"],
    )


@pytest.mark.asyncio
async def test_embed_returns_empty_list_for_empty_input(mock_settings: Settings) -> None:
    service = EmbeddingService(mock_settings)
    service.client = MagicMock()
    service.client.embeddings.create = AsyncMock(return_value=_make_embedding_response([]))

    result = await service.embed([])

    assert result == []


@pytest.mark.asyncio
async def test_embed_retries_on_rate_limit_then_succeeds(mock_settings: Settings) -> None:
    service = EmbeddingService(mock_settings)
    service.client = MagicMock()
    service.client.embeddings.create = AsyncMock(
        side_effect=[_rate_limit_error(), _make_embedding_response([[0.5]])]
    )

    result = await service.embed(["retry me"])

    assert result == [[0.5]]
    assert service.client.embeddings.create.await_count == 2


@pytest.mark.asyncio
async def test_embed_raises_after_exhausting_retries(mock_settings: Settings) -> None:
    service = EmbeddingService(mock_settings)
    service.client = MagicMock()
    service.client.embeddings.create = AsyncMock(side_effect=_rate_limit_error())

    with pytest.raises(RateLimitError):
        await service.embed(["always fails"])

    assert service.client.embeddings.create.await_count == 3


@pytest.mark.asyncio
async def test_embed_propagates_non_rate_limit_errors_without_retrying(mock_settings: Settings) -> None:
    service = EmbeddingService(mock_settings)
    service.client = MagicMock()
    service.client.embeddings.create = AsyncMock(side_effect=ValueError("boom"))

    with pytest.raises(ValueError, match="boom"):
        await service.embed(["broken"])

    assert service.client.embeddings.create.await_count == 1
