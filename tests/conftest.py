import pytest

from app.config import Settings


@pytest.fixture
def mock_settings() -> Settings:
    return Settings(
        openai_api_key="test-api-key",
        chroma_path="./test_chroma_db",
        embedding_model="text-embedding-3-small",
        top_k=5,
    )
