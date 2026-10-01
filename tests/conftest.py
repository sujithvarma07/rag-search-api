import os

os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")
os.environ.setdefault("API_KEY", "test-x-api-key")
os.environ.setdefault("CHROMA_PATH", "./test_chroma_db")

import pytest

from app.config import Settings


@pytest.fixture
def mock_settings() -> Settings:
    return Settings(
        openai_api_key="test-api-key",
        api_key="test-x-api-key",
        chroma_path="./test_chroma_db",
        embedding_model="text-embedding-3-small",
        top_k=5,
    )
