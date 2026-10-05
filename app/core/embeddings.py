from openai import AsyncOpenAI, RateLimitError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.config import Settings
from app.utils.cache import LRUCache


class EmbeddingService:
    def __init__(self, settings: Settings, cache_size: int = 1000) -> None:
        self.settings = settings
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.cache: LRUCache[str, list[float]] = LRUCache(max_size=cache_size)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        resolved: dict[str, list[float]] = {}
        missing: list[str] = []
        for text in dict.fromkeys(texts):
            cached = self.cache.get(text)
            if cached is None:
                missing.append(text)
            else:
                resolved[text] = cached

        if missing:
            vectors = await self._create_embeddings(missing)
            for text, vector in zip(missing, vectors):
                self.cache.set(text, vector)
                resolved[text] = vector

        return [resolved[text] for text in texts]

    @retry(
        retry=retry_if_exception_type(RateLimitError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def _create_embeddings(self, texts: list[str]) -> list[list[float]]:
        response = await self.client.embeddings.create(
            model=self.settings.embedding_model,
            input=texts,
        )
        return [item.embedding for item in response.data]
