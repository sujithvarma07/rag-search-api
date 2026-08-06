from openai import AsyncOpenAI

from app.config import Settings

ANSWER_SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions using only the "
    "provided context. If the context does not contain enough information "
    "to answer, say so instead of guessing."
)


class LLMService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)

    async def generate_answer(self, query: str, context_chunks: list[str]) -> str:
        context = "\n\n".join(context_chunks)
        user_prompt = f"Context:\n{context}\n\nQuestion: {query}"

        response = await self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content or ""
