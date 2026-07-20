from typing import Any

import chromadb
from chromadb.api.models.Collection import Collection

from app.config import Settings


class VectorStore:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = chromadb.PersistentClient(path=settings.chroma_path)
        self.collection: Collection = self.client.get_or_create_collection(
            name="documents"
        )

    def add(
        self,
        documents: list[str],
        embeddings: list[list[float]],
        ids: list[str],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        self.collection.add(
            documents=documents,
            embeddings=embeddings,
            ids=ids,
            metadatas=metadatas,
        )

    def query(self, embedding: list[float], n_results: int = 5) -> list[dict[str, Any]]:
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=n_results,
        )

        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        matches: list[dict[str, Any]] = []
        for i in range(len(ids)):
            matches.append(
                {
                    "id": ids[i],
                    "document": documents[i],
                    "metadata": metadatas[i],
                    "distance": distances[i],
                }
            )

        return matches

    def delete(self, doc_id: str) -> int:
        existing = self.collection.get(where={"parent_id": doc_id})
        ids = existing.get("ids", [])

        if not ids:
            return 0

        self.collection.delete(ids=ids)
        return len(ids)
