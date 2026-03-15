import logging
import hashlib
from typing import List

from sentence_transformers import SentenceTransformer

from app.config.settings import settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Embedding service using Mistral API (`mistral-embed`).
    """

    def __init__(self):
        self._cache: dict[str, List[float]] = {}
        self.dimension = settings.EMBEDDING_DIMENSION
        self.client = None

    def _get_client(self):
        if self.client is None:
            from mistralai.client.sdk import Mistral
            if not settings.MISTRAL_API_KEY:
                raise ValueError("MISTRAL_API_KEY is not set in settings/.env")
            self.client = Mistral(api_key=settings.MISTRAL_API_KEY)
        return self.client

    def embed_text(self, text: str) -> List[float]:
        """Embed a single string."""
        cache_key = self._hash(text)
        if cache_key in self._cache:
            return self._cache[cache_key]

        client = self._get_client()
        response = client.embeddings.create(
            inputs=[text],
            model=settings.MISTRAL_EMBEDDING_MODEL
        )
        embedding = response.data[0].embedding

        self._cache[cache_key] = embedding
        return embedding

    def embed_batch(self, texts: List[str], batch_size: int = 64) -> List[List[float]]:
        """Embed multiple texts with caching support."""
        results: List[Optional[List[float]]] = [None] * len(texts)
        uncached_indices: List[int] = []
        uncached_texts: List[str] = []

        for i, text in enumerate(texts):
            key = self._hash(text)
            if key in self._cache:
                results[i] = self._cache[key]
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)

        logger.info(
            f"Embedding batch: {len(texts)} texts "
            f"({len(texts) - len(uncached_texts)} cached, "
            f"{len(uncached_texts)} new)"
        )

        if uncached_texts:
            client = self._get_client()
            response = client.embeddings.create(
                inputs=uncached_texts,
                model=settings.MISTRAL_EMBEDDING_MODEL
            )
            embeddings = [d.embedding for d in response.data]

            for idx, embedding in zip(uncached_indices, embeddings):
                results[idx] = embedding
                self._cache[self._hash(texts[idx])] = embedding

        return results  # type: ignore

    def get_dimension(self) -> int:
        return self.dimension

    def clear_cache(self):
        self._cache.clear()

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()


# Singleton
embedding_service = EmbeddingService()
