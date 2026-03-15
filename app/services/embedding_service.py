import logging
import hashlib
from typing import List

from sentence_transformers import SentenceTransformer

from app.config.settings import settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Local embedding using SentenceTransformers.
    No API calls, no quota issues — runs fully on your machine.
    Model: google/gemma-embedding-300m
    """

    def __init__(self):
        logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL}")
        # Downloads model on first run, then caches locally
        self.model = SentenceTransformer(settings.EMBEDDING_MODEL)
        self.dimension = settings.EMBEDDING_DIMENSION
        self._cache: dict[str, List[float]] = {}
        logger.info(f"Embedding model loaded. Dimension: {self.dimension}")

    def embed_text(self, text: str) -> List[float]:
        """Embed a single string. Checks in-memory cache first."""
        cache_key = self._hash(text)

        if cache_key in self._cache:
            logger.debug("Embedding cache hit")
            return self._cache[cache_key]

        embedding = self.model.encode(text, convert_to_numpy=True).tolist()
        self._cache[cache_key] = embedding
        return embedding

    def embed_batch(self, texts: List[str], batch_size: int = 64) -> List[List[float]]:
        """
        Embed multiple texts efficiently using local model.
        Much faster than API calls for batches.
        """
        results: List[List[float]] = [None] * len(texts)
        uncached_indices: List[int] = []
        uncached_texts: List[str] = []

        # Check cache
        for i, text in enumerate(texts):
            key = self._hash(text)
            if key in self._cache:
                results[i] = self._cache[key]
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)

        logger.info(
            f"Embedding {len(texts)} texts: "
            f"{len(texts) - len(uncached_texts)} cached, "
            f"{len(uncached_texts)} to embed locally"
        )

        if uncached_texts:
            # SentenceTransformers handles batching internally
            embeddings = self.model.encode(
                uncached_texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                show_progress_bar=len(uncached_texts) > 10,
            ).tolist()

            for idx, embedding in zip(uncached_indices, embeddings):
                results[idx] = embedding
                self._cache[self._hash(texts[idx])] = embedding

        return results

    def get_dimension(self) -> int:
        return self.dimension

    def clear_cache(self):
        self._cache.clear()

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()


# Singleton — model loads once at startup
embedding_service = EmbeddingService()
