import logging
import pickle
from pathlib import Path
from typing import List, Optional

from rank_bm25 import BM25Okapi

from app.services.vector_store import SearchResult    # reuse same dataclass

logger = logging.getLogger(__name__)


class BM25Store:
    """
    In-memory BM25 index for exact keyword matching.
    Great for: names, dates, technical terms, exact phrases.
    BM25 doesn't support incremental updates, so we always rebuild on add.
    """

    def __init__(self):
        self.bm25: Optional[BM25Okapi] = None
        self.chunk_ids: List[str] = []
        self._corpus: List[List[str]] = []        # tokenized corpus kept for rebuilds
        self._persist_dir = Path("data/bm25")
        self._persist_dir.mkdir(parents=True, exist_ok=True)

    # ──────────────────────────────────────────
    # Indexing
    # ──────────────────────────────────────────

    def add_documents(self, texts: List[str], chunk_ids: List[str]):
        """
        Add documents and rebuild BM25 index.

        Args:
            texts:     Raw chunk texts
            chunk_ids: Corresponding chunk IDs (same order)
        """
        if len(texts) != len(chunk_ids):
            raise ValueError("texts and chunk_ids must have the same length")

        new_tokenized = [self._tokenize(t) for t in texts]

        self._corpus.extend(new_tokenized)
        self.chunk_ids.extend(chunk_ids)
        self.bm25 = BM25Okapi(self._corpus)        # always full rebuild

        logger.info(f"BM25 index rebuilt: {len(self.chunk_ids)} documents total")

    def reset(self):
        """Wipe the BM25 index."""
        self.bm25 = None
        self.chunk_ids = []
        self._corpus = []
        logger.info("BM25 index reset")

    # ──────────────────────────────────────────
    # Search
    # ──────────────────────────────────────────

    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """
        Find top-k most keyword-relevant chunks.

        Args:
            query: Raw query string
            top_k: Number of results
        Returns:
            List of SearchResult sorted by score descending
        """
        if self.bm25 is None or not self.chunk_ids:
            logger.warning("BM25 index is empty — no documents indexed yet")
            return []

        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        # Sort indices by score descending, take top_k
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:top_k]

        results = []
        for rank, idx in enumerate(top_indices):
            if scores[idx] <= 0:                   # skip zero-score (no keyword match)
                continue
            results.append(SearchResult(
                chunk_id=self.chunk_ids[idx],
                score=float(scores[idx]),
                rank=rank + 1,
            ))

        return results

    def total_documents(self) -> int:
        return len(self.chunk_ids)

    # ──────────────────────────────────────────
    # Private
    # ──────────────────────────────────────────

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Lowercase whitespace tokenizer. Can add stemming later."""
        return text.lower().split()

    # ──────────────────────────────────────────
    # Persistence
    # ──────────────────────────────────────────

    def save(self, name: str = "default"):
        with open(self._persist_dir / f"{name}_bm25.pkl", "wb") as f:
            pickle.dump({"corpus": self._corpus, "chunk_ids": self.chunk_ids}, f)
        logger.info(f"BM25 index saved: {name}")

    def load(self, name: str = "default") -> bool:
        path = self._persist_dir / f"{name}_bm25.pkl"
        if not path.exists():
            return False
        with open(path, "rb") as f:
            data = pickle.load(f)
        self._corpus   = data["corpus"]
        self.chunk_ids = data["chunk_ids"]
        self.bm25      = BM25Okapi(self._corpus)
        logger.info(f"BM25 index loaded: {name} ({len(self.chunk_ids)} docs)")
        return True


# Singleton
bm25_store = BM25Store()
