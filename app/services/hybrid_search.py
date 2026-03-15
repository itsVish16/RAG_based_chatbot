import logging
from typing import List, Dict, Optional
from dataclasses import dataclass, field

from app.services.vector_store import vector_store, SearchResult
from app.services.bm25_store import bm25_store
from app.services.embedding_service import embedding_service
from app.config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class HybridResult:
    """
    Final fused result after RRF.
    Contains chunk_id + combined score + source scores for transparency.
    """
    chunk_id: str
    rrf_score: float
    semantic_score: Optional[float] = None
    semantic_rank: Optional[int] = None
    bm25_score: Optional[float] = None
    bm25_rank: Optional[int] = None


class HybridSearch:
    """
    Combines FAISS (semantic) + BM25 (keyword) results using
    Reciprocal Rank Fusion (RRF) for 15-20% accuracy boost over either alone.

    RRF formula: score(d) = Σ 1 / (k + rank(d))
    where k=60 is a smoothing constant (standard for web search).
    """

    def __init__(self, rrf_k: int = 60):
        """
        Args:
            rrf_k: RRF smoothing constant.
                   Higher k → more weight to lower ranks.
                   60 is the standard value.
        """
        self.rrf_k = rrf_k

    def search(
        self,
        query: str,
        top_k: int = settings.TOP_K,
        semantic_weight: float = 0.6,    # slightly favour semantic for RAG
        bm25_weight: float = 0.4,
    ) -> List[HybridResult]:
        """
        Run hybrid search and return fused results.

        Args:
            query:           Raw query string
            top_k:           Number of final results
            semantic_weight: Weight for FAISS results (0-1)
            bm25_weight:     Weight for BM25 results (0-1)
        Returns:
            List of HybridResult sorted by RRF score descending
        """
        fetch_k = top_k * 3    # fetch more candidates for better fusion

        # 1. Semantic search (FAISS)
        query_embedding = embedding_service.embed_text(query)
        semantic_results: List[SearchResult] = vector_store.search(
            query_embedding, top_k=fetch_k
        )

        # 2. Keyword search (BM25)
        bm25_results: List[SearchResult] = bm25_store.search(
            query, top_k=fetch_k
        )

        # 3. RRF Fusion
        fused = self._rrf_fuse(
            semantic_results, bm25_results,
            semantic_weight, bm25_weight
        )

        logger.info(
            f"Hybrid search: {len(semantic_results)} semantic + "
            f"{len(bm25_results)} BM25 → {len(fused[:top_k])} fused results"
        )

        return fused[:top_k]

    # ──────────────────────────────────────────
    # Private
    # ──────────────────────────────────────────

    def _rrf_fuse(
        self,
        semantic_results: List[SearchResult],
        bm25_results: List[SearchResult],
        semantic_weight: float,
        bm25_weight: float,
    ) -> List[HybridResult]:
        """Apply Reciprocal Rank Fusion across both result sets."""

        # Build lookup maps: chunk_id → SearchResult
        semantic_map: Dict[str, SearchResult] = {r.chunk_id: r for r in semantic_results}
        bm25_map: Dict[str, SearchResult]     = {r.chunk_id: r for r in bm25_results}

        # All unique chunks seen across both searches
        all_chunk_ids = set(semantic_map.keys()) | set(bm25_map.keys())

        results: List[HybridResult] = []

        for chunk_id in all_chunk_ids:
            rrf_score = 0.0
            semantic_score = semantic_rank = bm25_score = bm25_rank = None

            if chunk_id in semantic_map:
                r = semantic_map[chunk_id]
                rrf_score     += semantic_weight * (1.0 / (self.rrf_k + r.rank))
                semantic_score = r.score
                semantic_rank  = r.rank

            if chunk_id in bm25_map:
                r = bm25_map[chunk_id]
                rrf_score  += bm25_weight * (1.0 / (self.rrf_k + r.rank))
                bm25_score = r.score
                bm25_rank  = r.rank

            results.append(HybridResult(
                chunk_id=chunk_id,
                rrf_score=rrf_score,
                semantic_score=semantic_score,
                semantic_rank=semantic_rank,
                bm25_score=bm25_score,
                bm25_rank=bm25_rank,
            ))

        # Sort by RRF score descending
        results.sort(key=lambda x: x.rrf_score, reverse=True)
        return results


# Singleton
hybrid_search = HybridSearch()
