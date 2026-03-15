import faiss
import numpy as np
import logging
import pickle
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    chunk_id : str
    score : float
    rank : int

class VectorStore:

    def __init__(self, dimension: int = 1536):
        self.dimension = dimension
        self.index : Optional[faiss.IndexFlatIP] = None
        self.chunk_ids: List[str] = []
        self._persist_dir = Path("data/faiss")
        self._persist_dir.mkdir(parents = True, exist_ok = True)




    def add_embeddings(self, embeddings : List[List[float]], chunk_ids: List[str]):

        if len(embeddings) != len(chunk_ids):
            raise ValueError(f"embeddinds and chunk_ids must have the same size")

        vectors = np.array(embeddings, dtype = np.float32)
        faiss.normalize_L2(vectors)

        if self.index is None:
            self.index  = faiss.IndexFlatIP(self.dimension)
            logger.info(f"created new faiss index (dim = {self.dimension})")
        self.index.add(vectors)
        self.chunk_ids.extend(chunk_ids)

        logger.info(
            f"Added {len(chunk_ids)} vectors."
            f"Total in index: {self.index.ntotal}"
        )

    def reset(self):
        self.index = None
        self.chunk_ids = []
        logger.info("FAISS index reset")

    def search(self, query_embeddings: List[float], top_k: int = 5) -> List[SearchResult]:
        if self.index is None or self.index.ntotal == 0:
            logger.warning("FAISS index is empty ")
            return []

        query = np.array([query_embeddings], dtype = np.float32)
        faiss.normalize_L2(query)

        K = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query, K)

        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx == -1:
                continue
            results.append(SearchResult(
                chunk_id = self.chunk_ids[idx],
                score = float(score),
                rank = rank + 1,
            ))

        return results

    def total_vectors(self) -> int:
        return self.index.ntotal if self.index else 0

    def save(self, name: str = "default"):
        if self.index is None:
            raise ValueError("No index to save")
        faiss.write_index(self.index, str(self._persist_dir/ f"{name}.index"))
        with open(self._persist_dir / f"{name}_ids.pkl", "wb") as f:
            pickle.dump(self.chunk_ids, f)

        logger.info(f"FAISS index saved: {name}")
    
    def load(self, name: str = "default") -> bool:
        index_path = self._persist_dir / f"{name}.index"
        ids_path = self._persist_dir / f"{name}_ids.pkl"
        if not index_path.exists() or not ids_path.exists():
            return False
        self.index = faiss.read_index(str(index_path))
        with open(ids_path, "rb") as f:
            self.chunk_ids = pickle.load(f)
        logger.info(f"FAISS index loaded : {name} ({self.index.ntotal} vectors)")
        return True

from app.config.settings import settings
vector_store = VectorStore(dimension=settings.EMBEDDING_DIMENSION)