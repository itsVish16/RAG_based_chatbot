import logging
import uuid
from typing import List, Optional
from dataclasses import dataclass
from qdrant_client import QdrantClient
from qdrant_client.http import models
from app.config.settings import settings

logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    chunk_id: str
    score: float
    rank: int

class VectorStore:
    """
    Metadata-Filtered Vector Store using Qdrant.
    Provides multi-tenancy by filtering with `user_id`.
    """

    def __init__(self):
        if not settings.QDRANT_URL or not settings.QDRANT_API_KEY:
            raise ValueError("QDRANT_URL or QDRANT_API_KEY is not set in settings/.env")

        self.client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY
        )
        self.collection_name = settings.QDRANT_COLLECTION_NAME

        # 1. Create collection if it doesn't exist
        try:
            if not self.client.collection_exists(self.collection_name):
                logger.info(f"Creating Qdrant collection: {self.collection_name}")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=settings.EMBEDDING_DIMENSION, 
                        distance=models.Distance.COSINE
                    )
                )
        except Exception as e:
            logger.warning(f"Could not verify/create collection, it may already exist. ({e})")

        logger.info(f"Connected to Qdrant collection: {self.collection_name}")
                # 2. Create Payload Index for user_id to enable filtering
        try:
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="user_id",
                field_schema=models.PayloadSchemaType.KEYWORD  # ⬅️ Speeds up & allows filtering
            )
            logger.info(f"Created/Verified Payload Index for 'user_id'")
        except Exception as e:
            # Safe to catch if index or collection doesn't exist yet/already there
            logger.info(f"Payload Index verification info: {e}")


    def add_embeddings(self, embeddings: List[List[float]], chunk_ids: List[str], user_id: str):
        """Add embeddings to Qdrant tagged with user_id to isolate tenants."""
        if len(embeddings) != len(chunk_ids):
            raise ValueError("Embeddings and chunk_ids must have the same size")

        points = []
        for embedding, chunk_id in zip(embeddings, chunk_ids):
            # Qdrant requires UUID or Int point ID. We generate it deterministically.
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))
            
            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={
                        "user_id": user_id, 
                        "chunk_id": chunk_id  # Save real text mapping for retrieval
                    }
                )
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        logger.info(f"Upserted {len(points)} points for {user_id}")

    def search(self, query_embeddings: List[float], user_id: str, top_k: int = 5) -> List[SearchResult]:
        """Query Qdrant using user_id metadata filtering."""
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embeddings,  # ⬅️ Changed query_vector to query
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="user_id",
                        match=models.MatchValue(value=user_id)
                    )
                ]
            ),
            limit=top_k,
            with_payload=True
        )

        results = []
        # Modern Qdrant `.points` list or directly iterable:
        points = response.points if hasattr(response, 'points') else response
        
        for rank, match in enumerate(points):
            results.append(SearchResult(
                chunk_id=match.payload.get("chunk_id", ""),
                score=match.score,
                rank=rank + 1
            ))

        return results


    def total_vectors(self) -> int:
        """Returns total vectors in the collection."""
        try:
            stats = self.client.get_collection(self.collection_name)
            return stats.points_count
        except:
            return 0

# Singleton
vector_store = VectorStore()
