import logging
from fastapi import APIRouter, HTTPException

from app.services.rag_pipeline import rag_pipeline
from app.schema.rag_schema import QueryRequest, QueryResponse, SourceChunkResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/rag", tags=["RAG"])


@router.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    Ask a question against all indexed documents.
    Returns an answer with source chunks used to generate it.
    """
    try:
        logger.info(f"Query received: '{request.question}'")

        result = rag_pipeline.query(
            question=request.question,
            top_k=request.top_k,
        )

        return QueryResponse(
            answer=result.answer,
            sources=[
                SourceChunkResponse(
                    chunk_id=s.chunk_id,
                    text=s.text,
                    rrf_score=round(s.rrf_score, 6),
                )
                for s in result.sources
            ],
            document_id=result.document_id,
            total_chunks_retrieved=result.total_chunks_retrieved,
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@router.get("/status")
async def index_status():
    """Health check — shows how many chunks are indexed."""
    from app.services.vector_store import vector_store
    from app.services.bm25_store import bm25_store

    return {
        "faiss_vectors": vector_store.total_vectors(),
        "bm25_documents": bm25_store.total_documents(),
        "ready": vector_store.total_vectors() > 0,
    }
