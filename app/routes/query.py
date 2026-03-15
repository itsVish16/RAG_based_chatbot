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


@router.get("/documents")
async def list_documents():
    """List all indexed documents and their chunk counts."""
    return {"documents": rag_pipeline.list_documents()}


@router.post("/summarize")
async def summarize_document(document_id: str):
    """
    Summarize an entire document using Map-Reduce.
    Use the document_id returned from /upload.

    Note: This calls the LLM once per chunk + once more for synthesis.
    Latency = N_chunks × LLM_latency. For a 10-chunk doc ≈ 10-15 seconds.
    """
    try:
        logger.info(f"Summarization requested for document: {document_id}")
        summary = rag_pipeline.summarize_document(document_id)
        return {
            "document_id": document_id,
            "summary": summary,
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Summarization failed: {e}")
        raise HTTPException(status_code=500, detail=f"Summarization failed: {str(e)}")
