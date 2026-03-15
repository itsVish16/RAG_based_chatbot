import logging
from fastapi import APIRouter, UploadFile, File, HTTPException

from app.services.rag_pipeline import rag_pipeline
from app.schema.rag_schema import UploadResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["Documents"])

# Allowed MIME types
ALLOWED_TYPES = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "text/plain",
}

MAX_FILE_SIZE = 50 * 1024 * 1024   # 50 MB


@router.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a PDF, DOCX, or TXT file.
    Parses, chunks, embeds, and indexes it for RAG queries.
    """

    # Validate file type
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type: {file.content_type}. "
                   f"Allowed: PDF, DOCX, TXT"
        )

    # Read file bytes
    file_bytes = await file.read()

    # Validate file size
    if len(file_bytes) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max size: 50MB"
        )

    if len(file_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    try:
        logger.info(f"Uploading: {file.filename} ({len(file_bytes)} bytes)")

        result = rag_pipeline.ingest_document(
            file_bytes=file_bytes,
            filename=file.filename or "unnamed",
            content_type=file.content_type,
        )

        return UploadResponse(
            document_id=result["document_id"],
            filename=result["filename"],
            total_chunks=result["total_chunks"],
            status=result["status"],
            message=f"Successfully indexed {result['total_chunks']} chunks. Ready to query!"
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")
