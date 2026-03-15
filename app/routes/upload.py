import logging
from fastapi import APIRouter, UploadFile, File, HTTPException

from app.services.s3_service import s3_service


from app.services.rag_pipeline import rag_pipeline
from app.schema.rag_schema import UploadResponse
from app.utils.security import get_current_user
from app.models.user import User
from fastapi import Depends


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["Documents"])

# Allowed MIME types
ALLOWED_TYPES = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "text/plain",
    "text/html",
    "image/png",
    "image/jpeg",
    "image/jpg",
    "image/webp",
}

MAX_FILE_SIZE = 50 * 1024 * 1024   # 50 MB


@router.post("/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...), 
    current_user: User = Depends(get_current_user)  # ⬅️ Added auth gate
):
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
        # 1. Store backup/proof on AWS S3 🔒
        s3_url = s3_service.upload_file_bytes(
            file_bytes=file_bytes,
            filename=file.filename or "unnamed",
            user_id=str(current_user.id),
            content_type=file.content_type
        )
        if s3_url:
            logger.info(f"File backed up to S3: {s3_url}")

        # 2. Add to RAG Index (Always run this!)
        result = rag_pipeline.ingest_document(
            file_bytes=file_bytes,
            filename=file.filename or "unnamed",
            content_type=file.content_type,
            user_id=str(current_user.id)
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
