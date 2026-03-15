from pydantic import BaseModel, Field
from typing import List, Optional


class UploadResponse(BaseModel):
    document_id: str
    filename: str
    total_chunks: int
    status: str
    message: str


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    top_k: int    = Field(default=5, ge=1, le=20)


class SourceChunkResponse(BaseModel):
    chunk_id: str
    text: str
    rrf_score: float


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceChunkResponse]
    document_id: str
    total_chunks_retrieved: int
