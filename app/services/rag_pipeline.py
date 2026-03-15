import logging
import uuid
from typing import List, Optional
from dataclasses import dataclass, field

from openai import OpenAI

from app.services.document_parser import document_parser
from app.services.chunker import text_chunker, Chunk
from app.services.embedding_service import embedding_service
from app.services.vector_store import vector_store
from app.services.bm25_store import bm25_store
from app.services.hybrid_search import hybrid_search, HybridResult
from app.config.settings import settings

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────
# Response models
# ──────────────────────────────────────────

@dataclass
class SourceChunk:
    """A retrieved chunk returned alongside the answer"""
    chunk_id: str
    text: str
    rrf_score: float


@dataclass
class RAGResponse:
    """Final response returned to the API layer"""
    answer: str
    sources: List[SourceChunk]
    document_id: str
    total_chunks_retrieved: int


# ──────────────────────────────────────────
# In-memory chunk text store
# (replaces PostgreSQL for now)
# ──────────────────────────────────────────

# chunk_id → chunk text
# e.g. {"doc_abc_0": "first chunk text...", "doc_abc_1": "second chunk..."}
_chunk_store: dict[str, str] = {}


class RAGPipeline:
    """
    Orchestrates the full RAG flow:
      Upload:  file bytes → parse → chunk → embed → index (FAISS + BM25)
      Query:   question → hybrid search → context assembly → LLM answer
    """

    def __init__(self):
        # LM Studio local endpoint — OpenAI-compatible, no real API key needed
        self.llm = OpenAI(
            base_url=settings.LM_STUDIO_BASE_URL,
            api_key=settings.LM_STUDIO_API_KEY,
        )

    # ══════════════════════════════════════════
    # INGESTION PIPELINE
    # ══════════════════════════════════════════

    def ingest_document(
        self,
        file_bytes: bytes,
        filename: str,
        content_type: str,
    ) -> dict:
        """
        Full ingestion pipeline: file → indexed and ready to query.

        Args:
            file_bytes:   Raw file content
            filename:     Original filename (e.g. "report.pdf")
            content_type: MIME type (e.g. "application/pdf")

        Returns:
            dict with document_id and chunk count
        """
        document_id = str(uuid.uuid4())
        logger.info(f"Starting ingestion for '{filename}' → doc_id={document_id}")

        # Step 1: Parse → extract text
        text = document_parser.parse(file_bytes, content_type, filename)
        logger.info(f"Parsed: {len(text)} characters")

        # Step 2: Chunk → split into overlapping token chunks
        chunks: List[Chunk] = text_chunker.chunk_text(text, document_id)
        logger.info(f"Chunked: {len(chunks)} chunks")

        # Step 3: Store chunk texts in memory (chunk_id → text)
        for chunk in chunks:
            _chunk_store[chunk.chunk_id] = chunk.text

        # Step 4: Embed all chunks in one batched API call
        chunk_texts = [chunk.text for chunk in chunks]
        chunk_ids   = [chunk.chunk_id for chunk in chunks]
        embeddings  = embedding_service.embed_batch(chunk_texts)
        logger.info(f"Embedded: {len(embeddings)} vectors")

        # Step 5: Add to FAISS index
        vector_store.add_embeddings(embeddings, chunk_ids)

        # Step 6: Add to BM25 index
        bm25_store.add_documents(chunk_texts, chunk_ids)

        logger.info(
            f"Ingestion complete: doc_id={document_id}, "
            f"{len(chunks)} chunks indexed in FAISS + BM25"
        )

        return {
            "document_id": document_id,
            "filename": filename,
            "total_chunks": len(chunks),
            "status": "indexed",
        }

    # ══════════════════════════════════════════
    # QUERY PIPELINE
    # ══════════════════════════════════════════

    def query(
        self,
        question: str,
        top_k: int = settings.TOP_K,
    ) -> RAGResponse:
        """
        Full query pipeline: question → hybrid search → GPT-4o answer.

        Args:
            question: User's natural language question
            top_k:    Number of chunks to retrieve and pass to LLM

        Returns:
            RAGResponse with answer + source chunks
        """
        if not question.strip():
            raise ValueError("Question cannot be empty")

        if vector_store.total_vectors() == 0:
            raise ValueError("No documents indexed yet. Please upload a document first.")

        logger.info(f"Query: '{question}'")

        # Step 1: Hybrid search → top-k chunk IDs
        hybrid_results: List[HybridResult] = hybrid_search.search(
            question, top_k=top_k
        )

        if not hybrid_results:
            return RAGResponse(
                answer="I couldn't find relevant information in the uploaded documents.",
                sources=[],
                document_id="",
                total_chunks_retrieved=0,
            )

        # Step 2: Fetch chunk texts from in-memory store
        source_chunks: List[SourceChunk] = []
        context_parts: List[str] = []

        for result in hybrid_results:
            text = _chunk_store.get(result.chunk_id, "")
            if text:
                source_chunks.append(SourceChunk(
                    chunk_id=result.chunk_id,
                    text=text,
                    rrf_score=result.rrf_score,
                ))
                context_parts.append(f"[Source {len(context_parts) + 1}]\n{text}")

        # Step 3: Assemble context
        context = "\n\n---\n\n".join(context_parts)

        # Step 4: LLM answer generation
        answer = self._generate_answer(question, context)

        # Extract document_id from first chunk_id (format: "{doc_id}_{index}")
        first_chunk_id = hybrid_results[0].chunk_id
        document_id = "_".join(first_chunk_id.split("_")[:-1])

        logger.info(f"Query answered with {len(source_chunks)} source chunks")

        return RAGResponse(
            answer=answer,
            sources=source_chunks,
            document_id=document_id,
            total_chunks_retrieved=len(source_chunks),
        )

    # ──────────────────────────────────────────
    # Private
    # ──────────────────────────────────────────

    def _generate_answer(self, question: str, context: str) -> str:
        """Call GPT-4o with retrieved context to generate the final answer."""

        system_prompt = """You are a precise and helpful assistant that answers questions 
based strictly on the provided document context.

Rules:
- Answer only from the context provided below
- If the answer is not in the context, say "I don't have enough information in the uploaded documents to answer this."
- Be concise and accurate
- Cite source numbers when referencing specific information (e.g. [Source 1])"""

        user_prompt = f"""Context from documents:
{context}

Question: {question}

Answer:"""

        try:
            response = self.llm.chat.completions.create(
                model=settings.LLM_MODEL,       # gpt-4o
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                temperature=0.1,                # low temp for factual accuracy
                max_tokens=1024,
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            raise


# Singleton
rag_pipeline = RAGPipeline()
