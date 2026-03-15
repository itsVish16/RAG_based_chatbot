import logging
import uuid
from typing import List, Optional
from dataclasses import dataclass, field
import time
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

# document_id → list of chunk_ids (for whole-doc summarization)
_document_chunks: dict[str, List[str]] = {}


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

        # Track chunks per document for summarization
        _document_chunks[document_id] = [chunk.chunk_id for chunk in chunks]

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
        t0 = time.perf_counter()

        if not question.strip():
            raise ValueError("Question cannot be empty")
        if vector_store.total_vectors() == 0:
            raise ValueError("No documents indexed yet. Please upload a document first.")

        logger.info(f"Query: '{question}'")

        # Step 1: Embed query
        t1 = time.perf_counter()
        # (embedding happens inside hybrid_search.search → embed_text)

        # Step 2: Hybrid search → top-k chunk IDs
        hybrid_results: List[HybridResult] = hybrid_search.search(
            question, top_k=top_k
        )
        t2 = time.perf_counter()

        if not hybrid_results:
            return RAGResponse(
                answer="I couldn't find relevant information in the uploaded documents.",
                sources=[],
                document_id="",
                total_chunks_retrieved=0,
            )

        # Step 3: Fetch chunk texts from in-memory store
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

        t3 = time.perf_counter()

        # Step 4: Assemble context + LLM answer generation
        context = "\n\n---\n\n".join(context_parts)
        answer = self._generate_answer(question, context)
        t4 = time.perf_counter()

        # ── Latency report ──────────────────────────
        logger.info(
            f"⏱ LATENCY REPORT | "
            f"Embed+Search: {(t2 - t0)*1000:.1f}ms | "
            f"Fetch chunks: {(t3 - t2)*1000:.1f}ms | "
            f"LFM 2.5 (LLM): {(t4 - t3)*1000:.1f}ms | "
            f"Total: {(t4 - t0)*1000:.1f}ms"
        )

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
                model=settings.LLM_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                temperature=0.1,
                max_tokens=1024,
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            raise

    # ══════════════════════════════════════════
    # SUMMARIZATION (Map-Reduce)
    # ══════════════════════════════════════════

    def summarize_document(self, document_id: str) -> str:
        """
        Summarize an entire document using Map-Reduce pattern.

        Why not RAG for this?
          RAG retrieves top-K chunks by similarity → misses most of the document.
          Map-Reduce summarizes EVERY chunk → combines them → full summary.

        Steps:
          1. MAP:    Summarize each chunk individually
          2. REDUCE: Synthesize all chunk summaries into one final summary

        Args:
            document_id: The document ID returned from /upload
        Returns:
            Full document summary string
        """
        t0 = time.perf_counter()

        if document_id not in _document_chunks:
            raise ValueError(
                f"Document '{document_id}' not found. "
                f"Available: {list(_document_chunks.keys())}"
            )

        chunk_ids = _document_chunks[document_id]
        chunks = [_chunk_store[cid] for cid in chunk_ids if cid in _chunk_store]

        if not chunks:
            raise ValueError(f"No text found for document '{document_id}'")

        logger.info(f"Summarizing document {document_id}: {len(chunks)} chunks")

        # ── STEP 1: MAP — summarize each chunk ──
        if len(chunks) == 1:
            # Only one chunk → summarize directly
            chunk_summaries = [self._summarize_chunk(chunks[0], chunk_num=1, total=1)]
        else:
            chunk_summaries = []
            for i, chunk in enumerate(chunks):
                summary = self._summarize_chunk(chunk, chunk_num=i + 1, total=len(chunks))
                chunk_summaries.append(summary)

        t_map = time.perf_counter()

        # ── STEP 2: HIERARCHICAL REDUCE — batch to avoid context overflow ──
        if len(chunk_summaries) == 1:
            final_summary = chunk_summaries[0]
        else:
            final_summary = self._hierarchical_reduce(chunk_summaries)

        t_reduce = time.perf_counter()

        logger.info(
            f"⏱ SUMMARY LATENCY | "
            f"Map ({len(chunks)} chunks): {(t_map - t0)*1000:.1f}ms | "
            f"Reduce: {(t_reduce - t_map)*1000:.1f}ms | "
            f"Total: {(t_reduce - t0)*1000:.1f}ms"
        )

        return final_summary

    def _summarize_chunk(self, chunk_text: str, chunk_num: int, total: int) -> str:
        """MAP step: Summarize a single chunk."""
        try:
            response = self.llm.chat.completions.create(
                model=settings.LLM_MODEL,
                messages=[
                    {"role": "system", "content":
                        "You are a precise summarizer. Summarize the given text "
                        "concisely, preserving all key facts, names, and numbers."},
                    {"role": "user", "content":
                        f"Summarize this section ({chunk_num}/{total}):\n\n{chunk_text}"},
                ],
                temperature=0.1,
                max_tokens=300,
            )
            summary = response.choices[0].message.content.strip()
            logger.info(f"Chunk {chunk_num}/{total} summarized")
            return summary
        except Exception as e:
            logger.error(f"Chunk {chunk_num} summarization failed: {e}")
            raise

    def _hierarchical_reduce(self, summaries: list, batch_size: int = 8) -> str:
        """
        HIERARCHICAL REDUCE: Batches summaries to avoid context overflow.

        Example for 86 chunks with batch_size=8:
          Round 1: 86 summaries → 11 intermediate summaries (8 per batch)
          Round 2: 11 summaries → 2 intermediate summaries
          Round 3: 2 summaries  → 1 final summary

        This handles documents of any size regardless of context window.
        """
        current = summaries
        round_num = 1

        while len(current) > 1:
            logger.info(f"Reduce round {round_num}: {len(current)} summaries → batches of {batch_size}")
            next_level = []

            for i in range(0, len(current), batch_size):
                batch = current[i : i + batch_size]
                combined = "\n\n".join(
                    f"[Section {i + j + 1}]: {s}" for j, s in enumerate(batch)
                )
                reduced = self._reduce_batch(combined, round_num)
                next_level.append(reduced)

            current = next_level
            round_num += 1

        logger.info(f"Hierarchical reduce complete in {round_num - 1} round(s)")
        return current[0]

    def _reduce_batch(self, combined_summaries: str, round_num: int) -> str:
        """Reduce a single batch of summaries into one."""
        try:
            response = self.llm.chat.completions.create(
                model=settings.LLM_MODEL,
                messages=[
                    {"role": "system", "content":
                        "You are a precise summarizer. Synthesize the given section summaries "
                        "into one coherent summary. Preserve all key facts, names, numbers, "
                        "and conclusions."},
                    {"role": "user", "content":
                        f"Synthesize these summaries into one (reduce round {round_num}):\n\n"
                        f"{combined_summaries}"},
                ],
                temperature=0.1,
                max_tokens=512,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Reduce batch failed (round {round_num}): {e}")
            raise

    def list_documents(self) -> list:
        """Return all indexed document IDs and their chunk counts."""
        return [
            {"document_id": doc_id, "total_chunks": len(chunk_ids)}
            for doc_id, chunk_ids in _document_chunks.items()
        ]


# Singleton
rag_pipeline = RAGPipeline()
