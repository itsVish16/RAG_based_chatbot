import logging
from dataclasses import dataclass
from typing import List

import tiktoken

from app.config.settings import settings
logger = logging.getLogger(__name__)

@dataclass
class Chunk:

    chunk_id : str
    text:str
    chunk_index : int
    token_count: int
    start_char: int
    end_char: int

class TextChunker:

    def __init__(
        self,
        chunk_size: int = settings.CHUNK_SIZE,
        chunk_overlap: int = settings.CHUNK_OVERLAP,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.tokenizer = tiktoken.get_encoding("cl100k_base")


    def chunk_text(self, text: str, document_id: str) -> List[Chunk]:
        """
        Split text into overlapping token-based chunks.
        Args:
            text:        Full extracted document text
            document_id: Used to build unique chunk_ids
        Returns:
            List[Chunk]: Ordered list of chunks
        """
        if not text or not text.strip():
            raise ValueError("Cannot chunk empty text")
        tokens = self.tokenizer.encode(text)
        total_tokens = len(tokens)
        logger.info(
            f"Chunking doc '{document_id}': {total_tokens} tokens → "
            f"chunk_size={self.chunk_size}, overlap={self.chunk_overlap}"
        )
        # Document short enough to fit in a single chunk
        if total_tokens <= self.chunk_size:
            return [Chunk(
                chunk_id=f"{document_id}_0",
                text=text.strip(),
                chunk_index=0,
                token_count=total_tokens,
                start_char=0,
                end_char=len(text),
            )]
        chunks: List[Chunk] = []
        start_token = 0
        chunk_index = 0
        step = self.chunk_size - self.chunk_overlap  # how far to advance each time
        while start_token < total_tokens:
            end_token = min(start_token + self.chunk_size, total_tokens)
            chunk_tokens = tokens[start_token:end_token]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            # Approximate character positions
            prefix_text = self.tokenizer.decode(tokens[:start_token])
            start_char = len(prefix_text)
            end_char = start_char + len(chunk_text)
            chunks.append(Chunk(
                chunk_id=f"{document_id}_{chunk_index}",
                text=chunk_text.strip(),
                chunk_index=chunk_index,
                token_count=len(chunk_tokens),
                start_char=start_char,
                end_char=end_char,
            ))
            # Stop if we've covered everything
            if end_token == total_tokens:
                break
            start_token += step
            chunk_index += 1
        logger.info(f"Created {len(chunks)} chunks for doc '{document_id}'")
        return chunks
# Singleton
text_chunker = TextChunker()