import logging
from io import BytesIO
from pathlib import Path
import tempfile
import os

logger = logging.getLogger(__name__)


class DocumentParser:
    """
    Universal document parser using Docling (IBM Research).
    Handles: PDF, DOCX, PPTX, Images, HTML — with full OCR.
    """

    SUPPORTED_TYPES = {
        "application/pdf":                                                          "pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation": "pptx",
        "text/plain":                                                               "txt",
        "text/html":                                                                "html",
        "image/png":                                                                "image",
        "image/jpeg":                                                               "image",
        "image/jpg":                                                                "image",
        "image/webp":                                                               "image",
    }

    def __init__(self):
        pass  # No heavy model initialized on start

    def parse(self, file_bytes: bytes, content_type: str, filename: str = "") -> str:
        """
        Parse any supported document and return clean text.
        Falls back to simple parsers for TXT/HTML.
        """
        doc_type = self.SUPPORTED_TYPES.get(content_type)

        # Extension-based fallback
        if not doc_type and filename:
            ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
            ext_map = {
                "pdf": "pdf", "docx": "docx", "pptx": "pptx",
                "txt": "txt", "html": "html", "htm": "html",
                "png": "image", "jpg": "image", "jpeg": "image", "webp": "image",
            }
            doc_type = ext_map.get(ext)

        if not doc_type:
            raise ValueError(
                f"Unsupported file type: {content_type}. "
                f"Supported: PDF, DOCX, PPTX, TXT, HTML, PNG, JPG"
            )

        from app.config.settings import settings

        logger.info(f"Parsing {doc_type.upper()}: {filename or 'unnamed'} ({len(file_bytes)} bytes)")

        # Simple parsers for text/html (no Docling needed)
        if doc_type == "txt":
            return self._parse_txt(file_bytes)
        if doc_type == "html":
            return self._parse_html(file_bytes)

        # ── Step 1: Lightweight PDF Parsing ─────────────────────────────
        if doc_type == "pdf":
            logger.info("Attempting lightweight PDF parsing with PyMuPDF")
            text = self._parse_with_pymupdf(file_bytes)
            if text.strip():
                return text
            logger.warning("PyMuPDF extracted 0 text (scanned doc).")

            # Try Mistral Vision OCR before Docling (which takes heavy RAM)
            if settings.MISTRAL_API_KEY:
                logger.info("Attempting Mistral Vision OCR for scanned PDF")
                mistral_text = self._parse_pdf_with_mistral(file_bytes)
                if mistral_text:
                    return mistral_text

        # ── Step 1b: Image Parsing with Mistral Vision ──────────────────
        if doc_type == "image":
            if settings.MISTRAL_API_KEY:
                logger.info("Attempting Mistral Vision OCR for Image")
                return self._parse_image_with_mistral(file_bytes)
            else:
                raise ValueError("MISTRAL_API_KEY is required for image parsing")

        # ── Step 2: Fallback ─────────────────────────────────────────────
        raise ValueError(f"Could not extract text from {filename or 'document'}. Fallback parser disabled.")

    # ──────────────────────────────────────────
    # Lightweight PDF (PyMuPDF)
    # ──────────────────────────────────────────

    def _parse_with_pymupdf(self, file_bytes: bytes) -> str:
        """Extract text quickly using PyMuPDF (fitz). Low RAM, perfect for Vercel."""
        try:
            import fitz
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            text_parts = []
            for i in range(len(doc)):
                page_text = doc[i].get_text().strip()
                if page_text:
                    text_parts.append(f"[Page {i+1}]\n{page_text}")
            doc.close()
            return "\n\n".join(text_parts)
        except ImportError:
            logger.warning("pymupdf not installed, install to speed up cloud parsing")
            return ""
        except Exception as e:
            logger.error(f"PyMuPDF parsing failed: {e}")
            return ""



    # ──────────────────────────────────────────
    # Mistral Vision API (Cloud OCR)
    # ──────────────────────────────────────────

    def _parse_image_with_mistral(self, file_bytes: bytes) -> str:
        """Use Mistral Vision model to extract text from a single image."""
        import base64
        from mistralai.client.sdk import Mistral
        from app.config.settings import settings

        client = Mistral(api_key=settings.MISTRAL_API_KEY)
        encoded = base64.b64encode(file_bytes).decode("utf-8")

        try:
            response = client.chat.complete(
                model="pixtral-12b",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Extract all text from this image exactly as written. Output only the text."},
                            {"type": "image_url", "image_url": f"data:image/jpeg;base64,{encoded}"}
                        ]
                    }
                ]
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Mistral Image OCR failed: {e}")
            return ""

    def _parse_pdf_with_mistral(self, file_bytes: bytes) -> str:
        """Convert PDF pages to images and run Mistral Vision on them."""
        try:
            import fitz
            import io
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            text_parts = []

            for i in range(len(doc)):
                page = doc[i]
                pix = page.get_pixmap()
                img_bytes = pix.tobytes("jpg")
                page_text = self._parse_image_with_mistral(img_bytes)
                if page_text:
                    text_parts.append(f"[Page {i+1}]\n{page_text}")

            doc.close()
            return "\n\n".join(text_parts)
        except Exception as e:
            logger.error(f"Mistral PDF OCR failed: {e}")
            return ""

    # ──────────────────────────────────────────
    # Simple parsers
    # ──────────────────────────────────────────

    def _parse_txt(self, file_bytes: bytes) -> str:
        try:
            text = file_bytes.decode("utf-8")
        except UnicodeDecodeError:
            text = file_bytes.decode("latin-1")
        text = text.strip()
        if not text:
            raise ValueError("TXT file is empty")
        logger.info(f"TXT parsed: {len(text)} chars")
        return text

    def _parse_html(self, file_bytes: bytes) -> str:
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(file_bytes, "html.parser")
            text = soup.get_text(separator="\n").strip()
        except ImportError:
            # Fallback without beautifulsoup
            text = file_bytes.decode("utf-8", errors="ignore").strip()
        if not text:
            raise ValueError("HTML file is empty")
        logger.info(f"HTML parsed: {len(text)} chars")
        return text


# Singleton
document_parser = DocumentParser()
