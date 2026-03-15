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
        self._converter = None   # lazy load — Docling is heavy

    def _get_converter(self):
        """Lazy-load Docling converter with OCR pipeline enabled."""
        if self._converter is None:
            try:
                from docling.document_converter import DocumentConverter, PdfFormatOption
                from docling.datamodel.pipeline_options import PdfPipelineOptions
                from docling.datamodel.base_models import InputFormat

                # PdfPipelineOptions is the correct class (not PipelineOptions)
                pipeline_options = PdfPipelineOptions()
                pipeline_options.do_ocr = True               # OCR for scanned pages
                pipeline_options.do_table_structure = True   # proper table extraction

                self._converter = DocumentConverter(
                    format_options={
                        InputFormat.PDF: PdfFormatOption(
                            pipeline_options=pipeline_options
                        )
                    }
                )
                logger.info("Docling converter initialized with OCR (PdfPipelineOptions)")
            except Exception as e:
                logger.warning(f"Docling OCR setup failed ({e}) — falling back to basic converter")
                from docling.document_converter import DocumentConverter
                self._converter = DocumentConverter()
        return self._converter

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

        logger.info(f"Parsing {doc_type.upper()}: {filename or 'unnamed'} ({len(file_bytes)} bytes)")

        # Simple parsers for text/html (no Docling needed)
        if doc_type == "txt":
            return self._parse_txt(file_bytes)
        if doc_type == "html":
            return self._parse_html(file_bytes)

        # Docling handles everything else (PDF, DOCX, PPTX, Images)
        return self._parse_with_docling(file_bytes, filename or f"file.{doc_type}")

    # ──────────────────────────────────────────
    # Docling (universal — PDF, DOCX, PPTX, Images)
    # ──────────────────────────────────────────

    def _parse_with_docling(self, file_bytes: bytes, filename: str) -> str:
        """
        Use Docling for OCR-capable parsing.
        Writes to a temp file (Docling requires a file path).
        """
        converter = self._get_converter()

        # Docling needs a real file path, not bytes
        suffix = f".{filename.rsplit('.', 1)[-1]}" if "." in filename else ".pdf"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        try:
            result = converter.convert(tmp_path)
            # Export to Markdown — preserves headers, tables, lists
            markdown_text = result.document.export_to_markdown()

            if not markdown_text or not markdown_text.strip():
                raise ValueError(f"Docling extracted no text from {filename}")

            logger.info(
                f"Docling parsed '{filename}': {len(markdown_text)} chars"
            )
            return markdown_text

        finally:
            os.unlink(tmp_path)   # always clean up temp file

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
