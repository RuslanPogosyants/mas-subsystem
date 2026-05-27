"""OCR adapter: protocol + in-process fake.

Real EasyOCR + PyMuPDF wrappers are built in M3. M2 uses the fake to keep CI
free of heavyweight ML dependencies.
"""

from __future__ import annotations

from typing import Literal, Protocol

from src.core.schemas import TextChunk

ExtractedSourceType = Literal["pdf_extracted", "image"]


class OcrAdapter(Protocol):
    """Async interface for any OCR/PDF text extraction backend."""

    async def extract(self, *, file_path: str, document_type: str) -> list[TextChunk]:
        """Return ordered TextChunks for a PDF or image file.

        `document_type` is one of {"pdf", "image"}. Implementations choose the
        right pipeline (PDF text layer first, OCR fallback for scans/images).
        """
        ...


class FakeOcrAdapter:
    """Returns pre-built TextChunks; used by tests and the demo pipeline."""

    def __init__(self, chunks: list[TextChunk] | None = None) -> None:
        self._chunks = chunks

    async def extract(self, *, file_path: str, document_type: str) -> list[TextChunk]:
        if self._chunks is not None:
            return self._chunks
        source: ExtractedSourceType = "pdf_extracted" if document_type == "pdf" else "image"
        return [
            TextChunk(
                id=f"chunk-{file_path}-0",
                task_id="",
                document_id="",
                source_type=source,
                content=f"[fake text extracted from {document_type} {file_path}]",
                chunk_index=0,
                confidence=0.9,
            )
        ]
