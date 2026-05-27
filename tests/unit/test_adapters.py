"""Unit tests for fake ML adapters."""

from __future__ import annotations

from src.adapters.ocr import FakeOcrAdapter
from src.adapters.transcriber import FakeTranscriberAdapter
from src.core.schemas import TextChunk


class TestFakeTranscriberAdapter:
    async def test_returns_one_chunk_by_default(self) -> None:
        adapter = FakeTranscriberAdapter()
        chunks = await adapter.transcribe(file_path="/x.mp3")
        assert len(chunks) == 1
        assert chunks[0].source_type == "audio"
        assert "/x.mp3" in chunks[0].content

    async def test_returns_injected_chunks(self) -> None:
        canned = [
            TextChunk(
                id="c1",
                task_id="",
                document_id="",
                source_type="audio",
                content="hello",
                chunk_index=0,
            )
        ]
        adapter = FakeTranscriberAdapter(chunks=canned)
        result = await adapter.transcribe(file_path="/y.mp3")
        assert result == canned


class TestFakeOcrAdapter:
    async def test_pdf_yields_extracted_source(self) -> None:
        adapter = FakeOcrAdapter()
        chunks = await adapter.extract(file_path="/x.pdf", document_type="pdf")
        assert chunks[0].source_type == "pdf_extracted"

    async def test_image_yields_image_source(self) -> None:
        adapter = FakeOcrAdapter()
        chunks = await adapter.extract(file_path="/x.jpg", document_type="image")
        assert chunks[0].source_type == "image"
