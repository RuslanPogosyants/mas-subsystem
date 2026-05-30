"""Pure chunk-mapping helpers of the real F1/F2 adapters (no model load)."""

from __future__ import annotations

from dataclasses import dataclass

from src.adapters.pymupdf_ocr import pages_to_chunks
from src.adapters.whisper_transcriber import segments_to_chunks


@dataclass
class _Seg:
    text: str
    start: float = 0.0
    end: float = 1.0


def test_segments_to_chunks_skips_whitespace_only_segments() -> None:
    # Whitespace-only segments are dropped; non-empty segments are accumulated.
    # With a small target_chars each non-empty segment flushes into its own chunk.
    segments = [
        _Seg(text="  hello  ", start=0.0, end=1.0),
        _Seg(text="   ", start=1.0, end=2.0),
        _Seg(text="world", start=2.0, end=3.0),
    ]
    chunks = segments_to_chunks(segments, target_chars=5)
    texts = [c.content for c in chunks]
    # "hello" is 5 chars — fills target exactly; "world" goes to the next chunk.
    assert "hello" in texts
    assert "world" in texts
    assert all(c.chunk_index == i for i, c in enumerate(chunks))


def test_segments_to_chunks_contiguous_chunk_index() -> None:
    segments = [_Seg(f"seg{i}") for i in range(4)]
    chunks = segments_to_chunks(segments, target_chars=10)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_pages_to_chunks_skips_blank_pages_and_strips() -> None:
    pages = ["page one text", "   ", "page three text"]
    chunks = pages_to_chunks(pages, source_type="pdf_extracted")
    assert [c.content for c in chunks] == ["page one text", "page three text"]
    assert [c.chunk_index for c in chunks] == [0, 1]


def test_pages_to_chunks_empty_sequence() -> None:
    assert pages_to_chunks([], source_type="image") == []
