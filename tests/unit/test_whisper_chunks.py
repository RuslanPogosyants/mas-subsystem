"""Unit tests for grouping faster-whisper segments into TextChunks."""

from __future__ import annotations

from dataclasses import dataclass

from src.adapters.whisper_transcriber import _enable_cuda_dll_search, segments_to_chunks


@dataclass
class _Seg:
    text: str
    start: float = 0.0
    end: float = 1.0
    avg_logprob: float = -0.2


def test_groups_segments_to_target_chars() -> None:
    segs = [_Seg("a" * 600), _Seg("b" * 600), _Seg("c" * 600)]
    chunks = segments_to_chunks(segs, target_chars=1000)
    # Greedy: chunk0=seg0 (600); seg1 would make 1201>1000 → flush,
    # chunk1=seg1 (600); seg2 would make 1201>1000 → flush, chunk2=seg2.
    assert len(chunks) == 3
    assert chunks[0].source_type == "audio"
    assert chunks[0].chunk_index == 0 and chunks[1].chunk_index == 1
    assert chunks[0].content.startswith("a")


def test_skips_empty_segments_and_strips() -> None:
    chunks = segments_to_chunks([_Seg("  hello  "), _Seg("   "), _Seg("world")], target_chars=10000)
    assert len(chunks) == 1
    assert chunks[0].content == "hello world"


def test_meta_carries_time_span() -> None:
    chunks = segments_to_chunks([_Seg("x", start=1.5, end=3.0)], target_chars=10)
    assert chunks[0].meta["start"] == 1.5
    assert chunks[0].meta["end"] == 3.0


def test_enable_cuda_dll_search_is_safe_and_idempotent() -> None:
    # No-op off Windows / when the NVIDIA wheels are absent; must never raise and
    # is safe to call repeatedly (it guards the lazy GPU model load).
    _enable_cuda_dll_search()
    _enable_cuda_dll_search()
