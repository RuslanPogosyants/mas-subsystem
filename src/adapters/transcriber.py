"""Transcription adapter: Protocol + in-process fake.

The real faster-whisper backend lives in `whisper_transcriber.py` and is selected
when `transcriber_backend="whisper"`. Tests and the default pipeline use
`FakeTranscriberAdapter` to keep CI lean and free of heavyweight ML dependencies.
"""

from __future__ import annotations

from typing import Protocol

from src.core.schemas import TextChunk


class TranscriberAdapter(Protocol):
    """Async interface for any speech-to-text backend."""

    async def transcribe(self, *, file_path: str, language: str = "ru") -> list[TextChunk]:
        """Return ordered TextChunks for the audio file."""
        ...


class FakeTranscriberAdapter:
    """Returns pre-built TextChunks; used by tests and the demo pipeline."""

    def __init__(self, chunks: list[TextChunk] | None = None) -> None:
        self._chunks = chunks

    async def transcribe(self, *, file_path: str, language: str = "ru") -> list[TextChunk]:
        if self._chunks is not None:
            return self._chunks
        return [
            TextChunk(
                id=f"chunk-{file_path}-0",
                task_id="",
                document_id="",
                source_type="audio",
                content=f"[fake transcript of {file_path} in {language}]",
                chunk_index=0,
                confidence=0.95,
            )
        ]
