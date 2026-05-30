"""Embedding adapter: protocol + deterministic in-process fake.

The adapter turns texts into (ideally normalised) embedding vectors. The
RecommenderAgent owns the corpus, cosine ranking, filtering, and Citation mapping,
so any backend (sentence-transformers, a stub) plugs in behind this async
interface. The real sentence-transformers adapter lives in
sentence_transformer.py; tests and CI use FakeEmbeddingAdapter.
"""

from __future__ import annotations

import math
from typing import Protocol

_FAKE_DIM = 16


class EmbeddingAdapter(Protocol):
    """Async interface for any text-embedding backend."""

    async def encode(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input text."""
        ...


class FakeEmbeddingAdapter:
    """Deterministic EmbeddingAdapter for tests and the demo pipeline.

    With explicit `vectors` it returns them in order (one per input text). With
    none, it derives a normalised character-histogram vector per text — stable
    across runs, so cosine ranking is reproducible.
    """

    def __init__(self, vectors: list[list[float]] | None = None) -> None:
        self._vectors = vectors

    async def encode(self, texts: list[str]) -> list[list[float]]:
        if self._vectors is not None:
            return [self._vectors[min(index, len(self._vectors) - 1)] for index, _ in enumerate(texts)]
        return [_char_histogram(text) for text in texts]


def _char_histogram(text: str, dim: int = _FAKE_DIM) -> list[float]:
    vector = [0.0] * dim
    for char in text.lower():
        vector[ord(char) % dim] += 1.0
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]
