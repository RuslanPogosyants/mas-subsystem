"""Real sentence-transformers EmbeddingAdapter (used for the live demo only).

`sentence-transformers` is an optional ML dependency imported lazily; the model
loads once and encoding runs in a worker thread to avoid stalling the event loop.
Embeddings are L2-normalised so the agent's cosine equals a dot product.
"""

from __future__ import annotations

import asyncio


class SentenceTransformerEmbeddingAdapter:
    """EmbeddingAdapter backed by a sentence-transformers model."""

    def __init__(self, model: str = "intfloat/multilingual-e5-base") -> None:
        from sentence_transformers import SentenceTransformer  # lazy: optional ml dependency

        self._model = SentenceTransformer(model)

    async def encode(self, texts: list[str]) -> list[list[float]]:
        vectors = await asyncio.to_thread(self._model.encode, texts, normalize_embeddings=True)
        return [[float(value) for value in vector] for vector in vectors]
