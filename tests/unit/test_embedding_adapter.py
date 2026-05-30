"""Unit tests for FakeEmbeddingAdapter."""

from __future__ import annotations

import math

from src.adapters.embedding import FakeEmbeddingAdapter


async def test_explicit_vectors_returned_per_text() -> None:
    fake = FakeEmbeddingAdapter(vectors=[[1.0, 0.0], [0.0, 1.0]])
    assert await fake.encode(["a", "b"]) == [[1.0, 0.0], [0.0, 1.0]]


async def test_default_vector_is_deterministic_and_normalised() -> None:
    fake = FakeEmbeddingAdapter()
    first = await fake.encode(["граф дерево"])
    second = await fake.encode(["граф дерево"])
    assert first == second  # stable across calls
    assert math.isclose(math.sqrt(sum(value * value for value in first[0])), 1.0)
