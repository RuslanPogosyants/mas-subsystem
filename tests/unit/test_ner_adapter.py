"""Unit tests for FakeNerAdapter."""

from __future__ import annotations

from src.adapters.ner import FakeNerAdapter, TermCandidate


async def test_default_derives_token_candidates() -> None:
    fake = FakeNerAdapter()
    result = await fake.extract("граф и дерево")
    assert [candidate.lemma for candidate in result] == ["граф", "дерево"]  # "и" dropped (len < 2)


async def test_explicit_candidates_returned_verbatim() -> None:
    candidates = [TermCandidate(text="Граф", lemma="граф", label="MISC")]
    fake = FakeNerAdapter(candidates=candidates)
    assert await fake.extract("anything") == candidates


async def test_extract_many_matches_per_text_extract() -> None:
    """extract_many must return the same results as calling extract per text."""
    fake = FakeNerAdapter()
    texts = ["граф и дерево", "алгоритм структура", "в"]
    many = await fake.extract_many(texts)
    per_text = [await fake.extract(t) for t in texts]
    assert many == per_text


async def test_extract_many_explicit_candidates() -> None:
    """With fixed candidates, extract_many returns them for every text."""
    candidates = [TermCandidate(text="Граф", lemma="граф", label="MISC")]
    fake = FakeNerAdapter(candidates=candidates)
    result = await fake.extract_many(["anything", "something"])
    assert result == [candidates, candidates]


async def test_extract_many_empty_list() -> None:
    fake = FakeNerAdapter()
    assert await fake.extract_many([]) == []
