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
