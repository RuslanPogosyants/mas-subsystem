"""NER/term-extraction adapter: protocol + deterministic in-process fake.

The adapter does only the NLP-specific work — turning text into term candidates
(surface form, lemma, optional NER label). TerminologyAgent owns aggregation,
filtering, categorisation, and ranking, so any backend (spaCy, a stub) plugs in
behind the same async interface. The real spaCy adapter lives in spacy_ner.py;
tests and CI use FakeNerAdapter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Protocol

_MIN_TOKEN_LEN: Final[int] = 2


@dataclass(frozen=True, slots=True)
class TermCandidate:
    """One extracted candidate: surface form, lemma, and NER label ("" if none)."""

    text: str
    lemma: str
    label: str = ""


class NerAdapter(Protocol):
    """Async interface for any term-candidate extraction backend."""

    async def extract(self, text: str) -> list[TermCandidate]:
        """Return term candidates found in `text`."""
        ...


class FakeNerAdapter:
    """Deterministic NerAdapter for tests and the demo pipeline.

    With explicit `candidates` it returns them verbatim (ignoring the input). With
    none, it derives one candidate per whitespace token of length >= 2, lemma =
    lowercased token, no NER label — enough for a deterministic demo round-trip.
    """

    def __init__(self, candidates: list[TermCandidate] | None = None) -> None:
        self._candidates = candidates

    async def extract(self, text: str) -> list[TermCandidate]:
        if self._candidates is not None:
            return self._candidates
        return [
            TermCandidate(text=token, lemma=token.lower()) for token in text.split() if len(token) >= _MIN_TOKEN_LEN
        ]
