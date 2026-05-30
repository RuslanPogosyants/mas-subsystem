"""RecommenderAgent (F6): summary + terms -> related Citations via embeddings.

A query is built from the summary introduction (first 200 chars) plus the top
terms, embedded through an EmbeddingAdapter, and scored by cosine similarity
against a precomputed corpus of (metadata, embedding) entries. The top-N within
the requested year filters become Citations with relevance_score = cosine. An
empty corpus or an empty query is refused. The corpus is loaded lazily so CI
(without numpy / corpus files) never imports the ML stack.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from src.agents.base import AgentBase
from src.core.schemas import Citation, Operation

if TYPE_CHECKING:
    from src.adapters.embedding import EmbeddingAdapter
    from src.core.bus import RedisStreamBus
    from src.core.messages import Message

_DEFAULT_N: Final[int] = 3
_INTRO_CHARS: Final[int] = 200
_TOP_TERMS: Final[int] = 5


@dataclass(frozen=True, slots=True)
class CorpusEntry:
    """One corpus paper: bibliographic metadata + its embedding vector."""

    title: str
    authors: str | None
    year: int | None
    url: str | None
    embedding: tuple[float, ...]


def load_corpus(path: str = "corpus") -> list[CorpusEntry]:
    """Load corpus entries from <path>/papers.jsonl + papers.npy, or [] if absent.

    numpy is imported lazily and only when both files exist, so CI (no corpus,
    no numpy) returns an empty corpus without touching the ML stack.
    """
    base = Path(path)
    jsonl, npy = base / "papers.jsonl", base / "papers.npy"
    if not (jsonl.exists() and npy.exists()):
        return []
    import numpy  # lazy: optional ml dependency, only on the production path

    embeddings = numpy.load(npy)
    entries: list[CorpusEntry] = []
    for line, row in zip(jsonl.read_text(encoding="utf-8").splitlines(), embeddings, strict=False):
        meta = json.loads(line)
        entries.append(
            CorpusEntry(
                title=str(meta["title"]),
                authors=meta.get("authors"),
                year=meta.get("year"),
                url=meta.get("url"),
                embedding=tuple(float(value) for value in row),
            )
        )
    return entries


class RecommenderAgent(AgentBase):
    name = "RecommenderAgent"

    def __init__(self, *, bus: RedisStreamBus, embedding: EmbeddingAdapter, corpus: list[CorpusEntry]) -> None:
        super().__init__(
            bus=bus,
            channel="agent.recommender",
            group="worker-recommender",
            operation=Operation.F6_RECOMMEND,
        )
        self._embedding = embedding
        self._corpus = corpus

    async def handle(self, message: Message) -> Message | None:
        if not self._corpus:
            return self._refuse(message, reason="corpus not loaded")
        query = _build_query(message.content.get("summary"), message.content.get("terms"))
        if not query:
            return self._refuse(message, reason="no summary or terms for recommendation")
        raw_n = message.content.get("n", _DEFAULT_N)
        top_n = raw_n if isinstance(raw_n, int) and not isinstance(raw_n, bool) and raw_n > 0 else _DEFAULT_N
        filters = message.content.get("filters")
        year_min, year_max = _year_bounds(filters if isinstance(filters, dict) else {})
        query_vector = (await self._embedding.encode([query]))[0]
        if len(query_vector) != len(self._corpus[0].embedding):
            return self._refuse(message, reason="embedding dimension mismatch")
        scored = [
            (entry, _cosine(query_vector, entry.embedding))
            for entry in self._corpus
            if _year_ok(entry.year, year_min, year_max)
        ]
        scored.sort(key=lambda pair: (-pair[1], pair[0].title))
        citations = [
            Citation(
                title=entry.title,
                authors=entry.authors,
                year=entry.year,
                url=entry.url,
                relevance_score=round(score, 4),
            )
            for entry, score in scored[:top_n]
        ]
        return self._inform(message, content={"citations": [citation.model_dump() for citation in citations]})


def _build_query(summary: object, terms: object) -> str:
    intro = ""
    if isinstance(summary, dict):
        sections = summary.get("sections")
        if isinstance(sections, list):
            for section in sections:
                if isinstance(section, dict) and section.get("type") == "introduction":
                    intro = str(section.get("text", ""))[:_INTRO_CHARS]
                    break
    top_terms: list[str] = []
    if isinstance(terms, list):
        for term in terms[:_TOP_TERMS]:
            if isinstance(term, dict):
                top_terms.append(str(term.get("term", term.get("lemma", ""))))
    return " ".join(part for part in [intro, *top_terms] if part).strip()


def _year_bounds(filters: dict[str, Any]) -> tuple[int | None, int | None]:
    year_min = filters.get("year_min")
    year_max = filters.get("year_max")
    return (year_min if isinstance(year_min, int) else None, year_max if isinstance(year_max, int) else None)


def _year_ok(year: int | None, year_min: int | None, year_max: int | None) -> bool:
    if year is None:
        return True
    if year_min is not None and year < year_min:
        return False
    return not (year_max is not None and year > year_max)


def _cosine(left: list[float], right: tuple[float, ...]) -> float:
    dot = sum(a * b for a, b in zip(left, right, strict=False))
    norm_left = math.sqrt(sum(a * a for a in left))
    norm_right = math.sqrt(sum(b * b for b in right))
    if norm_left == 0.0 or norm_right == 0.0:
        return 0.0
    return dot / (norm_left * norm_right)
