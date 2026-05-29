"""TerminologyAgent (F5): TextChunks -> ranked Term list via an NerAdapter.

Per-chunk candidate extraction keeps source-chunk attribution; candidates are
aggregated by lemma, filtered against a stop-list and a minimum length, scored by
frequency x idf (idf over the input chunks as a mini-corpus), and the top-N are
categorised (NER label first, then a domain dictionary, else "general"). Empty
chunk content is refused; content with no surviving terms informs an empty list.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Final

from src.agents.base import AgentBase
from src.core.schemas import Operation, Term

if TYPE_CHECKING:
    from src.adapters.ner import NerAdapter
    from src.core.bus import RedisStreamBus
    from src.core.messages import Message

_DATA_DIR: Final[Path] = Path(__file__).resolve().parents[2] / "data"
_DEFAULT_TOP_N: Final[int] = 10
_MIN_LEMMA_LEN: Final[int] = 2


def load_stopwords() -> set[str]:
    """Load lowercase stop-word lemmas from data/stopwords_ru.txt."""
    text = (_DATA_DIR / "stopwords_ru.txt").read_text(encoding="utf-8")
    return {line.strip().lower() for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")}


def load_domain_categories() -> dict[str, list[str]]:
    """Load the {category: [lemma, ...]} domain dictionary from data/."""
    raw = (_DATA_DIR / "domain_categories.json").read_text(encoding="utf-8")
    parsed: dict[str, list[str]] = json.loads(raw)
    return parsed


class TerminologyAgent(AgentBase):
    name = "TerminologyAgent"

    def __init__(
        self,
        *,
        bus: RedisStreamBus,
        ner: NerAdapter,
        stopwords: set[str] | None = None,
        domain_categories: dict[str, list[str]] | None = None,
        top_n_default: int = _DEFAULT_TOP_N,
    ) -> None:
        super().__init__(
            bus=bus,
            channel="agent.terminology",
            group="worker-terminology",
            operation=Operation.F5_TERMS,
        )
        self._ner = ner
        self._stopwords = stopwords if stopwords is not None else load_stopwords()
        self._domain = domain_categories if domain_categories is not None else load_domain_categories()
        self._top_n_default = top_n_default

    async def handle(self, message: Message) -> Message | None:
        raw_chunks = message.content.get("chunks")
        chunks = [chunk for chunk in raw_chunks if isinstance(chunk, dict)] if isinstance(raw_chunks, list) else []
        sources = [
            (str(chunk.get("id", "")), str(chunk.get("content", "")))
            for chunk in chunks
            if str(chunk.get("content", "")).strip()
        ]
        if not sources:
            return self._refuse(message, reason="no chunk content for terminology extraction")
        raw_top_n = message.content.get("top_n", self._top_n_default)
        top_n = raw_top_n if isinstance(raw_top_n, int) and raw_top_n > 0 else self._top_n_default
        terms = await self._extract_terms(sources, top_n)
        return self._inform(message, content={"terms": [term.model_dump() for term in terms]})

    async def _extract_terms(self, sources: list[tuple[str, str]], top_n: int) -> list[Term]:
        n_chunks = len(sources)
        frequency: dict[str, int] = {}
        chunk_ids: dict[str, set[str]] = {}
        first_seen: dict[str, tuple[str, str, str]] = {}  # lemma -> (surface, label, chunk_id)
        for chunk_id, content in sources:
            for candidate in await self._ner.extract(content):
                lemma = candidate.lemma.lower().strip()
                if len(lemma) < _MIN_LEMMA_LEN or lemma in self._stopwords:
                    continue
                frequency[lemma] = frequency.get(lemma, 0) + 1
                chunk_ids.setdefault(lemma, set()).add(chunk_id)
                if lemma not in first_seen:
                    first_seen[lemma] = (candidate.text, candidate.label, chunk_id)
        scored = sorted(
            frequency,
            key=lambda lemma: (-frequency[lemma] * math.log(1 + n_chunks / len(chunk_ids[lemma])), lemma),
        )
        terms: list[Term] = []
        for lemma in scored[:top_n]:
            surface, label, chunk_id = first_seen[lemma]
            terms.append(
                Term(
                    term=surface,
                    lemma=lemma,
                    frequency=frequency[lemma],
                    category=self._categorize(lemma, label),
                    context=None,
                    source_chunk_id=chunk_id,
                )
            )
        return terms

    def _categorize(self, lemma: str, label: str) -> str:
        if label:
            return label
        for category, lemmas in self._domain.items():
            if lemma in lemmas:
                return category
        return "general"
