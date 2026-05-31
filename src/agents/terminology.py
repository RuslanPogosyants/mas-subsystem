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
_MIN_SHARED_PREFIX: Final[int] = 4  # minimum shared-prefix length for near-duplicate last-token check
_MAX_INFLECTION_SUFFIX: Final[int] = 2  # maximum extra trailing characters in a prefix-relationship inflection

# Russian vowels and soft/hard signs that can appear as inflection endings.
# Merge is allowed only when BOTH differing final characters belong to this set,
# preventing over-merging of genuinely distinct words (e.g. «график»/«графит»).
_RU_INFLECTION_ENDINGS: Final[frozenset[str]] = frozenset(["а", "я", "о", "ё", "е", "у", "ю", "и", "ы", "э", "ь", "ъ"])


_DOTTED_INITIAL_LEN: Final[int] = 2  # e.g. «и.» → len 2, single char + period


def _is_noise_lemma(lemma: str) -> bool:
    """Return True when *lemma* looks like an initials artifact.

    A token is treated as an initial only when it is a single character
    immediately followed by a period (e.g. «и.», «А.»).  Bare single-letter
    tokens without a dot (e.g. «p» in «p значение», «t» in «t тест») are
    legitimate domain abbreviations and must NOT be dropped.
    """
    return any(len(token) == _DOTTED_INITIAL_LEN and token.endswith(".") for token in lemma.split())


def _rule1_same_length(last_a: str, last_b: str) -> bool:
    """Rule 1: last tokens identical length, differ only in one inflection-ending char."""
    shared_prefix = last_a[:-1]
    return (
        shared_prefix == last_b[:-1]
        and len(shared_prefix) >= _MIN_SHARED_PREFIX
        and last_a[-1] in _RU_INFLECTION_ENDINGS
        and last_b[-1] in _RU_INFLECTION_ENDINGS
    )


def _rule2_prefix_relationship(last_a: str, last_b: str) -> bool:
    """Rule 2: one last token is a prefix of the other; extra chars are all inflection endings."""
    shorter, longer = (last_a, last_b) if len(last_a) < len(last_b) else (last_b, last_a)
    extra_len = len(longer) - len(shorter)
    return (
        extra_len <= _MAX_INFLECTION_SUFFIX
        and len(shorter) >= _MIN_SHARED_PREFIX
        and longer.startswith(shorter)
        and all(ch in _RU_INFLECTION_ENDINGS for ch in longer[len(shorter) :])
    )


def _are_near_duplicate_lemmas(a: str, b: str) -> bool:
    """Return True when *a* and *b* should be treated as the same term.

    Two rules are applied in order; either is sufficient:

    1. **Single-character final-char substitution**: same token count, all
       non-final tokens identical, last tokens of equal length differ only in
       their final character, that character is a Russian inflection ending in
       both, and the shared leading prefix is at least ``_MIN_SHARED_PREFIX``
       characters long.
       Example: «двойной кавычка» / «двойной кавычки».

    2. **Suffix addition**: same token count, all non-final tokens identical,
       and the two last tokens are in a *prefix relationship* — one equals the
       other plus 1–``_MAX_INFLECTION_SUFFIX`` extra trailing characters, the
       shorter token is at least ``_MIN_SHARED_PREFIX`` characters long, and
       every extra trailing character is a Russian inflection ending.
       Example: «массив» / «массива» (extra «а» ∈ endings).

    Deliberately excluded: «тип» / «типаж» (extra «ж» ∉ endings);
    «код» / «кодер» (extra «р» ∉ endings); «график» / «графит» (neither is
    a prefix of the other and final chars к/т ∉ endings).
    """
    tokens_a = a.split()
    tokens_b = b.split()
    if len(tokens_a) != len(tokens_b) or tokens_a[:-1] != tokens_b[:-1]:
        return False
    last_a, last_b = tokens_a[-1], tokens_b[-1]
    if last_a == last_b:
        return True
    if len(last_a) == len(last_b):
        return _rule1_same_length(last_a, last_b)
    return _rule2_prefix_relationship(last_a, last_b)


_CandidateMaps = tuple[dict[str, int], dict[str, set[str]], dict[str, tuple[str, str, str]]]


def _merge_near_duplicates(
    frequency: dict[str, int],
    chunk_ids: dict[str, set[str]],
    first_seen: dict[str, tuple[str, str, str]],
) -> _CandidateMaps:
    """Group near-duplicate lemmas and return merged frequency/chunk_id/first_seen maps.

    Each group is reduced to a single representative: the member with the highest
    frequency (tie-break: lexicographically smaller lemma).  The emitted surface,
    category, and chunk_id all come from that representative's first_seen entry.
    """
    # 1. Union-find: map every lemma to a group key (the first-encountered member).
    group_key: dict[str, str] = {}
    all_lemmas = list(frequency)
    for i, lemma_a in enumerate(all_lemmas):
        if lemma_a in group_key:
            continue
        group_key[lemma_a] = lemma_a
        for lemma_b in all_lemmas[i + 1 :]:
            if lemma_b not in group_key and _are_near_duplicate_lemmas(lemma_a, lemma_b):
                group_key[lemma_b] = lemma_a

    # 2. Collect members per group.
    groups: dict[str, list[str]] = {}
    for lemma, key in group_key.items():
        groups.setdefault(key, []).append(lemma)

    # 3. For each group pick the best representative, then emit merged maps.
    merged_freq: dict[str, int] = {}
    merged_cids: dict[str, set[str]] = {}
    merged_fs: dict[str, tuple[str, str, str]] = {}
    for members in groups.values():
        # Highest frequency wins; ties broken by lexicographically smaller lemma.
        rep = min(members, key=lambda lemma: (-frequency[lemma], lemma))
        total_freq = sum(frequency[m] for m in members)
        union_cids: set[str] = set()
        for m in members:
            union_cids.update(chunk_ids[m])
        merged_freq[rep] = total_freq
        merged_cids[rep] = union_cids
        merged_fs[rep] = first_seen[rep]
    return merged_freq, merged_cids, merged_fs


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
                if _is_noise_lemma(lemma):
                    continue
                frequency[lemma] = frequency.get(lemma, 0) + 1
                chunk_ids.setdefault(lemma, set()).add(chunk_id)
                if lemma not in first_seen:
                    first_seen[lemma] = (candidate.text, candidate.label, chunk_id)

        merged_frequency, merged_chunk_ids, merged_first_seen = _merge_near_duplicates(frequency, chunk_ids, first_seen)
        scored = sorted(
            merged_frequency,
            key=lambda lemma: (
                -merged_frequency[lemma] * math.log(1 + n_chunks / len(merged_chunk_ids[lemma])),
                lemma,
            ),
        )
        terms: list[Term] = []
        for lemma in scored[:top_n]:
            surface, label, chunk_id = merged_first_seen[lemma]
            terms.append(
                Term(
                    term=surface,
                    lemma=lemma,
                    frequency=merged_frequency[lemma],
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
