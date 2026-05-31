"""Real spaCy implementation of NerAdapter (used for the live demo only).

`spacy` and at least one language model are optional ML dependencies imported
lazily, so the module imports cleanly in CI where they are absent. The blocking
spaCy pipeline runs in a worker thread to avoid stalling the asyncio event loop.
NER entities and adjective+noun bigrams become term candidates.

The model is chosen by the script of the input text: if Latin letters strictly
outnumber Cyrillic letters the English model (`en_core_web_sm` by default) is
used; otherwise the Russian model (`ru_core_news_lg` by default).  If the
preferred model is absent, the adapter falls back to the other configured model
(logging a warning) so the operation degrades instead of failing — e.g. an
English document is still processed by the Russian pipeline when the English
model is not installed.  Only if neither model can be loaded does the lazy
`spacy.load` raise ``OSError``, which ``AgentBase._safe_handle`` turns into a
graceful refuse.

Loaded pipelines are cached by language to avoid repeated model loading.

``extract_many`` groups texts by selected language and processes each
language-group via ``nlp.pipe`` for efficient batching.  Results are reassembled
in the original input order.
"""

from __future__ import annotations

import asyncio
import itertools
import time
from collections import defaultdict
from typing import Any, Final

from src.adapters.ner import TermCandidate
from src.core.metrics import MODEL_CALL_SECONDS

_LANG_RU: Final[str] = "ru"
_LANG_EN: Final[str] = "en"

# Unicode ranges for Cyrillic and Basic Latin alphabetic characters.
_CYRILLIC_START: Final[int] = 0x0400
_CYRILLIC_END: Final[int] = 0x04FF
_LATIN_START: Final[int] = 0x0041  # 'A'
_LATIN_UPPER_END: Final[int] = 0x005A  # 'Z'
_LATIN_LOWER_START: Final[int] = 0x0061  # 'a'
_LATIN_LOWER_END: Final[int] = 0x007A  # 'z'


def _is_cyrillic(ch: str) -> bool:
    code = ord(ch)
    return _CYRILLIC_START <= code <= _CYRILLIC_END


def _is_latin(ch: str) -> bool:
    code = ord(ch)
    return (_LATIN_START <= code <= _LATIN_UPPER_END) or (_LATIN_LOWER_START <= code <= _LATIN_LOWER_END)


class SpacyNerAdapter:
    """NerAdapter backed by spaCy pipelines chosen by text script."""

    def __init__(self, model: str = "ru_core_news_lg", en_model: str = "en_core_web_sm") -> None:
        self._model_name = model
        self._en_model_name = en_model
        self._pipelines: dict[str, Any] = {}  # language -> pipeline
        self._by_model: dict[str, Any] = {}  # model name -> pipeline (shared across languages)

    @staticmethod
    def _select_language(text: str) -> str:
        """Return ``"en"`` when Latin letters strictly outnumber Cyrillic; else ``"ru"``."""
        latin_count = 0
        cyrillic_count = 0
        for ch in text:
            if _is_latin(ch):
                latin_count += 1
            elif _is_cyrillic(ch):
                cyrillic_count += 1
        return _LANG_EN if latin_count > cyrillic_count else _LANG_RU

    def _ensure_nlp(self, language: str) -> Any:
        if language not in self._pipelines:
            preferred = self._en_model_name if language == _LANG_EN else self._model_name
            fallback = self._model_name if language == _LANG_EN else self._en_model_name
            self._pipelines[language] = self._load_model(preferred, fallback)
        return self._pipelines[language]

    def _load_model(self, preferred: str, fallback: str) -> Any:
        """Load ``preferred``; on OSError fall back to ``fallback``. Pipelines are
        cached by model name so two languages resolving to the same model (e.g. an
        English document handled by the Russian model after fallback) share one
        loaded instance instead of loading the heavy model twice."""
        import spacy  # lazy: optional ml dependency

        if preferred in self._by_model:
            return self._by_model[preferred]
        try:
            self._by_model[preferred] = spacy.load(preferred)
            return self._by_model[preferred]
        except OSError:
            if fallback == preferred:
                raise
            from loguru import logger

            logger.warning(
                "spaCy model '{}' unavailable; falling back to '{}'",
                preferred,
                fallback,
            )
            if fallback in self._by_model:
                return self._by_model[fallback]
            self._by_model[fallback] = spacy.load(fallback)  # may raise OSError -> graceful refuse
            return self._by_model[fallback]

    async def extract(self, text: str) -> list[TermCandidate]:
        language = self._select_language(text)
        nlp = self._ensure_nlp(language)
        start = time.perf_counter()
        try:
            doc = await asyncio.to_thread(nlp, text)
        finally:
            MODEL_CALL_SECONDS.labels(adapter="spacy", operation="F5").observe(time.perf_counter() - start)
        candidates: list[TermCandidate] = [
            TermCandidate(text=ent.text, lemma=ent.lemma_.lower(), label=ent.label_) for ent in doc.ents
        ]
        candidates.extend(self._adj_noun_bigrams(doc))
        return candidates

    async def extract_many(self, texts: list[str]) -> list[list[TermCandidate]]:
        """Extract candidates for all texts, batching same-language texts via nlp.pipe.

        Texts are grouped by detected language so each language-specific pipeline
        can process its group in one ``nlp.pipe`` call.  Results are reassembled
        in the original input order.
        """
        if not texts:
            return []

        # Group indices by language so we can batch per pipeline.
        lang_groups: dict[str, list[int]] = defaultdict(list)
        for idx, text in enumerate(texts):
            lang_groups[self._select_language(text)].append(idx)

        results: list[list[TermCandidate]] = [[] for _ in texts]

        for language, indices in lang_groups.items():
            nlp = self._ensure_nlp(language)
            group_texts = [texts[i] for i in indices]

            def _run_pipe(n: Any, t: list[str]) -> list[Any]:
                return list(n.pipe(t))

            start = time.perf_counter()
            try:
                docs = await asyncio.to_thread(_run_pipe, nlp, group_texts)
            finally:
                MODEL_CALL_SECONDS.labels(adapter="spacy", operation="F5").observe(time.perf_counter() - start)
            for idx, doc in zip(indices, docs, strict=True):
                candidates: list[TermCandidate] = [
                    TermCandidate(text=ent.text, lemma=ent.lemma_.lower(), label=ent.label_) for ent in doc.ents
                ]
                candidates.extend(self._adj_noun_bigrams(doc))
                results[idx] = candidates

        return results

    @staticmethod
    def _adj_noun_bigrams(doc: Any) -> list[TermCandidate]:
        tokens: list[Any] = list(doc)
        bigrams: list[TermCandidate] = []
        for left, right in itertools.pairwise(tokens):
            if getattr(left, "pos_", "") == "ADJ" and getattr(right, "pos_", "") == "NOUN":
                surface = f"{left.text} {right.text}"
                bigrams.append(TermCandidate(text=surface, lemma=f"{left.lemma_.lower()} {right.lemma_.lower()}"))
        return bigrams
