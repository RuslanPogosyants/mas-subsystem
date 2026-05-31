"""ROUGE (Recall-Oriented Understudy for Gisting Evaluation) metrics.

Computes ROUGE-1, ROUGE-2, and ROUGE-L F-measures via the ``rouge_score``
library with ``use_stemmer=False`` to avoid the English-only Porter stemmer
mangling Russian tokens.

The library's default tokenizer lowercases and then strips every character
outside ``[a-z0-9]`` — which deletes all Cyrillic, yielding empty token sets
and ROUGE 0.0 for Russian text. We pass a Unicode-aware tokenizer (``\\w+`` with
the ``re.UNICODE`` flag) so Cyrillic word characters survive.

All returned float values are rounded to 4 decimal places.
"""

from __future__ import annotations

import re

from rouge_score import rouge_scorer

_METRICS = ["rouge1", "rouge2", "rougeL"]
_WORD_RE = re.compile(r"\w+", re.UNICODE)


class _UnicodeTokenizer:
    """Whitespace/punctuation tokenizer that keeps Cyrillic (and any Unicode word char)."""

    def tokenize(self, text: str) -> list[str]:
        return _WORD_RE.findall(text.lower())


_SCORER = rouge_scorer.RougeScorer(_METRICS, use_stemmer=False, tokenizer=_UnicodeTokenizer())


def rouge_scores(reference: str, hypothesis: str) -> dict[str, float]:
    """Return ROUGE-1, ROUGE-2, and ROUGE-L F-measures for one pair.

    Args:
        reference: Ground-truth text.
        hypothesis: System output text.

    Returns:
        ``{"rouge1": f1, "rouge2": f1, "rougeL": f1}`` with values in [0, 1]
        rounded to 4 decimal places.
    """
    raw = _SCORER.score(reference, hypothesis)
    # rouge-score's LCS scorer returns int 0 (not 0.0) for no overlap; coerce to
    # float so the dict[str, float] contract holds on every path.
    return {metric: round(float(raw[metric].fmeasure), 4) for metric in _METRICS}


def corpus_rouge(pairs: list[tuple[str, str]]) -> dict[str, float]:
    """Return mean ROUGE F-measures across all (reference, hypothesis) pairs.

    Uses the arithmetic mean of per-pair F-measures (macro average).

    Returns ``{"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}`` for an empty
    list.
    """
    if not pairs:
        return dict.fromkeys(_METRICS, 0.0)

    totals: dict[str, float] = dict.fromkeys(_METRICS, 0.0)
    for ref, hyp in pairs:
        scores = rouge_scores(ref, hyp)
        for metric in _METRICS:
            totals[metric] += scores[metric]

    count = len(pairs)
    return {metric: round(totals[metric] / count, 4) for metric in _METRICS}
