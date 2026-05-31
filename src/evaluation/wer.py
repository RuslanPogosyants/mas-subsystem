"""Word Error Rate (WER) metrics for evaluating transcript/hypothesis quality.

Conventions for edge cases:
- Both reference and hypothesis are empty → WER = 0.0 (perfect match).
- Reference is empty but hypothesis is non-empty → WER = 1.0 (undefined in
  the classic formulation; 1.0 signals that the hypothesis is entirely wrong
  relative to a trivially-empty reference).

corpus_wer computes the *corpus-level* WER (total edit distance / total
reference word count), which is the standard aggregate metric and is NOT
the arithmetic mean of per-utterance WERs.
"""

from __future__ import annotations

import re
import unicodedata

import jiwer

# Characters to strip: anything that is not a letter (Unicode) or whitespace.
_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
# Collapse runs of whitespace to a single space.
_WHITESPACE_RE = re.compile(r"\s+")

# jiwer built-in transforms (version-agnostic approach: call wer directly).
_EMPTY_WER_IF_REF_EMPTY_HYP_NONEMPTY = 1.0
_EMPTY_WER_BOTH_EMPTY = 0.0


def normalize_text(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace.

    Preserves Cyrillic (and all other Unicode letter categories). Strips
    digits together with punctuation is intentional for speech-evaluation use:
    ``\\w`` also matches digits and underscores in Python's ``re``; underscores
    and digits are therefore retained. If callers need digit-free normalisation
    they should post-process the result.
    """
    # NFKC normalises e.g. fullwidth characters before any stripping.
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = _PUNCT_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Return WER after normalising both strings.

    WER = (substitutions + insertions + deletions) / len(reference_words).

    Edge cases (post-normalisation):
    - Both empty → 0.0.
    - Reference empty, hypothesis non-empty → 1.0 (documented convention).
    - Reference non-empty, hypothesis empty → 1.0 (all words deleted;
      jiwer returns this naturally for full-deletion).
    """
    ref_norm = normalize_text(reference)
    hyp_norm = normalize_text(hypothesis)

    if not ref_norm and not hyp_norm:
        return _EMPTY_WER_BOTH_EMPTY
    if not ref_norm:
        return _EMPTY_WER_IF_REF_EMPTY_HYP_NONEMPTY

    return float(jiwer.wer(ref_norm, hyp_norm))


def corpus_wer(pairs: list[tuple[str, str]]) -> float:
    """Corpus-level WER: total edit distance / total reference word count.

    This is the standard corpus WER and differs from the mean of per-utterance
    WERs because it weights each utterance by its reference length.

    Pairs with an empty reference cannot be passed to jiwer (it rejects empty
    references), but a non-empty hypothesis over an empty reference is a real
    insertion error (e.g. an ASR model hallucinating words over silence). Such
    insertions are added to the numerator (0 reference words → 0 to the
    denominator) so they are not silently hidden. The per-utterance
    ``word_error_rate`` already returns 1.0 for that case; this keeps the corpus
    aggregate consistent.

    Returns 0.0 for an empty list of pairs.
    """
    if not pairs:
        return 0.0

    refs = [normalize_text(ref) for ref, _ in pairs]
    hyps = [normalize_text(hyp) for _, hyp in pairs]

    # Insertions over empty references: every hypothesis word is a spurious insertion.
    empty_ref_insertions = sum(len(h.split()) for r, h in zip(refs, hyps, strict=True) if not r)

    non_empty = [(r, h) for r, h in zip(refs, hyps, strict=True) if r]
    if not non_empty:
        # No reference words anywhere: 0.0 if everything is empty, else all-insertion.
        return _EMPTY_WER_BOTH_EMPTY if empty_ref_insertions == 0 else _EMPTY_WER_IF_REF_EMPTY_HYP_NONEMPTY

    f_refs, f_hyps = zip(*non_empty, strict=True)
    out = jiwer.process_words(list(f_refs), list(f_hyps))
    edits = out.substitutions + out.deletions + out.insertions + empty_ref_insertions
    reference_words = out.substitutions + out.deletions + out.hits
    return float(edits / reference_words)
