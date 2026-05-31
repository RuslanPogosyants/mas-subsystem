"""Unit tests for WER, ROUGE, and term P/R/F1 metric modules.

All tests are deterministic and require no ML models or network access.
"""

from __future__ import annotations

import pytest
from src.evaluation.rouge import corpus_rouge, rouge_scores
from src.evaluation.term_prf import term_prf
from src.evaluation.wer import corpus_wer, normalize_text, word_error_rate

# ---------------------------------------------------------------------------
# normalize_text
# ---------------------------------------------------------------------------


def test_normalize_text_lowercases() -> None:
    assert normalize_text("Hello World") == "hello world"


def test_normalize_text_strips_punctuation() -> None:
    result = normalize_text("Привет, Мир!")
    assert result == "привет мир"


def test_normalize_text_collapses_whitespace() -> None:
    assert normalize_text("a   b\t\nc") == "a b c"


def test_normalize_text_cyrillic_preserved() -> None:
    result = normalize_text("Алгоритм — это чёткая инструкция.")
    assert "алгоритм" in result
    assert "чёткая" in result
    assert "инструкция" in result


# ---------------------------------------------------------------------------
# word_error_rate
# ---------------------------------------------------------------------------


def test_wer_identical_strings_is_zero() -> None:
    assert word_error_rate("a b c d", "a b c d") == 0.0


def test_wer_one_substitution() -> None:
    # "a b c d" vs "a b x d": 1 substitution out of 4 reference words
    result = word_error_rate("a b c d", "a b x d")
    assert result == pytest.approx(0.25)


def test_wer_normalisation_cyrillic() -> None:
    # After normalisation both sides become "привет мир" — identical
    assert word_error_rate("Привет, Мир!", "привет мир") == 0.0


def test_wer_both_empty_returns_zero() -> None:
    assert word_error_rate("", "") == 0.0


def test_wer_empty_reference_nonempty_hypothesis_returns_one() -> None:
    assert word_error_rate("", "something") == 1.0


def test_wer_full_deletion_returns_one() -> None:
    # All reference words deleted → WER == 1.0 (not higher for pure deletion)
    result = word_error_rate("a b c", "")
    assert result == pytest.approx(1.0)


def test_wer_full_insertion() -> None:
    # "a" vs "a b c": 2 insertions for 1 reference word → WER 2.0
    result = word_error_rate("a", "a b c")
    assert result == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# corpus_wer
# ---------------------------------------------------------------------------


def test_corpus_wer_aggregates_total_edits_over_total_ref_words() -> None:
    # Pair 1: "a b c d" vs "a b x d" → 1 edit / 4 ref words
    # Pair 2: "x y" vs "x y" → 0 edits / 2 ref words
    # Corpus WER: 1 / (4 + 2) = 1/6 ≈ 0.1667
    pairs = [("a b c d", "a b x d"), ("x y", "x y")]
    result = corpus_wer(pairs)
    assert result == pytest.approx(1 / 6, rel=1e-4)


def test_corpus_wer_single_pair() -> None:
    assert corpus_wer([("a b c d", "a b x d")]) == pytest.approx(0.25)


def test_corpus_wer_empty_list_returns_zero() -> None:
    assert corpus_wer([]) == 0.0


def test_corpus_wer_counts_insertions_over_empty_reference() -> None:
    # Regression: empty-ref pairs must not be silently dropped. A model that
    # hallucinates words over silence (empty reference) commits insertion errors.
    # Pair 1: "" vs "a b c" -> 3 insertions / 0 ref words
    # Pair 2: "x y" vs "x y" -> 0 edits / 2 ref words
    # Corpus WER: (3 + 0) / (0 + 2) = 1.5
    pairs = [("", "a b c"), ("x y", "x y")]
    assert corpus_wer(pairs) == pytest.approx(1.5)


def test_corpus_wer_all_empty_refs_with_hyps_is_one() -> None:
    assert corpus_wer([("", "a b"), ("", "c")]) == pytest.approx(1.0)


def test_corpus_wer_all_empty_is_zero() -> None:
    assert corpus_wer([("", ""), ("", "")]) == 0.0


# ---------------------------------------------------------------------------
# rouge_scores
# ---------------------------------------------------------------------------


def test_rouge_identical_text_all_ones() -> None:
    text = "The quick brown fox jumps over the lazy dog"
    scores = rouge_scores(text, text)
    assert scores["rouge1"] == pytest.approx(1.0)
    assert scores["rouge2"] == pytest.approx(1.0)
    assert scores["rougeL"] == pytest.approx(1.0)


def test_rouge_disjoint_hypothesis_is_zero() -> None:
    scores = rouge_scores("alpha beta gamma", "one two three")
    assert scores["rouge1"] == pytest.approx(0.0)
    assert scores["rouge2"] == pytest.approx(0.0)
    assert scores["rougeL"] == pytest.approx(0.0)


def test_rouge_partial_overlap_between_zero_and_one() -> None:
    scores = rouge_scores("a b c d e", "a b c x y")
    assert 0.0 < scores["rouge1"] < 1.0
    assert 0.0 < scores["rouge2"] < 1.0
    assert 0.0 < scores["rougeL"] < 1.0


def test_rouge_empty_hypothesis_returns_float_zeros() -> None:
    # rouge-score's LCS scorer returns int 0 for no overlap; rouge_scores must
    # coerce to float so the dict[str, float] contract holds.
    scores = rouge_scores("hello world", "")
    for metric in ("rouge1", "rouge2", "rougeL"):
        assert scores[metric] == 0.0
        assert isinstance(scores[metric], float)


def test_rouge_keys_present() -> None:
    scores = rouge_scores("hello world", "hello there")
    assert set(scores.keys()) == {"rouge1", "rouge2", "rougeL"}


def test_rouge_values_rounded_to_4dp() -> None:
    scores = rouge_scores("a b c d e", "a b c x y")
    for v in scores.values():
        # At most 4 decimal places
        assert round(v, 4) == v


def test_rouge_cyrillic_identical_all_ones() -> None:
    # Regression: rouge_score's default tokenizer strips non-[a-z0-9], deleting
    # all Cyrillic and yielding ROUGE 0.0. The Unicode tokenizer must keep it.
    text = "кошка сидела на тёплом ковре у окна"
    scores = rouge_scores(text, text)
    assert scores["rouge1"] == pytest.approx(1.0)
    assert scores["rougeL"] == pytest.approx(1.0)


def test_rouge_cyrillic_partial_overlap_nonzero() -> None:
    scores = rouge_scores("кошка сидела на ковре", "кошка лежала на ковре")
    assert scores["rouge1"] > 0.0
    assert scores["rougeL"] > 0.0


# ---------------------------------------------------------------------------
# corpus_rouge
# ---------------------------------------------------------------------------


def test_corpus_rouge_mean_of_per_pair_scores() -> None:
    # Pair 1: identical → all 1.0
    # Pair 2: disjoint → all 0.0
    # Mean → all 0.5
    pairs = [("a b c", "a b c"), ("a b c", "x y z")]
    result = corpus_rouge(pairs)
    assert result["rouge1"] == pytest.approx(0.5, abs=1e-4)
    assert result["rougeL"] == pytest.approx(0.5, abs=1e-4)


def test_corpus_rouge_empty_list_returns_zeros() -> None:
    result = corpus_rouge([])
    assert result == {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}


# ---------------------------------------------------------------------------
# term_prf
# ---------------------------------------------------------------------------


def test_term_prf_basic_partial_overlap() -> None:
    # pred=["граф","дерево","стек"], gold=["граф","дерево","очередь"]
    # intersection = {"граф","дерево"} → size 2
    # precision = 2/3, recall = 2/3, f1 = 2/3
    result = term_prf(["граф", "дерево", "стек"], ["граф", "дерево", "очередь"])
    expected = round(2 / 3, 4)
    assert result["precision"] == pytest.approx(expected, abs=1e-4)
    assert result["recall"] == pytest.approx(expected, abs=1e-4)
    assert result["f1"] == pytest.approx(expected, abs=1e-4)


def test_term_prf_normalisation_case_and_whitespace() -> None:
    # "Граф " and " дерево" should normalise to "граф" and "дерево"
    result = term_prf(["Граф ", " дерево"], ["граф", "дерево"])
    assert result["f1"] == pytest.approx(1.0)


def test_term_prf_empty_predicted_all_zero() -> None:
    result = term_prf([], ["граф", "дерево"])
    assert result["precision"] == 0.0
    assert result["recall"] == 0.0
    assert result["f1"] == 0.0


def test_term_prf_empty_gold_all_zero() -> None:
    result = term_prf(["граф", "дерево"], [])
    assert result["precision"] == 0.0
    assert result["recall"] == 0.0
    assert result["f1"] == 0.0


def test_term_prf_both_empty_all_zero() -> None:
    result = term_prf([], [])
    assert result["precision"] == 0.0
    assert result["recall"] == 0.0
    assert result["f1"] == 0.0


def test_term_prf_perfect_match() -> None:
    result = term_prf(["a", "b", "c"], ["a", "b", "c"])
    assert result["precision"] == pytest.approx(1.0)
    assert result["recall"] == pytest.approx(1.0)
    assert result["f1"] == pytest.approx(1.0)


def test_term_prf_no_overlap() -> None:
    result = term_prf(["x", "y"], ["a", "b"])
    assert result["precision"] == 0.0
    assert result["recall"] == 0.0
    assert result["f1"] == 0.0


def test_term_prf_deduplication() -> None:
    # Duplicate "граф" in pred should be treated as one unique term
    result = term_prf(["граф", "граф", "дерево"], ["граф", "дерево"])
    assert result["precision"] == pytest.approx(1.0)
    assert result["recall"] == pytest.approx(1.0)
    assert result["f1"] == pytest.approx(1.0)


def test_term_prf_keys_present() -> None:
    result = term_prf(["a"], ["a"])
    assert set(result.keys()) == {"precision", "recall", "f1"}


def test_term_prf_values_rounded_to_4dp() -> None:
    result = term_prf(["граф", "дерево", "стек"], ["граф", "дерево", "очередь"])
    for v in result.values():
        assert round(v, 4) == v
