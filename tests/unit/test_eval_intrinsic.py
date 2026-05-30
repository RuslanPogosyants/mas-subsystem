"""Unit tests for intrinsic (no-ground-truth) quality metrics."""

from __future__ import annotations

from src.evaluation.intrinsic import (
    citations_intrinsics,
    quiz_intrinsics,
    summary_intrinsics,
    terms_intrinsics,
)


def test_summary_intrinsics_compression_and_sections() -> None:
    summary = {
        "summary_id": "s",
        "sections": [{"type": "introduction", "text": "intro"}, {"type": "thesis", "text": "the main points here"}],
        "source_chunk_ids": [],
    }
    m = summary_intrinsics(summary, source_chars=1000)
    assert m["sections_present"] == ["introduction", "thesis"]
    assert m["summary_chars"] == len("intro") + len("the main points here")
    assert 0 < m["compression_ratio"] < 1
    assert m["section_count"] == 2


def test_summary_intrinsics_none() -> None:
    m = summary_intrinsics(None, source_chars=100)
    assert m["section_count"] == 0
    assert m["compression_ratio"] == 0.0


def test_terms_intrinsics_coverage_and_categories() -> None:
    source = "Граф состоит из узлов и рёбер. Граф важен."
    terms = [
        {"term": "Граф", "lemma": "граф", "frequency": 2, "category": "ds"},
        {"term": "Гиперпараметр", "lemma": "гиперпараметр", "frequency": 1, "category": "ml"},
    ]
    m = terms_intrinsics(terms, source)
    assert m["count"] == 2
    assert m["in_source_fraction"] == 0.5  # "Граф" present, "Гиперпараметр" not
    assert m["category_distribution"] == {"ds": 1, "ml": 1}
    assert m["avg_frequency"] == 1.5


def test_quiz_intrinsics_well_formed() -> None:
    quiz = [
        {"question": "Q1?", "type": "single_choice", "choices": ["a", "b"], "answer_idx": 1},
        # bad: idx out of range, fewer than 2 choices
        {"question": "Q2?", "type": "single_choice", "choices": ["a"], "answer_idx": 5},
        {"question": "Q3?", "type": "open_answer", "choices": []},
    ]
    m = quiz_intrinsics(quiz)
    assert m["count"] == 3
    assert m["type_distribution"] == {"single_choice": 2, "open_answer": 1}
    assert m["well_formed_count"] == 2  # Q1 ok, Q3 ok (open), Q2 malformed


def test_citations_intrinsics_scores() -> None:
    cits = [{"title": "A", "relevance_score": 0.8}, {"title": "B", "relevance_score": 0.4}]
    m = citations_intrinsics(cits)
    assert m["count"] == 2
    assert round(m["mean_relevance"], 2) == 0.6
    assert m["min_relevance"] == 0.4 and m["max_relevance"] == 0.8


# --- edge branches not covered above ---


def test_summary_intrinsics_empty_returns_zero_fields() -> None:
    out = summary_intrinsics(None, 100)
    assert out == {
        "section_count": 0,
        "sections_present": [],
        "summary_chars": 0,
        "compression_ratio": 0.0,
    }


def test_summary_intrinsics_zero_source_chars_gives_zero_ratio() -> None:
    out = summary_intrinsics({"sections": [{"type": "thesis", "text": "abc"}]}, 0)
    assert out["compression_ratio"] == 0.0
    assert out["summary_chars"] == 3


def test_terms_intrinsics_in_source_fraction_full_match() -> None:
    out = terms_intrinsics([{"term": "Graph", "frequency": 2, "category": "x"}], "a graph here")
    assert out["in_source_fraction"] == 1.0
    assert out["avg_frequency"] == 2.0


def test_terms_intrinsics_empty_list() -> None:
    assert terms_intrinsics([], "src")["count"] == 0


def test_quiz_intrinsics_mixed_types_well_formed_count() -> None:
    quiz = [
        {"type": "single_choice", "choices": ["A", "B"], "answer_idx": 0},
        {"type": "single_choice", "choices": ["A", "B"], "answer_idx": None},
        {"type": "multi_choice", "choices": ["A", "B"], "answer_indices": [0]},
        {"type": "open_answer", "question": "why?"},
    ]
    out = quiz_intrinsics(quiz)
    assert out["count"] == 4
    assert out["well_formed_count"] == 3


def test_citations_intrinsics_min_max_mean() -> None:
    out = citations_intrinsics([{"relevance_score": 0.2}, {"relevance_score": 0.8}])
    assert out["mean_relevance"] == 0.5
    assert out["min_relevance"] == 0.2
    assert out["max_relevance"] == 0.8
