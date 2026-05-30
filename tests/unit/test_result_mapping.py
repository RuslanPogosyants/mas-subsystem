"""Unit tests for the content->row result mapping."""

from __future__ import annotations

from src.db.result_mapping import citation_rows, quiz_row, summary_row, term_rows


def test_summary_thesis_section_maps_to_key_points() -> None:
    content = {
        "summary_id": "sum-1",
        "sections": [
            {"type": "introduction", "text": "intro"},
            {"type": "thesis", "text": "the thesis"},
            {"type": "conclusion", "text": "concl"},
        ],
        "source_chunk_ids": ["c1"],
    }
    row = summary_row("task-1", content)
    assert row.id == "sum-1"
    assert row.task_id == "task-1"
    assert row.introduction == "intro"
    assert row.key_points == "the thesis"  # thesis -> key_points
    assert row.conclusions == "concl"
    assert row.source_chunk_ids == ["c1"]


def test_term_rows_deterministic_ids_and_no_chunk_link() -> None:
    content = {"terms": [{"term": "граф", "lemma": "граф", "frequency": 2, "category": "ds", "source_chunk_id": "c1"}]}
    rows = term_rows("task-1", content)
    assert len(rows) == 1
    assert rows[0].id == "term-task-1-0"
    assert rows[0].term == "граф"
    assert rows[0].frequency == 2
    assert rows[0].text_chunk_id is None  # linkage deferred


def test_quiz_row_carries_questions_jsonb() -> None:
    content = {"quiz_id": "quiz-1", "questions": [{"question": "q", "type": "open_answer"}], "difficulty": "easy"}
    row = quiz_row("task-1", content)
    assert row.id == "quiz-1"
    assert row.questions[0]["type"] == "open_answer"
    assert row.difficulty == "easy"


def test_citation_rows_map_fields() -> None:
    content = {"citations": [{"title": "Graphs", "authors": "A", "year": 2020, "url": "u", "relevance_score": 0.9}]}
    rows = citation_rows("task-1", content)
    assert rows[0].id == "cit-task-1-0"
    assert rows[0].title == "Graphs"
    assert rows[0].relevance_score == 0.9


def test_missing_or_malformed_content_yields_empty_or_defaults() -> None:
    assert term_rows("t", {}) == []
    assert citation_rows("t", {"citations": "nope"}) == []
    assert quiz_row("t", {}).questions == []
    assert summary_row("t", {}).id == "sum-t"
