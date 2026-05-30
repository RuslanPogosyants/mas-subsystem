"""Round-trip tests for the content<->row bijection in result_mapping."""

from __future__ import annotations

from src.db.result_mapping import (
    chunk_rows,
    citation_rows,
    content_from_chunk_rows,
    content_from_citation_rows,
    content_from_quiz_row,
    content_from_summary_row,
    content_from_term_rows,
    quiz_row,
    summary_row,
    term_rows,
)


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


def test_term_rows_link_chunk_via_source_chunk_id() -> None:
    content = {"terms": [{"term": "граф", "lemma": "граф", "frequency": 2, "category": "ds", "source_chunk_id": "c1"}]}
    rows = term_rows("task-1", content)
    assert len(rows) == 1
    assert rows[0].id == "term-task-1-0"
    assert rows[0].term == "граф"
    assert rows[0].frequency == 2
    assert rows[0].text_chunk_id == "c1"  # now linked


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


def test_chunk_rows_and_reverse_round_trip() -> None:
    content = {
        "chunks": [
            {
                "id": "chunk-doc-t-0-0",
                "task_id": "t",
                "document_id": "doc-t-0",
                "source_type": "audio",
                "content": "hello world",
                "chunk_index": 0,
                "confidence": 0.9,
                "meta": {},
            }
        ]
    }
    rows = chunk_rows("t", content)
    assert len(rows) == 1
    assert rows[0].id == "chunk-doc-t-0-0"
    assert rows[0].document_id == "doc-t-0"
    back = content_from_chunk_rows(rows)
    assert back["chunks"][0]["id"] == "chunk-doc-t-0-0"
    assert back["chunks"][0]["content"] == "hello world"
    assert back["chunks"][0]["source_type"] == "audio"


def test_summary_round_trip_preserves_thesis_to_key_points() -> None:
    content = {
        "summary_id": "sum-t",
        "sections": [
            {"type": "introduction", "text": "intro"},
            {"type": "thesis", "text": "the points"},
            {"type": "conclusion", "text": "done"},
        ],
        "source_chunk_ids": ["chunk-doc-t-0-0"],
    }
    row = summary_row("t", content)
    assert row.key_points == "the points"
    back = content_from_summary_row(row)
    assert back["summary_id"] == "sum-t"
    by_type = {s["type"]: s["text"] for s in back["sections"]}
    assert by_type == {"introduction": "intro", "thesis": "the points", "conclusion": "done"}
    assert back["source_chunk_ids"] == ["chunk-doc-t-0-0"]


def test_term_rows_link_text_chunk_id_and_reverse() -> None:
    content = {
        "terms": [
            {
                "term": "Граф",
                "lemma": "граф",
                "frequency": 3,
                "category": "general",
                "source_chunk_id": "chunk-doc-t-0-0",
            },
            {"term": "Узел", "lemma": "узел", "frequency": 1, "category": "general", "source_chunk_id": None},
        ]
    }
    rows = term_rows("t", content)
    assert rows[0].text_chunk_id == "chunk-doc-t-0-0"
    assert rows[1].text_chunk_id is None
    back = content_from_term_rows(rows)
    assert [x["term"] for x in back["terms"]] == ["Граф", "Узел"]
    assert back["terms"][0]["source_chunk_id"] == "chunk-doc-t-0-0"


def test_quiz_round_trip() -> None:
    content = {
        "quiz_id": "quiz-t",
        "questions": [{"question": "Q?", "type": "single_choice", "choices": ["a", "b"], "answer_idx": 0}],
        "difficulty": "medium",
    }
    row = quiz_row("t", content)
    back = content_from_quiz_row(row)
    assert back["quiz_id"] == "quiz-t"
    assert back["difficulty"] == "medium"
    assert back["questions"][0]["question"] == "Q?"


def test_citation_round_trip() -> None:
    content = {
        "citations": [{"title": "Paper", "authors": "A", "year": 2020, "url": "http://x", "relevance_score": 0.7}]
    }
    rows = citation_rows("t", content)
    back = content_from_citation_rows(rows)
    assert back["citations"][0]["title"] == "Paper"
    assert back["citations"][0]["relevance_score"] == 0.7
