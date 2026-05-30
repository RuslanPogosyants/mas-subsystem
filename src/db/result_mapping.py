"""Single source of truth for mapping agent inform-content to ORM result rows.

Centralises the schema<->column bijection — notably the Summary 'thesis' section
maps to the SummaryRow.key_points column. Row ids are deterministic so that
re-persisting a result is an idempotent upsert.
"""

from __future__ import annotations

from typing import Any, Final

from src.db.models import CitationRow, QuizRow, SummaryRow, TermRow, TextChunkRow

# Summary section type (schema vocabulary) -> SummaryRow column.
_SECTION_TO_COLUMN: Final[dict[str, str]] = {
    "introduction": "introduction",
    "thesis": "key_points",
    "conclusion": "conclusions",
}

# SummaryRow column -> Summary section type (inverse of _SECTION_TO_COLUMN).
_COLUMN_TO_SECTION: Final[dict[str, str]] = {column: section for section, column in _SECTION_TO_COLUMN.items()}


def summary_row(task_id: str, content: dict[str, Any]) -> SummaryRow:
    columns: dict[str, str | None] = {"introduction": None, "key_points": None, "conclusions": None}
    sections = content.get("sections")
    if isinstance(sections, list):
        for section in sections:
            if isinstance(section, dict):
                column = _SECTION_TO_COLUMN.get(str(section.get("type")))
                if column is not None:
                    columns[column] = section.get("text")
    return SummaryRow(
        id=str(content.get("summary_id") or f"sum-{task_id}"),
        task_id=task_id,
        introduction=columns["introduction"],
        key_points=columns["key_points"],
        conclusions=columns["conclusions"],
        source_chunk_ids=content.get("source_chunk_ids"),
    )


def term_rows(task_id: str, content: dict[str, Any]) -> list[TermRow]:
    terms = content.get("terms")
    if not isinstance(terms, list):
        return []
    rows: list[TermRow] = []
    for index, term in enumerate(terms):
        if not isinstance(term, dict):
            continue
        rows.append(
            TermRow(
                id=f"term-{task_id}-{index}",
                task_id=task_id,
                text_chunk_id=term.get("source_chunk_id"),
                term=str(term.get("term", "")),
                lemma=term.get("lemma"),
                frequency=term.get("frequency"),
                category=term.get("category"),
            )
        )
    return rows


def quiz_row(task_id: str, content: dict[str, Any]) -> QuizRow:
    questions = content.get("questions")
    return QuizRow(
        id=str(content.get("quiz_id") or f"quiz-{task_id}"),
        task_id=task_id,
        questions=questions if isinstance(questions, list) else [],
        difficulty=content.get("difficulty"),
    )


def citation_rows(task_id: str, content: dict[str, Any]) -> list[CitationRow]:
    citations = content.get("citations")
    if not isinstance(citations, list):
        return []
    rows: list[CitationRow] = []
    for index, citation in enumerate(citations):
        if not isinstance(citation, dict):
            continue
        rows.append(
            CitationRow(
                id=f"cit-{task_id}-{index}",
                task_id=task_id,
                title=str(citation.get("title", "")),
                authors=citation.get("authors"),
                year=citation.get("year"),
                url=citation.get("url"),
                relevance_score=citation.get("relevance_score"),
            )
        )
    return rows


def chunk_rows(task_id: str, content: dict[str, Any]) -> list[TextChunkRow]:
    chunks = content.get("chunks")
    if not isinstance(chunks, list):
        return []
    rows: list[TextChunkRow] = []
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        rows.append(
            TextChunkRow(
                id=str(chunk.get("id")),
                task_id=task_id,
                document_id=str(chunk.get("document_id")),
                source_type=str(chunk.get("source_type")),
                content=str(chunk.get("content", "")),
                chunk_index=int(chunk.get("chunk_index", 0)),
                confidence=chunk.get("confidence"),
                meta=chunk.get("meta") if isinstance(chunk.get("meta"), dict) else None,
            )
        )
    return rows


def content_from_chunk_rows(rows: list[TextChunkRow]) -> dict[str, Any]:
    return {
        "chunks": [
            {
                "id": row.id,
                "task_id": row.task_id,
                "document_id": row.document_id,
                "source_type": row.source_type,
                "content": row.content,
                "chunk_index": row.chunk_index,
                "confidence": row.confidence,
                "meta": row.meta or {},
            }
            for row in rows
        ]
    }


def content_from_summary_row(row: SummaryRow) -> dict[str, Any]:
    columns = {"introduction": row.introduction, "key_points": row.key_points, "conclusions": row.conclusions}
    sections = [
        {"type": _COLUMN_TO_SECTION[column], "text": text} for column, text in columns.items() if text is not None
    ]
    return {"summary_id": row.id, "sections": sections, "source_chunk_ids": row.source_chunk_ids or []}


def content_from_term_rows(rows: list[TermRow]) -> dict[str, Any]:
    return {
        "terms": [
            {
                "term": row.term,
                "lemma": row.lemma,
                "frequency": row.frequency,
                "category": row.category,
                "context": None,  # no column on TermRow; lossy by design
                "source_chunk_id": row.text_chunk_id,
            }
            for row in rows
        ]
    }


def content_from_quiz_row(row: QuizRow) -> dict[str, Any]:
    return {"quiz_id": row.id, "questions": row.questions, "difficulty": row.difficulty}


def content_from_citation_rows(rows: list[CitationRow]) -> dict[str, Any]:
    return {
        "citations": [
            {
                "title": row.title,
                "authors": row.authors,
                "year": row.year,
                "url": row.url,
                "relevance_score": row.relevance_score,
            }
            for row in rows
        ]
    }
