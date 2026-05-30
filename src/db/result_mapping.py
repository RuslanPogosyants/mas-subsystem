"""Single source of truth for mapping agent inform-content to ORM result rows.

Centralises the schema<->column bijection — notably the Summary 'thesis' section
maps to the SummaryRow.key_points column. Row ids are deterministic so that
re-persisting a result is an idempotent upsert.
"""

from __future__ import annotations

from typing import Any, Final

from src.db.models import CitationRow, QuizRow, SummaryRow, TermRow

# Summary section type (schema vocabulary) -> SummaryRow column.
_SECTION_TO_COLUMN: Final[dict[str, str]] = {
    "introduction": "introduction",
    "thesis": "key_points",
    "conclusion": "conclusions",
}


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
                text_chunk_id=None,  # chunk linkage deferred (needs text_chunks persistence)
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
