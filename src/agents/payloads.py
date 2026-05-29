"""Request payload builders for Coordinator subtasks.

Roots (F1/F2) draw their input from the Task's documents. Downstream operations
(F3/F4/F5/F6) draw from the inform content of their upstream subtasks, looked up
by deterministic subtask id. Each builder returns the `content` dict published in
the request message to the target agent.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Final

from src.core.schemas import DocumentType, Operation
from src.plan import subtask_id_for

if TYPE_CHECKING:
    from src.core.schemas import Task
    from src.plan import Subtask

PayloadBuilder = Callable[["Subtask", "Task", dict[str, object | None]], dict[str, object]]

_SUMMARY_STRUCTURE: Final[list[str]] = ["introduction", "thesis", "conclusion"]


def _first_document(task: Task, *types: DocumentType) -> tuple[str, str] | None:
    for document in task.documents:
        if document.document_type in types:
            return document.id, document.file_path
    return None


def _content(results: dict[str, object | None], task_id: str, operation: Operation) -> dict[str, object]:
    value = results.get(subtask_id_for(task_id, operation))
    return value if isinstance(value, dict) else {}


def _chunks(results: dict[str, object | None], task_id: str) -> list[object]:
    chunks: list[object] = []
    for operation in (Operation.F1_TRANSCRIBE, Operation.F2_OCR):
        content = _content(results, task_id, operation)
        raw = content.get("chunks")
        if isinstance(raw, list):
            chunks.extend(raw)
    return chunks


def _build_transcriber(subtask: Subtask, task: Task, results: dict[str, object | None]) -> dict[str, object]:
    found = _first_document(task, DocumentType.AUDIO)
    if found is None:
        return {}
    document_id, file_path = found
    return {"document_id": document_id, "file_path": file_path, "language": "ru"}


def _build_ocr(subtask: Subtask, task: Task, results: dict[str, object | None]) -> dict[str, object]:
    for document in task.documents:
        if document.document_type in (DocumentType.PDF, DocumentType.IMAGE):
            return {
                "document_id": document.id,
                "file_path": document.file_path,
                "document_type": document.document_type.value,
            }
    return {}


def _build_summarizer(subtask: Subtask, task: Task, results: dict[str, object | None]) -> dict[str, object]:
    return {"chunks": _chunks(results, task.id), "max_length_chars": 4000, "structure": list(_SUMMARY_STRUCTURE)}


def _build_terminology(subtask: Subtask, task: Task, results: dict[str, object | None]) -> dict[str, object]:
    return {"chunks": _chunks(results, task.id), "top_n": 10}


def _build_test_generator(subtask: Subtask, task: Task, results: dict[str, object | None]) -> dict[str, object]:
    summary = dict(_content(results, task.id, Operation.F3_SUMMARIZE))
    return {"summary": summary, "num_questions": 5, "difficulty": "medium"}


def _build_recommender(subtask: Subtask, task: Task, results: dict[str, object | None]) -> dict[str, object]:
    summary = dict(_content(results, task.id, Operation.F3_SUMMARIZE))
    raw_terms = _content(results, task.id, Operation.F5_TERMS).get("terms", [])
    terms: list[object] = list(raw_terms) if isinstance(raw_terms, list) else []
    return {"summary": summary, "terms": terms, "n": 3, "filters": {}}


_BUILDERS: Final[dict[Operation, PayloadBuilder]] = {
    Operation.F1_TRANSCRIBE: _build_transcriber,
    Operation.F2_OCR: _build_ocr,
    Operation.F3_SUMMARIZE: _build_summarizer,
    Operation.F4_TEST: _build_test_generator,
    Operation.F5_TERMS: _build_terminology,
    Operation.F6_RECOMMEND: _build_recommender,
}


def build_payload(subtask: Subtask, task: Task, results: dict[str, object | None]) -> dict[str, object]:
    """Build the request content for `subtask`, given resolved upstream `results`.

    The `subtask` argument is part of the uniform builder contract; it is kept so
    individual builders can inspect per-subtask config in the future without a
    signature change.

    Returned payloads may still share *nested* references with upstream results
    and must be treated as read-only by consumers.
    """
    builder = _BUILDERS.get(subtask.operation)
    if builder is None:
        return {}
    return builder(subtask, task, results)
