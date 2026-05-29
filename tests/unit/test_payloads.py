"""Unit tests for Coordinator request payload builders."""

from __future__ import annotations

from src.agents.payloads import build_payload
from src.core.schemas import Document, DocumentType, Operation, Task, TaskStatus
from src.plan import build_plan, subtask_id_for


def _task(ops: list[Operation], *, audio: bool = True, pdf: bool = True) -> Task:
    docs: list[Document] = []
    if audio:
        docs.append(Document(id="doc-a", task_id="task-x", document_type=DocumentType.AUDIO, file_path="/x.mp3"))
    if pdf:
        docs.append(Document(id="doc-p", task_id="task-x", document_type=DocumentType.PDF, file_path="/x.pdf"))
    return Task(
        id="task-x",
        status=TaskStatus.PLANNING,
        requested_outputs=ops,
        conversation_id="conv-x",
        documents=docs,
    )


def test_transcriber_payload_has_audio_file() -> None:
    task = _task([Operation.F1_TRANSCRIBE])
    plan = build_plan(task)
    subtask = plan.subtasks[0]
    payload = build_payload(subtask, task, {})
    assert payload == {"document_id": "doc-a", "file_path": "/x.mp3", "language": "ru"}


def test_ocr_payload_has_pdf_file() -> None:
    task = _task([Operation.F2_OCR])
    plan = build_plan(task)
    subtask = plan.subtasks[0]
    payload = build_payload(subtask, task, {})
    assert payload["file_path"] == "/x.pdf"
    assert payload["document_type"] == "pdf"


def test_summarizer_payload_collects_chunks_from_roots() -> None:
    task = _task([Operation.F1_TRANSCRIBE, Operation.F2_OCR, Operation.F3_SUMMARIZE])
    plan = build_plan(task)
    f3 = next(s for s in plan.subtasks if s.operation == Operation.F3_SUMMARIZE)
    results = {
        subtask_id_for("task-x", Operation.F1_TRANSCRIBE): {"chunks": [{"content": "a"}]},
        subtask_id_for("task-x", Operation.F2_OCR): {"chunks": [{"content": "b"}]},
    }
    payload = build_payload(f3, task, results)
    assert payload["chunks"] == [{"content": "a"}, {"content": "b"}]
    assert payload["max_length_chars"] == 4000


def test_recommender_payload_uses_summary_and_terms() -> None:
    task = _task([Operation.F3_SUMMARIZE, Operation.F5_TERMS, Operation.F6_RECOMMEND], pdf=False)
    plan = build_plan(task)
    f6 = next(s for s in plan.subtasks if s.operation == Operation.F6_RECOMMEND)
    results = {
        subtask_id_for("task-x", Operation.F3_SUMMARIZE): {"summary_id": "s1", "sections": []},
        subtask_id_for("task-x", Operation.F5_TERMS): {"terms": [{"term": "sort"}]},
    }
    payload = build_payload(f6, task, results)
    assert payload["summary"] == {"summary_id": "s1", "sections": []}
    assert payload["terms"] == [{"term": "sort"}]
    assert payload["n"] == 3
