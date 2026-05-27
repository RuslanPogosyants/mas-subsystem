"""Property-based tests for build_plan via hypothesis. RED until M1."""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st
from src.core.schemas import (
    Document,
    DocumentType,
    Operation,
    Task,
    TaskStatus,
)
from src.plan import (
    OPERATION_TO_AGENT,
    OPTIONAL_AGENTS,
    REQUIRED_AGENTS,
    build_plan,
)

operation_strategy = st.sampled_from(list(Operation))
ops_set_strategy = st.lists(operation_strategy, min_size=1, max_size=6, unique=True)


def _make_task(operations: list[Operation], with_audio: bool, with_pdf: bool) -> Task:
    docs: list[Document] = []
    if with_audio:
        docs.append(
            Document(
                id="doc-a",
                task_id="task-x",
                document_type=DocumentType.AUDIO,
                file_path="/x.mp3",
            )
        )
    if with_pdf:
        docs.append(
            Document(
                id="doc-p",
                task_id="task-x",
                document_type=DocumentType.PDF,
                file_path="/x.pdf",
            )
        )
    return Task(
        id="task-x",
        status=TaskStatus.PLANNING,
        requested_outputs=operations,
        conversation_id="conv-x",
        documents=docs,
    )


@settings(max_examples=50)
@given(operations=ops_set_strategy)
def test_plan_subtasks_only_from_requested_operations(operations: list[Operation]) -> None:
    task = _make_task(operations, with_audio=True, with_pdf=True)
    plan = build_plan(task)
    plan_ops = {subtask.operation for subtask in plan.subtasks}
    # With both audio and pdf attached every requested operation is eligible:
    # subset alone is too weak — it would pass for build_plan -> empty.
    assert plan_ops == set(operations)


@settings(max_examples=50)
@given(operations=ops_set_strategy)
def test_plan_required_iff_agent_in_required_set(operations: list[Operation]) -> None:
    task = _make_task(operations, with_audio=True, with_pdf=True)
    plan = build_plan(task)
    for subtask in plan.subtasks:
        if subtask.required:
            assert subtask.agent in REQUIRED_AGENTS
        else:
            assert subtask.agent in OPTIONAL_AGENTS


def test_plan_audio_only_no_ocr() -> None:
    task = _make_task([Operation.F1_TRANSCRIBE, Operation.F2_OCR], with_audio=True, with_pdf=False)
    plan = build_plan(task)
    assert all(subtask.operation != Operation.F2_OCR for subtask in plan.subtasks)


def test_plan_pdf_only_no_transcriber() -> None:
    task = _make_task([Operation.F1_TRANSCRIBE, Operation.F2_OCR], with_audio=False, with_pdf=True)
    plan = build_plan(task)
    assert all(subtask.operation != Operation.F1_TRANSCRIBE for subtask in plan.subtasks)


def test_plan_test_generator_depends_on_summarizer() -> None:
    task = _make_task([Operation.F3_SUMMARIZE, Operation.F4_TEST], with_audio=False, with_pdf=False)
    task.documents.append(
        Document(
            id="doc-t",
            task_id="task-x",
            document_type=DocumentType.TEXT,
            file_path="/x.txt",
        )
    )
    plan = build_plan(task)
    test_subtask = next(subtask for subtask in plan.subtasks if subtask.operation == Operation.F4_TEST)
    summarize_subtask = next(subtask for subtask in plan.subtasks if subtask.operation == Operation.F3_SUMMARIZE)
    assert summarize_subtask.id in test_subtask.depends_on


def test_plan_subtask_id_unique() -> None:
    task = _make_task(list(Operation), with_audio=True, with_pdf=True)
    plan = build_plan(task)
    ids = [subtask.id for subtask in plan.subtasks]
    assert len(ids) == len(set(ids))


def test_operation_to_agent_total() -> None:
    for op in Operation:
        assert op in OPERATION_TO_AGENT
