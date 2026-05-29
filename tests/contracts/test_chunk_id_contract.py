"""Contract: TextChunks carry `id`, and build_payload preserves it into the
F3/F5 chunk inputs. Pins RISK 6 before F3/F5 consume source_chunk_ids.
"""

from __future__ import annotations

from src.agents.payloads import build_payload
from src.core.schemas import Document, DocumentType, Operation, Task, TaskStatus, TextChunk
from src.plan import build_plan, subtask_id_for


def _audio_task() -> Task:
    return Task(
        id="task-1",
        status=TaskStatus.PLANNING,
        requested_outputs=[Operation.F1_TRANSCRIBE, Operation.F3_SUMMARIZE, Operation.F5_TERMS],
        conversation_id="conv-1",
        documents=[Document(id="doc-a", task_id="task-1", document_type=DocumentType.AUDIO, file_path="/x.mp3")],
    )


def test_textchunk_dump_carries_id() -> None:
    chunk = TextChunk(
        id="chunk-1", task_id="task-1", document_id="doc-a", source_type="audio", content="hi", chunk_index=0
    )
    assert chunk.model_dump()["id"] == "chunk-1"


def test_build_payload_preserves_chunk_id_for_summarizer_and_terminology() -> None:
    task = _audio_task()
    plan = build_plan(task)
    f1_chunk = TextChunk(
        id="chunk-1", task_id="task-1", document_id="doc-a", source_type="audio", content="lecture", chunk_index=0
    )
    results: dict[str, object | None] = {
        subtask_id_for(task.id, Operation.F1_TRANSCRIBE): {"chunks": [f1_chunk.model_dump()]}
    }
    for operation in (Operation.F3_SUMMARIZE, Operation.F5_TERMS):
        subtask = next(s for s in plan.subtasks if s.operation == operation)
        payload = build_payload(subtask, task, results)
        chunks = payload["chunks"]
        assert isinstance(chunks, list) and chunks, f"{operation} got no chunks"
        assert chunks[0]["id"] == "chunk-1"
