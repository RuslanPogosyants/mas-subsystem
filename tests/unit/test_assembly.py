"""Unit tests for ResultArtifact assembly from coordinator task state."""

from __future__ import annotations

from dataclasses import dataclass, field

from src.agents.assembly import assemble_artifact, assemble_payload
from src.core.schemas import (
    Document,
    DocumentType,
    Operation,
    Task,
    TaskStatus,
)
from src.plan import Plan, build_plan, subtask_id_for


@dataclass
class _State:
    task: Task
    plan: Plan
    results: dict[str, object | None]
    retry_counts: dict[str, int] = field(default_factory=dict)
    first_attempt_at: dict[str, float] = field(default_factory=dict)
    resolved_at: dict[str, float] = field(default_factory=dict)
    fail_reason: dict[str, str] = field(default_factory=dict)
    messages_exchanged: int = 0
    started_at: float = 0.0


def _full_task() -> Task:
    docs = [
        Document(id="doc-a", task_id="task-x", document_type=DocumentType.AUDIO, file_path="/x.mp3"),
        Document(id="doc-p", task_id="task-x", document_type=DocumentType.PDF, file_path="/x.pdf"),
    ]
    return Task(
        id="task-x",
        status=TaskStatus.PLANNING,
        requested_outputs=list(Operation),
        conversation_id="conv-x",
        documents=docs,
    )


def _sid(op: Operation) -> str:
    return subtask_id_for("task-x", op)


def test_assemble_payload_maps_each_operation() -> None:
    task = _full_task()
    plan = build_plan(task)
    results: dict[str, object | None] = {
        _sid(Operation.F1_TRANSCRIBE): {"chunks": []},
        _sid(Operation.F2_OCR): {"chunks": []},
        _sid(Operation.F3_SUMMARIZE): {"summary_id": "s1", "sections": [{"type": "introduction", "text": "x"}]},
        _sid(Operation.F5_TERMS): {"terms": [{"term": "sort", "lemma": "sort", "frequency": 3, "category": "method"}]},
        _sid(Operation.F4_TEST): {"questions": [{"question": "q?", "type": "open_answer"}]},
        _sid(Operation.F6_RECOMMEND): {"citations": [{"title": "Paper", "relevance_score": 0.9}]},
    }
    payload = assemble_payload(plan, results)
    assert payload.summary is not None
    assert len(payload.summary.sections) == 1
    assert len(payload.terms) == 1
    assert len(payload.quiz) == 1
    assert len(payload.citations) == 1


def test_assemble_artifact_completed_counts_seven_agents() -> None:
    task = _full_task()
    plan = build_plan(task)
    results: dict[str, object | None] = {
        _sid(Operation.F1_TRANSCRIBE): {"chunks": []},
        _sid(Operation.F2_OCR): {"chunks": []},
        _sid(Operation.F3_SUMMARIZE): {"summary_id": "s1", "sections": []},
        _sid(Operation.F5_TERMS): {"terms": []},
        _sid(Operation.F4_TEST): {"questions": []},
        _sid(Operation.F6_RECOMMEND): {"citations": []},
    }
    state = _State(task=task, plan=plan, results=results, messages_exchanged=13, started_at=100.0)
    artifact = assemble_artifact(state, TaskStatus.COMPLETED, now=187.0)
    assert {o.value for o in artifact.operations} == {"F1", "F2", "F3", "F4", "F5", "F6"}
    assert artifact.degraded == []
    assert artifact.stats.agents_called == 7
    assert artifact.stats.failed_operations == []
    assert artifact.stats.duration_sec == 87.0


def test_assemble_artifact_partial_records_failure_details() -> None:
    task = _full_task()
    plan = build_plan(task)
    f6 = _sid(Operation.F6_RECOMMEND)
    results: dict[str, object | None] = {
        _sid(Operation.F1_TRANSCRIBE): {"chunks": []},
        _sid(Operation.F2_OCR): {"chunks": []},
        _sid(Operation.F3_SUMMARIZE): {"summary_id": "s1", "sections": []},
        _sid(Operation.F5_TERMS): {"terms": []},
        _sid(Operation.F4_TEST): {"questions": []},
        f6: None,
    }
    state = _State(
        task=task,
        plan=plan,
        results=results,
        retry_counts={f6: 2},
        first_attempt_at={f6: 100.0},
        resolved_at={f6: 106.2},
        fail_reason={f6: "refuse: force_refuse flag enabled"},
        messages_exchanged=13,
        started_at=100.0,
    )
    artifact = assemble_artifact(state, TaskStatus.PARTIAL_READY, now=110.0)
    assert [o.value for o in artifact.degraded] == ["F6"]
    assert artifact.result.citations == []
    assert len(artifact.stats.failed_operations) == 1
    failed = artifact.stats.failed_operations[0]
    assert failed.op == Operation.F6_RECOMMEND
    assert failed.agent == "RecommenderAgent"
    assert "refuse" in failed.reason
    assert failed.retries == 2
    assert failed.elapsed_sec == 6.2
