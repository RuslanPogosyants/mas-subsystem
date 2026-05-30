"""Coarse throughput guard for the coordinator dispatch loop (in-process, fakes)."""

from __future__ import annotations

import time

import pytest
from src.agents.coordinator import Coordinator
from src.core.bus import channel_for_agent
from src.core.messages import Message, Performative, make_message
from src.core.schemas import Document, DocumentType, Operation, Task, TaskStatus

from tests.support.fake_bus import FakeBus, FakeTaskStore


class Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


_FAST_TIMEOUTS = {
    "transcriber": 5.0,
    "ocr": 5.0,
    "summarizer": 5.0,
    "test_generator": 5.0,
    "terminology": 5.0,
    "recommender": 2.0,
}


def _task(task_id: str) -> Task:
    return Task(
        id=task_id,
        status=TaskStatus.PLANNING,
        requested_outputs=[Operation.F1_TRANSCRIBE],
        conversation_id=f"conv-{task_id}",
        documents=[Document(id=f"{task_id}-a", task_id=task_id, document_type=DocumentType.AUDIO, file_path="/x.mp3")],
    )


def _inform(request: Message, content: dict[str, object]) -> Message:
    return make_message(
        performative=Performative.INFORM,
        sender=request.receiver,
        receiver="CoordinatorAgent",
        task_id=request.task_id,
        conversation_id=request.conversation_id,
        content=content,
        in_reply_to=request.message_id,
        subtask_id=request.subtask_id,
    )


@pytest.mark.perf
async def test_dispatch_many_tasks_completes_under_budget() -> None:
    n = 50
    bus, store, clock = FakeBus(), FakeTaskStore(), Clock()
    coordinator = Coordinator(bus=bus, store=store, agent_timeouts=_FAST_TIMEOUTS, clock=clock)

    start = time.perf_counter()
    for i in range(n):
        await coordinator.submit(_task(f"task-{i}"))
    # One transcriber REQUEST per task; reply INFORM to each so all finalise.
    for request in bus.requests_for(channel_for_agent("transcriber")):
        bus.feed_inbox(_inform(request, {"chunks": [{"content": "lecture"}]}))
    # Drive ticks until every task has finalised (bounded — a stuck loop fails the budget).
    for _ in range(n + 5):
        await coordinator._tick()
        if len(store.artifacts) == n:
            break
    elapsed = time.perf_counter() - start

    print(f"dispatch throughput: {n / elapsed:.1f} tasks/sec ({elapsed:.3f}s for {n} tasks)")
    assert len(store.artifacts) == n, "all tasks must finalise"
    assert all(status == "completed" for status in store.status.values())
    assert elapsed < 5.0  # generous CI-safe budget; the point is a number + regression guard
