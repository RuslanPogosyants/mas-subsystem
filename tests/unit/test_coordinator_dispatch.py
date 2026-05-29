"""Unit tests for the Coordinator persistent dispatch loop (FakeBus + fake clock)."""

from __future__ import annotations

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

    def advance(self, seconds: float) -> None:
        self.t += seconds


_FAST_TIMEOUTS = {
    "transcriber": 5.0,
    "ocr": 5.0,
    "summarizer": 5.0,
    "test_generator": 5.0,
    "terminology": 5.0,
    "recommender": 2.0,
}


def _coordinator(bus: FakeBus, store: FakeTaskStore, clock: Clock) -> Coordinator:
    return Coordinator(bus=bus, store=store, agent_timeouts=_FAST_TIMEOUTS, clock=clock)


def _task(task_id: str, ops: list[Operation], *, audio: bool = True, pdf: bool = True) -> Task:
    docs: list[Document] = []
    if audio:
        docs.append(Document(id=f"{task_id}-a", task_id=task_id, document_type=DocumentType.AUDIO, file_path="/x.mp3"))
    if pdf:
        docs.append(Document(id=f"{task_id}-p", task_id=task_id, document_type=DocumentType.PDF, file_path="/x.pdf"))
    return Task(
        id=task_id,
        status=TaskStatus.PLANNING,
        requested_outputs=ops,
        conversation_id=f"conv-{task_id}",
        documents=docs,
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


def _refuse(request: Message, reason: str) -> Message:
    return make_message(
        performative=Performative.REFUSE,
        sender=request.receiver,
        receiver="CoordinatorAgent",
        task_id=request.task_id,
        conversation_id=request.conversation_id,
        content={"reason": reason},
        in_reply_to=request.message_id,
        subtask_id=request.subtask_id,
    )


async def test_roots_published_on_submit_downstream_waits() -> None:
    bus, store, clock = FakeBus(), FakeTaskStore(), Clock()
    coordinator = _coordinator(bus, store, clock)
    await coordinator.submit(_task("task-1", [Operation.F1_TRANSCRIBE, Operation.F3_SUMMARIZE]))
    assert len(bus.requests_for(channel_for_agent("transcriber"))) == 1
    assert bus.requests_for(channel_for_agent("summarizer")) == []
    assert store.status["task-1"] == "running"


async def test_summarizer_published_after_transcriber_informs() -> None:
    bus, store, clock = FakeBus(), FakeTaskStore(), Clock()
    coordinator = _coordinator(bus, store, clock)
    await coordinator.submit(_task("task-1", [Operation.F1_TRANSCRIBE, Operation.F3_SUMMARIZE], pdf=False))
    f1_request = bus.requests_for(channel_for_agent("transcriber"))[0]
    bus.feed_inbox(_inform(f1_request, {"chunks": [{"content": "lecture"}]}))
    await coordinator._tick()
    summarizer_requests = bus.requests_for(channel_for_agent("summarizer"))
    assert len(summarizer_requests) == 1
    assert summarizer_requests[0].content["chunks"] == [{"content": "lecture"}]


async def test_completed_when_all_inform() -> None:
    bus, store, clock = FakeBus(), FakeTaskStore(), Clock()
    coordinator = _coordinator(bus, store, clock)
    await coordinator.submit(_task("task-1", [Operation.F1_TRANSCRIBE], pdf=False))
    f1_request = bus.requests_for(channel_for_agent("transcriber"))[0]
    bus.feed_inbox(_inform(f1_request, {"chunks": [{"content": "x"}]}))
    await coordinator._tick()
    assert store.status["task-1"] == "completed"
    assert store.artifacts["task-1"]["status"] == "completed"


async def test_refuse_retries_twice_then_fails_partial() -> None:
    bus, store, clock = FakeBus(), FakeTaskStore(), Clock()
    coordinator = _coordinator(bus, store, clock)
    # F6's deps (F3, F5) are not requested, so F6 is a root alongside F1 here.
    await coordinator.submit(_task("task-1", [Operation.F1_TRANSCRIBE, Operation.F6_RECOMMEND], pdf=False))
    f1_request = bus.requests_for(channel_for_agent("transcriber"))[0]
    bus.feed_inbox(_inform(f1_request, {"chunks": []}))
    refused: set[str] = set()
    for _ in range(10):
        for request in bus.requests_for(channel_for_agent("recommender")):
            if request.message_id not in refused:
                refused.add(request.message_id)
                bus.feed_inbox(_refuse(request, "force_refuse flag enabled"))
        await coordinator._tick()
        clock.advance(5.0)
        if "task-1" in store.artifacts:
            break
    artifact = store.artifacts["task-1"]
    assert artifact["status"] == "partial_ready"
    failed = artifact["stats"]["failed_operations"]
    assert len(failed) == 1
    assert failed[0]["op"] == "F6"
    assert "refuse" in failed[0]["reason"]
    assert failed[0]["retries"] == 2


async def test_timeout_retries_twice_then_fails_with_elapsed() -> None:
    bus, store, clock = FakeBus(), FakeTaskStore(), Clock()
    coordinator = _coordinator(bus, store, clock)
    await coordinator.submit(_task("task-1", [Operation.F1_TRANSCRIBE, Operation.F6_RECOMMEND], pdf=False))
    f1_request = bus.requests_for(channel_for_agent("transcriber"))[0]
    bus.feed_inbox(_inform(f1_request, {"chunks": []}))
    # Never feed a recommender reply -> drive timeouts by advancing the clock.
    for _ in range(15):
        clock.advance(2.0)
        await coordinator._tick()
        if "task-1" in store.artifacts:
            break
    failed = store.artifacts["task-1"]["stats"]["failed_operations"][0]
    assert failed["op"] == "F6"
    assert "timeout" in failed["reason"]
    assert failed["retries"] == 2
    assert failed["elapsed_sec"] >= 3.0


async def test_required_failure_marks_task_failed_and_cascades() -> None:
    bus, store, clock = FakeBus(), FakeTaskStore(), Clock()
    coordinator = _coordinator(bus, store, clock)
    await coordinator.submit(_task("task-1", [Operation.F1_TRANSCRIBE, Operation.F3_SUMMARIZE], pdf=False))
    refused: set[str] = set()
    for _ in range(10):
        for request in bus.requests_for(channel_for_agent("transcriber")):
            if request.message_id not in refused:
                refused.add(request.message_id)
                bus.feed_inbox(_refuse(request, "bad audio"))
        await coordinator._tick()
        clock.advance(6.0)
        if "task-1" in store.artifacts:
            break
    assert store.artifacts["task-1"]["status"] == "failed"
    assert bus.requests_for(channel_for_agent("summarizer")) == []  # never published


async def test_concurrent_tasks_do_not_steal_replies() -> None:
    bus, store, clock = FakeBus(), FakeTaskStore(), Clock()
    coordinator = _coordinator(bus, store, clock)
    await coordinator.submit(_task("task-A", [Operation.F1_TRANSCRIBE], pdf=False))
    await coordinator.submit(_task("task-B", [Operation.F1_TRANSCRIBE], pdf=False))
    requests = bus.requests_for(channel_for_agent("transcriber"))
    by_task = {r.task_id: r for r in requests}
    bus.feed_inbox(_inform(by_task["task-A"], {"chunks": [{"content": "A"}]}))
    bus.feed_inbox(_inform(by_task["task-B"], {"chunks": [{"content": "B"}]}))
    await coordinator._tick()
    assert store.status["task-A"] == "completed"
    assert store.status["task-B"] == "completed"


async def test_finalize_error_does_not_wedge_loop_or_starve_siblings() -> None:
    bus, store, clock = FakeBus(), FakeTaskStore(), Clock()
    coordinator = _coordinator(bus, store, clock)
    # Poison task: F3 will return content that fails Summary validation at finalize.
    await coordinator.submit(_task("task-poison", [Operation.F1_TRANSCRIBE, Operation.F3_SUMMARIZE], pdf=False))
    # Healthy sibling task: F1 only.
    await coordinator.submit(_task("task-ok", [Operation.F1_TRANSCRIBE], pdf=False))
    for request in bus.requests_for(channel_for_agent("transcriber")):
        bus.feed_inbox(_inform(request, {"chunks": []}))
    await coordinator._tick()  # resolves both F1s; task-ok completes; poison publishes F3
    assert store.status["task-ok"] == "completed"
    f3_request = bus.requests_for(channel_for_agent("summarizer"))[0]
    bus.feed_inbox(_inform(f3_request, {"not_a_summary": True}))  # fails Summary validation
    await coordinator._tick()  # must NOT raise; poison task abandoned, loop survives
    assert store.status["task-poison"] == "failed"
    assert "task-poison" not in coordinator._tasks
    # A further tick is stable (loop not wedged).
    await coordinator._tick()


async def test_summarizer_runs_on_one_parent_when_other_refuses() -> None:
    # audio + pdf both eligible; OCR (F2) refuses to exhaustion, transcriber (F1) informs.
    bus, store, clock = FakeBus(), FakeTaskStore(), Clock()
    coordinator = _coordinator(bus, store, clock)
    await coordinator.submit(_task("task-1", [Operation.F1_TRANSCRIBE, Operation.F2_OCR, Operation.F3_SUMMARIZE]))
    f1_request = bus.requests_for(channel_for_agent("transcriber"))[0]
    bus.feed_inbox(_inform(f1_request, {"chunks": [{"id": "c1", "content": "from audio"}]}))
    refused: set[str] = set()
    for _ in range(10):
        for request in bus.requests_for(channel_for_agent("ocr")):
            if request.message_id not in refused:
                refused.add(request.message_id)
                bus.feed_inbox(_refuse(request, "ocr down"))
        await coordinator._tick()
        clock.advance(6.0)
        if bus.requests_for(channel_for_agent("summarizer")):
            break
    summarizer_requests = bus.requests_for(channel_for_agent("summarizer"))
    assert len(summarizer_requests) == 1, "F3 must run on the available F1 chunks, not be skipped"
    assert summarizer_requests[0].content["chunks"] == [{"id": "c1", "content": "from audio"}]


async def test_summarizer_runs_on_ocr_chunks_when_transcriber_refuses() -> None:
    bus, store, clock = FakeBus(), FakeTaskStore(), Clock()
    coordinator = _coordinator(bus, store, clock)
    await coordinator.submit(_task("task-1", [Operation.F1_TRANSCRIBE, Operation.F2_OCR, Operation.F3_SUMMARIZE]))
    f2_request = bus.requests_for(channel_for_agent("ocr"))[0]
    bus.feed_inbox(_inform(f2_request, {"chunks": [{"id": "c2", "content": "from pdf"}]}))
    refused: set[str] = set()
    for _ in range(10):
        for request in bus.requests_for(channel_for_agent("transcriber")):
            if request.message_id not in refused:
                refused.add(request.message_id)
                bus.feed_inbox(_refuse(request, "audio down"))
        await coordinator._tick()
        clock.advance(6.0)
        if bus.requests_for(channel_for_agent("summarizer")):
            break
    summarizer_requests = bus.requests_for(channel_for_agent("summarizer"))
    assert len(summarizer_requests) == 1
    assert summarizer_requests[0].content["chunks"] == [{"id": "c2", "content": "from pdf"}]


async def test_summarizer_skipped_when_all_parents_fail() -> None:
    bus, store, clock = FakeBus(), FakeTaskStore(), Clock()
    coordinator = _coordinator(bus, store, clock)
    await coordinator.submit(_task("task-1", [Operation.F1_TRANSCRIBE, Operation.F2_OCR, Operation.F3_SUMMARIZE]))
    refused: set[str] = set()
    for _ in range(12):
        for agent in ("transcriber", "ocr"):
            for request in bus.requests_for(channel_for_agent(agent)):
                if request.message_id not in refused:
                    refused.add(request.message_id)
                    bus.feed_inbox(_refuse(request, "down"))
        await coordinator._tick()
        clock.advance(6.0)
        if "task-1" in store.artifacts:
            break
    assert bus.requests_for(channel_for_agent("summarizer")) == [], "F3 must be skipped when no chunks exist"
    # F1/F2 are both required and both failed -> task failed (see status FSM, unchanged).
    assert store.artifacts["task-1"]["status"] == "failed"


async def test_recommender_skipped_when_one_parent_fails_under_all_join() -> None:
    # F6 (join="all") needs both F3 and F5; F5 fails -> F6 must be skipped, not run on F3 alone.
    bus, store, clock = FakeBus(), FakeTaskStore(), Clock()
    coordinator = _coordinator(bus, store, clock)
    await coordinator.submit(
        _task(
            "task-1",
            [Operation.F1_TRANSCRIBE, Operation.F3_SUMMARIZE, Operation.F5_TERMS, Operation.F6_RECOMMEND],
            pdf=False,
        )
    )
    f1_request = bus.requests_for(channel_for_agent("transcriber"))[0]
    bus.feed_inbox(_inform(f1_request, {"chunks": [{"id": "c1", "content": "x"}]}))
    await coordinator._tick()  # F1 resolves -> F3 and F5 publish (join="any" on the audio chunks)
    f3_request = bus.requests_for(channel_for_agent("summarizer"))[0]
    bus.feed_inbox(_inform(f3_request, {"summary_id": "s1", "sections": [], "source_chunk_ids": []}))
    refused: set[str] = set()
    for _ in range(10):
        for request in bus.requests_for(channel_for_agent("terminology")):
            if request.message_id not in refused:
                refused.add(request.message_id)
                bus.feed_inbox(_refuse(request, "terms down"))
        await coordinator._tick()
        clock.advance(6.0)
        if "task-1" in store.artifacts:
            break
    assert bus.requests_for(channel_for_agent("recommender")) == [], "F6 must be skipped when F5 failed (join=all)"
    artifact = store.artifacts["task-1"]
    assert artifact["status"] == "partial_ready"
    failed_ops = {failure["op"] for failure in artifact["stats"]["failed_operations"]}
    assert {"F5", "F6"} <= failed_ops
