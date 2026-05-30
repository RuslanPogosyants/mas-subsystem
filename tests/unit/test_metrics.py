"""Unit tests for the metrics module: metric objects exist with the right shape."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from src.adapters.transcriber import FakeTranscriberAdapter
from src.agents.coordinator import Coordinator
from src.agents.recovery import RecoveredTask
from src.agents.transcriber import TranscriberAgent
from src.core import metrics
from src.core.bus import channel_for_agent
from src.core.messages import Performative, make_message
from src.core.schemas import Document, DocumentType, Operation, Task
from src.plan import subtask_id_for

from tests.support.fake_bus import FakeBus, FakeTaskStore
from tests.unit.test_coordinator_dispatch import Clock, _coordinator, _inform, _task

if TYPE_CHECKING:
    from prometheus_client import Counter, Histogram


def test_metric_objects_exist_and_are_labelled() -> None:
    # Observing with the documented labels must not raise (labels are declared).
    metrics.AGENT_HANDLE_SECONDS.labels(agent="TranscriberAgent", operation="F1", outcome="inform").observe(0.01)
    metrics.TASK_DURATION_SECONDS.labels(status="completed").observe(1.0)
    metrics.SUBTASK_DURATION_SECONDS.labels(operation="F3").observe(0.5)
    metrics.TASKS_TOTAL.labels(status="completed").inc()
    metrics.SUBTASK_OUTCOMES_TOTAL.labels(operation="F6", outcome="refuse").inc()
    metrics.RETRIES_TOTAL.labels(operation="F4").inc()
    metrics.INFLIGHT_TASKS.inc()
    metrics.INFLIGHT_TASKS.dec()
    metrics.LLM_CALL_SECONDS.labels(outcome="parsed").observe(0.2)
    metrics.RECOVERED_TASKS_TOTAL.inc()


def test_render_returns_prometheus_text() -> None:
    metrics.TASKS_TOTAL.labels(status="failed").inc()
    text = metrics.render().decode()
    assert "mas_tasks_total" in text
    assert "# HELP" in text


def _sample_count(histogram: Histogram, **labels: str) -> float:
    """Read a histogram's _count child value for the given labels (0.0 if absent)."""
    histogram.labels(**labels)
    for metric in histogram.collect():
        for s in metric.samples:
            if s.name.endswith("_count") and s.labels == {**labels}:
                return s.value
    return 0.0


@pytest.mark.asyncio
async def test_agent_handle_latency_is_observed() -> None:
    before = _sample_count(metrics.AGENT_HANDLE_SECONDS, agent="TranscriberAgent", operation="F1", outcome="inform")
    agent = TranscriberAgent(bus=FakeBus(), transcriber=FakeTranscriberAdapter())
    request = make_message(
        performative=Performative.REQUEST,
        sender="CoordinatorAgent",
        receiver="TranscriberAgent",
        task_id="t",
        conversation_id="c",
        content={"document_id": "doc-t-0", "file_path": "/x.mp3"},
        subtask_id="st-t-F1",
    )
    await agent._safe_handle(request)
    after = _sample_count(metrics.AGENT_HANDLE_SECONDS, agent="TranscriberAgent", operation="F1", outcome="inform")
    assert after == before + 1


def _counter_value(counter: Counter, **labels: str) -> float:
    """Read a counter child's `_total` value for the given labels (0.0 if absent)."""
    counter.labels(**labels)
    for metric in counter.collect():
        for s in metric.samples:
            if s.name == counter._name + "_total" and s.labels == {**labels}:
                return s.value
    return 0.0


@pytest.mark.asyncio
async def test_task_finalize_records_duration_and_count() -> None:
    bus, store, clock = FakeBus(), FakeTaskStore(), Clock()
    coordinator = _coordinator(bus, store, clock)

    tasks_before = _counter_value(metrics.TASKS_TOTAL, status="completed")
    subtasks_before = _counter_value(metrics.SUBTASK_OUTCOMES_TOTAL, operation="F1", outcome="inform")
    duration_before = _sample_count(metrics.TASK_DURATION_SECONDS, status="completed")
    gauge_before = metrics.INFLIGHT_TASKS._value.get()

    await coordinator.submit(_task("task-metrics", [Operation.F1_TRANSCRIBE], pdf=False))
    # While the single subtask is pending, the task is registered and counted in-flight.
    assert metrics.INFLIGHT_TASKS._value.get() == gauge_before + 1
    f1_request = bus.requests_for(channel_for_agent("transcriber"))[0]
    bus.feed_inbox(_inform(f1_request, {"chunks": [{"content": "x"}]}))
    await coordinator._tick()

    assert store.status["task-metrics"] == "completed"
    assert _counter_value(metrics.TASKS_TOTAL, status="completed") == tasks_before + 1
    assert _counter_value(metrics.SUBTASK_OUTCOMES_TOTAL, operation="F1", outcome="inform") == subtasks_before + 1
    assert _sample_count(metrics.TASK_DURATION_SECONDS, status="completed") == duration_before + 1
    # The in-flight gauge must return exactly to its starting value once the task finalises.
    assert metrics.INFLIGHT_TASKS._value.get() == gauge_before


@pytest.mark.asyncio
async def test_inflight_gauge_returns_to_baseline_after_abandon() -> None:
    bus, store, clock = FakeBus(), FakeTaskStore(), Clock()
    coordinator = _coordinator(bus, store, clock)
    gauge_before = metrics.INFLIGHT_TASKS._value.get()

    # F3 returns content that fails Summary validation at finalize -> the task is abandoned.
    await coordinator.submit(_task("task-abandon", [Operation.F1_TRANSCRIBE, Operation.F3_SUMMARIZE], pdf=False))
    assert metrics.INFLIGHT_TASKS._value.get() == gauge_before + 1
    f1_request = bus.requests_for(channel_for_agent("transcriber"))[0]
    bus.feed_inbox(_inform(f1_request, {"chunks": []}))
    await coordinator._tick()  # resolves F1, publishes F3
    f3_request = bus.requests_for(channel_for_agent("summarizer"))[0]
    bus.feed_inbox(_inform(f3_request, {"not_a_summary": True}))
    await coordinator._tick()  # finalize raises -> task abandoned

    assert store.status["task-abandon"] == "failed"
    assert "task-abandon" not in coordinator._tasks
    # Abandon must dec the gauge exactly once, restoring the baseline.
    assert metrics.INFLIGHT_TASKS._value.get() == gauge_before


class _FakeRecovery:
    def __init__(self, items: list[RecoveredTask]) -> None:
        self._items = items

    async def load_in_flight(self) -> list[RecoveredTask]:
        return self._items


@pytest.mark.asyncio
async def test_llm_call_latency_is_observed() -> None:
    from pydantic import BaseModel
    from src.adapters.llm import FakeLlmAdapter
    from src.agents._llm_json import parse_with_retry

    class _M(BaseModel):
        introduction: str
        key_points: str
        conclusions: str

    before = _sample_count(metrics.LLM_CALL_SECONDS, outcome="parsed")
    await parse_with_retry(FakeLlmAdapter(), system="s", user="u", model_cls=_M, retries=1)
    after = _sample_count(metrics.LLM_CALL_SECONDS, outcome="parsed")
    assert after >= before + 1


def test_model_call_seconds_exists() -> None:
    from src.core import metrics

    metrics.MODEL_CALL_SECONDS.labels(adapter="whisper", operation="F1").observe(0.5)
    assert "mas_model_call_seconds" in metrics.render().decode()


@pytest.mark.asyncio
async def test_immediately_finalized_recovery_does_not_leak_gauge() -> None:
    # A recovered task whose results are all present finalizes without ever entering
    # self._tasks; it must NOT inc the in-flight gauge (only RECOVERED_TASKS_TOTAL moves).
    task = Task(
        id="task-recover",
        requested_outputs=[Operation.F1_TRANSCRIBE],
        conversation_id="conv-task-recover",
        documents=[
            Document(id="doc-r-0", task_id="task-recover", document_type=DocumentType.AUDIO, file_path="/a.mp3")
        ],
    )
    results: dict[str, object] = {
        subtask_id_for("task-recover", Operation.F1_TRANSCRIBE): {"chunks": [{"id": "c0", "content": "c"}]},
    }
    coordinator = Coordinator(
        bus=FakeBus(),
        store=FakeTaskStore(),
        recovery=_FakeRecovery([RecoveredTask(task=task, results=results)]),
    )
    gauge_before = metrics.INFLIGHT_TASKS._value.get()
    recovered_before = metrics.RECOVERED_TASKS_TOTAL._value.get()

    await coordinator._recover()

    assert "task-recover" not in coordinator._tasks
    # Immediately-finalized recovery path must leave the gauge untouched...
    assert metrics.INFLIGHT_TASKS._value.get() == gauge_before
    # ...but still count the resumed task.
    assert metrics.RECOVERED_TASKS_TOTAL._value.get() == recovered_before + 1
