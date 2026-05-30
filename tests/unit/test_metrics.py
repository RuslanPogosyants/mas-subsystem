"""Unit tests for the metrics module: metric objects exist with the right shape."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from src.adapters.transcriber import FakeTranscriberAdapter
from src.agents.transcriber import TranscriberAgent
from src.core import metrics
from src.core.messages import Performative, make_message

from tests.support.fake_bus import FakeBus

if TYPE_CHECKING:
    from prometheus_client import Histogram


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
