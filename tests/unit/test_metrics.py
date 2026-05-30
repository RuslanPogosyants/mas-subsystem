"""Unit tests for the metrics module: metric objects exist with the right shape."""

from __future__ import annotations

from src.core import metrics


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
