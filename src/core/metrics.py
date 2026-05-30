"""Prometheus metrics for the multi-agent pipeline (default registry).

A deliberately small, low-cardinality set covering the two questions that matter
for the demo: how fast is each stage, and what were the outcomes. Metrics are
always-on singletons; instrumentation lives at the existing chokepoints
(AgentBase handle, Coordinator finalize/route/retry/recover, the LLM-JSON helper).
"""

from __future__ import annotations

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

# Seconds buckets spanning fake (sub-ms) to real GPU/LLM latencies (minutes).
_LATENCY_BUCKETS = (0.005, 0.025, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0)

AGENT_HANDLE_SECONDS = Histogram(
    "mas_agent_handle_seconds",
    "Wall-clock seconds an agent spent handling one request.",
    ("agent", "operation", "outcome"),
    buckets=_LATENCY_BUCKETS,
)
TASK_DURATION_SECONDS = Histogram(
    "mas_task_duration_seconds",
    "End-to-end seconds from task submit to finalize.",
    ("status",),
    buckets=_LATENCY_BUCKETS,
)
SUBTASK_DURATION_SECONDS = Histogram(
    "mas_subtask_duration_seconds",
    "Seconds from first dispatch to a successful subtask reply.",
    ("operation",),
    buckets=_LATENCY_BUCKETS,
)
TASKS_TOTAL = Counter("mas_tasks_total", "Tasks finalized, by terminal status.", ("status",))
SUBTASK_OUTCOMES_TOTAL = Counter(
    "mas_subtask_outcomes_total", "Subtask resolutions, by operation and outcome.", ("operation", "outcome")
)
RETRIES_TOTAL = Counter("mas_retries_total", "Subtask retries scheduled, by operation.", ("operation",))
INFLIGHT_TASKS = Gauge("mas_inflight_tasks", "Tasks currently being dispatched by the coordinator.")
LLM_CALL_SECONDS = Histogram(
    "mas_llm_call_seconds",
    "Seconds per LLM completion call, by parse outcome.",
    ("outcome",),
    buckets=_LATENCY_BUCKETS,
)
RECOVERED_TASKS_TOTAL = Counter("mas_recovered_tasks_total", "In-flight tasks resumed on coordinator startup.")


def render() -> bytes:
    """Serialize the default registry to Prometheus text exposition format."""
    return generate_latest()


CONTENT_TYPE = CONTENT_TYPE_LATEST
