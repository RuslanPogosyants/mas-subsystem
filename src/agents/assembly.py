"""Assemble a ResultArtifact from a finished Coordinator task state.

`assemble_payload` maps each completed operation's inform content to the
corresponding ResultPayload field. `assemble_artifact` adds real failure stats
(reason / retries / elapsed) and counts participating agents (specialised + the
coordinator).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from src.core.schemas import (
    Citation,
    FailedOperation,
    Operation,
    QuizQuestion,
    ResultArtifact,
    ResultPayload,
    ResultStats,
    Summary,
    Term,
)
from src.plan import AGENT_CLASS_NAMES

if TYPE_CHECKING:
    from src.core.schemas import Task, TaskStatus
    from src.plan import Plan


class TaskStateView(Protocol):
    """Read-only view of the per-task coordinator state needed for assembly."""

    task: Task
    plan: Plan
    results: dict[str, object | None]
    retry_counts: dict[str, int]
    first_attempt_at: dict[str, float]
    resolved_at: dict[str, float]
    fail_reason: dict[str, str]
    messages_exchanged: int
    started_at: float


def _as_list(raw: object) -> list[object]:
    """Return `raw` as a list, or an empty list if it is not a list."""
    return list(raw) if isinstance(raw, list) else []


def assemble_payload(plan: Plan, results: dict[str, object | None]) -> ResultPayload:
    """Map completed operation contents onto the four ResultPayload fields."""
    payload = ResultPayload()
    for subtask in plan.subtasks:
        content = results.get(subtask.id)
        if not isinstance(content, dict):
            continue
        if subtask.operation == Operation.F3_SUMMARIZE:
            payload.summary = Summary.model_validate(content)
        elif subtask.operation == Operation.F5_TERMS:
            payload.terms = [Term.model_validate(item) for item in _as_list(content.get("terms"))]
        elif subtask.operation == Operation.F4_TEST:
            payload.quiz = [QuizQuestion.model_validate(item) for item in _as_list(content.get("questions"))]
        elif subtask.operation == Operation.F6_RECOMMEND:
            payload.citations = [Citation.model_validate(item) for item in _as_list(content.get("citations"))]
    return payload


def _elapsed(state: TaskStateView, subtask_id: str, now: float) -> float:
    """Return elapsed seconds for a subtask, or 0.0 if not tracked."""
    started = state.first_attempt_at.get(subtask_id)
    if started is None:
        return 0.0
    resolved = state.resolved_at.get(subtask_id, now)
    return round(resolved - started, 3)


def assemble_artifact(state: TaskStateView, status: TaskStatus, now: float) -> ResultArtifact:
    """Build the ResultArtifact for a finished task.

    Args:
        state: Read-only view of the coordinator task state.
        status: Final status to stamp on the artifact.
        now: Current epoch timestamp (seconds) used for duration and elapsed
            calculations when a subtask has no recorded resolved_at time.

    Returns:
        A fully populated ResultArtifact.
    """
    plan = state.plan
    results = state.results
    completed = [s.operation for s in plan.subtasks if results.get(s.id) is not None]
    degraded = [s.operation for s in plan.subtasks if not s.required and results.get(s.id) is None]
    failed = [
        FailedOperation(
            op=subtask.operation,
            agent=AGENT_CLASS_NAMES[subtask.agent],
            reason=state.fail_reason.get(subtask.id, "no_reply"),
            retries=state.retry_counts.get(subtask.id, 0),
            elapsed_sec=_elapsed(state, subtask.id, now),
        )
        for subtask in plan.subtasks
        if results.get(subtask.id) is None
    ]
    stats = ResultStats(
        duration_sec=round(now - state.started_at, 3),
        agents_called=len({subtask.agent for subtask in plan.subtasks}) + 1,
        messages_exchanged=state.messages_exchanged,
        failed_operations=failed,
    )
    return ResultArtifact(
        task_id=state.task.id,
        status=status,
        operations=completed,
        result=assemble_payload(plan, results),
        degraded=degraded,
        stats=stats,
    )
