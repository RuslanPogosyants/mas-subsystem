"""Coordinator: a single long-lived dispatch loop over the agent bus.

Owns COORDINATOR_INBOX. Each submitted Task gets a TaskState; replies are routed
by task_id so concurrent tasks never steal each other's replies. Subtasks are
dispatched by dependency readiness (general DAG); refuse and timeout are retried
up to RETRY_MAX via a non-blocking retry_at schedule. On completion the task is
assembled into a ResultArtifact and persisted through a TaskStore.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final

from loguru import logger

from src.agents.assembly import assemble_artifact
from src.agents.payloads import build_payload
from src.core.bus import COORDINATOR_INBOX, channel_for_agent
from src.core.idempotency import IdempotentReceiver
from src.core.messages import Performative, make_message
from src.core.metrics import (
    INFLIGHT_TASKS,
    RECOVERED_TASKS_TOTAL,
    RETRIES_TOTAL,
    SUBTASK_DURATION_SECONDS,
    SUBTASK_OUTCOMES_TOTAL,
    TASK_DURATION_SECONDS,
    TASKS_TOTAL,
)
from src.core.retry import AGENT_TIMEOUTS_SEC, BACKOFF_SECONDS, RETRY_MAX
from src.core.schemas import Operation, TaskStatus
from src.core.status_fsm import determine_final_status
from src.plan import build_plan

if TYPE_CHECKING:
    from collections.abc import Callable

    from src.agents.recovery import TaskRecovery
    from src.agents.store import TaskStore
    from src.core.bus import RedisStreamBus
    from src.core.messages import Message
    from src.core.schemas import Task
    from src.plan import Plan, Subtask

COORDINATOR_GROUP: Final[str] = "coordinator"
COORDINATOR_NAME: Final[str] = "CoordinatorAgent"
_INBOX_BLOCK_MS: Final[int] = 200
_INBOX_BATCH: Final[int] = 32
_PERSIST_RETRIES: Final[int] = 3
_PERSIST_RETRY_DELAY_SEC: Final[float] = 0.05
# Effectively-unbounded default so tasks/tests that construct a Coordinator without
# a deadline never expire spuriously; production passes settings.coord_global_deadline_sec.
_DEFAULT_GLOBAL_DEADLINE_SEC: Final[float] = 1e9


@dataclass(slots=True)
class TaskState:
    """Mutable per-task dispatch state owned by the Coordinator run loop."""

    task: Task
    plan: Plan
    pending: set[str]
    started_at: float
    published: set[str] = field(default_factory=set)
    results: dict[str, object | None] = field(default_factory=dict)
    retry_counts: dict[str, int] = field(default_factory=dict)
    deadlines: dict[str, float] = field(default_factory=dict)
    retry_at: dict[str, float] = field(default_factory=dict)
    first_attempt_at: dict[str, float] = field(default_factory=dict)
    resolved_at: dict[str, float] = field(default_factory=dict)
    fail_reason: dict[str, str] = field(default_factory=dict)
    # Counts REQUESTs published (including retries) plus replies routed
    # (including refuses) — not request/response pairs.
    messages_exchanged: int = 0


class Coordinator:
    """Persistent dispatch loop. One instance per process; many tasks."""

    name = COORDINATOR_NAME

    def __init__(
        self,
        *,
        bus: RedisStreamBus,
        store: TaskStore,
        recovery: TaskRecovery | None = None,
        agent_timeouts: dict[str, float] | None = None,
        clock: Callable[[], float] = time.monotonic,
        global_deadline_sec: float = _DEFAULT_GLOBAL_DEADLINE_SEC,
    ) -> None:
        self._bus = bus
        self._store = store
        self._recovery = recovery
        self._timeouts = agent_timeouts or dict(AGENT_TIMEOUTS_SEC)
        self._clock = clock
        self._global_deadline_sec = global_deadline_sec
        self._tasks: dict[str, TaskState] = {}
        self._idempotency = IdempotentReceiver()
        self._lock = asyncio.Lock()
        self._shutdown = asyncio.Event()

    def shutdown(self) -> None:
        """Signal the run loop to stop after the current tick."""
        self._shutdown.set()

    async def run(self) -> None:
        """Main loop; cancel-safe via shutdown() or Task.cancel()."""
        await self._bus.ensure_group(COORDINATOR_INBOX, COORDINATOR_GROUP)
        await self._recover()
        logger.info("coordinator run loop started")
        while not self._shutdown.is_set():
            try:
                await self._tick()
            except asyncio.CancelledError:
                raise
            except Exception as error:
                logger.exception(f"coordinator tick error: {error}")
                await asyncio.sleep(0.5)

    async def submit(self, task: Task) -> None:
        """Register a task, persist RUNNING, and publish its ready roots."""
        plan = build_plan(task)
        await self._store.set_status(task.id, TaskStatus.RUNNING)
        async with self._lock:
            now = self._clock()
            state = TaskState(
                task=task,
                plan=plan,
                pending={subtask.id for subtask in plan.subtasks},
                started_at=now,
            )
            if not state.pending:
                await self._finalize(state, now)
                return
            self._tasks[task.id] = state
            INFLIGHT_TASKS.inc()
            await self._advance(state, now)
            await self._maybe_finalize(task.id, state, now)

    async def _recover(self) -> None:
        """Rebuild in-flight tasks from persistence and resume their dispatch."""
        if self._recovery is None:
            return
        recovered = await self._recovery.load_in_flight()
        async with self._lock:
            now = self._clock()
            for item in recovered:
                await self._resume(item.task, item.results, now)

    async def _resume(self, task: Task, results: dict[str, object], now: float) -> None:
        RECOVERED_TASKS_TOTAL.inc()
        plan = build_plan(task)
        valid_ids = {subtask.id for subtask in plan.subtasks}
        loaded = {sid: content for sid, content in results.items() if sid in valid_ids}
        state = TaskState(
            task=task,
            plan=plan,
            pending={subtask.id for subtask in plan.subtasks if subtask.id not in loaded},
            started_at=now,
            published=set(loaded),
            results=dict(loaded),
        )
        for subtask_id in loaded:
            state.resolved_at[subtask_id] = now
        logger.info(f"recovering task {task.id}: {len(loaded)} results reloaded, {len(state.pending)} pending")
        if not state.pending:
            await self._finalize(state, now)
            return
        self._tasks[task.id] = state
        INFLIGHT_TASKS.inc()
        await self._advance(state, now)
        await self._maybe_finalize(task.id, state, now)

    async def _tick(self) -> None:
        replies: list[Message] = []
        async for entry_id, reply in self._bus.read(
            COORDINATOR_INBOX, COORDINATOR_GROUP, count=_INBOX_BATCH, block_ms=_INBOX_BLOCK_MS
        ):
            await self._bus.ack(COORDINATOR_INBOX, COORDINATOR_GROUP, entry_id)
            if self._idempotency.accept(reply.message_id):
                replies.append(reply)
        persist_jobs: list[tuple[str, Operation, dict[str, object]]] = []
        async with self._lock:
            now = self._clock()
            for reply in replies:
                state = self._tasks.get(reply.task_id)
                if state is not None:
                    self._route_reply(state, reply, now, persist_jobs)
            for task_id in list(self._tasks):
                state = self._tasks.get(task_id)
                if state is None:
                    continue
                try:
                    await self._advance(state, now)
                    await self._maybe_finalize(task_id, state, now)
                    # Hard cap: a task still pending after _maybe_finalize (i.e. not removed
                    # this tick) that has run past the global deadline is finalized with
                    # whatever results exist, rather than waiting on stuck subtasks forever.
                    if task_id in self._tasks and now - state.started_at > self._global_deadline_sec:
                        await self._deadline_finalize(task_id, state, now)
                except Exception as error:
                    logger.exception(f"coordinator abandoning task {task_id}: {error}")
                    await self._abandon(task_id)
        # Persist durability writes only after releasing the lock, so a slow or retrying DB
        # write never blocks submit/recover/dispatch (no head-of-line blocking). The served
        # result is the self-contained final_artifact written by _finalize, not these per-op
        # rows. For a task still in flight, recovery (M4.2b) re-drives any subtask missing from
        # the DB on restart; for a task that finalized in this tick the artifact is already the
        # durable record, so the per-op rows are best-effort (COMPLETED tasks are not re-driven).
        for task_id, operation, content in persist_jobs:
            await self._persist_result(task_id, operation, content)

    def _route_reply(
        self,
        state: TaskState,
        reply: Message,
        now: float,
        persist_jobs: list[tuple[str, Operation, dict[str, object]]],
    ) -> None:
        subtask_id = reply.subtask_id
        if subtask_id is None or subtask_id not in state.pending:
            return
        state.messages_exchanged += 1
        state.deadlines.pop(subtask_id, None)
        if reply.performative == Performative.REFUSE:
            reason = f"refuse: {reply.content.get('reason')}"
            self._schedule_retry_or_fail(state, subtask_id, reason, now)
            return
        state.results[subtask_id] = reply.content
        state.resolved_at[subtask_id] = now
        state.retry_at.pop(subtask_id, None)
        state.pending.discard(subtask_id)
        subtask = state.plan.get(subtask_id)
        started = state.first_attempt_at.get(subtask_id, now)
        SUBTASK_DURATION_SECONDS.labels(operation=subtask.operation.value).observe(max(now - started, 0.0))
        SUBTASK_OUTCOMES_TOTAL.labels(operation=subtask.operation.value, outcome="inform").inc()
        persist_jobs.append((state.task.id, subtask.operation, reply.content))

    async def _persist_result(self, task_id: str, operation: Operation, content: dict[str, object]) -> None:
        """Persist one agent result reliably: bounded retry, never raises.

        On definitive failure the result remains in memory so the live run still
        completes; durability is reconciled on restart, where the database is the
        source of truth and any unpersisted subtask is re-driven (M4.2b).
        """
        for attempt in range(1, _PERSIST_RETRIES + 1):
            try:
                await self._store.save_result(task_id, operation, content)
                return
            except Exception as error:
                if attempt == _PERSIST_RETRIES:
                    logger.error(f"persist gave up for {task_id}/{operation.value} after {attempt} tries: {error}")
                    return
                logger.warning(f"persist retry {attempt} for {task_id}/{operation.value}: {error}")
                await asyncio.sleep(_PERSIST_RETRY_DELAY_SEC)

    async def _advance(self, state: TaskState, now: float) -> None:
        changed = True
        while changed:
            changed = False
            for subtask in state.plan.subtasks:
                if subtask.id in state.published or subtask.id not in state.pending:
                    continue
                if not all(dep in state.results for dep in subtask.depends_on):
                    continue
                succeeded = [dep for dep in subtask.depends_on if state.results.get(dep) is not None]
                if subtask.depends_on and not succeeded:
                    self._fail_subtask(state, subtask.id, "skipped: all upstream failed", now)
                elif subtask.join == "all" and len(succeeded) < len(subtask.depends_on):
                    self._fail_subtask(state, subtask.id, "skipped: upstream failed", now)
                else:
                    await self._publish(state, subtask, now)
                changed = True
            if self._expire_timeouts(state, now):
                changed = True
            if await self._republish_due(state, now):
                changed = True

    def _expire_timeouts(self, state: TaskState, now: float) -> bool:
        changed = False
        for subtask_id in list(state.deadlines):
            if now > state.deadlines[subtask_id]:
                del state.deadlines[subtask_id]
                self._schedule_retry_or_fail(state, subtask_id, "timeout", now)
                changed = True
        return changed

    async def _republish_due(self, state: TaskState, now: float) -> bool:
        changed = False
        for subtask_id in list(state.retry_at):
            if now >= state.retry_at[subtask_id]:
                del state.retry_at[subtask_id]
                await self._publish(state, state.plan.get(subtask_id), now)
                changed = True
        return changed

    def _schedule_retry_or_fail(self, state: TaskState, subtask_id: str, reason: str, now: float) -> None:
        if state.retry_counts.get(subtask_id, 0) >= RETRY_MAX:
            self._fail_subtask(state, subtask_id, reason, now)
            return
        state.retry_counts[subtask_id] = state.retry_counts.get(subtask_id, 0) + 1
        state.retry_at[subtask_id] = now + BACKOFF_SECONDS[state.retry_counts[subtask_id] - 1]
        state.fail_reason[subtask_id] = reason
        RETRIES_TOTAL.labels(operation=state.plan.get(subtask_id).operation.value).inc()

    def _fail_subtask(self, state: TaskState, subtask_id: str, reason: str, now: float) -> None:
        state.results[subtask_id] = None
        state.fail_reason[subtask_id] = reason
        state.resolved_at[subtask_id] = now
        state.deadlines.pop(subtask_id, None)
        state.retry_at.pop(subtask_id, None)
        state.published.add(subtask_id)
        state.pending.discard(subtask_id)
        SUBTASK_OUTCOMES_TOTAL.labels(operation=state.plan.get(subtask_id).operation.value, outcome="failed").inc()

    async def _publish(self, state: TaskState, subtask: Subtask, now: float) -> None:
        payload = build_payload(subtask, state.task, state.results)
        request = make_message(
            performative=Performative.REQUEST,
            sender=self.name,
            receiver=subtask.agent,
            task_id=state.task.id,
            conversation_id=f"conv-{state.task.id}-{subtask.operation.value}",
            content=payload,
            subtask_id=subtask.id,
        )
        await self._bus.publish(channel_for_agent(subtask.agent), request)
        state.published.add(subtask.id)
        state.messages_exchanged += 1
        state.deadlines[subtask.id] = now + self._timeout_for(subtask.agent)
        state.first_attempt_at.setdefault(subtask.id, now)

    def _timeout_for(self, agent: str) -> float:
        """Return the deadline for one agent request.

        Uses the injected per-agent override first; falls back to the global
        AGENT_TIMEOUTS_SEC default for that agent.
        """
        return self._timeouts.get(agent, AGENT_TIMEOUTS_SEC[agent])

    async def _maybe_finalize(self, task_id: str, state: TaskState, now: float) -> None:
        if state.pending:
            return
        await self._finalize(state, now)
        self._tasks.pop(task_id, None)
        INFLIGHT_TASKS.dec()

    async def _deadline_finalize(self, task_id: str, state: TaskState, now: float) -> None:
        """Finalize a task that overran the global deadline with its partial results.

        The status comes from determine_final_status over whatever results exist
        (typically PARTIAL_READY, or FAILED if nothing succeeded). Mirrors the
        removal / inflight-gauge bookkeeping of _maybe_finalize.
        """
        logger.warning(
            f"task {task_id} exceeded global deadline of {self._global_deadline_sec}s "
            f"({len(state.pending)} subtask(s) still pending); finalizing with partial results"
        )
        await self._finalize(state, now)
        self._tasks.pop(task_id, None)
        INFLIGHT_TASKS.dec()

    async def _abandon(self, task_id: str) -> None:
        """Drop a task whose processing raised; best-effort mark it failed.

        A single task that throws during advance/finalize (e.g. a malformed agent
        payload that fails validation) must never wedge the run loop or starve
        sibling tasks. The task is removed and, where possible, persisted as failed.
        """
        if self._tasks.pop(task_id, None) is not None:
            INFLIGHT_TASKS.dec()
        try:
            await self._store.set_status(task_id, TaskStatus.FAILED)
        except Exception as error:
            logger.exception(f"coordinator could not mark {task_id} failed: {error}")

    async def _finalize(self, state: TaskState, now: float) -> None:
        status = determine_final_status(state.plan, state.results)
        TASK_DURATION_SECONDS.labels(status=status.value).observe(max(now - state.started_at, 0.0))
        TASKS_TOTAL.labels(status=status.value).inc()
        artifact = assemble_artifact(state, status, now)
        await self._store.save_artifact(
            state.task.id,
            final_artifact=artifact.model_dump(mode="json"),
            stats=artifact.stats.model_dump(mode="json"),
        )
        await self._store.set_status(state.task.id, status)
        logger.info(f"task {state.task.id} finalised as {status.value}")
