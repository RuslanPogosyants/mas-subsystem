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
from src.core.retry import AGENT_TIMEOUTS_SEC, BACKOFF_SECONDS, RETRY_MAX
from src.core.schemas import TaskStatus
from src.core.status_fsm import determine_final_status
from src.plan import build_plan

if TYPE_CHECKING:
    from collections.abc import Callable

    from src.agents.store import TaskStore
    from src.core.bus import RedisStreamBus
    from src.core.messages import Message
    from src.core.schemas import Task
    from src.plan import Plan, Subtask

COORDINATOR_GROUP: Final[str] = "coordinator"
COORDINATOR_NAME: Final[str] = "CoordinatorAgent"
_INBOX_BLOCK_MS: Final[int] = 200
_INBOX_BATCH: Final[int] = 32


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
    messages_exchanged: int = 0


class Coordinator:
    """Persistent dispatch loop. One instance per process; many tasks."""

    name = COORDINATOR_NAME

    def __init__(
        self,
        *,
        bus: RedisStreamBus,
        store: TaskStore,
        agent_timeouts: dict[str, float] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._bus = bus
        self._store = store
        self._timeouts = agent_timeouts or dict(AGENT_TIMEOUTS_SEC)
        self._clock = clock
        self._tasks: dict[str, TaskState] = {}
        self._idempotency = IdempotentReceiver()
        self._lock = asyncio.Lock()
        self._shutdown = asyncio.Event()

    def shutdown(self) -> None:
        self._shutdown.set()

    async def run(self) -> None:
        """Main loop; cancel-safe via shutdown() or Task.cancel()."""
        await self._bus.ensure_group(COORDINATOR_INBOX, COORDINATOR_GROUP)
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
        async with self._lock:
            now = self._clock()
            for reply in replies:
                state = self._tasks.get(reply.task_id)
                if state is not None:
                    self._route_reply(state, reply, now)
            for task_id in list(self._tasks):
                state = self._tasks.get(task_id)
                if state is None:
                    continue
                await self._advance(state, now)
                await self._maybe_finalize(task_id, state, now)

    def _route_reply(self, state: TaskState, reply: Message, now: float) -> None:
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

    async def _advance(self, state: TaskState, now: float) -> None:
        changed = True
        while changed:
            changed = False
            for subtask in state.plan.subtasks:
                if subtask.id in state.published or subtask.id not in state.pending:
                    continue
                if not all(dep in state.results for dep in subtask.depends_on):
                    continue
                if any(state.results.get(dep) is None for dep in subtask.depends_on):
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

    def _fail_subtask(self, state: TaskState, subtask_id: str, reason: str, now: float) -> None:
        state.results[subtask_id] = None
        state.fail_reason[subtask_id] = reason
        state.resolved_at[subtask_id] = now
        state.deadlines.pop(subtask_id, None)
        state.retry_at.pop(subtask_id, None)
        state.published.add(subtask_id)
        state.pending.discard(subtask_id)

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
        return self._timeouts.get(agent, AGENT_TIMEOUTS_SEC[agent])

    async def _maybe_finalize(self, task_id: str, state: TaskState, now: float) -> None:
        if state.pending:
            return
        await self._finalize(state, now)
        self._tasks.pop(task_id, None)

    async def _finalize(self, state: TaskState, now: float) -> None:
        status = determine_final_status(state.plan, state.results)
        artifact = assemble_artifact(state, status, now)
        await self._store.save_artifact(
            state.task.id,
            final_artifact=artifact.model_dump(mode="json"),
            stats=artifact.stats.model_dump(mode="json"),
        )
        await self._store.set_status(state.task.id, status)
        logger.info(f"task {state.task.id} finalised as {status.value}")
