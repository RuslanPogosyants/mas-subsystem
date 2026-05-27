"""CoordinatorAgent: orchestrates a Task through its Plan.

M2 scope: happy-path dispatch of stage-1 subtasks (F1/F2). Refuse marks the
subtask as failed; retry / timeout / stage-2 chaining ship in M3.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

from loguru import logger

from src.core.bus import COORDINATOR_INBOX, channel_for_agent
from src.core.idempotency import IdempotentReceiver
from src.core.messages import Performative, make_message
from src.core.retry import GLOBAL_DEADLINE_SEC
from src.core.schemas import Operation, TaskStatus
from src.core.status_fsm import determine_final_status
from src.plan import OPERATION_TO_AGENT, build_plan

if TYPE_CHECKING:
    from src.core.bus import RedisStreamBus
    from src.core.schemas import Task
    from src.db.repos import DocumentRepo, TaskRepo
    from src.plan import Plan, Subtask


CoordinatorGroup = "coordinator"


class CoordinatorAgent:
    name = "CoordinatorAgent"

    def __init__(
        self,
        *,
        bus: RedisStreamBus,
        task_repo: TaskRepo,
        document_repo: DocumentRepo,
    ) -> None:
        self._bus = bus
        self._task_repo = task_repo
        self._document_repo = document_repo
        self._idempotency = IdempotentReceiver()

    async def dispatch(self, task: Task) -> dict[str, object | None]:
        """Run the plan for `task` and return the per-subtask results map."""
        await self._bus.ensure_group(COORDINATOR_INBOX, CoordinatorGroup)
        plan = build_plan(task)
        await self._task_repo.update_status(task.id, TaskStatus.RUNNING)
        await self._task_repo.commit()
        logger.info(f"coordinator dispatching task {task.id} with {len(plan.subtasks)} subtasks")

        pending = {subtask.id for subtask in plan.subtasks}
        results: dict[str, object | None] = {}
        if not pending:
            await self._finalize(task, plan, results)
            return results

        await self._publish_stage(plan.stage1(), plan, task)
        deadline = time.monotonic() + GLOBAL_DEADLINE_SEC
        await self._drain_replies(pending, results, plan, task, deadline)
        for subtask_id in pending:
            results[subtask_id] = None
        await self._finalize(task, plan, results)
        return results

    async def _drain_replies(
        self,
        pending: set[str],
        results: dict[str, object | None],
        plan: Plan,
        task: Task,
        deadline: float,
    ) -> None:
        while pending and time.monotonic() < deadline:
            found_any = False
            async for entry_id, reply in self._bus.read(COORDINATOR_INBOX, CoordinatorGroup, count=10, block_ms=500):
                found_any = True
                await self._bus.ack(COORDINATOR_INBOX, CoordinatorGroup, entry_id)
                if not self._idempotency.accept(reply.message_id):
                    continue
                subtask_id = reply.subtask_id
                if subtask_id is None or subtask_id not in pending:
                    continue
                if reply.performative == Performative.REFUSE:
                    logger.warning(f"subtask {subtask_id} refused: {reply.content.get('reason')}")
                    results[subtask_id] = None
                else:
                    results[subtask_id] = reply.content
                pending.discard(subtask_id)
            if not found_any:
                await asyncio.sleep(0.05)

    async def _publish_stage(self, subtasks: list[Subtask], plan: Plan, task: Task) -> None:
        for subtask in subtasks:
            payload = await self._payload_for(subtask, task)
            request = make_message(
                performative=Performative.REQUEST,
                sender=self.name,
                receiver=OPERATION_TO_AGENT[subtask.operation],
                task_id=task.id,
                conversation_id=f"conv-{task.id}-{subtask.operation.value}",
                content=payload,
                subtask_id=subtask.id,
            )
            await self._bus.publish(channel_for_agent(subtask.agent), request)

    async def _payload_for(self, subtask: Subtask, task: Task) -> dict[str, object]:
        if subtask.operation == Operation.F1_TRANSCRIBE:
            audio = next(
                (doc for doc in task.documents if doc.document_type.value == "audio"),
                None,
            )
            if audio is None:
                return {}
            return {"document_id": audio.id, "file_path": audio.file_path, "language": "ru"}
        if subtask.operation == Operation.F2_OCR:
            for doc in task.documents:
                if doc.document_type.value in ("pdf", "image"):
                    return {
                        "document_id": doc.id,
                        "file_path": doc.file_path,
                        "document_type": doc.document_type.value,
                    }
        return {}

    async def _finalize(
        self,
        task: Task,
        plan: Plan,
        results: dict[str, object | None],
    ) -> None:
        status = determine_final_status(plan, results)
        await self._task_repo.update_status(task.id, status)
        await self._task_repo.commit()
        logger.info(f"task {task.id} finalised as {status.value}")
