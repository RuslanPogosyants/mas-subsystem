"""Persistence boundary for the Coordinator.

`TaskStore` is the minimal write surface the dispatch loop needs. `DbTaskStore`
opens a short-lived session per call via the app's sessionmaker, so the
long-lived Coordinator never holds a session across tasks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from src.core.schemas import Operation, TaskStatus
from src.db.repos import ResultRepo, TaskRepo

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker


class TaskStore(Protocol):
    """Write surface the Coordinator uses to persist task progress."""

    async def set_status(self, task_id: str, status: TaskStatus) -> None: ...

    async def save_artifact(self, task_id: str, *, final_artifact: dict[str, Any], stats: dict[str, Any]) -> None: ...

    async def save_result(self, task_id: str, operation: Operation, content: dict[str, Any]) -> None: ...


class DbTaskStore:
    """TaskStore backed by SQLAlchemy; one session per operation."""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def set_status(self, task_id: str, status: TaskStatus) -> None:
        async with self._session_factory() as session:
            await TaskRepo(session).update_status(task_id, status)
            await session.commit()

    async def save_artifact(self, task_id: str, *, final_artifact: dict[str, Any], stats: dict[str, Any]) -> None:
        async with self._session_factory() as session:
            await TaskRepo(session).save_artifact(task_id, final_artifact=final_artifact, stats=stats)
            await session.commit()

    async def save_result(self, task_id: str, operation: Operation, content: dict[str, Any]) -> None:
        async with self._session_factory() as session:
            repo = ResultRepo(session)
            if operation in (Operation.F1_TRANSCRIBE, Operation.F2_OCR):
                await repo.save_chunks(task_id, content)
            elif operation == Operation.F3_SUMMARIZE:
                await repo.save_summary(task_id, content)
            elif operation == Operation.F5_TERMS:
                await repo.save_terms(task_id, content)
            elif operation == Operation.F4_TEST:
                await repo.save_quiz(task_id, content)
            elif operation == Operation.F6_RECOMMEND:
                await repo.save_citations(task_id, content)
            else:
                return
            await session.commit()
