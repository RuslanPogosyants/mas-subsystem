"""Async repositories for Task / Document / TextChunk."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sqlalchemy import delete, select

from src.core.schemas import DocumentType, Operation, TaskStatus
from src.db.models import CitationRow, DocumentRow, TaskRow, TermRow, TextChunkRow
from src.db.result_mapping import chunk_rows, citation_rows, quiz_row, summary_row, term_rows

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


class TaskRepo:
    """CRUD over the tasks table."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def commit(self) -> None:
        """Flush pending changes to the database."""
        await self._session.commit()

    async def create(self, *, task_id: str, requested_outputs: list[Operation]) -> TaskRow:
        row = TaskRow(
            id=task_id,
            status=TaskStatus.PLANNING.value,
            requested_outputs=[operation.value for operation in requested_outputs],
        )
        self._session.add(row)
        await self._session.flush()
        return row

    async def get(self, task_id: str) -> TaskRow | None:
        result = await self._session.execute(select(TaskRow).where(TaskRow.id == task_id))
        return result.scalar_one_or_none()

    async def update_status(self, task_id: str, status: TaskStatus) -> None:
        row = await self.get(task_id)
        if row is None:
            raise KeyError(f"task not found: {task_id}")
        row.status = status.value
        await self._session.flush()

    async def save_artifact(self, task_id: str, *, final_artifact: dict[str, Any], stats: dict[str, Any]) -> None:
        row = await self.get(task_id)
        if row is None:
            raise KeyError(f"task not found: {task_id}")
        row.final_artifact = final_artifact
        row.stats = stats
        await self._session.flush()


class DocumentRepo:
    """CRUD over the documents table."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(
        self,
        *,
        document_id: str,
        task_id: str,
        document_type: DocumentType,
        file_path: str,
        original_name: str | None = None,
    ) -> DocumentRow:
        row = DocumentRow(
            id=document_id,
            task_id=task_id,
            document_type=document_type.value,
            file_path=file_path,
            original_name=original_name,
        )
        self._session.add(row)
        await self._session.flush()
        return row

    async def list_for_task(self, task_id: str) -> list[DocumentRow]:
        result = await self._session.execute(select(DocumentRow).where(DocumentRow.task_id == task_id))
        return list(result.scalars().all())


class ResultRepo:
    """Upserts agent result rows (idempotent by deterministic primary key)."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def save_summary(self, task_id: str, content: dict[str, Any]) -> None:
        await self._session.merge(summary_row(task_id, content))
        await self._session.flush()

    async def save_chunks(self, task_id: str, content: dict[str, Any]) -> None:
        rows = chunk_rows(task_id, content)
        document_ids = {row.document_id for row in rows}
        if document_ids:
            await self._session.execute(delete(TextChunkRow).where(TextChunkRow.document_id.in_(document_ids)))
        for row in rows:
            self._session.add(row)
        await self._session.flush()

    async def save_terms(self, task_id: str, content: dict[str, Any]) -> None:
        await self._session.execute(delete(TermRow).where(TermRow.task_id == task_id))
        for row in term_rows(task_id, content):
            self._session.add(row)
        await self._session.flush()

    async def save_quiz(self, task_id: str, content: dict[str, Any]) -> None:
        await self._session.merge(quiz_row(task_id, content))
        await self._session.flush()

    async def save_citations(self, task_id: str, content: dict[str, Any]) -> None:
        await self._session.execute(delete(CitationRow).where(CitationRow.task_id == task_id))
        for row in citation_rows(task_id, content):
            self._session.add(row)
        await self._session.flush()
