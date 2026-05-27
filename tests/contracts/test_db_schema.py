"""Contract tests: Postgres schema matches the VKR ER diagram (figure 2.9)."""

from __future__ import annotations

import asyncio
import os
import sys
from typing import TYPE_CHECKING, Final

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine

EXPECTED_TABLES: Final[frozenset[str]] = frozenset(
    {"tasks", "documents", "text_chunks", "summaries", "terms", "quizzes", "citations"}
)


@pytest.fixture
async def engine_with_migrations(postgres_url: str) -> AsyncEngine:
    """Engine pointing at a database with alembic migrations applied."""
    env = dict(os.environ)
    env["DATABASE_URL"] = postgres_url

    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "alembic",
        "upgrade",
        "head",
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()
    assert proc.returncode == 0, f"alembic failed: {stderr.decode()}"

    return create_async_engine(postgres_url)


@pytest.mark.integration
class TestSchemaMatchesERDiagram:
    async def test_all_seven_tables_exist(self, engine_with_migrations: AsyncEngine) -> None:
        async with engine_with_migrations.connect() as conn:
            result = await conn.execute(
                text(
                    "SELECT table_name FROM information_schema.tables "
                    "WHERE table_schema = 'public' AND table_type = 'BASE TABLE'"
                )
            )
            tables = {row[0] for row in result.all()}
        assert EXPECTED_TABLES.issubset(tables), f"missing tables: {EXPECTED_TABLES - tables}"

    async def test_tasks_has_required_columns(self, engine_with_migrations: AsyncEngine) -> None:
        async with engine_with_migrations.connect() as conn:
            result = await conn.execute(
                text("SELECT column_name FROM information_schema.columns " "WHERE table_name = 'tasks'")
            )
            columns = {row[0] for row in result.all()}
        expected = {
            "id",
            "user_id",
            "status",
            "created_at",
            "updated_at",
            "requested_outputs",
            "final_artifact",
            "stats",
        }
        assert expected.issubset(columns), f"tasks missing: {expected - columns}"

    async def test_text_chunks_has_task_and_document_fk(self, engine_with_migrations: AsyncEngine) -> None:
        async with engine_with_migrations.connect() as conn:
            result = await conn.execute(
                text("SELECT column_name FROM information_schema.columns " "WHERE table_name = 'text_chunks'")
            )
            columns = {row[0] for row in result.all()}
        expected = {
            "id",
            "task_id",
            "document_id",
            "source_type",
            "content",
            "chunk_index",
            "confidence",
        }
        assert expected.issubset(columns)

    async def test_indices_exist(self, engine_with_migrations: AsyncEngine) -> None:
        async with engine_with_migrations.connect() as conn:
            result = await conn.execute(text("SELECT indexname FROM pg_indexes WHERE schemaname = 'public'"))
            indices = {row[0] for row in result.all()}
        assert "idx_text_chunks_task" in indices
        assert "idx_terms_task" in indices
