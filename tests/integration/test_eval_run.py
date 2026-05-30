"""Integration: evaluation.run scores a seeded finished task and writes reports."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from src.db.models import Base
from src.db.repos import TaskRepo
from src.evaluation.run import _run

if TYPE_CHECKING:
    from pathlib import Path


@pytest_asyncio.fixture
async def seeded_url(postgres_url: str) -> str:
    engine = create_async_engine(postgres_url)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as session:
        repo = TaskRepo(session)
        await repo.create(task_id="task-eval", requested_outputs=[])
        artifact = {
            "summary": {"sections": [{"type": "thesis", "text": "Граф — структура."}]},
            "terms": [{"term": "граф", "frequency": 2, "category": "x"}],
            "quiz": [{"type": "open_answer", "question": "Что такое граф?"}],
            "citations": [{"title": "Graphs", "relevance_score": 0.7}],
        }
        await repo.save_artifact(
            "task-eval",
            final_artifact={"result": artifact, "stats": {}},
            stats={},
        )
        await session.commit()
    try:
        yield postgres_url
    finally:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
        await engine.dispose()


@pytest.mark.integration
async def test_run_writes_markdown_and_json(
    seeded_url: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DATABASE_URL", seeded_url)
    out = tmp_path / "eval.md"
    await _run("task-eval", source_path="", out_path=str(out))
    assert out.exists()
    data = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
    assert data["task_id"] == "task-eval"
