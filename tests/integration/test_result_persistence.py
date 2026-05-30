"""Integration: agent results persist into the four output tables."""

from __future__ import annotations

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from src.agents.store import DbTaskStore
from src.core.schemas import DocumentType, Operation
from src.db.models import Base, CitationRow, QuizRow, SummaryRow, TermRow, TextChunkRow
from src.db.repos import DocumentRepo, ResultRepo, TaskRepo


@pytest_asyncio.fixture
async def session_factory(postgres_url: str):
    engine = create_async_engine(postgres_url)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    try:
        yield factory
    finally:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
        await engine.dispose()


@pytest.mark.integration
class TestResultPersistence:
    async def test_results_persist_into_tables(self, postgres_url: str) -> None:
        engine = create_async_engine(postgres_url)
        try:
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            session_factory = async_sessionmaker(engine, expire_on_commit=False)
            async with session_factory() as session:
                await TaskRepo(session).create(task_id="task-1", requested_outputs=[Operation.F3_SUMMARIZE])
                await session.commit()

            store = DbTaskStore(session_factory)
            await store.save_result(
                "task-1",
                Operation.F3_SUMMARIZE,
                {"summary_id": "sum-1", "sections": [{"type": "thesis", "text": "T"}], "source_chunk_ids": []},
            )
            await store.save_result(
                "task-1", Operation.F4_TEST, {"quiz_id": "quiz-1", "questions": [{"q": 1}], "difficulty": "easy"}
            )
            await store.save_result("task-1", Operation.F5_TERMS, {"terms": [{"term": "граф", "frequency": 1}]})
            await store.save_result(
                "task-1", Operation.F6_RECOMMEND, {"citations": [{"title": "Graphs", "relevance_score": 0.5}]}
            )

            async with session_factory() as session:
                summary = (await session.execute(select(SummaryRow))).scalar_one()
                assert summary.key_points == "T"  # thesis -> key_points
                assert (await session.execute(select(QuizRow))).scalar_one().difficulty == "easy"
                assert (await session.execute(select(TermRow))).scalar_one().term == "граф"
                assert (await session.execute(select(CitationRow))).scalar_one().title == "Graphs"

            await store.save_result(
                "task-1",
                Operation.F3_SUMMARIZE,
                {"summary_id": "sum-1", "sections": [{"type": "thesis", "text": "T2"}], "source_chunk_ids": []},
            )
            async with session_factory() as session:
                summaries = (await session.execute(select(SummaryRow))).scalars().all()
                assert len(summaries) == 1
                assert summaries[0].key_points == "T2"
        finally:
            await engine.dispose()


@pytest.mark.integration
async def test_save_chunks_persists_and_replaces_per_document(session_factory) -> None:
    async with session_factory() as session:
        await TaskRepo(session).create(task_id="t1", requested_outputs=[Operation.F1_TRANSCRIBE])
        await DocumentRepo(session).create(
            document_id="doc-t1-0", task_id="t1", document_type=DocumentType.AUDIO, file_path="/a.mp3"
        )
        await session.commit()

    two = {
        "chunks": [
            {
                "id": "chunk-doc-t1-0-0",
                "task_id": "t1",
                "document_id": "doc-t1-0",
                "source_type": "audio",
                "content": "a",
                "chunk_index": 0,
                "confidence": None,
                "meta": {},
            },
            {
                "id": "chunk-doc-t1-0-1",
                "task_id": "t1",
                "document_id": "doc-t1-0",
                "source_type": "audio",
                "content": "b",
                "chunk_index": 1,
                "confidence": None,
                "meta": {},
            },
        ]
    }
    async with session_factory() as session:
        await ResultRepo(session).save_chunks("t1", two)
        await session.commit()
    async with session_factory() as session:
        rows = (await session.execute(select(TextChunkRow).where(TextChunkRow.task_id == "t1"))).scalars().all()
        assert len(rows) == 2

    # Re-run yields ONE chunk for the same document -> orphan (index 1) must be gone.
    async with session_factory() as session:
        await ResultRepo(session).save_chunks("t1", {"chunks": [two["chunks"][0]]})
        await session.commit()
    async with session_factory() as session:
        rows = (await session.execute(select(TextChunkRow).where(TextChunkRow.task_id == "t1"))).scalars().all()
        assert [r.id for r in rows] == ["chunk-doc-t1-0-0"]


@pytest.mark.integration
async def test_save_terms_replaces_and_drops_orphans(session_factory) -> None:
    async with session_factory() as session:
        await TaskRepo(session).create(task_id="t2", requested_outputs=[Operation.F5_TERMS])
        await session.commit()
    async with session_factory() as session:
        await ResultRepo(session).save_terms(
            "t2",
            {
                "terms": [
                    {"term": "A", "lemma": "a", "frequency": 1, "category": "general", "source_chunk_id": None},
                    {"term": "B", "lemma": "b", "frequency": 1, "category": "general", "source_chunk_id": None},
                ]
            },
        )
        await session.commit()
    # Re-run with fewer terms must not leave the orphan term-t2-1 behind.
    async with session_factory() as session:
        await ResultRepo(session).save_terms(
            "t2",
            {
                "terms": [
                    {"term": "A", "lemma": "a", "frequency": 2, "category": "general", "source_chunk_id": None},
                ]
            },
        )
        await session.commit()
    async with session_factory() as session:
        rows = (await session.execute(select(TermRow).where(TermRow.task_id == "t2"))).scalars().all()
        assert len(rows) == 1
        assert rows[0].frequency == 2
