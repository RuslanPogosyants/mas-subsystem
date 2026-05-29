"""Integration: Coordinator over real Redis with live F1/F2 agents (fake adapters).

Regression guard for the M2 cross-task reply-loss bug: two concurrent tasks must
each receive their own replies and finalise independently.
"""

from __future__ import annotations

import asyncio
import contextlib

import pytest
from redis.asyncio import Redis
from src.adapters.ocr import FakeOcrAdapter
from src.adapters.transcriber import FakeTranscriberAdapter
from src.agents.coordinator import Coordinator
from src.agents.ocr import OcrAgent
from src.agents.transcriber import TranscriberAgent
from src.core.bus import RedisStreamBus
from src.core.schemas import Document, DocumentType, Operation, Task, TaskStatus

from tests.support.fake_bus import FakeTaskStore


def _task(task_id: str) -> Task:
    return Task(
        id=task_id,
        status=TaskStatus.PLANNING,
        requested_outputs=[Operation.F1_TRANSCRIBE, Operation.F2_OCR],
        conversation_id=f"conv-{task_id}",
        documents=[
            Document(id=f"{task_id}-a", task_id=task_id, document_type=DocumentType.AUDIO, file_path="/x.mp3"),
            Document(id=f"{task_id}-p", task_id=task_id, document_type=DocumentType.PDF, file_path="/x.pdf"),
        ],
    )


@pytest.mark.integration
class TestCoordinatorConcurrentDispatch:
    async def test_two_tasks_finalise_completed_without_stealing_replies(self, clean_redis: str) -> None:
        redis = Redis.from_url(clean_redis, decode_responses=True)
        try:
            bus = RedisStreamBus(redis)
            store = FakeTaskStore()
            coordinator = Coordinator(bus=bus, store=store)
            transcriber = TranscriberAgent(bus=bus, transcriber=FakeTranscriberAdapter())
            ocr = OcrAgent(bus=bus, ocr=FakeOcrAdapter())

            runners = [
                asyncio.create_task(coordinator.run()),
                asyncio.create_task(transcriber.run()),
                asyncio.create_task(ocr.run()),
            ]
            try:
                await coordinator.submit(_task("task-A"))
                await coordinator.submit(_task("task-B"))
                async with asyncio.timeout(20):
                    while store.status.get("task-A") != "completed" or store.status.get("task-B") != "completed":
                        await asyncio.sleep(0.1)
            finally:
                coordinator.shutdown()
                transcriber.shutdown()
                ocr.shutdown()
                for runner in runners:
                    runner.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await asyncio.gather(*runners)

            assert store.artifacts["task-A"]["status"] == "completed"
            assert store.artifacts["task-B"]["status"] == "completed"
            assert set(store.artifacts["task-A"]["operations"]) == {"F1", "F2"}
            assert set(store.artifacts["task-B"]["operations"]) == {"F1", "F2"}
        finally:
            await redis.aclose()
