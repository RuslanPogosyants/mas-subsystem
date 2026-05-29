"""FastAPI application entry point with lifespan-managed agents."""

from __future__ import annotations

import asyncio
import contextlib
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from fastapi import FastAPI
from redis.asyncio import Redis

from src.adapters.llm import FakeLlmAdapter
from src.adapters.ocr import FakeOcrAdapter
from src.adapters.transcriber import FakeTranscriberAdapter
from src.agents.coordinator import Coordinator
from src.agents.ocr import OcrAgent
from src.agents.store import DbTaskStore
from src.agents.summarizer import SummarizerAgent
from src.agents.test_generator import TestGeneratorAgent
from src.agents.transcriber import TranscriberAgent
from src.api.routes import router as api_router
from src.config import get_settings
from src.core.bus import RedisStreamBus
from src.db.session import create_engine_and_session

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from src.agents.base import AgentBase
    from src.config import Settings


def _agent_timeouts(settings: Settings) -> dict[str, float]:
    """Per-agent timeout overrides from settings (honours COORD_TIMEOUT_* env)."""
    return {
        "transcriber": float(settings.coord_timeout_transcriber),
        "ocr": float(settings.coord_timeout_ocr),
        "summarizer": float(settings.coord_timeout_summarizer),
        "test_generator": float(settings.coord_timeout_test_generator),
        "terminology": float(settings.coord_timeout_terminology),
        "recommender": float(settings.coord_timeout_recommender),
    }


async def _teardown(
    coordinator: Coordinator,
    agents: list[AgentBase],
    coordinator_task: asyncio.Task[None],
    agent_tasks: list[asyncio.Task[None]],
    redis: Redis,
) -> None:
    """Shut down coordinator, agents, tasks, and connections in order."""
    coordinator.shutdown()
    for agent in agents:
        agent.shutdown()
    all_tasks = [coordinator_task, *agent_tasks]
    for task in all_tasks:
        task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await asyncio.gather(*all_tasks, return_exceptions=False)
    await redis.aclose()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Wire DB / Redis / agents / coordinator on startup; tear down on shutdown."""
    settings = get_settings()
    engine, session_factory = create_engine_and_session(settings.database_url)
    redis = Redis.from_url(settings.redis_url, decode_responses=True)
    bus = RedisStreamBus(redis)

    transcriber_agent = TranscriberAgent(bus=bus, transcriber=FakeTranscriberAdapter())
    ocr_agent = OcrAgent(bus=bus, ocr=FakeOcrAdapter())
    summarizer_agent = SummarizerAgent(
        bus=bus,
        llm=FakeLlmAdapter(),
        block_chars=settings.summarizer_block_chars,
        overlap=settings.summarizer_overlap,
    )
    test_generator_agent = TestGeneratorAgent(bus=bus, llm=FakeLlmAdapter())
    agents = [transcriber_agent, ocr_agent, summarizer_agent, test_generator_agent]
    coordinator = Coordinator(
        bus=bus,
        store=DbTaskStore(session_factory),
        agent_timeouts=_agent_timeouts(settings),
    )

    agent_tasks = [asyncio.create_task(agent.run()) for agent in agents]
    coordinator_task = asyncio.create_task(coordinator.run())

    app.state.engine = engine
    app.state.session_factory = session_factory
    app.state.redis = redis
    app.state.bus = bus
    app.state.agents = agents
    app.state.coordinator = coordinator
    app.state.agent_tasks = agent_tasks
    app.state.coordinator_task = coordinator_task

    try:
        yield
    finally:
        await _teardown(coordinator, agents, coordinator_task, agent_tasks, redis)
        await engine.dispose()


app = FastAPI(
    title="mas-subsystem",
    description="Multi-agent subsystem for intelligent processing of educational data",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(api_router)


@app.get("/")
async def root() -> dict[str, str]:
    """Health check and pointer to interactive docs."""
    return {"service": "mas-subsystem", "docs": "/docs", "status": "ok"}
