"""FastAPI application entry point with lifespan-managed agents."""

from __future__ import annotations

import asyncio
import contextlib
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from fastapi import FastAPI
from redis.asyncio import Redis

from src.adapters.ocr import FakeOcrAdapter
from src.adapters.transcriber import FakeTranscriberAdapter
from src.agents.ocr import OcrAgent
from src.agents.transcriber import TranscriberAgent
from src.api.routes import router as api_router
from src.config import get_settings
from src.core.bus import RedisStreamBus
from src.db.session import create_engine_and_session

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Wire DB / Redis / agents on startup; tear them down on shutdown."""
    settings = get_settings()
    engine, session_factory = create_engine_and_session(settings.database_url)
    redis = Redis.from_url(settings.redis_url, decode_responses=True)
    bus = RedisStreamBus(redis)
    transcriber_agent = TranscriberAgent(bus=bus, transcriber=FakeTranscriberAdapter())
    ocr_agent = OcrAgent(bus=bus, ocr=FakeOcrAdapter())
    agents = [transcriber_agent, ocr_agent]
    tasks = [asyncio.create_task(agent.run()) for agent in agents]

    app.state.engine = engine
    app.state.session_factory = session_factory
    app.state.redis = redis
    app.state.bus = bus
    app.state.agents = agents
    app.state.agent_tasks = tasks
    app.state.dispatch_tasks = set()

    try:
        yield
    finally:
        dispatch_tasks: set[asyncio.Task[None]] = app.state.dispatch_tasks
        for dispatch in list(dispatch_tasks):
            dispatch.cancel()
        for agent in agents:
            agent.shutdown()
        for task in tasks:
            task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.gather(*tasks, *dispatch_tasks, return_exceptions=False)
        await redis.aclose()
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
