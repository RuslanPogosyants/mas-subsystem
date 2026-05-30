"""FastAPI application entry point with lifespan-managed agents."""

from __future__ import annotations

import asyncio
import contextlib
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Final

from fastapi import FastAPI, Response
from redis.asyncio import Redis

from src.adapters.embedding import FakeEmbeddingAdapter
from src.adapters.llm import FakeLlmAdapter
from src.adapters.ner import FakeNerAdapter
from src.adapters.ocr import FakeOcrAdapter
from src.adapters.transcriber import FakeTranscriberAdapter
from src.agents.coordinator import Coordinator
from src.agents.ocr import OcrAgent
from src.agents.recommender import CorpusEntry, RecommenderAgent, load_corpus
from src.agents.recovery import DbTaskRecovery
from src.agents.store import DbTaskStore
from src.agents.summarizer import SummarizerAgent
from src.agents.terminology import TerminologyAgent
from src.agents.test_generator import TestGeneratorAgent
from src.agents.transcriber import TranscriberAgent
from src.api.routes import router as api_router
from src.config import get_settings
from src.core.bus import RedisStreamBus
from src.core.logging import configure_logging
from src.core.metrics import CONTENT_TYPE, render
from src.db.session import create_engine_and_session

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from src.adapters.embedding import EmbeddingAdapter
    from src.adapters.llm import LlmAdapter
    from src.adapters.ner import NerAdapter
    from src.adapters.ocr import OcrAdapter
    from src.adapters.transcriber import TranscriberAdapter
    from src.agents.base import AgentBase
    from src.config import Settings


def _build_llm(settings: Settings) -> LlmAdapter:
    """Real GigaChat when credentials are configured, else the in-process fake.

    Tests blank GIGACHAT_CREDENTIALS (see tests/conftest.py) so they always get the
    fake and never reach the live API; CI has no credentials configured either.
    """
    if settings.gigachat_credentials:
        from src.adapters.gigachat import GigaChatAdapter

        return GigaChatAdapter(settings)
    return FakeLlmAdapter()


def _build_embedding(settings: Settings) -> EmbeddingAdapter:
    """Real sentence-transformers adapter when a corpus is present, else the fake.

    A real corpus carries real-dimensional vectors, so the query must be embedded
    by the same model — pairing the real adapter with the corpus prevents a
    dimension mismatch. Without corpus files (CI/dev) the fake keeps imports light.
    """
    base = Path(settings.corpus_path)
    if (base / "papers.jsonl").exists() and (base / "papers.npy").exists():
        from src.adapters.sentence_transformer import SentenceTransformerEmbeddingAdapter

        return SentenceTransformerEmbeddingAdapter(settings.embedding_model)
    return FakeEmbeddingAdapter()


def _build_transcriber(settings: Settings) -> TranscriberAdapter:
    """Real Whisper adapter when backend is configured, else the in-process fake."""
    if settings.transcriber_backend == "whisper":
        from src.adapters.whisper_transcriber import WhisperTranscriberAdapter

        return WhisperTranscriberAdapter(
            model_size=settings.whisper_model,
            device=settings.whisper_device,
            compute_type=settings.whisper_compute_type,
        )
    return FakeTranscriberAdapter()


def _build_ocr(settings: Settings) -> OcrAdapter:
    """Real PyMuPDF+EasyOCR adapter when backend is configured, else the fake."""
    if settings.ocr_backend == "pymupdf":
        from src.adapters.pymupdf_ocr import PymupdfOcrAdapter

        return PymupdfOcrAdapter(languages=[s.strip() for s in settings.ocr_languages.split(",") if s.strip()])
    return FakeOcrAdapter()


def _build_ner(settings: Settings) -> NerAdapter:
    """Real spaCy NER adapter when backend is configured, else the in-process fake."""
    if settings.ner_backend == "spacy":
        from src.adapters.spacy_ner import SpacyNerAdapter

        return SpacyNerAdapter(model=settings.spacy_model)
    return FakeNerAdapter()


_DEMO_PAPERS: Final[tuple[tuple[str, int, str], ...]] = (
    ("Graph algorithms and data structures", 2021, "https://example.org/graphs"),
    ("An introduction to machine learning", 2020, "https://example.org/ml"),
    ("Methods in natural language processing", 2022, "https://example.org/nlp"),
    ("Information retrieval foundations", 2019, "https://example.org/ir"),
)


async def _demo_corpus(embedding: EmbeddingAdapter) -> list[CorpusEntry]:
    """A tiny built-in corpus so F6 returns citations without a real corpus.

    Embeddings come from the active embedding adapter, so their dimension always
    matches the query (fake 16-dim in CI/demo, real model dims otherwise).
    """
    vectors = await embedding.encode([title for title, _, _ in _DEMO_PAPERS])
    return [
        CorpusEntry(title=title, authors=None, year=year, url=url, embedding=tuple(vector))
        for (title, year, url), vector in zip(_DEMO_PAPERS, vectors, strict=False)
    ]


async def _build_recommender(bus: RedisStreamBus, settings: Settings) -> RecommenderAgent:
    """Build F6 with a real corpus when present; in demo_mode fall back to a
    built-in demo corpus, otherwise an empty corpus (the agent then refuses)."""
    embedding = _build_embedding(settings)
    corpus = load_corpus(settings.corpus_path)
    if not corpus and settings.demo_mode:
        corpus = await _demo_corpus(embedding)
    return RecommenderAgent(bus=bus, embedding=embedding, corpus=corpus)


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


def _attach_state(
    app: FastAPI,
    engine: object,
    session_factory: object,
    redis: Redis,
    bus: RedisStreamBus,
    agents: list[AgentBase],
    coordinator: Coordinator,
    agent_tasks: list[asyncio.Task[None]],
    coordinator_task: asyncio.Task[None],
) -> None:
    """Store runtime objects on app.state for access by request handlers."""
    app.state.engine = engine
    app.state.session_factory = session_factory
    app.state.redis = redis
    app.state.bus = bus
    app.state.agents = agents
    app.state.coordinator = coordinator
    app.state.agent_tasks = agent_tasks
    app.state.coordinator_task = coordinator_task


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Wire DB / Redis / agents / coordinator on startup; tear down on shutdown."""
    settings = get_settings()
    configure_logging(settings)
    engine, session_factory = create_engine_and_session(settings.database_url)
    redis = Redis.from_url(settings.redis_url, decode_responses=True)
    bus = RedisStreamBus(redis)

    llm = _build_llm(settings)
    transcriber_agent = TranscriberAgent(bus=bus, transcriber=_build_transcriber(settings))
    ocr_agent = OcrAgent(bus=bus, ocr=_build_ocr(settings))
    summarizer_agent = SummarizerAgent(
        bus=bus,
        llm=llm,
        block_chars=settings.summarizer_block_chars,
        overlap=settings.summarizer_overlap,
    )
    test_generator_agent = TestGeneratorAgent(bus=bus, llm=llm)
    terminology_agent = TerminologyAgent(bus=bus, ner=_build_ner(settings))
    recommender_agent = await _build_recommender(bus, settings)
    agents = [
        transcriber_agent,
        ocr_agent,
        summarizer_agent,
        test_generator_agent,
        terminology_agent,
        recommender_agent,
    ]
    coordinator = Coordinator(
        bus=bus,
        store=DbTaskStore(session_factory),
        recovery=DbTaskRecovery(session_factory),
        agent_timeouts=_agent_timeouts(settings),
    )

    agent_tasks = [asyncio.create_task(agent.run()) for agent in agents]
    coordinator_task = asyncio.create_task(coordinator.run())
    _attach_state(app, engine, session_factory, redis, bus, agents, coordinator, agent_tasks, coordinator_task)

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


@app.get("/metrics")
async def metrics() -> Response:
    """Prometheus exposition endpoint (default registry)."""
    return Response(content=render(), media_type=CONTENT_TYPE)
