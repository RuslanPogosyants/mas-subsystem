"""Root conftest: session fixtures for Redis and Postgres testcontainers."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest
import pytest_asyncio

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator
    from pathlib import Path

    from testcontainers.postgres import PostgresContainer
    from testcontainers.redis import RedisContainer


@pytest.fixture(autouse=True)
def _force_fake_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests never call the live GigaChat API: blank the credentials so the app
    lifespan selects FakeLlmAdapter regardless of the developer's local .env."""
    monkeypatch.setenv("GIGACHAT_CREDENTIALS", "")


@pytest.fixture(autouse=True)
def _force_fake_backends(monkeypatch: pytest.MonkeyPatch) -> None:
    """Runtime defaults are the real ML backends; tests force the in-process Fakes
    so the suite needs no GPU, model weights, or network. Tests that want a real
    backend override these via their own monkeypatch."""
    monkeypatch.setenv("TRANSCRIBER_BACKEND", "fake")
    monkeypatch.setenv("OCR_BACKEND", "fake")
    monkeypatch.setenv("NER_BACKEND", "fake")


@pytest.fixture(autouse=True)
def _enable_demo_corpus(monkeypatch: pytest.MonkeyPatch) -> None:
    """Production defaults demo_mode off, so F6 refuses without a real corpus. The
    test suite exercises the full F6 path with the built-in demo corpus, so enable
    it here. Tests asserting the off behaviour override DEMO_MODE via monkeypatch."""
    monkeypatch.setenv("DEMO_MODE", "true")


@pytest.fixture(autouse=True)
def _isolate_corpus(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A real recommender corpus is committed at the default 'corpus/' path, which
    would make the app load the e5 model. Point tests at an empty dir so F6 stays on
    the fake/demo path and no model loads. Tests override CORPUS_PATH as needed."""
    monkeypatch.setenv("CORPUS_PATH", str(tmp_path))


@pytest.fixture(scope="session")
def event_loop_policy() -> asyncio.AbstractEventLoopPolicy:
    """Session-scoped event loop policy."""
    return asyncio.DefaultEventLoopPolicy()


@pytest.fixture(scope="session")
def postgres_container() -> Iterator[PostgresContainer]:
    """Postgres 16 container, shared across the test session."""
    from testcontainers.postgres import PostgresContainer

    container = PostgresContainer("postgres:16-alpine", username="test", password="test", dbname="test")
    container.start()
    try:
        yield container
    finally:
        container.stop()


@pytest.fixture(scope="session")
def redis_container() -> Iterator[RedisContainer]:
    """Redis 7 container, shared across the test session."""
    from testcontainers.redis import RedisContainer

    container = RedisContainer("redis:7-alpine")
    container.start()
    try:
        yield container
    finally:
        container.stop()


@pytest.fixture
def postgres_url(postgres_container: PostgresContainer) -> str:
    """asyncpg URL for the test Postgres."""
    url = postgres_container.get_connection_url()
    return url.replace("postgresql+psycopg2://", "postgresql+asyncpg://")


@pytest.fixture
def redis_url(redis_container: RedisContainer) -> str:
    """Redis URL for the test container."""
    host = redis_container.get_container_host_ip()
    port = redis_container.get_exposed_port(6379)
    return f"redis://{host}:{port}/0"


@pytest_asyncio.fixture
async def clean_redis(redis_url: str) -> AsyncIterator[str]:
    """Redis URL guaranteed to have an empty keyspace."""
    import redis.asyncio as aioredis

    client = aioredis.from_url(redis_url, decode_responses=True)
    await client.flushdb()
    yield redis_url
    await client.flushdb()
    await client.aclose()
