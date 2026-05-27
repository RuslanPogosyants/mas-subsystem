"""Async session factory for SQLAlchemy 2.0."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


def create_engine_and_session(
    database_url: str,
) -> tuple[Any, async_sessionmaker[AsyncSession]]:
    """Create async engine and sessionmaker for the given database URL."""
    engine = create_async_engine(database_url, echo=False, pool_pre_ping=True)
    session_factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    return engine, session_factory


async def yield_session(
    session_factory: async_sessionmaker[AsyncSession],
) -> AsyncIterator[AsyncSession]:
    """FastAPI Depends-compatible async session generator."""
    async with session_factory() as session:
        yield session
