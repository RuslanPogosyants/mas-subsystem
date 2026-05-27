"""Redis Streams transport for FIPA-ACL-like agent messages.

Channels follow the convention `agent.<name>`:
    agent.transcriber, agent.ocr, agent.summarizer, agent.test_generator,
    agent.terminology, agent.recommender, agent.coordinator.inbox.

XADD publishes; consumer groups + XREADGROUP read; XACK marks done; XPENDING
+ XCLAIM handle a stuck consumer (spec section 3.3). Each agent is its own
consumer group; the consumer name defaults to a per-process uuid.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from redis.exceptions import ResponseError

from src.core.messages import Message

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from redis.asyncio import Redis


def channel_for_agent(agent: str) -> str:
    """Channel name an agent reads from."""
    if agent == "coordinator":
        return "agent.coordinator.inbox"
    return f"agent.{agent}"


COORDINATOR_INBOX = channel_for_agent("coordinator")


class RedisStreamBus:
    """Async Redis Streams transport wrapping XADD / XREADGROUP / XACK."""

    def __init__(self, redis: Redis, consumer_name: str | None = None) -> None:
        self._redis = redis
        self._consumer = consumer_name or f"consumer-{uuid.uuid4().hex[:8]}"

    @property
    def consumer_name(self) -> str:
        return self._consumer

    async def publish(self, channel: str, message: Message) -> str:
        """XADD a Message to the channel; returns the assigned stream entry id."""
        entry_id = await self._redis.xadd(channel, {"payload": message.model_dump_json()})
        return str(entry_id)

    async def ensure_group(self, channel: str, group: str) -> None:
        """Create the consumer group for `channel` (idempotent)."""
        try:
            await self._redis.xgroup_create(channel, group, id="0", mkstream=True)
        except ResponseError as error:
            if "BUSYGROUP" not in str(error):
                raise

    async def read(
        self, channel: str, group: str, *, count: int = 1, block_ms: int = 5000
    ) -> AsyncIterator[tuple[str, Message]]:
        """XREADGROUP one batch and yield (entry_id, message) pairs.

        Async generator: caller iterates `async for ... in bus.read(...)`. After
        a successful handle, the caller must call `ack(channel, group, entry_id)`.
        """
        response = await self._redis.xreadgroup(
            groupname=group,
            consumername=self._consumer,
            streams={channel: ">"},
            count=count,
            block=block_ms,
        )
        if not response:
            return
        for _stream_name, entries in response:
            for entry_id, payload in entries:
                yield str(entry_id), Message.model_validate_json(payload["payload"])

    async def ack(self, channel: str, group: str, entry_id: str) -> None:
        """XACK one entry. Call only after the handler finished successfully."""
        await self._redis.xack(channel, group, entry_id)

    async def pending_count(self, channel: str, group: str) -> int:
        """Number of XPENDING entries for the group; used by recovery.

        The XPENDING summary form returns `[count, min_id, max_id, consumers]`
        (a list), not a dict — see Redis docs. `count` is element 0.
        """
        summary = await self._redis.xpending(channel, group)
        if not summary:
            return 0
        return int(summary[0])
