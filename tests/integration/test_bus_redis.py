"""Integration tests for RedisStreamBus against a real Redis testcontainer."""

from __future__ import annotations

import pytest
from redis.asyncio import Redis
from src.core.bus import RedisStreamBus, channel_for_agent
from src.core.messages import Performative, make_message


@pytest.mark.integration
class TestRedisStreamBus:
    async def test_publish_and_read_round_trip(self, clean_redis: str) -> None:
        redis = Redis.from_url(clean_redis, decode_responses=True)
        try:
            bus = RedisStreamBus(redis)
            channel = channel_for_agent("transcriber")
            group = "worker-transcriber"
            await bus.ensure_group(channel, group)

            outgoing = make_message(
                performative=Performative.REQUEST,
                sender="CoordinatorAgent",
                receiver="TranscriberAgent",
                task_id="task-7c41",
                conversation_id="conv-7c41-1",
                content={"file_path": "/x.mp3"},
                subtask_id="st-task-7c41-F1",
            )
            await bus.publish(channel, outgoing)

            received: list[tuple[str, str]] = []
            async for entry_id, message in bus.read(channel, group, block_ms=500):
                received.append((entry_id, message.message_id))
                await bus.ack(channel, group, entry_id)

            assert len(received) == 1
            assert received[0][1] == outgoing.message_id
        finally:
            await redis.aclose()

    async def test_ensure_group_is_idempotent(self, clean_redis: str) -> None:
        redis = Redis.from_url(clean_redis, decode_responses=True)
        try:
            bus = RedisStreamBus(redis)
            channel = channel_for_agent("ocr")
            await bus.ensure_group(channel, "worker-ocr")
            await bus.ensure_group(channel, "worker-ocr")  # must not raise
        finally:
            await redis.aclose()

    async def test_two_consumers_share_one_group(self, clean_redis: str) -> None:
        """A second consumer in the same group sees messages the first did not ack."""
        redis_a = Redis.from_url(clean_redis, decode_responses=True)
        redis_b = Redis.from_url(clean_redis, decode_responses=True)
        try:
            bus_a = RedisStreamBus(redis_a, consumer_name="a")
            bus_b = RedisStreamBus(redis_b, consumer_name="b")
            channel = channel_for_agent("summarizer")
            group = "worker-summarizer"
            await bus_a.ensure_group(channel, group)

            for index in range(3):
                await bus_a.publish(
                    channel,
                    make_message(
                        performative=Performative.REQUEST,
                        sender="CoordinatorAgent",
                        receiver="SummarizerAgent",
                        task_id=f"task-{index}",
                        conversation_id=f"conv-{index}",
                        content={},
                    ),
                )

            seen_a: list[str] = []
            async for entry_id, _msg in bus_a.read(channel, group, count=2, block_ms=500):
                seen_a.append(entry_id)
                await bus_a.ack(channel, group, entry_id)

            seen_b: list[str] = []
            async for entry_id, _msg in bus_b.read(channel, group, count=10, block_ms=500):
                seen_b.append(entry_id)
                await bus_b.ack(channel, group, entry_id)

            assert len(seen_a) + len(seen_b) == 3
            assert set(seen_a).isdisjoint(set(seen_b))
        finally:
            await redis_a.aclose()
            await redis_b.aclose()
