"""In-memory test doubles for Coordinator unit tests.

FakeBus records published messages per channel and serves coordinator-inbox
replies fed by the test, with no real Redis. FakeTaskStore records the last
persisted status and artifact.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Any

from src.core.bus import COORDINATOR_INBOX

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from src.core.messages import Message


class FakeBus:
    """Minimal RedisStreamBus stand-in for deterministic loop tests."""

    def __init__(self) -> None:
        self.published: dict[str, list[Message]] = defaultdict(list)
        self.acked: list[str] = []
        self._inbox: list[tuple[str, Message]] = []
        self._counter = 0

    async def ensure_group(self, channel: str, group: str) -> None:
        return None

    async def publish(self, channel: str, message: Message) -> str:
        self.published[channel].append(message)
        self._counter += 1
        return f"fake-{self._counter}"

    def feed_inbox(self, reply: Message) -> None:
        """Queue a reply as if an agent had published it to the coordinator inbox."""
        self._counter += 1
        self._inbox.append((f"in-{self._counter}", reply))

    async def read(
        self, channel: str, group: str, *, count: int = 1, block_ms: int = 5000
    ) -> AsyncIterator[tuple[str, Message]]:
        if channel != COORDINATOR_INBOX:
            return
        batch = self._inbox[:count]
        del self._inbox[:count]
        for entry_id, message in batch:
            yield entry_id, message

    async def ack(self, channel: str, group: str, entry_id: str) -> None:
        self.acked.append(entry_id)

    def requests_for(self, channel: str) -> list[Message]:
        return list(self.published.get(channel, []))


class FakeTaskStore:
    """Records the last persisted status and artifact; no DB."""

    def __init__(self) -> None:
        self.status: dict[str, str] = {}
        self.artifacts: dict[str, dict[str, Any]] = {}
        self.stats: dict[str, dict[str, Any]] = {}
        self.results: list[tuple[str, Any, dict[str, Any]]] = []

    async def set_status(self, task_id: str, status: Any) -> None:
        self.status[task_id] = status.value

    async def save_artifact(self, task_id: str, *, final_artifact: dict[str, Any], stats: dict[str, Any]) -> None:
        self.artifacts[task_id] = final_artifact
        self.stats[task_id] = stats

    async def save_result(self, task_id: str, operation: Any, content: dict[str, Any]) -> None:
        self.results.append((task_id, operation, content))
