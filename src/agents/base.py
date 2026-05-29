"""AgentBase: XREADGROUP loop with idempotency, handler dispatch, ack."""

from __future__ import annotations

import asyncio
import os
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Final

from loguru import logger

from src.core.bus import COORDINATOR_INBOX
from src.core.idempotency import IdempotentReceiver
from src.core.messages import Message, Performative, make_message

if TYPE_CHECKING:
    from src.core.bus import RedisStreamBus
    from src.core.schemas import Operation

_HANG_SECONDS: Final[float] = 86_400.0


class AgentBase(ABC):
    """Read its inbox, run a handler, publish the reply to coordinator inbox.

    Subclasses implement `handle(message)` and return either an `inform`-typed
    Message (success) or a `refuse`-typed Message (failure with reason). The
    correlator `subtask_id` is echoed back automatically by `_reply_for`.
    """

    name: str

    def __init__(self, *, bus: RedisStreamBus, channel: str, group: str, operation: Operation) -> None:
        self._bus = bus
        self._channel = channel
        self._group = group
        self._operation = operation
        self._idempotency = IdempotentReceiver()
        self._shutdown = asyncio.Event()

    async def run(self) -> None:
        """Main loop. Cancel-safe via `shutdown()` or asyncio.Task.cancel()."""
        await self._bus.ensure_group(self._channel, self._group)
        logger.info(f"agent {self.name} listening on {self._channel}")
        while not self._shutdown.is_set():
            try:
                await self._read_once()
            except asyncio.CancelledError:
                raise
            except Exception as error:
                logger.exception(f"agent {self.name} loop error: {error}")
                await asyncio.sleep(0.5)

    def shutdown(self) -> None:
        self._shutdown.set()

    async def _read_once(self) -> None:
        async for entry_id, message in self._bus.read(self._channel, self._group, block_ms=1000):
            if not self._idempotency.accept(message.message_id):
                await self._bus.ack(self._channel, self._group, entry_id)
                continue
            reply = await self._safe_handle(message)
            if reply is not None:
                await self._bus.publish(COORDINATOR_INBOX, reply)
            await self._bus.ack(self._channel, self._group, entry_id)

    async def _safe_handle(self, message: Message) -> Message | None:
        """Handle a message; honour resilience flags; convert adapter failures to refuse.

        Resilience flags FORCE_REFUSE / HANG_AGENT (env, value is an op code like
        "F6") are read fresh per message so tests may set them after construction.
        Programming errors (AttributeError, TypeError, ValueError) propagate to the
        outer run() loop and surface in logs — they must not be masked as refuse.
        """
        if self._is_flagged("FORCE_REFUSE"):
            return self._refuse(message, reason="force_refuse flag enabled")
        if self._is_flagged("HANG_AGENT"):
            await asyncio.sleep(_HANG_SECONDS)
        try:
            return await self.handle(message)
        except (OSError, ConnectionError, TimeoutError, RuntimeError) as error:
            logger.exception(f"agent {self.name} adapter error: {error}")
            return self._refuse(message, reason=f"adapter error: {error.__class__.__name__}")

    def _is_flagged(self, env_var: str) -> bool:
        return os.environ.get(env_var, "") == self._operation.value

    def _inform(self, request: Message, *, content: dict[str, object]) -> Message:
        return make_message(
            performative=Performative.INFORM,
            sender=self.name,
            receiver=request.sender,
            task_id=request.task_id,
            conversation_id=request.conversation_id,
            content=content,
            in_reply_to=request.message_id,
            subtask_id=request.subtask_id,
        )

    def _refuse(self, request: Message, *, reason: str) -> Message:
        return make_message(
            performative=Performative.REFUSE,
            sender=self.name,
            receiver=request.sender,
            task_id=request.task_id,
            conversation_id=request.conversation_id,
            content={"reason": reason},
            in_reply_to=request.message_id,
            subtask_id=request.subtask_id,
        )

    @abstractmethod
    async def handle(self, message: Message) -> Message | None:
        """Process one request message; return inform/refuse, or None to skip."""
