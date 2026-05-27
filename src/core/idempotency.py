"""Receive-side idempotency: LRU cache of message_id values seen recently.

Protects workers from processing duplicate messages delivered by Redis Streams.
"""

from __future__ import annotations

from collections import OrderedDict


class IdempotentReceiver:
    """LRU cache over the last N message_id values."""

    def __init__(self, cache_size: int = 1000) -> None:
        self._cache_size = cache_size
        self._seen: OrderedDict[str, None] = OrderedDict()

    def accept(self, message_id: str) -> bool:
        """Accept a message. Returns True if new, False if duplicate."""
        raise NotImplementedError("accept: implemented in M1")
