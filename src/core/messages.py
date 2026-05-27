"""Pydantic models for Redis Streams bus messages.

Conforms to listings 2.3-2.4 of the VKR thesis: a subset of FIPA-ACL
performatives used between CoordinatorAgent and specialised agents.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict


class Performative(StrEnum):
    """Allowed FIPA-ACL speech acts (VKR thesis, table 2.2)."""

    REQUEST = "request"
    INFORM = "inform"
    REFUSE = "refuse"
    QUERY_REF = "query-ref"


AgentName = str


class Message(BaseModel):
    """Bus message. Field set matches listings 2.3-2.4 of the VKR thesis (v6)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    message_id: str
    performative: Performative
    sender: AgentName
    receiver: AgentName
    task_id: str
    conversation_id: str
    content: dict[str, Any]
    reply_to: str = "agent.coordinator.inbox"
    timestamp: str
    in_reply_to: str | None = None


def make_message(
    *,
    performative: Performative,
    sender: AgentName,
    receiver: AgentName,
    task_id: str,
    conversation_id: str,
    content: dict[str, Any],
    in_reply_to: str | None = None,
) -> Message:
    """Build a Message, auto-populating message_id, timestamp, reply_to."""
    return Message(
        message_id=f"msg-{uuid.uuid4().hex[:8]}",
        performative=performative,
        sender=sender,
        receiver=receiver,
        task_id=task_id,
        conversation_id=conversation_id,
        content=content,
        timestamp=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        in_reply_to=in_reply_to,
    )
