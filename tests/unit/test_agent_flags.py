"""Unit tests for the AgentBase resilience flags (FORCE_REFUSE / HANG_AGENT)
and adapter-error handling (ImportError -> graceful refuse)."""

from __future__ import annotations

import asyncio

import pytest
from src.adapters.transcriber import FakeTranscriberAdapter
from src.agents.base import AgentBase
from src.agents.transcriber import TranscriberAgent
from src.core.messages import Message, Performative, make_message
from src.core.schemas import Operation

from tests.support.fake_bus import FakeBus


def _request() -> Message:
    return make_message(
        performative=Performative.REQUEST,
        sender="CoordinatorAgent",
        receiver="TranscriberAgent",
        task_id="task-x",
        conversation_id="conv-x",
        content={"document_id": "doc-task-x-0", "file_path": "/x.mp3", "language": "ru"},
        subtask_id="st-task-x-F1",
    )


def _agent() -> TranscriberAgent:
    return TranscriberAgent(bus=FakeBus(), transcriber=FakeTranscriberAdapter())


async def test_force_refuse_makes_agent_refuse(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FORCE_REFUSE", "F1")
    reply = await _agent()._safe_handle(_request())
    assert reply is not None
    assert reply.performative == Performative.REFUSE
    assert "force_refuse" in reply.content["reason"]


async def test_force_refuse_other_op_does_not_affect(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FORCE_REFUSE", "F6")
    reply = await _agent()._safe_handle(_request())
    assert reply is not None
    assert reply.performative == Performative.INFORM


async def test_hang_agent_never_returns(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HANG_AGENT", "F1")
    with pytest.raises(TimeoutError):
        async with asyncio.timeout(0.3):
            await _agent()._safe_handle(_request())


class _ImportErrorAgent(AgentBase):
    """Stub agent whose handle() raises ModuleNotFoundError — simulates a missing ML lib."""

    name = "ImportErrorAgent"

    def __init__(self) -> None:
        super().__init__(
            bus=FakeBus(),
            channel="agent.import_error_stub",
            group="worker-import_error_stub",
            operation=Operation.F1_TRANSCRIBE,
        )

    async def handle(self, message: Message) -> Message | None:
        raise ModuleNotFoundError("faster_whisper")


async def test_module_not_found_becomes_refuse() -> None:
    """ImportError from a handle() call must produce a refuse, not an unhandled exception."""
    agent = _ImportErrorAgent()
    reply = await agent._safe_handle(_request())
    assert reply is not None
    assert reply.performative == Performative.REFUSE
    assert "adapter error" in reply.content["reason"]
