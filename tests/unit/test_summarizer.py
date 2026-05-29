"""Unit tests for SummarizerAgent (map-reduce, parse/retry, refuse-on-empty)."""

from __future__ import annotations

import json

from src.adapters.llm import FakeLlmAdapter
from src.agents.summarizer import SummarizerAgent
from src.core.messages import Message, Performative, make_message
from src.core.schemas import Summary

from tests.support.fake_bus import FakeBus


def _request(content: dict[str, object]) -> Message:
    return make_message(
        performative=Performative.REQUEST,
        sender="CoordinatorAgent",
        receiver="SummarizerAgent",
        task_id="task-1",
        conversation_id="conv-1",
        content=content,
        subtask_id="st-task-1-F3",
    )


def _valid_json(intro: str = "i", kp: str = "k", concl: str = "c") -> str:
    return json.dumps({"introduction": intro, "key_points": kp, "conclusions": concl})


def _agent(llm: FakeLlmAdapter, *, block_chars: int = 6000, overlap: int = 500) -> SummarizerAgent:
    # FakeBus is a typed bus double; handle() never touches the bus, so this never publishes.
    return SummarizerAgent(bus=FakeBus(), llm=llm, block_chars=block_chars, overlap=overlap)


async def test_refuses_when_no_chunk_content() -> None:
    agent = _agent(FakeLlmAdapter())
    reply = await agent.handle(_request({"chunks": []}))
    assert reply is not None and reply.performative == Performative.REFUSE
    assert "no chunk content" in reply.content["reason"]


async def test_refuses_when_chunks_have_only_empty_content() -> None:
    agent = _agent(FakeLlmAdapter())
    reply = await agent.handle(_request({"chunks": [{"id": "c1", "content": ""}]}))
    assert reply is not None and reply.performative == Performative.REFUSE


async def test_single_call_produces_valid_summary_with_thesis_mapping() -> None:
    llm = FakeLlmAdapter(responses=[_valid_json(intro="вступление", kp="тезис", concl="вывод")])
    agent = _agent(llm)
    reply = await agent.handle(_request({"chunks": [{"id": "c1", "content": "короткий текст"}]}))
    assert reply is not None and reply.performative == Performative.INFORM
    assert len(llm.calls) == 1
    summary = Summary.model_validate(reply.content)
    by_type = {section.type: section.text for section in summary.sections}
    assert by_type["thesis"] == "тезис"
    assert summary.source_chunk_ids == ["c1"]


async def test_map_reduce_calls_llm_per_block_plus_reduce() -> None:
    llm = FakeLlmAdapter()
    agent = _agent(llm, block_chars=10, overlap=2)
    text = "x" * 25
    reply = await agent.handle(_request({"chunks": [{"id": "c1", "content": text}]}))
    assert reply is not None and reply.performative == Performative.INFORM
    assert len(llm.calls) == 5


async def test_retries_once_on_invalid_json_then_succeeds() -> None:
    llm = FakeLlmAdapter(responses=["not json", _valid_json()])
    agent = _agent(llm)
    reply = await agent.handle(_request({"chunks": [{"id": "c1", "content": "текст"}]}))
    assert reply is not None and reply.performative == Performative.INFORM
    assert len(llm.calls) == 2


async def test_refuses_after_retries_exhausted() -> None:
    llm = FakeLlmAdapter(responses=["bad", "bad", "bad", "bad"])
    agent = _agent(llm)
    reply = await agent.handle(_request({"chunks": [{"id": "c1", "content": "текст"}]}))
    assert reply is not None and reply.performative == Performative.REFUSE
    assert "invalid summary json" in reply.content["reason"]
    assert len(llm.calls) == 3
