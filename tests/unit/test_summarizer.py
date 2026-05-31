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


async def test_skips_declined_block_and_summarizes_from_survivor() -> None:
    # Two blocks: the first parses; the second is declined on all 3 attempts and
    # is skipped. A single survivor is returned directly (no reduce) — graceful
    # degradation instead of failing the whole summary (real GigaChat filter case).
    llm = FakeLlmAdapter(responses=[_valid_json(kp="из выжившего блока"), "отказ", "отказ", "отказ"])
    agent = _agent(llm, block_chars=10, overlap=2)
    reply = await agent.handle(_request({"chunks": [{"id": "c1", "content": "x" * 12}]}))
    assert reply is not None and reply.performative == Performative.INFORM
    by_type = {section.type: section.text for section in Summary.model_validate(reply.content).sections}
    assert by_type["thesis"] == "из выжившего блока"


async def test_refuses_only_when_all_blocks_declined() -> None:
    llm = FakeLlmAdapter(responses=["отказ"])  # clamps to last -> every call declined
    agent = _agent(llm, block_chars=10, overlap=2)
    reply = await agent.handle(_request({"chunks": [{"id": "c1", "content": "x" * 12}]}))
    assert reply is not None and reply.performative == Performative.REFUSE


async def test_reduces_when_multiple_blocks_survive() -> None:
    llm = FakeLlmAdapter(responses=[_valid_json(kp="a"), _valid_json(kp="b"), _valid_json(kp="итог")])
    agent = _agent(llm, block_chars=10, overlap=2)
    reply = await agent.handle(_request({"chunks": [{"id": "c1", "content": "x" * 12}]}))
    assert reply is not None and reply.performative == Performative.INFORM
    by_type = {section.type: section.text for section in Summary.model_validate(reply.content).sections}
    assert by_type["thesis"] == "итог"
    assert len(llm.calls) == 3


async def test_falls_back_to_partial_when_reduce_declined() -> None:
    # Both blocks parse, but the reduce step is declined -> the first surviving
    # partial is returned so a usable summary still reaches the user.
    llm = FakeLlmAdapter(responses=[_valid_json(kp="первый"), _valid_json(kp="второй"), "отказ", "отказ", "отказ"])
    agent = _agent(llm, block_chars=10, overlap=2)
    reply = await agent.handle(_request({"chunks": [{"id": "c1", "content": "x" * 12}]}))
    assert reply is not None and reply.performative == Performative.INFORM
    by_type = {section.type: section.text for section in Summary.model_validate(reply.content).sections}
    assert by_type["thesis"] == "первый"


async def test_system_prompt_requires_full_key_points() -> None:
    """_SYSTEM_PROMPT must instruct fuller key_points (all key ideas, not one sentence)."""
    from src.agents.summarizer import _SYSTEM_PROMPT

    prompt_lower = _SYSTEM_PROMPT.lower()
    assert "перечисли все" in prompt_lower
    assert "ключевые тезисы" in prompt_lower


# ---------------------------------------------------------------------------
# Coercion tests (Layer 1): structured fields flattened to plain strings
# ---------------------------------------------------------------------------


async def test_dict_key_points_coerced_to_string() -> None:
    """GigaChat sometimes returns key_points as a dict; must be coerced to non-empty str."""
    import json

    structured_response = json.dumps(
        {
            "introduction": "вступление",
            "key_points": {"a": "тезис один", "b": "тезис два"},
            "conclusions": "итог",
        }
    )
    llm = FakeLlmAdapter(responses=[structured_response])
    agent = _agent(llm)
    reply = await agent.handle(_request({"chunks": [{"id": "c1", "content": "текст про ML"}]}))
    assert reply is not None and reply.performative == Performative.INFORM
    summary = Summary.model_validate(reply.content)
    by_type = {section.type: section.text for section in summary.sections}
    thesis = by_type["thesis"]
    assert "тезис один" in thesis
    assert "тезис два" in thesis


async def test_list_key_points_coerced_to_string() -> None:
    """GigaChat sometimes returns key_points as a list; must be coerced to non-empty str."""
    import json

    structured_response = json.dumps(
        {
            "introduction": "введение",
            "key_points": ["первый", "второй"],
            "conclusions": "заключение",
        }
    )
    llm = FakeLlmAdapter(responses=[structured_response])
    agent = _agent(llm)
    reply = await agent.handle(_request({"chunks": [{"id": "c1", "content": "текст"}]}))
    assert reply is not None and reply.performative == Performative.INFORM
    summary = Summary.model_validate(reply.content)
    by_type = {section.type: section.text for section in summary.sections}
    thesis = by_type["thesis"]
    assert "первый" in thesis
    assert "второй" in thesis


async def test_string_key_points_unchanged() -> None:
    """A plain string key_points must round-trip unchanged."""
    llm = FakeLlmAdapter(responses=[_valid_json(kp="нормальный тезис")])
    agent = _agent(llm)
    reply = await agent.handle(_request({"chunks": [{"id": "c1", "content": "текст"}]}))
    assert reply is not None and reply.performative == Performative.INFORM
    summary = Summary.model_validate(reply.content)
    by_type = {section.type: section.text for section in summary.sections}
    assert by_type["thesis"] == "нормальный тезис"


def test_system_prompt_demands_plain_strings() -> None:
    """_SYSTEM_PROMPT must explicitly instruct plain strings, not objects or arrays."""
    from src.agents.summarizer import _SYSTEM_PROMPT

    prompt_lower = _SYSTEM_PROMPT.lower()
    assert "строк" in prompt_lower
    assert "не объект" in prompt_lower


def test_raw_summary_model_validate_coercion() -> None:
    """Direct _RawSummary.model_validate tests for the coercion logic."""
    from src.agents.summarizer import _RawSummary

    # dict coercion
    raw = _RawSummary.model_validate(
        {"introduction": "i", "key_points": {"a": "тезис один", "b": "тезис два"}, "conclusions": "c"}
    )
    assert "тезис один" in raw.key_points
    assert "тезис два" in raw.key_points
    assert isinstance(raw.key_points, str)

    # list coercion
    raw2 = _RawSummary.model_validate({"introduction": "i", "key_points": ["первый", "второй"], "conclusions": "c"})
    assert "первый" in raw2.key_points
    assert "второй" in raw2.key_points
    assert isinstance(raw2.key_points, str)

    # plain string passthrough
    raw3 = _RawSummary.model_validate({"introduction": "i", "key_points": "уже строка", "conclusions": "c"})
    assert raw3.key_points == "уже строка"

    # nested dict coercion
    raw4 = _RawSummary.model_validate(
        {
            "introduction": "i",
            "key_points": {"outer": {"inner": "вложенный тезис"}},
            "conclusions": "c",
        }
    )
    assert "вложенный тезис" in raw4.key_points
    assert isinstance(raw4.key_points, str)
