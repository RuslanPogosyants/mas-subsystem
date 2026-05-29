"""Integration tests: agents over a real Redis bus + fake ML adapters."""

from __future__ import annotations

import asyncio
import contextlib

import pytest
from redis.asyncio import Redis
from src.adapters.llm import FakeLlmAdapter
from src.adapters.ner import FakeNerAdapter, TermCandidate
from src.adapters.ocr import FakeOcrAdapter
from src.adapters.transcriber import FakeTranscriberAdapter
from src.agents.ocr import OcrAgent
from src.agents.summarizer import SummarizerAgent
from src.agents.terminology import TerminologyAgent
from src.agents.test_generator import TestGeneratorAgent
from src.agents.transcriber import TranscriberAgent
from src.core.bus import COORDINATOR_INBOX, RedisStreamBus
from src.core.messages import Message, Performative, make_message
from src.core.schemas import Summary


async def _run_agent_until_replied(
    agent: TranscriberAgent | OcrAgent | SummarizerAgent | TestGeneratorAgent | TerminologyAgent,
    bus: RedisStreamBus,
    request: Message,
    *,
    channel: str,
) -> Message:
    await bus.ensure_group(COORDINATOR_INBOX, "coordinator")
    await bus.publish(channel, request)
    agent_task = asyncio.create_task(agent.run())
    try:
        for _ in range(30):
            async for entry_id, reply in bus.read(COORDINATOR_INBOX, "coordinator", block_ms=500):
                await bus.ack(COORDINATOR_INBOX, "coordinator", entry_id)
                return reply
            await asyncio.sleep(0.1)
        raise TimeoutError(f"no reply on {COORDINATOR_INBOX} in time")
    finally:
        agent.shutdown()
        agent_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await agent_task


@pytest.mark.integration
class TestTranscriberAgentRoundTrip:
    async def test_transcriber_replies_inform_with_chunks(self, clean_redis: str) -> None:
        redis = Redis.from_url(clean_redis, decode_responses=True)
        try:
            bus = RedisStreamBus(redis)
            agent = TranscriberAgent(bus=bus, transcriber=FakeTranscriberAdapter())
            request = make_message(
                performative=Performative.REQUEST,
                sender="CoordinatorAgent",
                receiver=agent.name,
                task_id="task-7c41",
                conversation_id="conv-7c41-1",
                content={"file_path": "/x.mp3", "language": "ru"},
                subtask_id="st-task-7c41-F1",
            )
            reply = await _run_agent_until_replied(agent, bus, request, channel="agent.transcriber")
            assert reply.performative == Performative.INFORM
            assert reply.subtask_id == "st-task-7c41-F1"
            assert reply.in_reply_to == request.message_id
            assert "chunks" in reply.content
            assert reply.content["language"] == "ru"
        finally:
            await redis.aclose()

    async def test_transcriber_refuses_when_file_path_missing(self, clean_redis: str) -> None:
        redis = Redis.from_url(clean_redis, decode_responses=True)
        try:
            bus = RedisStreamBus(redis)
            agent = TranscriberAgent(bus=bus, transcriber=FakeTranscriberAdapter())
            request = make_message(
                performative=Performative.REQUEST,
                sender="CoordinatorAgent",
                receiver=agent.name,
                task_id="task-x",
                conversation_id="conv-x",
                content={},
                subtask_id="st-x-F1",
            )
            reply = await _run_agent_until_replied(agent, bus, request, channel="agent.transcriber")
            assert reply.performative == Performative.REFUSE
            assert "file_path" in reply.content["reason"]
        finally:
            await redis.aclose()


@pytest.mark.integration
class TestOcrAgentRoundTrip:
    async def test_ocr_replies_inform_with_pdf_chunks(self, clean_redis: str) -> None:
        redis = Redis.from_url(clean_redis, decode_responses=True)
        try:
            bus = RedisStreamBus(redis)
            agent = OcrAgent(bus=bus, ocr=FakeOcrAdapter())
            request = make_message(
                performative=Performative.REQUEST,
                sender="CoordinatorAgent",
                receiver=agent.name,
                task_id="task-x",
                conversation_id="conv-x",
                content={"file_path": "/paper.pdf", "document_type": "pdf"},
                subtask_id="st-x-F2",
            )
            reply = await _run_agent_until_replied(agent, bus, request, channel="agent.ocr")
            assert reply.performative == Performative.INFORM
            assert reply.subtask_id == "st-x-F2"
            chunks = reply.content["chunks"]
            assert isinstance(chunks, list)
            assert chunks[0]["source_type"] == "pdf_extracted"
        finally:
            await redis.aclose()

    async def test_ocr_refuses_unknown_document_type(self, clean_redis: str) -> None:
        redis = Redis.from_url(clean_redis, decode_responses=True)
        try:
            bus = RedisStreamBus(redis)
            agent = OcrAgent(bus=bus, ocr=FakeOcrAdapter())
            request = make_message(
                performative=Performative.REQUEST,
                sender="CoordinatorAgent",
                receiver=agent.name,
                task_id="task-x",
                conversation_id="conv-x",
                content={"file_path": "/x.docx", "document_type": "docx"},
                subtask_id="st-x-F2",
            )
            reply = await _run_agent_until_replied(agent, bus, request, channel="agent.ocr")
            assert reply.performative == Performative.REFUSE
            assert "pdf or image" in reply.content["reason"]
        finally:
            await redis.aclose()


@pytest.mark.integration
class TestSummarizerAgentRoundTrip:
    async def test_summarizer_replies_inform_with_summary(self, clean_redis: str) -> None:
        redis = Redis.from_url(clean_redis, decode_responses=True)
        try:
            bus = RedisStreamBus(redis)
            agent = SummarizerAgent(bus=bus, llm=FakeLlmAdapter())
            request = make_message(
                performative=Performative.REQUEST,
                sender="CoordinatorAgent",
                receiver=agent.name,
                task_id="task-s1",
                conversation_id="conv-s1",
                content={"chunks": [{"id": "c1", "content": "лекция про графы"}]},
                subtask_id="st-task-s1-F3",
            )
            reply = await _run_agent_until_replied(agent, bus, request, channel="agent.summarizer")
            assert reply.performative == Performative.INFORM
            assert reply.subtask_id == "st-task-s1-F3"
            summary = Summary.model_validate(reply.content)
            assert summary.source_chunk_ids == ["c1"]
        finally:
            await redis.aclose()

    async def test_summarizer_refuses_on_empty_chunks(self, clean_redis: str) -> None:
        redis = Redis.from_url(clean_redis, decode_responses=True)
        try:
            bus = RedisStreamBus(redis)
            agent = SummarizerAgent(bus=bus, llm=FakeLlmAdapter())
            request = make_message(
                performative=Performative.REQUEST,
                sender="CoordinatorAgent",
                receiver=agent.name,
                task_id="task-s2",
                conversation_id="conv-s2",
                content={"chunks": []},
                subtask_id="st-task-s2-F3",
            )
            reply = await _run_agent_until_replied(agent, bus, request, channel="agent.summarizer")
            assert reply.performative == Performative.REFUSE
        finally:
            await redis.aclose()


@pytest.mark.integration
class TestTestGeneratorAgentRoundTrip:
    async def test_test_generator_replies_inform_with_quiz(self, clean_redis: str) -> None:
        redis = Redis.from_url(clean_redis, decode_responses=True)
        try:
            bus = RedisStreamBus(redis)
            quiz_json = (
                '{"questions": [{"question": "Q1?", "type": "single_choice", "choices": ["A", "B"], "answer_idx": 1}]}'
            )
            agent = TestGeneratorAgent(bus=bus, llm=FakeLlmAdapter(responses=[quiz_json]))
            request = make_message(
                performative=Performative.REQUEST,
                sender="CoordinatorAgent",
                receiver=agent.name,
                task_id="task-q1",
                conversation_id="conv-q1",
                content={
                    "summary": {
                        "summary_id": "s1",
                        "sections": [{"type": "thesis", "text": "тема"}],
                        "source_chunk_ids": [],
                    },
                    "num_questions": 1,
                    "difficulty": "medium",
                },
                subtask_id="st-task-q1-F4",
            )
            reply = await _run_agent_until_replied(agent, bus, request, channel="agent.test_generator")
            assert reply.performative == Performative.INFORM
            assert reply.subtask_id == "st-task-q1-F4"
            assert reply.content["questions"][0]["type"] == "single_choice"
        finally:
            await redis.aclose()


@pytest.mark.integration
class TestTerminologyAgentRoundTrip:
    async def test_terminology_replies_inform_with_terms(self, clean_redis: str) -> None:
        redis = Redis.from_url(clean_redis, decode_responses=True)
        try:
            bus = RedisStreamBus(redis)
            ner = FakeNerAdapter(candidates=[TermCandidate(text="граф", lemma="граф", label="MISC")])
            agent = TerminologyAgent(bus=bus, ner=ner, stopwords=set(), domain_categories={})
            request = make_message(
                performative=Performative.REQUEST,
                sender="CoordinatorAgent",
                receiver=agent.name,
                task_id="task-t1",
                conversation_id="conv-t1",
                content={"chunks": [{"id": "c1", "content": "граф"}], "top_n": 5},
                subtask_id="st-task-t1-F5",
            )
            reply = await _run_agent_until_replied(agent, bus, request, channel="agent.terminology")
            assert reply.performative == Performative.INFORM
            assert reply.subtask_id == "st-task-t1-F5"
            assert reply.content["terms"][0]["lemma"] == "граф"
            assert reply.content["terms"][0]["category"] == "MISC"
        finally:
            await redis.aclose()
