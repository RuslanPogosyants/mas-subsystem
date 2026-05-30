"""Unit tests for TestGeneratorAgent."""

from __future__ import annotations

import json

from src.adapters.llm import FakeLlmAdapter
from src.agents.test_generator import TestGeneratorAgent
from src.core.messages import Message, Performative, make_message
from src.core.schemas import QuizQuestion

from tests.support.fake_bus import FakeBus


def _request(content: dict[str, object]) -> Message:
    return make_message(
        performative=Performative.REQUEST,
        sender="CoordinatorAgent",
        receiver="TestGeneratorAgent",
        task_id="task-1",
        conversation_id="conv-1",
        content=content,
        subtask_id="st-task-1-F4",
    )


def _summary(text: str = "граф это структура данных") -> dict[str, object]:
    return {"summary_id": "sum-1", "sections": [{"type": "thesis", "text": text}], "source_chunk_ids": ["c1"]}


def _quiz_json() -> str:
    return json.dumps(
        {
            "questions": [
                {"question": "Что такое граф?", "type": "single_choice", "choices": ["A", "B"], "answer_idx": 0},
                {"question": "Опишите обход.", "type": "open_answer"},
            ]
        }
    )


def _agent(llm: FakeLlmAdapter) -> TestGeneratorAgent:
    return TestGeneratorAgent(bus=FakeBus(), llm=llm)


async def test_refuses_when_summary_empty() -> None:
    agent = _agent(FakeLlmAdapter(responses=[_quiz_json()]))
    reply = await agent.handle(_request({"summary": {}, "num_questions": 5}))
    assert reply is not None and reply.performative == Performative.REFUSE
    assert "summary" in reply.content["reason"]


async def test_produces_valid_quiz() -> None:
    llm = FakeLlmAdapter(responses=[_quiz_json()])
    agent = _agent(llm)
    reply = await agent.handle(_request({"summary": _summary(), "num_questions": 2, "difficulty": "easy"}))
    assert reply is not None and reply.performative == Performative.INFORM
    assert len(llm.calls) == 1
    questions = reply.content["questions"]
    assert [QuizQuestion.model_validate(q).type for q in questions] == ["single_choice", "open_answer"]
    assert reply.content["difficulty"] == "easy"
    assert reply.content["quiz_id"] == "quiz-task-1"


async def test_retries_once_then_succeeds() -> None:
    llm = FakeLlmAdapter(responses=["not json", _quiz_json()])
    agent = _agent(llm)
    reply = await agent.handle(_request({"summary": _summary()}))
    assert reply is not None and reply.performative == Performative.INFORM
    assert len(llm.calls) == 2


async def test_refuses_on_unsupported_question_type() -> None:
    bad = json.dumps({"questions": [{"question": "q", "type": "essay", "choices": []}]})
    llm = FakeLlmAdapter(responses=[bad, bad])
    agent = _agent(llm)
    reply = await agent.handle(_request({"summary": _summary()}))
    assert reply is not None and reply.performative == Performative.REFUSE
    assert len(llm.calls) == 2


async def test_refuses_on_empty_question_list() -> None:
    llm = FakeLlmAdapter(responses=[json.dumps({"questions": []}), json.dumps({"questions": []})])
    agent = _agent(llm)
    reply = await agent.handle(_request({"summary": _summary()}))
    assert reply is not None and reply.performative == Performative.REFUSE


async def test_drops_malformed_questions_keeps_well_formed() -> None:
    mixed = json.dumps(
        {
            "questions": [
                {"question": "ok?", "type": "single_choice", "choices": ["A", "B"], "answer_idx": 0},
                {"question": "bad?", "type": "single_choice", "choices": ["A", "B"], "answer_idx": None},
            ]
        }
    )
    agent = _agent(FakeLlmAdapter(responses=[mixed]))
    reply = await agent.handle(_request({"summary": _summary()}))
    assert reply is not None and reply.performative == Performative.INFORM
    assert len(reply.content["questions"]) == 1
    assert reply.content["questions"][0]["question"] == "ok?"


async def test_refuses_when_all_questions_malformed() -> None:
    bad = json.dumps(
        {"questions": [{"question": "bad?", "type": "single_choice", "choices": ["A", "B"], "answer_idx": 9}]}
    )
    agent = _agent(FakeLlmAdapter(responses=[bad, bad]))
    reply = await agent.handle(_request({"summary": _summary()}))
    assert reply is not None and reply.performative == Performative.REFUSE
    assert "well-formed" in reply.content["reason"]


async def test_tolerates_numeric_source_chunk_id_and_extra_fields() -> None:
    # Real GigaChat fills source_chunk_id with an integer and may add stray fields;
    # the LLM-facing model ignores them (the chunk linkage is internal, set to None).
    raw = json.dumps(
        {
            "questions": [
                {
                    "question": "Что выведет код?",
                    "type": "multi_choice",
                    "choices": ["A", "B", "C"],
                    "answer_idx": 0,
                    "answer_indices": [0, 1],
                    "source_chunk_id": 1,
                    "explanation": "лишнее поле",
                }
            ]
        }
    )
    agent = _agent(FakeLlmAdapter(responses=[raw]))
    reply = await agent.handle(_request({"summary": _summary()}))
    assert reply is not None and reply.performative == Performative.INFORM
    question = reply.content["questions"][0]
    assert question["type"] == "multi_choice"
    assert question["source_chunk_id"] is None  # bogus int dropped; linkage set internally
    QuizQuestion.model_validate(question)  # canonical schema validates clean
