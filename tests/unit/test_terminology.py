"""Unit tests for TerminologyAgent (extraction, ranking, filtering, refuse)."""

from __future__ import annotations

from src.adapters.ner import FakeNerAdapter, TermCandidate
from src.agents.terminology import TerminologyAgent
from src.core.messages import Message, Performative, make_message
from src.core.schemas import Term

from tests.support.fake_bus import FakeBus


def _request(content: dict[str, object]) -> Message:
    return make_message(
        performative=Performative.REQUEST,
        sender="CoordinatorAgent",
        receiver="TerminologyAgent",
        task_id="task-1",
        conversation_id="conv-1",
        content=content,
        subtask_id="st-task-1-F5",
    )


def _agent(candidates: list[TermCandidate]) -> TerminologyAgent:
    return TerminologyAgent(
        bus=FakeBus(),
        ner=FakeNerAdapter(candidates=candidates),
        stopwords={"и", "в"},
        domain_categories={"структура_данных": ["граф"]},
    )


def _chunk(chunk_id: str, content: str) -> dict[str, object]:
    return {"id": chunk_id, "content": content}


async def test_refuses_when_no_chunk_content() -> None:
    agent = _agent([TermCandidate(text="граф", lemma="граф")])
    reply = await agent.handle(_request({"chunks": [{"id": "c1", "content": "   "}]}))
    assert reply is not None and reply.performative == Performative.REFUSE
    assert "terminology" in reply.content["reason"]


async def test_extracts_ranks_and_categorizes() -> None:
    candidates = [
        TermCandidate(text="граф", lemma="граф"),
        TermCandidate(text="граф", lemma="граф"),
        TermCandidate(text="дерево", lemma="дерево"),
        TermCandidate(text="и", lemma="и"),  # stop-word -> dropped
    ]
    agent = _agent(candidates)
    reply = await agent.handle(_request({"chunks": [_chunk("c1", "граф граф дерево и")], "top_n": 10}))
    assert reply is not None and reply.performative == Performative.INFORM
    terms = [Term.model_validate(item) for item in reply.content["terms"]]
    assert [t.lemma for t in terms] == ["граф", "дерево"]  # граф first (higher frequency)
    assert terms[0].frequency == 2
    assert terms[0].category == "структура_данных"  # from domain dict
    assert terms[0].source_chunk_id == "c1"


async def test_respects_top_n() -> None:
    candidates = [TermCandidate(text=w, lemma=w) for w in ["альфа", "бета", "гамма", "дельта"]]
    agent = _agent(candidates)
    reply = await agent.handle(_request({"chunks": [_chunk("c1", "x")], "top_n": 2}))
    assert reply is not None and reply.performative == Performative.INFORM
    assert len(reply.content["terms"]) == 2


async def test_ner_label_used_as_category() -> None:
    agent = _agent([TermCandidate(text="Москва", lemma="москва", label="LOC")])
    reply = await agent.handle(_request({"chunks": [_chunk("c1", "Москва")]}))
    assert reply is not None and reply.performative == Performative.INFORM
    assert reply.content["terms"][0]["category"] == "LOC"


async def test_informs_empty_when_all_filtered() -> None:
    agent = _agent([TermCandidate(text="и", lemma="и"), TermCandidate(text="в", lemma="в")])
    reply = await agent.handle(_request({"chunks": [_chunk("c1", "и в")]}))
    assert reply is not None and reply.performative == Performative.INFORM
    assert reply.content["terms"] == []
