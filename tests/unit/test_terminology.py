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
    # Equal score (freq 1, single chunk) -> deterministic lemma-ascending tie-break.
    assert [term["lemma"] for term in reply.content["terms"]] == ["альфа", "бета"]


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


# ---------------------------------------------------------------------------
# NER noise filtering
# ---------------------------------------------------------------------------


async def test_ner_noise_initial_dropped() -> None:
    """A candidate whose lemma contains a single-char initial token is dropped."""
    candidates = [
        TermCandidate(text="И. Вот", lemma="и. вот", label="PER"),
        TermCandidate(text="граф", lemma="граф"),
    ]
    agent = _agent(candidates)
    reply = await agent.handle(_request({"chunks": [_chunk("c1", "И. Вот граф")]}))
    assert reply is not None and reply.performative == Performative.INFORM
    terms = [Term.model_validate(item) for item in reply.content["terms"]]
    lemmas = [t.lemma for t in terms]
    assert "и. вот" not in lemmas
    assert "граф" in lemmas


# ---------------------------------------------------------------------------
# Near-duplicate merging
# ---------------------------------------------------------------------------


async def test_near_dup_merged_into_one_term() -> None:
    """Two lemmas that differ only in the final character of the last token are merged."""
    candidates = [
        TermCandidate(text="двойные кавычки", lemma="двойной кавычки"),
        TermCandidate(text="двойные кавычки", lemma="двойной кавычки"),
        TermCandidate(text="двойных кавычек", lemma="двойной кавычка"),
    ]
    agent = _agent(candidates)
    reply = await agent.handle(_request({"chunks": [_chunk("c1", "x")]}))
    assert reply is not None and reply.performative == Performative.INFORM
    terms = [Term.model_validate(item) for item in reply.content["terms"]]
    # Only one term for the concept (merging near-dups)
    assert len(terms) == 1
    # Frequency must be summed (2 + 1 = 3)
    assert terms[0].frequency == 3


async def test_distinct_concepts_not_merged() -> None:
    """Two terms differing in a non-final token must NOT be merged."""
    candidates = [
        TermCandidate(text="целые числа", lemma="целый число"),
        TermCandidate(text="вещественные числа", lemma="вещественный число"),
    ]
    agent = _agent(candidates)
    reply = await agent.handle(_request({"chunks": [_chunk("c1", "x")]}))
    assert reply is not None and reply.performative == Performative.INFORM
    terms = [Term.model_validate(item) for item in reply.content["terms"]]
    lemmas = [t.lemma for t in terms]
    assert "целый число" in lemmas
    assert "вещественный число" in lemmas


async def test_single_token_prefix_not_merged() -> None:
    """Single-token lemmas that share fewer than 4 chars of prefix are NOT merged."""
    candidates = [
        TermCandidate(text="строка", lemma="строка"),
        TermCandidate(text="строфа", lemma="строфа"),
    ]
    agent = _agent(candidates)
    reply = await agent.handle(_request({"chunks": [_chunk("c1", "x")]}))
    assert reply is not None and reply.performative == Performative.INFORM
    terms = [Term.model_validate(item) for item in reply.content["terms"]]
    lemmas = [t.lemma for t in terms]
    assert "строка" in lemmas
    assert "строфа" in lemmas
