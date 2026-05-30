"""Unit tests for RecommenderAgent (query, cosine ranking, filters, refuse)."""

from __future__ import annotations

from src.adapters.embedding import FakeEmbeddingAdapter
from src.agents.recommender import CorpusEntry, RecommenderAgent
from src.core.messages import Message, Performative, make_message
from src.core.schemas import Citation

from tests.support.fake_bus import FakeBus


def _request(content: dict[str, object]) -> Message:
    return make_message(
        performative=Performative.REQUEST,
        sender="CoordinatorAgent",
        receiver="RecommenderAgent",
        task_id="task-1",
        conversation_id="conv-1",
        content=content,
        subtask_id="st-task-1-F6",
    )


def _summary(intro: str = "графы и деревья") -> dict[str, object]:
    return {"summary_id": "s1", "sections": [{"type": "introduction", "text": intro}], "source_chunk_ids": []}


_CORPUS = [
    CorpusEntry(title="Graphs", authors="A", year=2020, url="u1", embedding=(1.0, 0.0)),
    CorpusEntry(title="Trees", authors="B", year=2010, url="u2", embedding=(0.0, 1.0)),
]


def _agent(corpus: list[CorpusEntry], query_vector: list[float]) -> RecommenderAgent:
    return RecommenderAgent(bus=FakeBus(), embedding=FakeEmbeddingAdapter(vectors=[query_vector]), corpus=corpus)


async def test_refuses_when_corpus_empty() -> None:
    agent = _agent([], [1.0, 0.0])
    reply = await agent.handle(_request({"summary": _summary(), "terms": []}))
    assert reply is not None and reply.performative == Performative.REFUSE
    assert "corpus" in reply.content["reason"]


async def test_refuses_when_no_query() -> None:
    agent = _agent(_CORPUS, [1.0, 0.0])
    reply = await agent.handle(_request({"summary": {}, "terms": []}))
    assert reply is not None and reply.performative == Performative.REFUSE
    assert "summary or terms" in reply.content["reason"]


async def test_refuses_on_embedding_dimension_mismatch() -> None:
    # 2-dim corpus but a 3-dim query embedding -> refuse, never silently score garbage.
    agent = _agent(_CORPUS, [1.0, 0.0, 0.0])
    reply = await agent.handle(_request({"summary": _summary(), "terms": []}))
    assert reply is not None and reply.performative == Performative.REFUSE
    assert "dimension" in reply.content["reason"]


async def test_ranks_by_cosine_and_sets_relevance() -> None:
    agent = _agent(_CORPUS, [1.0, 0.0])  # aligned with "Graphs"
    reply = await agent.handle(_request({"summary": _summary(), "terms": [], "n": 2}))
    assert reply is not None and reply.performative == Performative.INFORM
    citations = [Citation.model_validate(item) for item in reply.content["citations"]]
    assert [c.title for c in citations] == ["Graphs", "Trees"]
    assert citations[0].relevance_score == 1.0
    assert citations[1].relevance_score == 0.0


async def test_respects_top_n() -> None:
    agent = _agent(_CORPUS, [1.0, 1.0])
    reply = await agent.handle(_request({"summary": _summary(), "terms": [], "n": 1}))
    assert reply is not None and reply.performative == Performative.INFORM
    assert len(reply.content["citations"]) == 1


async def test_year_filter_excludes_out_of_range() -> None:
    agent = _agent(_CORPUS, [1.0, 0.0])
    reply = await agent.handle(_request({"summary": _summary(), "terms": [], "filters": {"year_min": 2015}}))
    assert reply is not None and reply.performative == Performative.INFORM
    titles = [item["title"] for item in reply.content["citations"]]
    assert titles == ["Graphs"]  # Trees (2010) filtered out


async def test_query_uses_terms_when_summary_missing() -> None:
    # D2 degradation: F5 succeeded, F3 failed -> query from terms only, still works.
    agent = _agent(_CORPUS, [0.0, 1.0])
    reply = await agent.handle(_request({"summary": {}, "terms": [{"term": "дерево"}], "n": 1}))
    assert reply is not None and reply.performative == Performative.INFORM
    assert reply.content["citations"][0]["title"] == "Trees"
