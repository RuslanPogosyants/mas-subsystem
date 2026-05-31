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


# ---------------------------------------------------------------------------
# Fix 1: representative is the HIGHEST-frequency member of a merge group
# ---------------------------------------------------------------------------


async def test_merge_rep_is_highest_frequency_member() -> None:
    """In a 3-member near-dup group the emitted surface/lemma belongs to the
    highest-frequency member even when the first-encountered member has the
    lowest frequency.

    Group members all share prefix «двойной кавычк» (≥ 4 chars) and differ
    only in the final vowel — all three endings (а, и, е) are Russian inflection
    endings so all three correctly form one merge group.
    """
    # «двойной кавычка» freq 1 seen first — should NOT be the representative
    # «двойной кавычки» freq 5               — highest freq, MUST be representative
    # «двойной кавычке» freq 3
    candidates: list[TermCandidate] = (
        [TermCandidate(text="двойных кавычек", lemma="двойной кавычка")]  # freq 1, first
        + [TermCandidate(text="двойные кавычки", lemma="двойной кавычки")] * 5  # freq 5
        + [TermCandidate(text="двойных кавычке", lemma="двойной кавычке")] * 3  # freq 3
    )
    agent = _agent(candidates)
    reply = await agent.handle(_request({"chunks": [_chunk("c1", "x")]}))
    assert reply is not None and reply.performative == Performative.INFORM
    terms = [Term.model_validate(item) for item in reply.content["terms"]]
    assert len(terms) == 1
    # Frequency is sum of all members: 1 + 5 + 3 = 9
    assert terms[0].frequency == 9
    # Surface and lemma come from the highest-frequency member
    assert terms[0].term == "двойные кавычки"
    assert terms[0].lemma == "двойной кавычки"


# ---------------------------------------------------------------------------
# Fix 2: bare single-letter token (e.g. «p значение») is NOT noise
# ---------------------------------------------------------------------------


async def test_noise_filter_keeps_bare_single_letter_token() -> None:
    """A lemma with a bare single-letter (no dot) token must NOT be dropped.

    «p значение» (p-value), «t тест» (t-test) etc. are legitimate domain terms.
    """
    candidates = [
        TermCandidate(text="p значение", lemma="p значение"),
        TermCandidate(text="t тест", lemma="t тест"),
        TermCandidate(text="граф", lemma="граф"),
    ]
    agent = _agent(candidates)
    reply = await agent.handle(_request({"chunks": [_chunk("c1", "x")]}))
    assert reply is not None and reply.performative == Performative.INFORM
    terms = [Term.model_validate(item) for item in reply.content["terms"]]
    lemmas = [t.lemma for t in terms]
    assert "p значение" in lemmas, "p-value term must survive noise filter"
    assert "t тест" in lemmas, "t-test term must survive noise filter"


# ---------------------------------------------------------------------------
# Fix 3: near-dup merge only fires when differing final chars are inflections
# ---------------------------------------------------------------------------


async def test_non_inflection_final_chars_not_merged() -> None:
    """«график» / «графит» differ in final к/т — neither is a Russian inflection
    ending — so they must remain as two separate terms.
    """
    candidates = [
        TermCandidate(text="график", lemma="график"),
        TermCandidate(text="графит", lemma="графит"),
    ]
    agent = _agent(candidates)
    reply = await agent.handle(_request({"chunks": [_chunk("c1", "x")]}))
    assert reply is not None and reply.performative == Performative.INFORM
    terms = [Term.model_validate(item) for item in reply.content["terms"]]
    lemmas = [t.lemma for t in terms]
    assert "график" in lemmas, "график must not be merged with графит"
    assert "графит" in lemmas, "графит must not be merged with график"


# ---------------------------------------------------------------------------
# B2: Suffix-addition inflection merging (prefix relationship)
# ---------------------------------------------------------------------------


async def test_suffix_addition_inflection_merged() -> None:
    """«массив» and «массива» differ by one trailing «а» (an inflection ending)
    and «массив» is a prefix of «массива» — they must collapse into ONE term.
    """
    candidates = [
        TermCandidate(text="массив", lemma="массив"),
        TermCandidate(text="массива", lemma="массива"),
        TermCandidate(text="массива", lemma="массива"),
    ]
    agent = _agent(candidates)
    reply = await agent.handle(_request({"chunks": [_chunk("c1", "x")]}))
    assert reply is not None and reply.performative == Performative.INFORM
    terms = [Term.model_validate(item) for item in reply.content["terms"]]
    assert len(terms) == 1, "массив/массива must merge into one term"
    # Frequency must be summed (1 + 2 = 3)
    assert terms[0].frequency == 3


async def test_non_inflection_extra_char_not_merged() -> None:
    """«тип» / «типаж»: extra chars «аж» — «ж» is NOT a Russian inflection ending.
    Must remain two separate terms.
    """
    candidates = [
        TermCandidate(text="тип", lemma="тип"),
        TermCandidate(text="типаж", lemma="типаж"),
    ]
    agent = _agent(candidates)
    reply = await agent.handle(_request({"chunks": [_chunk("c1", "x")]}))
    assert reply is not None and reply.performative == Performative.INFORM
    terms = [Term.model_validate(item) for item in reply.content["terms"]]
    lemmas = [t.lemma for t in terms]
    assert "тип" in lemmas, "тип must not merge with типаж"
    assert "типаж" in lemmas, "типаж must not merge with тип"


async def test_non_inflection_extra_chars_two_not_merged() -> None:
    """«код» / «кодер»: extra chars «ер» — «р» is NOT a Russian inflection ending.
    Must remain two separate terms.
    """
    candidates = [
        TermCandidate(text="код", lemma="код"),
        TermCandidate(text="кодер", lemma="кодер"),
    ]
    agent = _agent(candidates)
    reply = await agent.handle(_request({"chunks": [_chunk("c1", "x")]}))
    assert reply is not None and reply.performative == Performative.INFORM
    terms = [Term.model_validate(item) for item in reply.content["terms"]]
    lemmas = [t.lemma for t in terms]
    assert "код" in lemmas, "код must not merge with кодер"
    assert "кодер" in lemmas, "кодер must not merge with код"


async def test_suffix_addition_two_token_phrase_merged() -> None:
    """Verify prefix-relationship rule works for two-token phrases.

    «двойной массив» (base form) vs «двойной массива» (genitive, extra «а» ∈ endings).
    «массив» is a prefix of «массива»; «двойной» == «двойной» → must merge.
    """
    candidates = [
        TermCandidate(text="двойной массив", lemma="двойной массив"),
        TermCandidate(text="двойного массива", lemma="двойной массива"),
        TermCandidate(text="двойного массива", lemma="двойной массива"),
    ]
    agent = _agent(candidates)
    reply = await agent.handle(_request({"chunks": [_chunk("c1", "x")]}))
    assert reply is not None and reply.performative == Performative.INFORM
    terms = [Term.model_validate(item) for item in reply.content["terms"]]
    assert len(terms) == 1, "двойной массив/массива must merge"
    assert terms[0].frequency == 3


async def test_existing_grafik_grafit_still_separate_under_b2() -> None:
    """«график» / «графит»: equal length, final chars к/т not in endings.
    Rule 1 rejects (к/т not endings). Rule 2 cannot apply (equal length).
    Must remain separate.
    """
    candidates = [
        TermCandidate(text="график", lemma="график"),
        TermCandidate(text="графит", lemma="графит"),
    ]
    agent = _agent(candidates)
    reply = await agent.handle(_request({"chunks": [_chunk("c1", "x")]}))
    assert reply is not None and reply.performative == Performative.INFORM
    terms = [Term.model_validate(item) for item in reply.content["terms"]]
    lemmas = [t.lemma for t in terms]
    assert "график" in lemmas
    assert "графит" in lemmas
