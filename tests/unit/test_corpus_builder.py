"""Unit tests for the F6 corpus builder (network + model stubbed)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_MISSING: Any = object()  # sentinel for "not provided"


def _openalex_work(
    *,
    title: str | None = "A Paper",
    abstract_inverted_index: Any = _MISSING,
    publication_year: int | None = 2023,
    authorships: list[dict[str, Any]] | None = None,
    doi: str | None = None,
    oa_id: str = "https://openalex.org/W1",
) -> dict[str, Any]:
    """Build a minimal OpenAlex work dict."""
    if abstract_inverted_index is _MISSING:
        abstract_inverted_index = {"Hello": [0], "world": [1]}
    if authorships is None:
        authorships = [{"author": {"display_name": "Alice"}}]
    return {
        "title": title,
        "abstract_inverted_index": abstract_inverted_index,
        "publication_year": publication_year,
        "authorships": authorships,
        "doi": doi,
        "id": oa_id,
    }


def _make_client(responses: list[dict[str, Any]]) -> type:
    """Return a fake httpx.Client class that cycles through *responses* (each is a JSON payload)."""
    call_index = 0

    class _Resp:
        def __init__(self, payload: dict[str, Any], status: int = 200) -> None:
            self.status_code = status
            self._payload = payload

        def raise_for_status(self) -> None: ...

        def json(self) -> dict[str, Any]:
            return self._payload

    class _Client:
        def __init__(self, *a: Any, **k: Any) -> None: ...
        def __enter__(self) -> _Client:
            return self

        def __exit__(self, *a: Any) -> None: ...

        def get(self, *a: Any, **k: Any) -> _Resp:
            nonlocal call_index
            entry = responses[call_index]
            call_index += 1
            status = entry.pop("__status__", 200)
            return _Resp(entry, status)

    return _Client


# ---------------------------------------------------------------------------
# Test 1 – _reconstruct_abstract: direct unit test
# ---------------------------------------------------------------------------


def test_reconstruct_abstract_direct() -> None:
    """Words placed by position should reconstruct to the original sentence."""
    from src.corpus_builder.build import _reconstruct_abstract

    result = _reconstruct_abstract({"Python": [0], "is": [1], "great": [2]})
    assert result == "Python is great"


def test_reconstruct_abstract_unsorted_positions() -> None:
    """Positions don't have to be given in order."""
    from src.corpus_builder.build import _reconstruct_abstract

    result = _reconstruct_abstract({"world": [1], "Hello": [0]})
    assert result == "Hello world"


# ---------------------------------------------------------------------------
# Test 2 – dedup by title + skip missing abstract or title
# ---------------------------------------------------------------------------


def test_fetch_papers_dedupes_and_requires_abstract(monkeypatch: pytest.MonkeyPatch) -> None:
    """Duplicate title and null abstract_inverted_index both cause the work to be skipped."""
    from src.corpus_builder import build as builder

    works = [
        _openalex_work(title="Graphs", abstract_inverted_index={"about": [0], "graphs": [1]}, oa_id="W1"),
        _openalex_work(title="Graphs", abstract_inverted_index={"duplicate": [0]}, oa_id="W2"),  # dup title
        _openalex_work(title="NoAbstract", abstract_inverted_index=None, oa_id="W3"),  # no abstract
        _openalex_work(title=None, abstract_inverted_index={"no": [0], "title": [1]}, oa_id="W4"),  # no title
    ]
    payload = {"results": works}

    monkeypatch.setattr(builder.httpx, "Client", _make_client([payload]))
    papers = builder.fetch_papers(queries=["q"], per_query=10)

    assert [p["title"] for p in papers] == ["Graphs"]


# ---------------------------------------------------------------------------
# Test 3 – authors join from authorships[].author.display_name
# ---------------------------------------------------------------------------


def test_fetch_papers_authors_join(monkeypatch: pytest.MonkeyPatch) -> None:
    """Multiple authorships are joined with ', '."""
    from src.corpus_builder import build as builder

    work = _openalex_work(
        authorships=[
            {"author": {"display_name": "Alice"}},
            {"author": {"display_name": "Bob"}},
            {"author": {}},  # missing display_name entry – should be skipped
        ],
    )
    monkeypatch.setattr(builder.httpx, "Client", _make_client([{"results": [work]}]))
    papers = builder.fetch_papers(queries=["q"], per_query=10)

    assert len(papers) == 1
    assert papers[0]["authors"] == "Alice, Bob"


# ---------------------------------------------------------------------------
# Test 4 – url = doi when present, else id
# ---------------------------------------------------------------------------


def test_fetch_papers_url_prefers_doi(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.corpus_builder import build as builder

    with_doi = _openalex_work(doi="https://doi.org/10.1234/example", oa_id="https://openalex.org/W10")
    without_doi = _openalex_work(doi=None, oa_id="https://openalex.org/W11", title="NoDoi Paper")

    monkeypatch.setattr(builder.httpx, "Client", _make_client([{"results": [with_doi, without_doi]}]))
    papers = builder.fetch_papers(queries=["q"], per_query=10)

    assert papers[0]["url"] == "https://doi.org/10.1234/example"
    assert papers[1]["url"] == "https://openalex.org/W11"


# ---------------------------------------------------------------------------
# Test 5 – retry on 429
# ---------------------------------------------------------------------------


def test_fetch_papers_retries_on_429(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 429 on the first attempt triggers a retry that succeeds."""
    from src.corpus_builder import build as builder

    work = _openalex_work(title="Retry Paper", oa_id="https://openalex.org/W99")
    # First response: 429 (no results needed), second: 200 with our work
    responses: list[dict[str, Any]] = [
        {"__status__": 429, "results": []},
        {"results": [work]},
    ]

    call_count = 0

    class _Resp:
        def __init__(self, payload: dict[str, Any], status: int) -> None:
            self.status_code = status
            self._payload = payload

        def raise_for_status(self) -> None: ...

        def json(self) -> dict[str, Any]:
            return self._payload

    class _Client:
        def __init__(self, *a: Any, **k: Any) -> None: ...
        def __enter__(self) -> _Client:
            return self

        def __exit__(self, *a: Any) -> None: ...

        def get(self, *a: Any, **k: Any) -> _Resp:
            nonlocal call_count
            resp = responses[call_count]
            call_count += 1
            status = resp.pop("__status__", 200)
            return _Resp(resp, status)

    monkeypatch.setattr(builder.httpx, "Client", _Client)
    monkeypatch.setattr(builder.time, "sleep", lambda *_: None)

    papers = builder.fetch_papers(queries=["q"], per_query=10)

    assert call_count == 2, f"expected 2 calls (429 + 200), got {call_count}"
    assert len(papers) == 1
    assert papers[0]["title"] == "Retry Paper"


# ---------------------------------------------------------------------------
# Test 6 – end-to-end abstract reconstruction through fetch_papers
# ---------------------------------------------------------------------------


def test_fetch_papers_reconstructs_abstract(monkeypatch: pytest.MonkeyPatch) -> None:
    """Abstract is correctly reconstructed from abstract_inverted_index."""
    from src.corpus_builder import build as builder

    work = _openalex_work(
        title="Inverted Index Paper",
        abstract_inverted_index={"Python": [0], "is": [1], "great": [2]},
    )
    monkeypatch.setattr(builder.httpx, "Client", _make_client([{"results": [work]}]))
    papers = builder.fetch_papers(queries=["q"], per_query=10)

    assert len(papers) == 1
    assert papers[0]["abstract"] == "Python is great"


# ---------------------------------------------------------------------------
# Test 7 – build() prefixes passages and writes files (largely unchanged)
# ---------------------------------------------------------------------------


def test_build_prefixes_passages_and_writes_files(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from src.corpus_builder import build as builder

    monkeypatch.setattr(
        builder,
        "fetch_papers",
        lambda: [{"title": "T", "abstract": "abs", "authors": "A", "year": 2021, "url": "u"}],
    )
    seen: list[str] = []

    class _Encoder:
        def __init__(self, name: str) -> None: ...

        def encode(self, texts: list[str], normalize_embeddings: bool = False) -> Any:
            import numpy

            seen.extend(texts)
            return numpy.array([[0.1, 0.2]])

    import sentence_transformers

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", _Encoder)

    count = builder.build(out_dir=str(tmp_path))

    assert count == 1
    assert seen == ["passage: abs"]  # e5 passage prefix applied to abstracts
    meta = json.loads((tmp_path / "papers.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert meta == {"title": "T", "authors": "A", "year": 2021, "url": "u"}
    assert (tmp_path / "papers.npy").exists()
