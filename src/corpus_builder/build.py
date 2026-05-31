"""Build the recommender corpus from the OpenAlex Works API (no key required).

Run manually before the live F6 demo:
    python -m src.corpus_builder.build [out_dir]
Fetches paper metadata + abstracts for a set of queries via the OpenAlex
polite-pool endpoint, embeds the abstracts with the configured model, and
writes <out_dir>/papers.jsonl + papers.npy.
Network + ML deps required; NOT part of CI. Queries are spaced out and
retried with exponential backoff to handle transient HTTP 429/5xx responses.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import httpx

_API = "https://api.openalex.org/works"
_MAILTO = "ruslanpogosyants9594@gmail.com"
_SELECT = "title,abstract_inverted_index,publication_year,authorships,doi,id"
_QUERIES = [
    "machine learning",
    "natural language processing",
    "algorithms data structures",
    "information retrieval",
    "multi-agent systems",
]
_PER_QUERY = 100
_INTER_QUERY_DELAY_SEC = 3.0
_MAX_RETRIES = 5
_BACKOFF_BASE_SEC = 5.0
_HTTP_TOO_MANY = 429
_HTTP_SERVER_ERROR = 500


def _reconstruct_abstract(inverted_index: dict[str, list[int]]) -> str:
    """Reconstruct plain-text abstract from an OpenAlex abstract_inverted_index.

    The inverted index maps each word to the list of positions where it appears.
    We expand each (word, positions) pair into (pos, word) tuples, sort by
    position, then join to produce the original sentence.
    """
    positions: list[tuple[int, str]] = []
    for word, where in inverted_index.items():
        for pos in where:
            positions.append((pos, word))
    positions.sort()
    return " ".join(word for _, word in positions)


def _search(client: httpx.Client, query: str, per_query: int) -> list[dict[str, Any]]:
    """Run one OpenAlex search request, retrying on 429/5xx with exponential backoff."""
    params: dict[str, str | int] = {
        "search": query,
        "per-page": per_query,
        "select": _SELECT,
        "mailto": _MAILTO,
    }
    for attempt in range(_MAX_RETRIES):
        response = client.get(_API, params=params)
        if response.status_code == _HTTP_TOO_MANY or response.status_code >= _HTTP_SERVER_ERROR:
            wait = _BACKOFF_BASE_SEC * (2**attempt)
            print(f"  {query!r}: HTTP {response.status_code}, retry in {wait:.0f}s", flush=True)
            time.sleep(wait)
            continue
        response.raise_for_status()
        data: list[dict[str, Any]] = response.json().get("results", [])
        return data
    raise RuntimeError(f"rate-limited after {_MAX_RETRIES} retries for query {query!r}")


def fetch_papers(queries: list[str] = _QUERIES, per_query: int = _PER_QUERY) -> list[dict[str, Any]]:
    """Fetch deduplicated papers (with abstracts) for the given queries."""
    seen: set[str] = set()
    papers: list[dict[str, Any]] = []
    with httpx.Client(timeout=30.0) as client:
        for index, query in enumerate(queries):
            if index > 0:
                time.sleep(_INTER_QUERY_DELAY_SEC)  # space out requests to respect the polite-pool limit
            for item in _search(client, query, per_query):
                title = item.get("title")
                inverted = item.get("abstract_inverted_index")
                if not title or not inverted or title in seen:
                    continue
                abstract = _reconstruct_abstract(inverted)
                seen.add(title)
                raw_authors = item.get("authorships") or []
                authors = ", ".join(
                    a["author"]["display_name"]
                    for a in raw_authors
                    if a.get("author") and a["author"].get("display_name")
                )
                year: int | None = item.get("publication_year")
                url: str | None = item.get("doi") or item.get("id")
                papers.append(
                    {
                        "title": title,
                        "abstract": abstract,
                        "authors": authors or None,
                        "year": year,
                        "url": url,
                    }
                )
    return papers


def build(out_dir: str = "corpus", model: str = "intfloat/multilingual-e5-base") -> int:
    """Fetch, embed, and persist the corpus. Returns the number of papers written."""
    import numpy
    from sentence_transformers import SentenceTransformer

    papers = fetch_papers()
    if not papers:
        return 0
    encoder = SentenceTransformer(model)
    embeddings = encoder.encode([f"passage: {paper['abstract']}" for paper in papers], normalize_embeddings=True)
    base = Path(out_dir)
    base.mkdir(parents=True, exist_ok=True)
    with (base / "papers.jsonl").open("w", encoding="utf-8") as handle:
        for paper in papers:
            meta = {key: paper[key] for key in ("title", "authors", "year", "url")}
            handle.write(json.dumps(meta, ensure_ascii=False) + "\n")
    numpy.save(base / "papers.npy", embeddings)
    return len(papers)


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "corpus"
    count = build(out_dir=out)
    print(f"wrote {count} papers to {out}/")
