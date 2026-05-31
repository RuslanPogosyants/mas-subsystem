"""Real-mode load test: ramp concurrency against the app running on REAL adapters
(PyMuPDF F2 + GigaChat Lite F3 + spaCy F5), measure end-to-end per-document latency
percentiles and throughput, and find where latency degrades. Writes
experiments/results/loadtest-real.csv.

By default the load document is a self-contained synthetic PDF generated in-memory
(via PyMuPDF) from a fixed block of technical English text, so the test is fully
reproducible from a fresh clone with no external/gitignored asset. Pass a path to
use your own document instead.

Real mode is bounded by the shared GigaChat client and CPU NLP, so it ramps only
[1, 2, 4] with a fixed task count per level (count-based, not time-based: real
documents take seconds, so a time window would yield too few samples). No audio op
(F1) so Whisper does not load — avoids GPU contention; the LLM is the real
bottleneck under study. p95/p99 from a dozen samples are indicative, not precise.

Start the app first (separate terminal):
    python experiments/serve_real.py
Then run (optionally pass a PDF path; defaults to the bundled synthetic document):
    python experiments/loadtest_real.py [path/to/document.pdf]
"""

from __future__ import annotations

import asyncio
import csv
import sys
import time
from pathlib import Path

import httpx

_LEVELS = [1, 2, 4]
_TASKS_PER_LEVEL = 12
_POLL_S = 0.5
_OPS = ["F2", "F3", "F5"]
_TERMINAL = {"completed", "partial_ready", "failed"}
_PAGES = 3
_OUT = Path("experiments/results/loadtest-real.csv")

# Neutral technical paragraph (Latin script -> renders with PDF base-14 fonts, fully
# portable). Repeated to fill several pages so F3 does real map-reduce work.
_PARAGRAPH = (
    "A multi-agent data processing subsystem decomposes an incoming document into "
    "independent operations that specialised agents execute concurrently. A coordinator "
    "dispatches each operation as soon as its dependencies are satisfied, retries transient "
    "failures, and finalises the task once every reachable operation has produced a result. "
    "Text extraction parses the source file into ordered chunks; the summarisation agent "
    "condenses those chunks into a structured abstract; the terminology agent identifies "
    "salient entities and noun phrases. Throughput is governed by the slowest shared "
    "resource rather than by the number of agents, so measuring per-stage latency under "
    "increasing concurrency reveals the true saturation point of the pipeline. "
)


def _synthetic_pdf() -> bytes:
    import fitz  # PyMuPDF (ml extra)

    # Line-by-line insertion (reliable; insert_textbox silently drops everything on
    # overflow). ~12 words/line, ~46 lines/page → ~1600 words over 3 pages, enough
    # for F3 to chunk and summarise real content.
    margin, line_step, page_bottom = 72.0, 15.0, 760.0
    words = _PARAGRAPH.split() * 24
    lines = [" ".join(words[i : i + 12]) for i in range(0, len(words), 12)]
    doc = fitz.open()
    line_idx = 0
    for _ in range(_PAGES):
        page = doc.new_page()
        y = margin
        while y < page_bottom and line_idx < len(lines):
            page.insert_text((margin, y), lines[line_idx], fontsize=10, fontname="helv")
            y += line_step
            line_idx += 1
    data: bytes = doc.tobytes()
    doc.close()
    return data


async def _submit_and_wait(client: httpx.AsyncClient, pdf: bytes) -> tuple[float, str]:
    start = time.perf_counter()
    resp = await client.post(
        "/api/tasks",
        files=[("files", ("doc.pdf", pdf, "application/pdf"))],
        data={"ops": _OPS},
    )
    tid = resp.json()["task_id"]
    while True:
        await asyncio.sleep(_POLL_S)
        status = (await client.get(f"/api/tasks/{tid}")).json()["status"]
        if status in _TERMINAL:
            return time.perf_counter() - start, status


def _pct(sorted_lats: list[float], p: float) -> float:
    if not sorted_lats:
        return 0.0
    return round(sorted_lats[min(len(sorted_lats) - 1, int(len(sorted_lats) * p))], 2)


async def _run_level(concurrency: int, pdf: bytes) -> dict[str, float | int | str]:
    lats: list[float] = []
    errs = [0]
    statuses: dict[str, int] = {}
    sem = asyncio.Semaphore(concurrency)
    limits = httpx.Limits(max_connections=concurrency * 2 + 10)

    async def _one(client: httpx.AsyncClient) -> None:
        async with sem:
            try:
                latency, status = await _submit_and_wait(client, pdf)
                lats.append(latency)
                statuses[status] = statuses.get(status, 0) + 1
            except Exception:
                errs[0] += 1

    wall_start = time.perf_counter()
    async with httpx.AsyncClient(base_url="http://localhost:8000", timeout=600.0, limits=limits) as client:
        await asyncio.gather(*[_one(client) for _ in range(_TASKS_PER_LEVEL)])
    wall = time.perf_counter() - wall_start
    lats.sort()
    return {
        "concurrency": concurrency,
        "tasks": _TASKS_PER_LEVEL,
        "completed": len(lats),
        "throughput_per_s": round(len(lats) / wall, 3) if wall else 0.0,
        "p50_s": _pct(lats, 0.50),
        "p95_s": _pct(lats, 0.95),
        "p99_s": _pct(lats, 0.99),
        "errors": errs[0],
        "statuses": ";".join(f"{k}={v}" for k, v in sorted(statuses.items())),
    }


async def main() -> None:
    if len(sys.argv) > 1:
        pdf = Path(sys.argv[1]).read_bytes()
        source = sys.argv[1]
    else:
        pdf = _synthetic_pdf()
        source = f"synthetic ({_PAGES} pages)"
    print(f"document={source} bytes={len(pdf)} ops={_OPS} tasks_per_level={_TASKS_PER_LEVEL}", flush=True)
    rows: list[dict[str, float | int | str]] = []
    for concurrency in _LEVELS:
        row = await _run_level(concurrency, pdf)
        rows.append(row)
        print(row, flush=True)
    with _OUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {_OUT}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
