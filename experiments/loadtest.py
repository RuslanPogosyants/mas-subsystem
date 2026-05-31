"""Orchestration-mode load test: ramp concurrency against the running app, measure
end-to-end task latency percentiles (p50/p95/p99) and throughput per level, find the
saturation point. Writes experiments/results/loadtest.csv.

Orchestration mode = the app runs on fake backends (no GPU / no live LLM), so this
isolates the multi-agent framework's own scheduling/bus/DB capacity from ML cost.

Start the app first (separate terminal):
    python experiments/serve_orchestration.py
Then run:
    python experiments/loadtest.py
"""

from __future__ import annotations

import asyncio
import csv
import time
from pathlib import Path

import httpx

_LEVELS = [1, 2, 4, 8, 16, 32]
_DURATION_S = 12.0
_POLL_S = 0.2
_PDF = b"%PDF-1.4\n1 0 obj<<>>endobj\ntrailer<<>>\n%%EOF\n"
_TERMINAL = {"completed", "partial_ready", "failed"}
_OUT = Path("experiments/results/loadtest.csv")


async def _submit_and_wait(client: httpx.AsyncClient) -> float:
    start = time.perf_counter()
    resp = await client.post(
        "/api/tasks",
        files=[("files", ("l.pdf", _PDF, "application/pdf"))],
        data={"ops": ["F2", "F3", "F4"]},
    )
    tid = resp.json()["task_id"]
    while True:
        await asyncio.sleep(_POLL_S)
        status = (await client.get(f"/api/tasks/{tid}")).json()["status"]
        if status in _TERMINAL:
            return time.perf_counter() - start


async def _worker(client: httpx.AsyncClient, deadline: float, lats: list[float], errs: list[int]) -> None:
    while time.perf_counter() < deadline:
        try:
            lats.append(await _submit_and_wait(client))
        except Exception:
            errs[0] += 1


def _pct(sorted_lats: list[float], p: float) -> float:
    if not sorted_lats:
        return 0.0
    return round(sorted_lats[min(len(sorted_lats) - 1, int(len(sorted_lats) * p))], 3)


async def _run_level(concurrency: int) -> dict[str, float | int]:
    lats: list[float] = []
    errs = [0]
    limits = httpx.Limits(max_connections=concurrency * 2 + 10, max_keepalive_connections=concurrency * 2)
    async with httpx.AsyncClient(base_url="http://localhost:8000", timeout=120.0, limits=limits) as client:
        deadline = time.perf_counter() + _DURATION_S
        await asyncio.gather(*[_worker(client, deadline, lats, errs) for _ in range(concurrency)])
    lats.sort()
    return {
        "concurrency": concurrency,
        "completed": len(lats),
        "throughput_per_s": round(len(lats) / _DURATION_S, 2),
        "p50_s": _pct(lats, 0.50),
        "p95_s": _pct(lats, 0.95),
        "p99_s": _pct(lats, 0.99),
        "errors": errs[0],
    }


async def main() -> None:
    rows: list[dict[str, float | int]] = []
    for concurrency in _LEVELS:
        row = await _run_level(concurrency)
        rows.append(row)
        print(row, flush=True)
    with _OUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {_OUT}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
