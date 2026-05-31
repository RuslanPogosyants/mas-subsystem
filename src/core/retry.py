"""Retry policy with exponential backoff and per-agent timeouts.

On subtask timeout the coordinator re-publishes up to RETRY_MAX times.
Per-agent timeouts encode the expected upper bound for a single attempt;
override per-agent via env (e.g. COORD_TIMEOUT_RECOMMENDER) is wired in M3.
"""

from __future__ import annotations

from typing import Final

BACKOFF_SECONDS: Final[tuple[float, ...]] = (1.0, 4.0)
RETRY_MAX: Final[int] = len(BACKOFF_SECONDS)

AGENT_TIMEOUTS_SEC: Final[dict[str, float]] = {
    "transcriber": 600.0,
    "ocr": 120.0,
    "summarizer": 90.0,
    "test_generator": 60.0,
    "terminology": 30.0,
    "recommender": 15.0,
}

# Per-agent timeouts + RETRY_MAX bound a single attempt; the Coordinator also
# enforces a hard per-task global deadline (coord_global_deadline_sec, D3) that
# finalises an overrunning task with whatever partial results it has.


def compute_backoff(retry_count: int) -> float:
    """Return the backoff delay before retry number `retry_count`.

    Args:
        retry_count: 0 for the first retry, 1 for the second, and so on.

    Returns:
        Delay in seconds.

    Raises:
        ValueError: if `retry_count` is negative or >= RETRY_MAX.
    """
    if retry_count < 0:
        raise ValueError(f"retry_count must be non-negative, got {retry_count}")
    if retry_count >= RETRY_MAX:
        raise ValueError(f"retry_count must be < {RETRY_MAX}, got {retry_count}")
    return BACKOFF_SECONDS[retry_count]
