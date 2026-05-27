"""Retry policy with exponential backoff.

On subtask timeout the coordinator re-publishes up to RETRY_MAX times.
"""

from __future__ import annotations

from typing import Final

RETRY_MAX: Final[int] = 2
BACKOFF_SECONDS: Final[tuple[float, ...]] = (1.0, 4.0)


def compute_backoff(retry_count: int) -> float:
    """Return the backoff delay before retry number `retry_count`.

    Args:
        retry_count: 0 for the first retry, 1 for the second, and so on.

    Returns:
        Delay in seconds.

    Raises:
        ValueError: if `retry_count` is negative or >= RETRY_MAX.
    """
    raise NotImplementedError("compute_backoff: implemented in M1")
