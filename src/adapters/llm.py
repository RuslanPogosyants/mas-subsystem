"""LLM adapter: provider-agnostic protocol + deterministic in-process fake.

The protocol is a thin completion boundary: given a system + user prompt, return
the model's text. The map-reduce / JSON-parsing logic lives in SummarizerAgent,
so the adapter stays trivial and any backend (GigaChat, a local model, a stub)
plugs in behind the same interface. The real GigaChat implementation lives in
gigachat.py; tests and CI use FakeLlmAdapter.
"""

from __future__ import annotations

import json
from typing import Protocol


class LlmAdapter(Protocol):
    """Async text-completion interface for any LLM backend."""

    async def complete(self, *, system: str, user: str) -> str:
        """Return the model's text response to the system + user prompt."""
        ...


class FakeLlmAdapter:
    """Deterministic LlmAdapter stand-in for tests and the demo pipeline.

    Returns scripted `responses` in order; after the list is exhausted it keeps
    returning the last entry. With no script it returns a valid structured-summary
    JSON. Every call is recorded in `calls` for assertions.
    """

    def __init__(self, responses: list[str] | None = None) -> None:
        self._responses = responses
        self.calls: list[dict[str, str]] = []

    async def complete(self, *, system: str, user: str) -> str:
        self.calls.append({"system": system, "user": user})
        if self._responses:
            index = min(len(self.calls) - 1, len(self._responses) - 1)
            return self._responses[index]
        return json.dumps(
            {
                "introduction": f"intro for {len(user)} chars",
                "key_points": "key points",
                "conclusions": "conclusions",
            }
        )
