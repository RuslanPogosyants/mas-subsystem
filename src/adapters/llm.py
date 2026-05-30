"""LLM adapter: provider-agnostic protocol + deterministic in-process fake.

The protocol is a thin completion boundary: given a system + user prompt, return
the model's text. The map-reduce / JSON-parsing logic lives in SummarizerAgent,
so the adapter stays trivial and any backend (GigaChat, a local model, a stub)
plugs in behind the same interface. The real GigaChat implementation lives in
gigachat.py; tests and CI use FakeLlmAdapter.
"""

from __future__ import annotations

import json
from typing import Final, Protocol

_FAKE_SUMMARY: Final[str] = json.dumps(
    {"introduction": "введение", "key_points": "ключевые тезисы", "conclusions": "выводы"}
)
_FAKE_QUIZ: Final[str] = json.dumps(
    {"questions": [{"question": "Что главное?", "type": "single_choice", "choices": ["A", "B"], "answer_idx": 0}]}
)


class LlmAdapter(Protocol):
    """Async text-completion interface for any LLM backend."""

    async def complete(self, *, system: str, user: str) -> str:
        """Return the model's text response to the system + user prompt."""
        ...


class FakeLlmAdapter:
    """Deterministic LlmAdapter stand-in for tests and the demo pipeline.

    Returns scripted `responses` in order; after the list is exhausted it keeps
    returning the last entry. With no script the default is role-aware: a quiz
    JSON when the system prompt asks for questions (F4), otherwise a structured
    summary JSON (F3) — so the whole Fake pipeline produces valid end-to-end
    output. Every call is recorded in `calls` for assertions.
    """

    def __init__(self, responses: list[str] | None = None) -> None:
        self._responses = responses
        self.calls: list[dict[str, str]] = []

    async def complete(self, *, system: str, user: str) -> str:
        self.calls.append({"system": system, "user": user})
        if self._responses:
            index = min(len(self.calls) - 1, len(self._responses) - 1)
            return self._responses[index]
        return _FAKE_QUIZ if "questions" in system else _FAKE_SUMMARY
