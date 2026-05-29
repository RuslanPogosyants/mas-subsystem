"""Real GigaChat implementation of LlmAdapter (used for the live demo only).

The `gigachat` SDK is an optional ML dependency and is imported lazily so the
module imports cleanly in CI where the package is absent. SummarizerAgent owns
the map-reduce and JSON parsing; this adapter only performs one chat completion
per call.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from src.config import Settings


class GigaChatAdapter:
    """LlmAdapter backed by Sber GigaChat. Constructed from Settings."""

    def __init__(self, settings: Settings) -> None:
        from gigachat import GigaChat  # lazy: optional ml dependency

        self._client = GigaChat(
            credentials=settings.gigachat_credentials,
            scope=settings.gigachat_scope,
            model=settings.gigachat_model,
            verify_ssl_certs=settings.gigachat_verify_ssl,
        )
        self._temperature = settings.gigachat_temperature

    async def complete(self, *, system: str, user: str) -> str:
        from gigachat.models import Chat, Messages, MessagesRole  # lazy

        chat = Chat(
            messages=[
                Messages(role=MessagesRole.SYSTEM, content=system),
                Messages(role=MessagesRole.USER, content=user),
            ],
            temperature=self._temperature,
        )
        # Normalise transport/HTTP failures into a ConnectionError so AgentBase
        # converts them to a refuse instead of crashing the agent loop.
        try:
            response = await self._client.achat(chat)
        except httpx.HTTPError as error:
            raise ConnectionError(f"gigachat request failed: {error}") from error
        return str(response.choices[0].message.content)
