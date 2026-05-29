"""SummarizerAgent (F3): TextChunks -> structured Summary via an LlmAdapter.

Map-reduce: if the concatenated chunk text fits in one block, summarise in a
single LLM call; otherwise summarise each overlapping block and reduce the
partials into one final summary. The LLM returns raw JSON
{introduction, key_points, conclusions}; the agent validates it, retries up to
twice on malformed output, then maps it to the schemas.Summary vocabulary
(key_points -> thesis section) and echoes the source chunk ids. Empty input
chunks are refused, never silently summarised into nothing.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Final

from pydantic import BaseModel, ValidationError

from src.agents.base import AgentBase
from src.core.schemas import Operation, Summary, SummarySection

if TYPE_CHECKING:
    from src.adapters.llm import LlmAdapter
    from src.core.bus import RedisStreamBus
    from src.core.messages import Message

_MAX_PARSE_RETRIES: Final[int] = 2
_SYSTEM_PROMPT: Final[str] = (
    "Ты делаешь структурированное саммари учебного текста. Ответь СТРОГО одним "
    "JSON-объектом с ключами introduction, key_points, conclusions; все значения "
    "— непустые строки на русском языке. Без markdown и без пояснений."
)


class _RawSummary(BaseModel):
    """The raw JSON shape the LLM is asked to emit (spec section 7.3)."""

    introduction: str
    key_points: str
    conclusions: str


class SummarizerAgent(AgentBase):
    name = "SummarizerAgent"

    def __init__(self, *, bus: RedisStreamBus, llm: LlmAdapter, block_chars: int = 6000, overlap: int = 500) -> None:
        super().__init__(
            bus=bus,
            channel="agent.summarizer",
            group="worker-summarizer",
            operation=Operation.F3_SUMMARIZE,
        )
        self._llm = llm
        self._block_chars = block_chars
        self._overlap = overlap

    async def handle(self, message: Message) -> Message | None:
        raw_chunks = message.content.get("chunks")
        chunks = [chunk for chunk in raw_chunks if isinstance(chunk, dict)] if isinstance(raw_chunks, list) else []
        texts = [str(chunk.get("content", "")) for chunk in chunks]
        combined = "\n".join(text for text in texts if text).strip()
        if not combined:
            return self._refuse(message, reason="no chunk content to summarize")
        raw = await self._summarize(combined)
        if raw is None:
            return self._refuse(message, reason="llm returned invalid summary json")
        chunk_ids = [str(chunk["id"]) for chunk in chunks if "id" in chunk]
        summary = Summary(
            summary_id=f"sum-{message.task_id}",
            sections=[
                SummarySection(type="introduction", text=raw.introduction),
                SummarySection(type="thesis", text=raw.key_points),
                SummarySection(type="conclusion", text=raw.conclusions),
            ],
            source_chunk_ids=chunk_ids,
        )
        return self._inform(message, content=summary.model_dump())

    async def _summarize(self, text: str) -> _RawSummary | None:
        if len(text) <= self._block_chars:
            return await self._call_json(text)
        partials: list[_RawSummary] = []
        for block in self._split(text):
            partial = await self._call_json(block)
            if partial is None:
                return None
            partials.append(partial)
        return await self._call_json(self._reduce_prompt(partials))

    def _split(self, text: str) -> list[str]:
        step = max(1, self._block_chars - self._overlap)
        return [text[start : start + self._block_chars] for start in range(0, len(text), step)]

    def _reduce_prompt(self, partials: list[_RawSummary]) -> str:
        joined = "\n\n".join(
            f"Введение: {part.introduction}\nТезисы: {part.key_points}\nВыводы: {part.conclusions}" for part in partials
        )
        return f"Объедини частичные саммари в одно итоговое саммари:\n{joined}"

    async def _call_json(self, user: str) -> _RawSummary | None:
        prompt = user
        for _ in range(_MAX_PARSE_RETRIES + 1):
            response = await self._llm.complete(system=_SYSTEM_PROMPT, user=prompt)
            parsed = _parse_raw_summary(response)
            if parsed is not None:
                return parsed
            prompt = (
                f"{user}\n\nПредыдущий ответ не был валидным JSON с ключами "
                "introduction, key_points, conclusions. Верни строго такой JSON."
            )
        return None


def _parse_raw_summary(response: str) -> _RawSummary | None:
    try:
        data = json.loads(response)
    except (json.JSONDecodeError, TypeError):
        return None
    try:
        return _RawSummary.model_validate(data)
    except ValidationError:
        return None
