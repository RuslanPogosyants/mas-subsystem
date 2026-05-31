"""SummarizerAgent (F3): TextChunks -> structured Summary via an LlmAdapter.

Map-reduce: if the concatenated chunk text fits in one block, summarise in a
single LLM call; otherwise summarise each overlapping block and reduce the
partials into one final summary. The LLM returns raw JSON
{introduction, key_points, conclusions}; the agent validates it (via the shared
parse_with_retry helper), then maps it to the schemas.Summary vocabulary
(key_points -> thesis section) and echoes the source chunk ids. Empty input
chunks are refused, never silently summarised into nothing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from loguru import logger
from pydantic import BaseModel

from src.agents._llm_json import parse_with_retry
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
    "— непустые строки на русском языке. Без markdown и без пояснений.\n"
    "Требования к полноте:\n"
    "- introduction: одно-два предложения о теме и цели материала.\n"
    "- key_points: подробно перечисли все ключевые тезисы, понятия и факты из текста "
    "(не одно предложение; по пункту на каждую значимую идею). "
    "Не сокращай и не объединяй разные идеи в одну строку.\n"
    "- conclusions: одно-два предложения с итоговым выводом."
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
        """Map-reduce with graceful per-block degradation.

        A block the LLM declines (e.g. a content-filter refusal on a sensitive
        passage) or returns unparseable JSON for is skipped, not fatal — the
        summary is reduced from the blocks that succeeded. Only an all-blocks
        failure refuses. If the reduce step itself is declined, the first
        surviving partial is returned so a usable summary still reaches the user.
        """
        if len(text) <= self._block_chars:
            return await self._call(text)
        blocks = self._split(text)
        partials: list[_RawSummary] = []
        for block in blocks:
            partial = await self._call(block)
            if partial is not None:
                partials.append(partial)
        skipped = len(blocks) - len(partials)
        if skipped:
            logger.warning(f"summarizer skipped {skipped}/{len(blocks)} block(s): LLM declined or invalid JSON")
        if not partials:
            return None
        if len(partials) == 1:
            return partials[0]
        reduced = await self._call(self._reduce_prompt(partials))
        return reduced if reduced is not None else partials[0]

    def _split(self, text: str) -> list[str]:
        step = max(1, self._block_chars - self._overlap)
        return [text[start : start + self._block_chars] for start in range(0, len(text), step)]

    def _reduce_prompt(self, partials: list[_RawSummary]) -> str:
        joined = "\n\n".join(
            f"Введение: {part.introduction}\nТезисы: {part.key_points}\nВыводы: {part.conclusions}" for part in partials
        )
        return (
            "Объедини частичные саммари в одно итоговое саммари. "
            "Сохрани ВСЕ отдельные ключевые тезисы из каждого частичного саммари — "
            "не объединяй разные идеи и не выбрасывай детали:\n" + joined
        )

    async def _call(self, user: str) -> _RawSummary | None:
        return await parse_with_retry(
            self._llm, system=_SYSTEM_PROMPT, user=user, model_cls=_RawSummary, retries=_MAX_PARSE_RETRIES
        )
