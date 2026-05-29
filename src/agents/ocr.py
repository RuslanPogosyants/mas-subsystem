"""OCRAgent (F2): pdf/image -> TextChunks via OcrAdapter."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.agents.base import AgentBase
from src.core.schemas import Operation

if TYPE_CHECKING:
    from src.adapters.ocr import OcrAdapter
    from src.core.bus import RedisStreamBus
    from src.core.messages import Message


class OcrAgent(AgentBase):
    name = "OCRAgent"

    def __init__(self, *, bus: RedisStreamBus, ocr: OcrAdapter) -> None:
        super().__init__(
            bus=bus,
            channel="agent.ocr",
            group="worker-ocr",
            operation=Operation.F2_OCR,
        )
        self._ocr = ocr

    async def handle(self, message: Message) -> Message | None:
        file_path = message.content.get("file_path")
        document_type = message.content.get("document_type")
        if not isinstance(file_path, str):
            return self._refuse(message, reason="missing or invalid file_path")
        if document_type not in ("pdf", "image"):
            return self._refuse(message, reason="document_type must be pdf or image")
        chunks = await self._ocr.extract(file_path=file_path, document_type=document_type)
        return self._inform(
            message,
            content={"chunks": [chunk.model_dump() for chunk in chunks]},
        )
