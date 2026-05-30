import pytest
from src.adapters.ocr import FakeOcrAdapter
from src.agents.ocr import OcrAgent
from src.core.messages import Performative, make_message

from tests.support.fake_bus import FakeBus


def _ocr_request(content: dict[str, object]) -> object:
    return make_message(
        performative=Performative.REQUEST,
        sender="CoordinatorAgent",
        receiver="OCRAgent",
        task_id="task-xyz",
        conversation_id="conv-task-xyz-F2",
        content=content,
        subtask_id="st-task-xyz-F2",
    )


@pytest.mark.asyncio
async def test_ocr_stamps_chunks_with_task_and_document() -> None:
    agent = OcrAgent(bus=FakeBus(), ocr=FakeOcrAdapter())
    reply = await agent.handle(
        _ocr_request({"document_id": "doc-task-xyz-1", "file_path": "/x/p.pdf", "document_type": "pdf"})
    )
    assert reply is not None and reply.performative == Performative.INFORM
    chunks = reply.content["chunks"]
    assert chunks and all(c["task_id"] == "task-xyz" for c in chunks)
    assert all(c["document_id"] == "doc-task-xyz-1" for c in chunks)
    assert chunks[0]["id"] == "chunk-doc-task-xyz-1-0"


@pytest.mark.asyncio
async def test_ocr_refuses_without_document_id() -> None:
    agent = OcrAgent(bus=FakeBus(), ocr=FakeOcrAdapter())
    reply = await agent.handle(_ocr_request({"file_path": "/x/p.pdf", "document_type": "pdf"}))
    assert reply is not None and reply.performative == Performative.REFUSE
