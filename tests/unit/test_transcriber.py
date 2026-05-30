import pytest
from src.adapters.transcriber import FakeTranscriberAdapter
from src.agents.transcriber import TranscriberAgent
from src.core.messages import Performative, make_message

from tests.support.fake_bus import FakeBus


def _request(content: dict[str, object]) -> object:
    return make_message(
        performative=Performative.REQUEST,
        sender="CoordinatorAgent",
        receiver="TranscriberAgent",
        task_id="task-xyz",
        conversation_id="conv-task-xyz-F1",
        content=content,
        subtask_id="st-task-xyz-F1",
    )


@pytest.mark.asyncio
async def test_transcriber_stamps_chunks_with_task_and_document() -> None:
    agent = TranscriberAgent(bus=FakeBus(), transcriber=FakeTranscriberAdapter())
    reply = await agent.handle(
        _request({"document_id": "doc-task-xyz-0", "file_path": "/x/lecture.mp3", "language": "ru"})
    )
    assert reply is not None
    assert reply.performative == Performative.INFORM
    chunks = reply.content["chunks"]
    assert chunks and all(c["task_id"] == "task-xyz" for c in chunks)
    assert all(c["document_id"] == "doc-task-xyz-0" for c in chunks)
    assert chunks[0]["id"] == "chunk-doc-task-xyz-0-0"


@pytest.mark.asyncio
async def test_transcriber_refuses_without_document_id() -> None:
    agent = TranscriberAgent(bus=FakeBus(), transcriber=FakeTranscriberAdapter())
    reply = await agent.handle(_request({"file_path": "/x/lecture.mp3"}))
    assert reply is not None
    assert reply.performative == Performative.REFUSE
