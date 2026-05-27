"""Contract tests for the bus message format (VKR listings 2.3-2.4)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from src.core.messages import Message, Performative, make_message


class TestRequestMessageShape:
    def test_make_request_includes_all_listing_2_3_fields(self) -> None:
        msg = make_message(
            performative=Performative.REQUEST,
            sender="CoordinatorAgent",
            receiver="SummarizerAgent",
            task_id="task-7c41",
            conversation_id="conv-7c41-3",
            content={"operation": "summarize", "chunks": ["chunk-001"]},
        )
        dumped = msg.model_dump()
        expected_fields = {
            "message_id",
            "performative",
            "sender",
            "receiver",
            "task_id",
            "conversation_id",
            "content",
            "reply_to",
            "timestamp",
            "in_reply_to",
            "subtask_id",
        }
        assert set(dumped.keys()) == expected_fields

    def test_request_has_no_in_reply_to(self) -> None:
        msg = make_message(
            performative=Performative.REQUEST,
            sender="CoordinatorAgent",
            receiver="SummarizerAgent",
            task_id="task-7c41",
            conversation_id="conv-7c41-3",
            content={},
        )
        assert msg.in_reply_to is None

    def test_reply_to_defaults_to_coordinator_inbox(self) -> None:
        msg = make_message(
            performative=Performative.REQUEST,
            sender="CoordinatorAgent",
            receiver="SummarizerAgent",
            task_id="task-7c41",
            conversation_id="conv-7c41-3",
            content={},
        )
        assert msg.reply_to == "agent.coordinator.inbox"

    def test_message_id_has_msg_prefix(self) -> None:
        msg = make_message(
            performative=Performative.REQUEST,
            sender="CoordinatorAgent",
            receiver="SummarizerAgent",
            task_id="task-7c41",
            conversation_id="conv-7c41-3",
            content={},
        )
        assert msg.message_id.startswith("msg-")

    def test_timestamp_is_iso_utc(self) -> None:
        msg = make_message(
            performative=Performative.REQUEST,
            sender="CoordinatorAgent",
            receiver="SummarizerAgent",
            task_id="task-7c41",
            conversation_id="conv-7c41-3",
            content={},
        )
        assert msg.timestamp.endswith("Z")
        assert "T" in msg.timestamp


class TestInformMessageShape:
    def test_inform_with_in_reply_to(self) -> None:
        msg = make_message(
            performative=Performative.INFORM,
            sender="SummarizerAgent",
            receiver="CoordinatorAgent",
            task_id="task-7c41",
            conversation_id="conv-7c41-3",
            content={"summary_id": "sum-7c41", "sections": []},
            in_reply_to="msg-9f2a",
            subtask_id="st-task-7c41-F3",
        )
        assert msg.in_reply_to == "msg-9f2a"
        assert msg.subtask_id == "st-task-7c41-F3"

    def test_request_omits_subtask_id_by_default(self) -> None:
        msg = make_message(
            performative=Performative.REQUEST,
            sender="CoordinatorAgent",
            receiver="SummarizerAgent",
            task_id="task-7c41",
            conversation_id="conv-7c41-3",
            content={},
        )
        assert msg.subtask_id is None


class TestPerformativeWhitelist:
    def test_only_four_performatives_allowed(self) -> None:
        allowed = {p.value for p in Performative}
        assert allowed == {"request", "inform", "refuse", "query-ref"}

    def test_invalid_performative_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Message.model_validate(
                {
                    "message_id": "msg-1",
                    "performative": "propose",
                    "sender": "A",
                    "receiver": "B",
                    "task_id": "t",
                    "conversation_id": "c",
                    "content": {},
                    "timestamp": "2026-05-20T10:14:22Z",
                }
            )


class TestImmutability:
    def test_message_is_frozen(self) -> None:
        msg = make_message(
            performative=Performative.REQUEST,
            sender="CoordinatorAgent",
            receiver="SummarizerAgent",
            task_id="task-7c41",
            conversation_id="conv-7c41-3",
            content={},
        )
        with pytest.raises(ValidationError):
            msg.message_id = "tampered"


class TestExtraFieldsForbidden:
    def test_extra_field_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Message.model_validate(
                {
                    "message_id": "msg-1",
                    "performative": "request",
                    "sender": "A",
                    "receiver": "B",
                    "task_id": "t",
                    "conversation_id": "c",
                    "content": {},
                    "timestamp": "2026-05-20T10:14:22Z",
                    "priority": "high",
                }
            )
