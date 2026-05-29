"""Unit tests for FakeLlmAdapter."""

from __future__ import annotations

import json

from src.adapters.llm import FakeLlmAdapter


async def test_default_returns_valid_summary_json() -> None:
    fake = FakeLlmAdapter()
    response = await fake.complete(system="s", user="hello")
    data = json.loads(response)
    assert set(data) == {"introduction", "key_points", "conclusions"}


async def test_scripted_responses_returned_in_order_then_repeat_last() -> None:
    fake = FakeLlmAdapter(responses=["a", "b"])
    assert await fake.complete(system="s", user="1") == "a"
    assert await fake.complete(system="s", user="2") == "b"
    assert await fake.complete(system="s", user="3") == "b"  # exhausted -> last


async def test_records_calls() -> None:
    fake = FakeLlmAdapter()
    await fake.complete(system="sys", user="usr")
    assert fake.calls == [{"system": "sys", "user": "usr"}]
