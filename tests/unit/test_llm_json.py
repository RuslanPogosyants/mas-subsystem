"""Unit tests for the shared parse_with_retry helper."""

from __future__ import annotations

import json

from pydantic import BaseModel
from src.adapters.llm import FakeLlmAdapter
from src.agents._llm_json import parse_with_retry


class _Model(BaseModel):
    a: int
    b: str


async def test_parses_valid_json_first_try() -> None:
    llm = FakeLlmAdapter(responses=[json.dumps({"a": 1, "b": "x"})])
    result = await parse_with_retry(llm, system="s", user="u", model_cls=_Model, retries=1)
    assert result == _Model(a=1, b="x")
    assert len(llm.calls) == 1


async def test_retries_then_succeeds() -> None:
    llm = FakeLlmAdapter(responses=["nonsense", json.dumps({"a": 2, "b": "y"})])
    result = await parse_with_retry(llm, system="s", user="u", model_cls=_Model, retries=1)
    assert result == _Model(a=2, b="y")
    assert len(llm.calls) == 2


async def test_returns_none_when_exhausted() -> None:
    llm = FakeLlmAdapter(responses=["bad", "bad", "bad"])
    result = await parse_with_retry(llm, system="s", user="u", model_cls=_Model, retries=1)
    assert result is None
    assert len(llm.calls) == 2  # initial + 1 retry


async def test_returns_none_on_schema_mismatch() -> None:
    llm = FakeLlmAdapter(responses=[json.dumps({"a": "not-an-int", "b": "x"})])
    result = await parse_with_retry(llm, system="s", user="u", model_cls=_Model, retries=0)
    assert result is None
