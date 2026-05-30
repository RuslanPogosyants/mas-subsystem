"""F6 corpus selection respects demo_mode when no real corpus is present."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.config import get_settings
from src.main import _build_recommender

from tests.support.fake_bus import FakeBus

if TYPE_CHECKING:
    import pytest


async def test_demo_mode_true_yields_demo_corpus(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv("CORPUS_PATH", str(tmp_path))  # empty dir -> no real corpus files
    monkeypatch.setenv("DEMO_MODE", "true")
    agent = await _build_recommender(FakeBus(), get_settings())
    assert len(agent._corpus) > 0


async def test_demo_mode_false_yields_empty_corpus(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv("CORPUS_PATH", str(tmp_path))
    monkeypatch.setenv("DEMO_MODE", "false")
    agent = await _build_recommender(FakeBus(), get_settings())
    assert agent._corpus == []
