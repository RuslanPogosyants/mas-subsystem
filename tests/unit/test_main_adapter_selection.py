"""The ingestion adapter selectors honour the backend setting."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pytest

from src.adapters.ocr import FakeOcrAdapter
from src.adapters.transcriber import FakeTranscriberAdapter
from src.config import Settings
from src.main import _build_ner, _build_ocr, _build_transcriber


def test_fake_env_override_selects_fake_adapters() -> None:
    """When backend env vars are forced to 'fake' (as in the test suite conftest),
    the selectors must return the in-process Fake adapters."""
    # The autouse _force_fake_backends fixture already sets the env vars to 'fake',
    # so Settings() picks them up. Pass them explicitly to document the dependency.
    settings = Settings(transcriber_backend="fake", ocr_backend="fake", ner_backend="fake", gigachat_credentials="")
    assert isinstance(_build_transcriber(settings), FakeTranscriberAdapter)
    assert isinstance(_build_ocr(settings), FakeOcrAdapter)
    assert type(_build_ner(settings)).__name__ == "FakeNerAdapter"


def test_class_defaults_are_real_backends(monkeypatch: pytest.MonkeyPatch) -> None:
    """When backend env vars are absent, Settings uses the real backend class defaults."""
    monkeypatch.delenv("TRANSCRIBER_BACKEND", raising=False)
    monkeypatch.delenv("OCR_BACKEND", raising=False)
    monkeypatch.delenv("NER_BACKEND", raising=False)
    settings = Settings()
    assert settings.transcriber_backend == "whisper"
    assert settings.ocr_backend == "pymupdf"
    assert settings.ner_backend == "spacy"


def test_whisper_backend_builds_real_adapter() -> None:
    settings = Settings(transcriber_backend="whisper")
    adapter = _build_transcriber(settings)
    assert type(adapter).__name__ == "WhisperTranscriberAdapter"


def test_pymupdf_backend_builds_real_adapter() -> None:
    settings = Settings(ocr_backend="pymupdf")
    adapter = _build_ocr(settings)
    assert type(adapter).__name__ == "PymupdfOcrAdapter"


def test_spacy_backend_builds_real_adapter() -> None:
    settings = Settings(ner_backend="spacy")
    adapter = _build_ner(settings)
    assert type(adapter).__name__ == "SpacyNerAdapter"
