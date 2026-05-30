"""The ingestion adapter selectors default to Fake and honour the backend setting."""

from __future__ import annotations

from src.adapters.ocr import FakeOcrAdapter
from src.adapters.transcriber import FakeTranscriberAdapter
from src.config import Settings
from src.main import _build_ner, _build_ocr, _build_transcriber


def test_defaults_are_fake() -> None:
    settings = Settings(gigachat_credentials="")
    assert isinstance(_build_transcriber(settings), FakeTranscriberAdapter)
    assert isinstance(_build_ocr(settings), FakeOcrAdapter)
    # NER fake: assert the type name to avoid importing the real spaCy adapter
    assert type(_build_ner(settings)).__name__ == "FakeNerAdapter"


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
