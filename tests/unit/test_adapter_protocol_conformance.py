"""The real adapter classes structurally expose their Protocol methods."""

from __future__ import annotations

from src.adapters.embedding import EmbeddingAdapter
from src.adapters.llm import LlmAdapter
from src.adapters.ner import NerAdapter
from src.adapters.ocr import OcrAdapter
from src.adapters.transcriber import TranscriberAdapter


def test_real_adapters_expose_protocol_methods() -> None:
    from src.adapters.gigachat import GigaChatAdapter
    from src.adapters.pymupdf_ocr import PymupdfOcrAdapter
    from src.adapters.sentence_transformer import SentenceTransformerEmbeddingAdapter
    from src.adapters.spacy_ner import SpacyNerAdapter
    from src.adapters.whisper_transcriber import WhisperTranscriberAdapter

    assert callable(GigaChatAdapter.complete)
    assert callable(WhisperTranscriberAdapter.transcribe)
    assert callable(PymupdfOcrAdapter.extract)
    assert callable(SpacyNerAdapter.extract)
    assert callable(SentenceTransformerEmbeddingAdapter.encode)
    # Protocols themselves are importable and non-None (structural conformance marker)
    assert all([EmbeddingAdapter, LlmAdapter, NerAdapter, OcrAdapter, TranscriberAdapter])
