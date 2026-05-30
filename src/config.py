"""Subsystem configuration via pydantic-settings."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    database_url: str = "postgresql+asyncpg://mas:mas@localhost:5432/mas_subsystem"
    redis_url: str = "redis://localhost:6379/0"

    gigachat_credentials: str = ""
    gigachat_scope: str = "GIGACHAT_API_PERS"
    gigachat_model: str = "GigaChat-Pro"
    gigachat_temperature: float = 0.3
    # Sber's GigaChat endpoint uses a Russian CA absent from default trust stores;
    # verification is off by default for the demo and can be enabled in production.
    gigachat_verify_ssl: bool = False
    summarizer_block_chars: int = 6000
    summarizer_overlap: int = 500

    coord_global_deadline_sec: int = 1800
    coord_retry_max: int = 2

    coord_timeout_transcriber: int = 600
    coord_timeout_ocr: int = 120
    coord_timeout_summarizer: int = 90
    coord_timeout_test_generator: int = 60
    coord_timeout_terminology: int = 30
    coord_timeout_recommender: int = 15

    embedding_model: str = "intfloat/multilingual-e5-base"
    corpus_path: str = "corpus"

    transcriber_backend: str = "fake"  # "fake" | "whisper"
    whisper_model: str = "large-v3"
    whisper_device: str = "cuda"
    whisper_compute_type: str = "float16"
    ocr_backend: str = "fake"  # "fake" | "pymupdf"
    ocr_languages: str = "ru,en"
    ner_backend: str = "fake"  # "fake" | "spacy"
    spacy_model: str = "ru_core_news_lg"

    log_level: str = "INFO"
    log_format: str = "coordinator"

    force_refuse: str = ""
    hang_agent: str = ""


def get_settings() -> Settings:
    """Settings factory; cacheable via FastAPI Depends."""
    return Settings()
