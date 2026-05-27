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

    coord_global_deadline_sec: int = 1800
    coord_retry_max: int = 2

    coord_timeout_transcriber: int = 600
    coord_timeout_ocr: int = 120
    coord_timeout_summarizer: int = 90
    coord_timeout_test_generator: int = 60
    coord_timeout_terminology: int = 30
    coord_timeout_recommender: int = 15

    log_level: str = "INFO"
    log_format: str = "coordinator"

    force_refuse: str = ""
    hang_agent: str = ""


def get_settings() -> Settings:
    """Settings factory; cacheable via FastAPI Depends."""
    return Settings()
