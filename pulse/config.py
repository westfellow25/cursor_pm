"""Centralised application settings loaded from env / .env file."""

from __future__ import annotations

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Core ──────────────────────────────────────────
    app_name: str = "Pulse"
    debug: bool = False
    database_url: str = f"sqlite:///{BASE_DIR / 'pulse.db'}"
    secret_key: str = "change-me"

    # ── LLM providers ─────────────────────────────────
    # Claude (Anthropic) — preferred for product/business analysis
    anthropic_api_key: str = ""
    claude_model: str = "claude-sonnet-4-6"

    # OpenAI — used for embeddings + LLM fallback
    openai_api_key: str = ""
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 256
    llm_model: str = "gpt-4o-mini"

    # ── Integrations ──────────────────────────────────
    intercom_access_token: str = ""
    zendesk_subdomain: str = ""
    zendesk_api_token: str = ""
    slack_bot_token: str = ""
    slack_signing_secret: str = ""

    # ── Server ────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    log_level: str = "info"

    # ── Analysis defaults ─────────────────────────────
    min_clusters: int = 3
    max_clusters: int = 20
    similarity_threshold: float = 0.62
    trend_window_days: int = 30
    anomaly_sensitivity: float = 2.0  # std-devs for anomaly detection


settings = Settings()
