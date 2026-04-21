"""Runtime configuration loaded from environment variables via pydantic-settings."""

import logging
import sys
from pathlib import Path

from pydantic import ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="", case_sensitive=False)

    log_path: Path = Path("./logs")
    default_seed: int | None = None
    host: str = "0.0.0.0"
    port: int = 8000
    curriculum_start_tier: int = 0

    # Optional LLM judge (Layer 2). When set, TrialJudge uses a real LLM for
    # qualitative workflow scoring. Falls back to rule-based stub when absent.
    judge_llm_model: str | None = None   # e.g. "gpt-4o-mini", "claude-3-haiku-20240307"
    judge_llm_api_key: str | None = None  # OpenAI or Anthropic API key
    judge_llm_base_url: str | None = None  # optional: custom base URL (e.g. local vLLM)


def get_settings() -> Settings:
    """Load and return Settings, exiting with code 1 on validation failure."""
    try:
        return Settings()
    except ValidationError as exc:
        logger.error("Startup configuration error: %s", exc)
        sys.exit(1)


settings = get_settings()
