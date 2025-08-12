from __future__ import annotations

from functools import lru_cache
from pydantic import BaseSettings


class Settings(BaseSettings):
    # App
    RUN_MODE: str = "MEMSYM"
    LOG_LEVEL: str = "INFO"
    CORS_ALLOW_ORIGINS: str = "*"

    # Embeddings
    EMBEDDING_MODEL_NAME: str = "gpt4o-mini"   # or "all-MiniLM-L6-v2"
    EMBEDDING_VECTOR_DIM: int = 768

    # OpenAI (optional)
    OPENAI_API_KEY: str | None = None
    OPENAI_EMBEDDING_MODEL: str | None = None  # e.g., "text-embedding-3-small"

    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    return Settings()


# 👉 this is what `from backend.config.config import settings` expects
settings: Settings = get_settings()
