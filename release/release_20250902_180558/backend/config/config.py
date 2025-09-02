# backend/config/config.py
from __future__ import annotations
from typing import Optional

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict  # v2
    from pydantic import Field
    _V2 = True
except Exception:
    from pydantic import BaseSettings, Field  # v1 fallback
    SettingsConfigDict = dict  # type: ignore
    _V2 = False

class Settings(BaseSettings):
    # add BOTH names so old code is happy; MiniLM default = 384
    EMBEDDING_VECTOR_DIM: int = Field(default=384)
    EMBEDDING_MODEL_NAME: Optional[str] = Field(default=None)  # legacy name
    EMBED_MODEL_NAME: Optional[str] = Field(default="sentence-transformers/all-MiniLM-L6-v2")  # preferred
    OPENAI_API_KEY: Optional[str] = Field(default=None)
    ENV: str = Field(default="dev")

    if _V2:
        model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    else:
        class Config:
            env_file = ".env"
            extra = "ignore"

settings = Settings()
