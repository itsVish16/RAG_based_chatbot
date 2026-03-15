from pydantic_settings import BaseSettings
from typing import Optional
from functools import lru_cache


class Settings(BaseSettings):
    """Settings for the application"""

    # ── LM Studio (Local LLM) ──────────────────
    LM_STUDIO_BASE_URL: str = "http://localhost:1234/v1"
    LM_STUDIO_API_KEY: str = "lm-studio"   # LM Studio accepts any non-empty string
    LLM_MODEL: str = "lfm-2.5"            # must match model name shown in LM Studio

    # ── Embedding (local SentenceTransformers) ──
    # BAAI/bge-small-en-v1.5 → free, fast, top MTEB benchmark, 384-dim
    # Switch to google/gemma-embedding-300m only after HF login + model access
    EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"
    EMBEDDING_DIMENSION: int = 384         # output dim for bge-small-en-v1.5

    # ── RAG Settings ───────────────────────────
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 128
    TOP_K: int = 5
    SIMILARITY_THRESHOLD: float = 0.7
    CACHE_TTL: int = 3600

    # ── Optional Services (not needed right now) ─
    OPENAI_API_KEY: str = ""
    TOOL_ROUTER_MODEL: str = "gpt-4o-mini"
    TAVILY_API_KEY: str = ""
    AWS_ACCESS_KEY_ID: str = ""
    AWS_SECRET_ACCESS_KEY: str = ""
    AWS_REGION: str = "ap-south-1"
    S3_BUCKET_NAME: str = ""
    POSTGRES_URL: str = ""
    REDIS_URL: str = "redis://localhost:6379"

    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """Cached settings instance"""
    return Settings()


settings = get_settings()