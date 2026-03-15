from pydantic_settings import BaseSettings
from typing import Optional
from functools import lru_cache


class Settings(BaseSettings):
    """Settings for the application"""

    # ── Mistral API Configs (Primary) ───────────
    MISTRAL_API_KEY: str = ""
    MISTRAL_LLM_MODEL: str = "mistral-small-latest"
    MISTRAL_EMBEDDING_MODEL: str = "mistral-embed"
    # Mistral embed is 1024 dimension
    EMBEDDING_DIMENSION: int = 1024

    # ── RAG Settings ───────────────────────────
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 128
    TOP_K: int = 5
    SIMILARITY_THRESHOLD: float = 0.7
    CACHE_TTL: int = 3600

    SECRET_KEY: str = "your-super-secret-key-change-this-in-prod"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 

    # ── External Services (Optional) ────────────
    TAVILY_API_KEY: str = ""
    AWS_ACCESS_KEY_ID: str = ""
    AWS_SECRET_ACCESS_KEY: str = ""
    AWS_REGION: str = "ap-south-1"
    S3_BUCKET_NAME: str = ""
    POSTGRES_URL: str = ""
    REDIS_URL: str = "redis://localhost:6379"

        # ── Qdrant Configs ──────────────────────────
    QDRANT_URL: str = ""
    QDRANT_API_KEY: str = ""
    QDRANT_COLLECTION_NAME: str = "rag_chatbot"


    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    """Cached settings instance"""
    return Settings()


settings = get_settings()