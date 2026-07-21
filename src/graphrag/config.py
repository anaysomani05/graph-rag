from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql://graphrag:graphrag@localhost:5432/graphrag"
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384

    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"


settings = Settings()
