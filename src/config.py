from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_api_key: str
    chroma_persist_dir: str = "./chroma_data"
    chunk_size: int = 500
    chunk_overlap: int = 50
    embedding_model: str = "text-embedding-3-small"
    llm_model: str = "gpt-4o-mini"
    max_retrieved_chunks: int = 5


@lru_cache
def get_settings() -> Settings:
    return Settings()
