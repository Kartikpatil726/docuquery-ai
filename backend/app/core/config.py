from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=('../.env', '.env'),
        env_file_encoding='utf-8',
        extra='ignore',
    )

    app_name: str = 'DocuQuery AI'
    environment: Literal['development', 'production'] = 'development'
    log_level: str = 'INFO'

    gemini_api_key: str = ''
    gemini_model: str = 'gemini-2.5-flash'
    llm_temperature: float = 0.0
    llm_max_output_tokens: int = 2048

    embedding_provider: Literal['local', 'gemini'] = 'local'
    local_embedding_model: str = 'sentence-transformers/all-MiniLM-L6-v2'
    gemini_embedding_model: str = 'gemini-embedding-001'
    gemini_embedding_dim: int = 768

    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k: int = 5
    min_similarity: float = 0.25
    max_context_chars: int = 12000

    data_dir: Path = Path('../data')
    max_upload_mb: int = 25
    max_pdf_pages: int = 500

    api_key: str = ''
    cors_origins: str = 'http://localhost:8501'
    rate_limit: str = '30/minute'

    @property
    def raw_docs_dir(self) -> Path:
        return self.data_dir / 'raw_docs'

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / 'chroma'

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(',') if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
