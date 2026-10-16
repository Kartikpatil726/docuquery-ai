import logging
from functools import lru_cache
from typing import Protocol

import numpy as np

from app.core.config import get_settings
from app.core.retry import call_with_retry

log = logging.getLogger(__name__)


class Embedder(Protocol):
    name: str
    dim: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


def _normalize(values) -> list[float]:
    vec = np.asarray(values, dtype=np.float64)
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec.tolist()
    return (vec / norm).tolist()


class LocalEmbedder:
    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer

        log.info('Loading local embedding model %s ...', model_name)
        self.name = model_name
        self._model = SentenceTransformer(model_name)
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(
            texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


class GeminiEmbedder:
    BATCH = 50

    def __init__(self, api_key: str, model_name: str, dim: int) -> None:
        if not api_key:
            raise RuntimeError('GEMINI_API_KEY is required when EMBEDDING_PROVIDER=gemini')
        from google import genai
        from google.genai import types

        self._client = genai.Client(api_key=api_key)
        self._types = types
        self.name = model_name
        self.dim = dim

    def _embed(self, texts: list[str], task_type: str) -> list[list[float]]:
        out: list[list[float]] = []
        for start in range(0, len(texts), self.BATCH):
            batch = texts[start:start + self.BATCH]
            config = self._types.EmbedContentConfig(
                task_type=task_type, output_dimensionality=self.dim
            )
            response = call_with_retry(
                lambda: self._client.models.embed_content(
                    model=self.name, contents=batch, config=config
                )
            )
            out.extend(_normalize(item.values) for item in response.embeddings)
        return out

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts, 'RETRIEVAL_DOCUMENT')

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], 'RETRIEVAL_QUERY')[0]


@lru_cache
def get_embedder() -> Embedder:
    s = get_settings()
    if s.embedding_provider == 'gemini':
        return GeminiEmbedder(s.gemini_api_key, s.gemini_embedding_model, s.gemini_embedding_dim)
    return LocalEmbedder(s.local_embedding_model)
