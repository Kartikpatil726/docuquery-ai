import hashlib
from dataclasses import dataclass
from functools import lru_cache
from typing import Protocol

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.core.config import get_settings
from app.services.chunking import Chunk


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    doc_id: str
    filename: str
    page: int
    text: str
    score: float


class VectorStore(Protocol):
    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None: ...

    def search(
        self, query_embedding: list[float], k: int, doc_id: str | None = None
    ) -> list[RetrievedChunk]: ...

    def doc_exists(self, doc_id: str) -> bool: ...

    def list_documents(self) -> list[dict]: ...

    def delete_document(self, doc_id: str) -> int: ...

    def count(self) -> int: ...


class ChromaVectorStore:
    WRITE_BATCH = 256

    def __init__(self, persist_dir: str, collection_name: str) -> None:
        self._client = chromadb.PersistentClient(
            path=persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={'hnsw:space': 'cosine'},
        )

    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError('chunks and embeddings must have the same length')
        for start in range(0, len(chunks), self.WRITE_BATCH):
            part = chunks[start:start + self.WRITE_BATCH]
            self._collection.upsert(
                ids=[c.chunk_id for c in part],
                embeddings=embeddings[start:start + self.WRITE_BATCH],
                documents=[c.text for c in part],
                metadatas=[
                    {
                        'doc_id': c.doc_id,
                        'filename': c.filename,
                        'page': c.page,
                        'chunk_index': c.chunk_index,
                    }
                    for c in part
                ],
            )

    def search(
        self, query_embedding: list[float], k: int, doc_id: str | None = None
    ) -> list[RetrievedChunk]:
        total = self._collection.count()
        if total == 0:
            return []
        where = {'doc_id': doc_id} if doc_id else None
        result = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(k, total),
            where=where,
            include=['documents', 'metadatas', 'distances'],
        )
        hits: list[RetrievedChunk] = []
        for chunk_id, text, meta, distance in zip(
            result['ids'][0],
            result['documents'][0],
            result['metadatas'][0],
            result['distances'][0],
        ):
            hits.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    doc_id=meta['doc_id'],
                    filename=meta['filename'],
                    page=int(meta['page']),
                    text=text,
                    score=round(1.0 - float(distance), 4),
                )
            )
        return hits

    def doc_exists(self, doc_id: str) -> bool:
        found = self._collection.get(where={'doc_id': doc_id}, limit=1, include=[])
        return len(found['ids']) > 0

    def list_documents(self) -> list[dict]:
        data = self._collection.get(include=['metadatas'])
        docs: dict[str, dict] = {}
        for meta in data['metadatas']:
            entry = docs.setdefault(
                meta['doc_id'],
                {'doc_id': meta['doc_id'], 'filename': meta['filename'], 'chunks': 0},
            )
            entry['chunks'] += 1
        return sorted(docs.values(), key=lambda d: d['filename'].lower())

    def delete_document(self, doc_id: str) -> int:
        found = self._collection.get(where={'doc_id': doc_id}, include=[])
        ids = found['ids']
        if ids:
            self._collection.delete(ids=ids)
        return len(ids)

    def count(self) -> int:
        return self._collection.count()


@lru_cache
def get_vector_store() -> VectorStore:
    from app.services.embeddings import get_embedder

    s = get_settings()
    embedder = get_embedder()
    tag = hashlib.sha1(embedder.name.encode()).hexdigest()[:6]
    collection_name = f'docs_{s.embedding_provider}_{embedder.dim}_{tag}'
    s.chroma_dir.mkdir(parents=True, exist_ok=True)
    return ChromaVectorStore(str(s.chroma_dir), collection_name)
