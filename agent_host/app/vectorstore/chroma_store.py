"""Local ChromaVectorStore: file-backed, no server process."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.vectorstore.base import (
    Chunk,
    Filters,
    MetadataValue,
    RetrievedChunk,
    VectorStoreError,
)
from app.vectorstore.embeddings import DEFAULT_MODEL, Embedder, SentenceTransformerEmbedder

logger = logging.getLogger("vectorstore.chroma")

BATCH_SIZE = 256
MAX_K = 50


def _clean_metadata(metadata: dict[str, Any]) -> dict[str, MetadataValue]:
    clean: dict[str, MetadataValue] = {}
    for key, value in metadata.items():
        if value is None:
            continue
        clean[str(key)] = value if isinstance(value, (str, int, float, bool)) else str(value)
    return clean


def _build_where(filters: Filters | None) -> dict[str, Any] | None:
    if not filters:
        return None
    clauses: list[dict[str, Any]] = []
    for key, value in filters.items():
        if isinstance(value, (list, tuple, set)):
            clauses.append({key: {"$in": list(value)}})
        else:
            clauses.append({key: value})
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


class ChromaVectorStore:
    def __init__(
        self,
        path: str | Path,
        collection: str = "filings",
        embedder: Embedder | None = None,
        model_name: str = DEFAULT_MODEL,
    ) -> None:
        self._embedder: Embedder = embedder or SentenceTransformerEmbedder(model_name)
        Path(path).mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(
            path=str(path), settings=ChromaSettings(anonymized_telemetry=False)
        )
        self._col = client.get_or_create_collection(
            name=collection,
            embedding_function=None,
            metadata={"hnsw:space": "cosine", "embedding_model": self._embedder.name},
        )
        stored = (self._col.metadata or {}).get("embedding_model")
        if stored and stored != self._embedder.name:
            raise VectorStoreError(
                f"Collection '{collection}' was built with embedding model '{stored}', but "
                f"'{self._embedder.name}' is configured. Vectors from different models are "
                "not comparable. Use a different CHROMA_COLLECTION or delete the chroma_db "
                "folder and re-ingest."
            )

    async def upsert(self, chunks: Sequence[Chunk]) -> int:
        return await asyncio.to_thread(self._upsert_sync, chunks)

    async def query(
        self, text: str, k: int = 5, filters: Filters | None = None
    ) -> list[RetrievedChunk]:
        if not text or not text.strip():
            raise ValueError("query text must not be empty")
        k = max(1, min(int(k), MAX_K))
        return await asyncio.to_thread(self._query_sync, text, k, filters)

    async def count(self) -> int:
        return await asyncio.to_thread(self._col.count)

    def _upsert_sync(self, chunks: Sequence[Chunk]) -> int:
        unique = {chunk.id: chunk for chunk in chunks}
        items = list(unique.values())
        stamp = datetime.now(UTC).isoformat(timespec="seconds")
        for start in range(0, len(items), BATCH_SIZE):
            batch = items[start : start + BATCH_SIZE]
            self._col.upsert(
                ids=[chunk.id for chunk in batch],
                documents=[chunk.text for chunk in batch],
                embeddings=self._embedder.embed([chunk.text for chunk in batch]),
                metadatas=[
                    {**_clean_metadata(chunk.metadata), "indexed_at": stamp} for chunk in batch
                ],
            )
        logger.info("Upserted %d chunks", len(items))
        return len(items)

    def _query_sync(self, text: str, k: int, filters: Filters | None) -> list[RetrievedChunk]:
        total = self._col.count()
        if total == 0:
            return []
        result = self._col.query(
            query_embeddings=self._embedder.embed([text]),
            n_results=min(k, total),
            where=_build_where(filters),
            include=["documents", "metadatas", "distances"],
        )
        return [
            RetrievedChunk(
                id=chunk_id,
                text=document,
                score=round(1.0 - float(distance), 4),
                metadata=dict(metadata or {}),
            )
            for chunk_id, document, metadata, distance in zip(
                result["ids"][0],
                result["documents"][0],
                result["metadatas"][0],
                result["distances"][0],
            )
        ]
