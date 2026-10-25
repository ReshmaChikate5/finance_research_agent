"""Local ChromaVectorStore: file-backed, no server process.

We compute embeddings ourselves (embeddings.py) and hand Chroma the vectors, instead of
registering a Chroma embedding function. That keeps the code independent of Chroma's
embedding-function API, and ingestion can reuse the same embedder.

Chroma's client is synchronous, so every call runs in a worker thread.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from datetime import datetime, timezone
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
    """Chroma only accepts str/int/float/bool values: drop None, stringify anything else."""
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
            embedding_function=None,  # we supply vectors ourselves
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

    # ---- public async API -------------------------------------------------
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
        """Number of stored chunks (Chroma-specific helper, not part of the interface)."""
        return await asyncio.to_thread(self._col.count)

    # ---- sync internals ---------------------------------------------------
    def _upsert_sync(self, chunks: Sequence[Chunk]) -> int:
        unique = {c.id: c for c in chunks}  # duplicate ids in one call: last one wins
        items = list(unique.values())
        stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        for start in range(0, len(items), BATCH_SIZE):
            batch = items[start : start + BATCH_SIZE]
            self._col.upsert(
                ids=[c.id for c in batch],
                documents=[c.text for c in batch],
                embeddings=self._embedder.embed([c.text for c in batch]),
                # indexed_at also guarantees metadata is never empty (Chroma rejects {})
                metadatas=[{**_clean_metadata(c.metadata), "indexed_at": stamp} for c in batch],
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
                id=cid,
                text=doc,
                score=round(1.0 - float(dist), 4),  # cosine distance -> similarity
                metadata=dict(meta or {}),
            )
            for cid, doc, meta, dist in zip(
                result["ids"][0],
                result["documents"][0],
                result["metadatas"][0],
                result["distances"][0],
            )
        ]
