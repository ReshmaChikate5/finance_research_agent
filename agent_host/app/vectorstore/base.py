"""VectorStore interface: the ONLY thing the LangGraph rag_node should depend on.

Any backend (local Chroma now, Bedrock Knowledge Bases on AWS later) implements these two
async methods. The graph never imports a concrete store, so swapping is a config change.

Conventional metadata keys (keep them consistent across ingestion and retrieval so the
synthesizer can cite sources): ticker, form_type, accession_number, filing_date, section,
source_url.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

MetadataValue = str | int | float | bool
Filters = dict[str, MetadataValue | list[MetadataValue]]


class VectorStoreError(Exception):
    """A vector-store problem whose message is safe to show (setup, config, mismatch)."""


@dataclass(frozen=True)
class Chunk:
    """One piece of text to index. `id` must be stable so re-ingesting replaces, not duplicates."""

    id: str
    text: str
    metadata: dict[str, MetadataValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id or not self.id.strip():
            raise ValueError("Chunk.id must be a non-empty string")
        if not self.text or not self.text.strip():
            raise ValueError(f"Chunk {self.id!r} has empty text")


@dataclass(frozen=True)
class RetrievedChunk:
    """A search hit. `score` is cosine similarity (higher is more relevant, about 0 to 1)."""

    id: str
    text: str
    score: float
    metadata: dict[str, MetadataValue]


@runtime_checkable
class VectorStore(Protocol):
    async def upsert(self, chunks: Sequence[Chunk]) -> int:
        """Insert or replace chunks by id. Returns how many distinct chunks were written."""
        ...

    async def query(
        self, text: str, k: int = 5, filters: Filters | None = None
    ) -> list[RetrievedChunk]:
        """Return the k chunks most similar to `text`, best first.

        `filters` narrows by metadata equality, e.g. {"ticker": "NVDA"}; a list value
        means "any of", e.g. {"form_type": ["10-K", "10-Q"]}.
        """
        ...
