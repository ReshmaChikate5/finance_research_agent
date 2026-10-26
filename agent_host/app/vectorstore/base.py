"""Shared vector-store types and protocol."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

MetadataValue = str | int | float | bool
Filters = dict[str, MetadataValue | list[MetadataValue]]


class VectorStoreError(Exception):
    """A vector-store problem whose message is safe to show."""


@dataclass(frozen=True)
class Chunk:
    """One piece of text to index, identified by a stable id."""

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
    """A search hit, ordered by cosine similarity."""

    id: str
    text: str
    score: float
    metadata: dict[str, MetadataValue]


@runtime_checkable
class VectorStore(Protocol):
    async def upsert(self, chunks: Sequence[Chunk]) -> int:
        """Insert or replace chunks by id and return the number written."""
        ...

    async def query(
        self, text: str, k: int = 5, filters: Filters | None = None
    ) -> list[RetrievedChunk]:
        """Return the most similar chunks, optionally filtered by metadata."""
        ...
