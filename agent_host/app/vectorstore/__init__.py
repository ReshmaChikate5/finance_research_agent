"""Vector store factory and public exports."""

from __future__ import annotations

from app.config import Settings, get_settings
from app.vectorstore.base import Chunk, RetrievedChunk, VectorStore, VectorStoreError
from app.vectorstore.chroma_store import ChromaVectorStore

__all__ = [
    "ChromaVectorStore",
    "Chunk",
    "RetrievedChunk",
    "Settings",
    "VectorStore",
    "VectorStoreError",
    "get_vector_store",
]


def get_vector_store(settings: Settings | None = None) -> VectorStore:
    resolved = settings or get_settings()

    if resolved.vector_store == "chroma":
        return ChromaVectorStore(
            path=resolved.chroma_path,
            collection=resolved.chroma_collection,
        )

    if resolved.vector_store == "bedrock_kb":
        if not resolved.bedrock_kb_id:
            raise ValueError("BEDROCK_KB_ID must be set when VECTOR_STORE=bedrock_kb")
        raise NotImplementedError("Bedrock Knowledge Base support is planned in Step 24.")

    raise ValueError(f"Unsupported VECTOR_STORE value: {resolved.vector_store!r}")
