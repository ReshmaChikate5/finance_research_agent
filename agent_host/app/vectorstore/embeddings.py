"""Local embedding model (sentence-transformers), loaded lazily.

Model choice: all-MiniLM-L6-v2 is small (~90 MB), fast on CPU, and produces 384-dim
vectors. Its limit is 256 word-pieces per input (roughly 800-1000 characters), and longer
text is silently truncated, so keep chunks about that size (Step 13 chunker).
A stronger drop-in alternative: BAAI/bge-small-en-v1.5 (also 384-dim, 512-token limit).

Changing the model changes the vector space: re-ingest into a fresh collection.
The model is loaded on first use, so importing/constructing this class is cheap.
"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from typing import Any, Protocol

from app.vectorstore.base import VectorStoreError

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class Embedder(Protocol):
    name: str  # recorded on the collection so mixed models are detected

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        self.name = model_name
        self._model: Any = None
        self._lock = threading.Lock()

    def _load(self) -> Any:
        with self._lock:
            if self._model is None:
                try:
                    from sentence_transformers import SentenceTransformer
                except ImportError as exc:
                    raise VectorStoreError(
                        "sentence-transformers is not installed. From agent_host run: "
                        "uv sync --extra local"
                    ) from exc
                self._model = SentenceTransformer(self.name, device="cpu")
            return self._model

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = self._load().encode(
            list(texts), normalize_embeddings=True, batch_size=32, show_progress_bar=False
        )
        return vectors.tolist()
