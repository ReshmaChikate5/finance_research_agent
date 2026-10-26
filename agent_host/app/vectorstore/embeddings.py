"""Local embedding model, loaded lazily."""

from __future__ import annotations

import threading
from collections.abc import Sequence
from typing import Any, Protocol

from app.vectorstore.base import VectorStoreError

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class Embedder(Protocol):
    name: str

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
