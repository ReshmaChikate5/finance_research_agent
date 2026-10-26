"""Embedding for ingestion: the SAME local model the agent uses at query time.

Documents and queries must be embedded by the same model, so this module deliberately adds
no model of its own. It extends the agent's SentenceTransformerEmbedder with two things the
chunker needs: the model's token window (max_tokens) and a token counter. all-MiniLM-L6-v2
silently ignores text beyond 256 tokens, and dense financial prose (numbers, tickers,
punctuation) can run well over 4 characters per token, so the chunker uses these to re-split
any chunk that would be truncated.
"""

from __future__ import annotations

from collections.abc import Sequence

from app.vectorstore.embeddings import DEFAULT_MODEL, SentenceTransformerEmbedder

__all__ = ["IngestionEmbedder", "get_embedder", "DEFAULT_MODEL"]


class IngestionEmbedder(SentenceTransformerEmbedder):
    @property
    def max_tokens(self) -> int:
        """The model's input window in tokens (256 for all-MiniLM-L6-v2)."""
        return int(self._load().max_seq_length)

    def count_tokens(self, texts: Sequence[str]) -> list[int]:
        """Tokens each text would use, including special tokens, without truncation."""
        tokenizer = self._load().tokenizer
        try:  # silence the "longer than max length" warning: we WANT to measure that
            tokenizer.deprecation_warnings["sequence-length-is-longer-than-the-specified-maximum"] = True
        except Exception:  # noqa: BLE001 - cosmetic only
            pass
        encoded = tokenizer(list(texts), add_special_tokens=True, truncation=False)
        return [len(ids) for ids in encoded["input_ids"]]


def get_embedder(model_name: str = DEFAULT_MODEL) -> IngestionEmbedder:
    return IngestionEmbedder(model_name)
