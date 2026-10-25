"""Unit tests for the vector store layer.

Uses a tiny deterministic fake embedder (hashed bag of words), so no model is downloaded
and tests run in about a second. Run from agent_host/:

    uv run --extra local python -m pytest ../tests/unit/test_vectorstore.py
"""

import asyncio
import hashlib
import math
import re

import pytest

from app.config import Settings
from app.vectorstore import get_vector_store
from app.vectorstore.base import Chunk, VectorStore, VectorStoreError
from app.vectorstore.chroma_store import ChromaVectorStore


class FakeEmbedder:
    """Word-overlap embedder: texts sharing words get similar vectors."""

    DIM = 64

    def __init__(self, name: str = "fake-hash-64") -> None:
        self.name = name

    def embed(self, texts):
        out = []
        for text in texts:
            vec = [0.0] * self.DIM
            for word in re.findall(r"[a-z0-9]+", text.lower()):
                vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.DIM] += 1.0
            norm = math.sqrt(sum(x * x for x in vec)) or 1.0
            out.append([x / norm for x in vec])
        return out


CHUNKS = [
    Chunk("a1", "supply chain disruption could delay chip shipments",
          {"ticker": "NVDA", "form_type": "10-K"}),
    Chunk("a2", "revenue grew strongly driven by data center demand",
          {"ticker": "NVDA", "form_type": "10-Q"}),
    Chunk("b1", "supply chain constraints affected our manufacturing partners",
          {"ticker": "AMD", "form_type": "10-K"}),
    Chunk("b2", "we are party to legal proceedings and patent litigation",
          {"ticker": "AMD", "form_type": "10-K"}),
]


@pytest.fixture
def store(tmp_path):
    return ChromaVectorStore(tmp_path / "db", "test", embedder=FakeEmbedder())


def run(coro):
    return asyncio.run(coro)


def test_implements_protocol(store):
    assert isinstance(store, VectorStore)


def test_query_returns_most_relevant_first(store):
    run(store.upsert(CHUNKS))
    hits = run(store.query("supply chain disruption chip shipments", k=3))
    assert hits[0].id == "a1"
    assert hits[0].score >= hits[1].score >= hits[2].score
    assert hits[0].metadata["ticker"] == "NVDA"
    assert "indexed_at" in hits[0].metadata


def test_upsert_same_id_replaces_not_duplicates(store):
    run(store.upsert(CHUNKS))
    run(store.upsert([Chunk("a1", "completely different text about dividends", {"ticker": "NVDA"})]))
    assert run(store.count()) == len(CHUNKS)
    hits = run(store.query("dividends", k=1))
    assert hits[0].id == "a1"
    assert "dividends" in hits[0].text


def test_duplicate_ids_within_one_call_last_wins(store):
    n = run(store.upsert([Chunk("x", "first version"), Chunk("x", "second version")]))
    assert n == 1
    assert run(store.query("second version", k=1))[0].text == "second version"


def test_filter_by_ticker(store):
    run(store.upsert(CHUNKS))
    hits = run(store.query("supply chain", k=5, filters={"ticker": "AMD"}))
    assert {h.metadata["ticker"] for h in hits} == {"AMD"}


def test_filter_multiple_keys_and_any_of(store):
    run(store.upsert(CHUNKS))
    both = run(store.query("supply chain", k=5, filters={"ticker": "NVDA", "form_type": "10-K"}))
    assert [h.id for h in both] == ["a1"]
    either = run(store.query("anything", k=5, filters={"form_type": ["10-K", "10-Q"]}))
    assert len(either) == 4


def test_query_empty_collection_returns_empty_list(store):
    assert run(store.query("anything")) == []


def test_k_larger_than_collection_is_fine(store):
    run(store.upsert(CHUNKS[:2]))
    assert len(run(store.query("supply chain", k=20))) == 2


def test_blank_query_rejected(store):
    with pytest.raises(ValueError):
        run(store.query("   "))


def test_chunk_validation():
    with pytest.raises(ValueError):
        Chunk("", "text")
    with pytest.raises(ValueError):
        Chunk("id", "   ")


def test_none_metadata_values_are_dropped(store):
    run(store.upsert([Chunk("n1", "some text here", {"ticker": "NVDA", "section": None})]))
    assert "section" not in run(store.query("some text", k=1))[0].metadata


def test_embedding_model_mismatch_is_rejected(tmp_path):
    ChromaVectorStore(tmp_path / "db", "test", embedder=FakeEmbedder("model-a"))
    with pytest.raises(VectorStoreError, match="embedding model"):
        ChromaVectorStore(tmp_path / "db", "test", embedder=FakeEmbedder("model-b"))


def test_data_persists_across_instances(tmp_path):
    ChromaVectorStore(tmp_path / "db", "test", embedder=FakeEmbedder())
    first = ChromaVectorStore(tmp_path / "db", "test", embedder=FakeEmbedder())
    run(first.upsert(CHUNKS))
    second = ChromaVectorStore(tmp_path / "db", "test", embedder=FakeEmbedder())
    assert run(second.count()) == len(CHUNKS)


def test_factory_selects_chroma(tmp_path):
    settings = Settings(vector_store="chroma", chroma_path=tmp_path / "c", _env_file=None)
    assert isinstance(get_vector_store(settings), ChromaVectorStore)


def test_factory_bedrock_flag_is_wired_but_not_implemented_yet():
    settings = Settings(vector_store="bedrock_kb", bedrock_kb_id="kb-123", _env_file=None)
    with pytest.raises(NotImplementedError, match="Step 24"):
        get_vector_store(settings)


def test_bedrock_requires_kb_id():
    with pytest.raises(ValueError, match="BEDROCK_KB_ID"):
        Settings(vector_store="bedrock_kb", _env_file=None)


def test_invalid_flag_value_rejected():
    with pytest.raises(ValueError):
        Settings(vector_store="pinecone", _env_file=None)
