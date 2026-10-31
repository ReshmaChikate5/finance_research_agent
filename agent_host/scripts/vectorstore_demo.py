"""Manual check with the REAL embedding model (downloads ~90 MB on first run).

Run from agent_host/:
    uv run --extra local python -m scripts.vectorstore_demo

Uses a throwaway temp database and obviously fictional text, so it never touches your
real chroma_db.
"""

import asyncio
import tempfile
import time

from app.vectorstore.base import Chunk
from app.vectorstore.chroma_store import ChromaVectorStore

SAMPLES = [
    Chunk(
        "acme-1",
        "ACME depends on a single supplier for key components, so a shortage "
        "could delay production and hurt sales.",
        {"ticker": "ACME", "section": "Risk Factors"},
    ),
    Chunk(
        "acme-2",
        "ACME quarterly revenue increased as customers bought more subscriptions.",
        {"ticker": "ACME", "section": "MD&A"},
    ),
    Chunk(
        "glbx-1",
        "Globex is involved in several lawsuits over patents that may require "
        "large settlement payments.",
        {"ticker": "GLBX", "section": "Legal Proceedings"},
    ),
    Chunk(
        "glbx-2",
        "Globex manufacturing partners reported shortages of raw materials.",
        {"ticker": "GLBX", "section": "Risk Factors"},
    ),
]


async def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store = ChromaVectorStore(tmp, "demo")
        t0 = time.perf_counter()
        await store.upsert(SAMPLES)
        print(
            f"Upserted {await store.count()} chunks in {time.perf_counter() - t0:.1f}s "
            "(includes model load)\n"
        )

        for question, filters in [
            ("What supply chain risks does the company face?", None),
            ("Are there any lawsuits?", None),
            ("Supply shortages", {"ticker": "GLBX"}),
        ]:
            print(f"Q: {question}   filters={filters}")
            for hit in await store.query(question, k=2, filters=filters):
                print(
                    f"   {hit.score:.3f}  [{hit.metadata['ticker']}/{hit.metadata['section']}] "
                    f"{hit.text[:70]}..."
                )
            print()


if __name__ == "__main__":
    asyncio.run(main())
