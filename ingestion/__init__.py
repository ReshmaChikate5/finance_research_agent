"""Ingestion pipeline: fetch filings/news -> chunk -> embed -> upsert into the vector store.

Importing this package puts agent_host/ on sys.path, so ingestion code can reuse the
agent's own `app.vectorstore` and `app.config` modules (one definition of Chunk,
one embedding model, one Chroma location) without installing agent_host as a package.
"""

import sys
from pathlib import Path

_AGENT_HOST = Path(__file__).resolve().parents[1] / "agent_host"
if str(_AGENT_HOST) not in sys.path:
    sys.path.insert(0, str(_AGENT_HOST))
