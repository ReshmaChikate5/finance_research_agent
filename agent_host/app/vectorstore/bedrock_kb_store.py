"""AWS implementation placeholder: Amazon Bedrock Knowledge Bases.

Implemented in Step 24 (after the CDK stacks create the Knowledge Base). It exists now so
the VECTOR_STORE=bedrock_kb flag is already wired and fails with a clear message.
"""

from __future__ import annotations

from collections.abc import Sequence

from app.vectorstore.base import Chunk, Filters, RetrievedChunk


class BedrockKnowledgeBaseStore:
    def __init__(self, knowledge_base_id: str, region: str) -> None:
        raise NotImplementedError(
            "BedrockKnowledgeBaseStore is implemented in Step 24. "
            "Set VECTOR_STORE=chroma until then."
        )

    async def upsert(self, chunks: Sequence[Chunk]) -> int:  # pragma: no cover
        raise NotImplementedError

    async def query(  # pragma: no cover
        self, text: str, k: int = 5, filters: Filters | None = None
    ) -> list[RetrievedChunk]:
        raise NotImplementedError
