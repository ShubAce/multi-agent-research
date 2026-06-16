"""
app/memory/long_term.py

Semantic long-term memory stored in ChromaDB.
After each research session the Synthesiser extracts key facts;
those facts are embedded and stored so future queries can recall them.

Collection:  research_memory  (separate from the arxiv_papers corpus)
"""

from __future__ import annotations

import time
import uuid
from typing import Mapping

import chromadb

from app.core.config import get_settings
from app.core.logging import get_logger

log = get_logger(__name__)

_COLLECTION_NAME = "research_memory"


class LongTermMemory:
    """
    Stores extracted facts from past sessions as embeddings.
    Uses the same ChromaDB instance as the RAG corpus but a different collection.

    Note: ChromaDB's built-in embedding handles the embedding step here,
    using the default sentence-transformers model it ships with.
    This keeps this module dependency-light.
    """

    def __init__(self) -> None:
        settings = get_settings()
        from chromadb.config import Settings as ChromaSettings
        self._client = chromadb.HttpClient(
            host=settings.chroma_host,
            port=settings.chroma_port,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    # ── Write ─────────────────────────────────────────────────────────────────

    async def memorise(
        self,
        session_id: str,
        facts: list[str],
    ) -> int:
        """
        Embed and store a list of facts extracted from a research session.

        Args:
            session_id: The session these facts came from.
            facts:      Short, self-contained factual statements.

        Returns:
            Number of facts stored.
        """
        if not facts:
            return 0

        ids = [str(uuid.uuid4()) for _ in facts]
        metadatas: list[Mapping[str, str | float | int | bool]] = [
            {"session_id": session_id, "stored_at": time.time()}
            for _ in facts
        ]

        # ChromaDB will embed using its own default model
        self._collection.add(documents=facts, ids=ids, metadatas=metadatas)
        log.info("facts_memorised", session_id=session_id, count=len(facts))
        return len(facts)

    # ── Read ──────────────────────────────────────────────────────────────────

    async def recall(
        self,
        query: str,
        n: int = 4,
    ) -> list[str]:
        """
        Retrieve the `n` most semantically relevant facts for a query.

        Returns an empty list if the memory store is empty.
        """
        if self._collection.count() == 0:
            return []

        results = self._collection.query(
            query_texts=[query],
            n_results=min(n, self._collection.count()),
        )
        facts: list[str] = results["documents"][0] if results["documents"] else []
        log.debug("facts_recalled", query=query[:60], count=len(facts))
        return facts

    async def format_for_prompt(self, query: str, n: int = 4) -> str:
        """Return recalled facts as a prompt-ready string."""
        facts = await self.recall(query, n)
        if not facts:
            return ""
        bullet_points = "\n".join(f"- {f}" for f in facts)
        return f"Relevant facts from past research:\n{bullet_points}"

    # ── Stats ─────────────────────────────────────────────────────────────────

    def count(self) -> int:
        return self._collection.count()
