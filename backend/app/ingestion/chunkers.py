"""
app/ingestion/chunkers.py

Two chunking strategies that can be swapped via config.
The sentence-window parser is the default — it stores a wider context window
alongside each node so the synthesiser has richer surrounding text,
while retrieval still operates on the focused sentence.

Run `make eval` to see RAGAS scores for each strategy and pick the winner.
"""

from __future__ import annotations

from typing import Literal

from llama_index.core import Document
from llama_index.core.node_parser import (
    SentenceWindowNodeParser,
    SimpleNodeParser,
)
from llama_index.core.schema import BaseNode

from app.core.config import get_settings
from app.core.logging import get_logger

log = get_logger(__name__)

ChunkStrategy = Literal["fixed", "sentence_window"]


def chunk_documents(
    documents: list[Document],
    strategy: ChunkStrategy = "sentence_window",
) -> list[BaseNode]:
    """
    Chunk a list of LlamaIndex Documents into indexable nodes.

    Args:
        documents: Raw documents from the loader.
        strategy:  "fixed" → SimpleNodeParser with token-based splitting.
                   "sentence_window" → SentenceWindowNodeParser (recommended).

    Returns:
        List of BaseNode objects ready to be embedded and stored.
    """
    settings = get_settings()

    if strategy == "fixed":
        parser = SimpleNodeParser.from_defaults(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        nodes = parser.get_nodes_from_documents(documents, show_progress=True)
        log.info(
            "chunked_fixed",
            strategy=strategy,
            chunk_size=settings.chunk_size,
            overlap=settings.chunk_overlap,
            num_nodes=len(nodes),
        )

    elif strategy == "sentence_window":
        parser = SentenceWindowNodeParser.from_defaults(
            # Each node = one sentence; window stores N surrounding sentences
            # as metadata for the synthesiser to reference
            window_size=3,
            window_metadata_key="window",
            original_text_metadata_key="original_text",
        )
        nodes = parser.get_nodes_from_documents(documents, show_progress=True)
        log.info(
            "chunked_sentence_window",
            strategy=strategy,
            window_size=3,
            num_nodes=len(nodes),
        )

    else:
        raise ValueError(f"Unknown chunking strategy: {strategy!r}")

    return nodes
