"""
app/ingestion/pipeline.py

Orchestrates the full ingestion flow:
  ArXiv API → Documents → Chunked Nodes → Embeddings → ChromaDB

Run directly:
    poetry run python -m app.ingestion.pipeline --query "flash attention"

Or via Makefile:
    make ingest
"""

from __future__ import annotations

import threading
from functools import lru_cache

import chromadb
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.schema import NodeWithScore
from llama_index.core.settings import Settings as LlamaSettings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from app.core.config import get_settings
from app.core.logging import get_logger
from app.ingestion.arxiv_loader import (
    ArxivPaper,
    base_arxiv_id,
    fetch_arxiv_papers,
    papers_to_documents,
)
from app.ingestion.chunkers import chunk_documents

log = get_logger(__name__)


# ── Default ArXiv queries for ML Research corpus ──────────────────────────────
DEFAULT_QUERIES = [
    "attention mechanism transformers",
    "large language models instruction tuning",
    "retrieval augmented generation RAG",
    "diffusion models image generation",
    "reinforcement learning from human feedback RLHF",
]


@lru_cache(maxsize=1)
def get_chroma_client() -> chromadb.HttpClient:
    """Lightweight Chroma client — no embedding model is loaded."""
    from chromadb.config import Settings as ChromaSettings

    settings = get_settings()
    return chromadb.HttpClient(
        host=settings.chroma_host,
        port=settings.chroma_port,
        settings=ChromaSettings(anonymized_telemetry=False),
    )


def get_corpus_collection():
    """The ArXiv corpus collection (created on first use)."""
    return get_chroma_client().get_or_create_collection(
        name=get_settings().chroma_collection,
        metadata={"hnsw:space": "cosine"},
    )


class IngestionPipeline:
    """
    Handles document ingestion into ChromaDB and exposes the index for retrieval.

    Construction loads the embedding model (seconds), so use `get_pipeline()`
    to share one instance per process instead of creating it per request.
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self._setup_embedding_model()
        self._setup_vector_store()
        self._nodes_lock = threading.Lock()
        self._nodes_cache: tuple[int, list[NodeWithScore]] | None = None

    def _setup_embedding_model(self) -> None:
        """
        Use a free HuggingFace embedding model — no API key required.
        BAAI/bge-small-en-v1.5 is fast and punches well above its weight.
        """
        log.info("loading_embedding_model", model=self.settings.embedding_model)
        LlamaSettings.embed_model = HuggingFaceEmbedding(
            model_name=self.settings.embedding_model,
            max_length=512,
        )
        # LlamaIndex falls back to OpenAI when no LLM is configured, so pin Groq
        try:
            from llama_index.llms.groq import Groq

            LlamaSettings.llm = Groq(
                model=self.settings.groq_model,
                api_key=self.settings.groq_api_key,
                temperature=0.1,
            )
        except Exception as exc:
            log.warning("llamaindex_llm_setup_failed", error=str(exc))

    def _setup_vector_store(self) -> None:
        """Connect to ChromaDB and get or create our collection."""
        log.info(
            "connecting_chromadb",
            host=self.settings.chroma_host,
            port=self.settings.chroma_port,
        )
        self.collection = get_corpus_collection()
        self.vector_store = ChromaVectorStore(chroma_collection=self.collection)
        log.info(
            "chromadb_ready",
            collection=self.settings.chroma_collection,
            existing_docs=self.collection.count(),
        )

    # ── Retrieval helpers ─────────────────────────────────────────────────────

    def get_index(self) -> VectorStoreIndex:
        """
        Return a VectorStoreIndex backed by the existing ChromaDB collection.
        Used by the RAG agent to query without re-ingesting.
        """
        return VectorStoreIndex.from_vector_store(self.vector_store)

    def corpus_nodes(self) -> list[NodeWithScore]:
        """
        Every chunk in the collection, rebuilt as LlamaIndex nodes (for BM25).

        The Chroma vector store does not populate LlamaIndex's docstore, so the
        nodes have to be reconstructed from Chroma's metadata. Cached until the
        collection size changes.
        """
        from llama_index.core.vector_stores.utils import metadata_dict_to_node

        count = self.collection.count()
        with self._nodes_lock:
            if self._nodes_cache and self._nodes_cache[0] == count:
                return self._nodes_cache[1]

            raw = self.collection.get(include=["documents", "metadatas"])
            nodes: list[NodeWithScore] = []
            for doc, meta in zip(raw["documents"] or [], raw["metadatas"] or [], strict=False):
                try:
                    node = metadata_dict_to_node(meta, text=doc)
                    nodes.append(NodeWithScore(node=node, score=1.0))
                except Exception:
                    continue
            self._nodes_cache = (count, nodes)
            log.info("corpus_nodes_loaded", count=len(nodes))
            return nodes

    # ── Ingestion ─────────────────────────────────────────────────────────────

    def existing_arxiv_ids(self) -> set[str]:
        """Base ArXiv IDs (no version suffix) already in the collection."""
        metas = self.collection.get(include=["metadatas"])["metadatas"] or []
        return {base_arxiv_id(str(m.get("arxiv_id", ""))) for m in metas if m.get("arxiv_id")}

    def ingest_papers(
        self,
        papers: list[ArxivPaper],
        strategy: str = "sentence_window",
    ) -> dict:
        """Chunk, embed and store papers, skipping ones already indexed."""
        existing = self.existing_arxiv_ids()
        new_papers = [p for p in papers if base_arxiv_id(p.arxiv_id) not in existing]
        stats = {
            "papers_found": len(papers),
            "papers_added": len(new_papers),
            "papers_skipped": len(papers) - len(new_papers),
            "chunks_indexed": 0,
        }
        if not new_papers:
            return stats

        documents = papers_to_documents(new_papers)
        nodes = chunk_documents(documents, strategy=strategy)  # type: ignore[arg-type]
        storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        VectorStoreIndex(nodes, storage_context=storage_context, show_progress=False)

        # Invalidate BM25 cache so the next request rebuilds with fresh documents
        from app.retrieval.hybrid import invalidate_bm25_cache

        invalidate_bm25_cache()
        stats["chunks_indexed"] = len(nodes)
        return stats

    def ingest_arxiv(
        self,
        query: str,
        max_papers: int = 20,
        strategy: str = "sentence_window",
    ) -> dict:
        """
        Fetch papers from ArXiv, chunk them, embed, and store in ChromaDB.

        Returns:
            {"papers_found", "papers_added", "papers_skipped", "chunks_indexed"}
        """
        log.info("ingestion_start", query=query, max_papers=max_papers)

        papers = fetch_arxiv_papers(query=query, max_results=max_papers)
        if not papers:
            log.warning("no_papers_found", query=query)
            return {"papers_found": 0, "papers_added": 0, "papers_skipped": 0, "chunks_indexed": 0}

        stats = self.ingest_papers(papers, strategy=strategy)
        log.info("ingestion_complete", query=query, **stats)
        return stats

    def ingest_default_corpus(self, papers_per_query: int = 10) -> int:
        """
        Seed the vector DB with a balanced ML Research corpus
        by running all DEFAULT_QUERIES. Returns the number of chunks indexed.
        """
        total = 0
        for query in DEFAULT_QUERIES:
            total += self.ingest_arxiv(query=query, max_papers=papers_per_query)["chunks_indexed"]
        log.info("default_corpus_ingested", total_nodes=total)
        return total

    def collection_stats(self) -> dict:
        """Return basic stats about the current ChromaDB collection."""
        return {
            "collection": self.settings.chroma_collection,
            "document_count": self.collection.count(),
        }


_pipeline: IngestionPipeline | None = None
_pipeline_lock = threading.Lock()


def get_pipeline() -> IngestionPipeline:
    """Process-wide pipeline so the embedding model loads exactly once."""
    global _pipeline
    with _pipeline_lock:
        if _pipeline is None:
            _pipeline = IngestionPipeline()
        return _pipeline


# ── CLI entry-point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingest ArXiv papers into ChromaDB")
    parser.add_argument("--query", type=str, default=None, help="Custom ArXiv query")
    parser.add_argument("--max-papers", type=int, default=20)
    parser.add_argument("--seed-corpus", action="store_true",
                        help="Ingest the full default ML Research corpus")
    args = parser.parse_args()

    pipeline = get_pipeline()

    if args.seed_corpus or not args.query:
        total = pipeline.ingest_default_corpus(papers_per_query=args.max_papers)
        print(f"\n[SUCCESS] Seeded corpus - {total} nodes indexed.")
    else:
        result = pipeline.ingest_arxiv(query=args.query, max_papers=args.max_papers)
        print(
            f"\n[SUCCESS] {result['papers_added']} new papers "
            f"({result['chunks_indexed']} chunks) for query: {args.query!r}"
        )
