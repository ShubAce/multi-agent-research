"""
app/ingestion/pipeline.py

Orchestrates the full ingestion flow:
  ArXiv API → Documents → Chunked Nodes → Embeddings → ChromaDB

Run directly:
    poetry run python -m app.ingestion.pipeline

Or via Makefile:
    make ingest
"""

from __future__ import annotations

import chromadb
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.settings import Settings as LlamaSettings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from app.core.config import get_settings
from app.core.logging import get_logger
from app.ingestion.arxiv_loader import fetch_arxiv_papers, papers_to_documents
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


class IngestionPipeline:
    """
    Handles document ingestion into ChromaDB.
    One instance is typically created at startup and reused.
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self._setup_embedding_model()
        self._setup_vector_store()

    def _setup_embedding_model(self) -> None:
        """
        Use a free HuggingFace embedding model — no API key required.
        BAAI/bge-small-en-v1.5 is fast and punches well above its weight.
        """
        log.info("loading_embedding_model", model=self.settings.embedding_model)
        embed_model = HuggingFaceEmbedding(
            model_name=self.settings.embedding_model,
            max_length=512,
        )
        # Set globally so all LlamaIndex components use it
        LlamaSettings.embed_model = embed_model
        
        from llama_index.llms.groq import Groq
        LlamaSettings.llm = Groq(
            model=self.settings.groq_model,
            api_key=self.settings.groq_api_key,
            temperature=0.1,
        )

    def _setup_vector_store(self) -> None:
        """Connect to ChromaDB and get or create our collection."""
        log.info(
            "connecting_chromadb",
            host=self.settings.chroma_host,
            port=self.settings.chroma_port,
        )
        from chromadb.config import Settings as ChromaSettings
        self.chroma_client = chromadb.HttpClient(
            host=self.settings.chroma_host,
            port=self.settings.chroma_port,
            settings=ChromaSettings(anonymized_telemetry=False)
        )
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.settings.chroma_collection,
            metadata={"hnsw:space": "cosine"},
        )
        self.vector_store = ChromaVectorStore(chroma_collection=self.collection)
        log.info(
            "chromadb_ready",
            collection=self.settings.chroma_collection,
            existing_docs=self.collection.count(),
        )

    def get_index(self) -> VectorStoreIndex:
        """
        Return a VectorStoreIndex backed by the existing ChromaDB collection.
        Used by the RAG agent to query without re-ingesting.
        """
        storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store
        )
        return VectorStoreIndex.from_vector_store(
            self.vector_store,
            storage_context=storage_context,
        )

    def ingest_arxiv(
        self,
        query: str,
        max_papers: int = 20,
        strategy: str = "sentence_window",
    ) -> int:
        """
        Fetch papers from ArXiv, chunk them, embed, and store in ChromaDB.

        Returns:
            Number of nodes (chunks) indexed.
        """
        log.info("ingestion_start", query=query, max_papers=max_papers)

        # 1. Fetch papers
        papers = fetch_arxiv_papers(query=query, max_results=max_papers)
        if not papers:
            log.warning("no_papers_found", query=query)
            return 0

        # 2. Convert to LlamaIndex Documents
        documents = papers_to_documents(papers)

        # 3. Chunk
        nodes = chunk_documents(documents, strategy=strategy)  # type: ignore[arg-type]

        # 4. Embed + store in ChromaDB
        storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store
        )
        VectorStoreIndex(nodes, storage_context=storage_context, show_progress=True)

        # Invalidate BM25 cache so next request rebuilds with fresh documents
        from app.retrieval.hybrid import invalidate_bm25_cache
        invalidate_bm25_cache()

        log.info(
            "ingestion_complete",
            query=query,
            papers=len(papers),
            nodes=len(nodes),
        )
        return len(nodes)

    def ingest_default_corpus(self, papers_per_query: int = 10) -> int:
        """
        Seed the vector DB with a balanced ML Research corpus
        by running all DEFAULT_QUERIES.
        """
        total = 0
        for query in DEFAULT_QUERIES:
            total += self.ingest_arxiv(query=query, max_papers=papers_per_query)
        log.info("default_corpus_ingested", total_nodes=total)
        return total

    def collection_stats(self) -> dict:
        """Return basic stats about the current ChromaDB collection."""
        return {
            "collection": self.settings.chroma_collection,
            "document_count": self.collection.count(),
        }


# ── CLI entry-point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingest ArXiv papers into ChromaDB")
    parser.add_argument("--query", type=str, default=None, help="Custom ArXiv query")
    parser.add_argument("--max-papers", type=int, default=20)
    parser.add_argument("--seed-corpus", action="store_true",
                        help="Ingest the full default ML Research corpus")
    args = parser.parse_args()

    pipeline = IngestionPipeline()

    if args.seed_corpus:
        total = pipeline.ingest_default_corpus(papers_per_query=args.max_papers)
        print(f"\n[SUCCESS] Seeded corpus - {total} nodes indexed.")
    elif args.query:
        total = pipeline.ingest_arxiv(query=args.query, max_papers=args.max_papers)
        print(f"\n[SUCCESS] Indexed {total} nodes for query: {args.query!r}")
    else:
        print("Pass --query <text> or --seed-corpus")
