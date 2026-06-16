"""
app/core/config.py
Central settings — loaded once at startup, available everywhere via `get_settings()`.
"""

from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── App ───────────────────────────────────────────────────────────────────
    environment: str = Field(default="development")
    log_level: str = Field(default="INFO")
    api_key: str = Field(default="dev-secret-key-change-in-prod")

    # ── LLM ──────────────────────────────────────────────────────────────────
    groq_api_key: str = Field(default="")
    groq_model: str = Field(default="llama-3.3-70b-versatile")
    groq_model_fast: str = Field(default="llama-3.1-8b-instant")   # cheaper, for simple tasks

    # ── Tools ────────────────────────────────────────────────────────────────
    tavily_api_key: str = Field(default="")

    # ── Services ─────────────────────────────────────────────────────────────
    redis_url: str = Field(default="redis://localhost:6379")
    chroma_host: str = Field(default="localhost")
    chroma_port: int = Field(default=8001)
    chroma_collection: str = Field(default="arxiv_papers")

    # ── Database ─────────────────────────────────────────────────────────────
    database_url: str = Field(default="sqlite+aiosqlite:///./research.db")

    # ── Embedding ────────────────────────────────────────────────────────────
    # Free HuggingFace model — no API key needed
    embedding_model: str = Field(default="BAAI/bge-small-en-v1.5")

    # ── RAG ──────────────────────────────────────────────────────────────────
    retrieval_top_k: int = Field(default=6)        # reduced: less reranker work, still good recall
    rerank_top_n: int = Field(default=3)           # after reranking
    chunk_size: int = Field(default=512)
    chunk_overlap: int = Field(default=50)

    # ── Hybrid Search ────────────────────────────────────────────────────────
    # Combines dense (vector) + sparse (BM25) retrieval via RRF fusion
    hybrid_search_enabled: bool = Field(default=True)
    bm25_top_k: int = Field(default=6)             # reduced from 10
    rrf_k: int = Field(default=60)                 # RRF constant (60 is standard)

    # ── Query Rewriting ──────────────────────────────────────────────────────
    # Rewrites the query into N variants before hitting the vector DB
    query_rewriting_enabled: bool = Field(default=True)
    query_rewrite_n: int = Field(default=3)        # number of rewritten variants

    # ── Faithfulness Filter ──────────────────────────────────────────────────
    # Post-processes the synthesiser output to remove ungrounded claims
    faithfulness_filter_enabled: bool = Field(default=True)
    faithfulness_threshold: float = Field(default=0.5)  # flag sentences below this

    # ── Agents ───────────────────────────────────────────────────────────────
    agent_max_iterations: int = Field(default=5)   # prevents infinite loops
    agent_timeout_seconds: int = Field(default=120)

    # ── LangSmith ────────────────────────────────────────────────────────────
    langchain_tracing_v2: bool = Field(default=False)
    langchain_api_key: str = Field(default="")
    langchain_project: str = Field(default="research-assistant")

    @property
    def is_production(self) -> bool:
        return self.environment == "production"


@lru_cache
def get_settings() -> Settings:
    """Cached — only one Settings object is ever created."""
    return Settings()
