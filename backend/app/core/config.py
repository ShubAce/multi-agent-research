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
    # Browser origins allowed to call the API (JSON list in .env)
    cors_origins: list[str] = Field(
        default=["http://localhost:3000", "http://127.0.0.1:3000"]
    )

    # ── LLM ──────────────────────────────────────────────────────────────────
    groq_api_key: str = Field(default="")
    # Groq retired llama-3.3-70b-versatile / llama-3.1-8b-instant — they now 404
    groq_model: str = Field(default="openai/gpt-oss-120b")
    groq_model_fast: str = Field(default="openai/gpt-oss-20b")     # cheaper, for simple tasks
    groq_reasoning_effort: str = Field(default="low")              # gpt-oss: low | medium | high

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
    rerank_top_n: int = Field(default=5)           # passages kept per sub-task after reranking
    rerank_model: str = Field(default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    rerank_min_score: float = Field(default=0.05)  # drop passages the reranker finds irrelevant (0–1)
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
    agent_timeout_seconds: int = Field(default=240)
    max_sources: int = Field(default=10)           # numbered sources shown to the synthesiser
    follow_ups_enabled: bool = Field(default=True) # suggest follow-up questions after each answer

    # ── Job events ───────────────────────────────────────────────────────────
    job_events_ttl_seconds: int = Field(default=3600)  # how long a job's event log is replayable

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
