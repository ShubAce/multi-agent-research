"""
app/main.py

FastAPI application factory.
Run with:  uvicorn app.main:app --reload --port 8000
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from app.api.v1 import research, health, papers
from app.core.config import get_settings
from app.core.logging import get_logger, setup_logging

# Configure logging before anything else
setup_logging()
log = get_logger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup / shutdown logic.
    - Pre-warm the embedding model so the first request isn't slow.
    - Log environment info for debugging.
    """
    log.info(
        "app_starting",
        environment=settings.environment,
        groq_model=settings.groq_model,
        chroma_collection=settings.chroma_collection,
    )

    # Pre-warm embedding model (downloads on first run, cached afterward)
    try:
        from app.ingestion.pipeline import IngestionPipeline
        _ = IngestionPipeline()   # triggers model load
        log.info("embedding_model_ready")
    except Exception as exc:
        log.warning("embedding_model_prewarm_failed", error=str(exc))

    yield  # app is running

    log.info("app_shutting_down")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Research Assistant API",
        description=(
            "Multi-agent RAG system: submit a research query, stream live agent "
            "events, receive a grounded answer with citations."
        ),
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # ── Middleware ────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:3000",          # Next.js dev server
            "https://your-domain.com",        # replace with your prod domain
        ],
        allow_credentials=True,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )
    app.add_middleware(GZipMiddleware, minimum_size=1000)

    # ── Routers ───────────────────────────────────────────────────────────────
    app.include_router(research.router, prefix="/api/v1")
    app.include_router(health.router,   prefix="/api/v1")
    app.include_router(papers.router,   prefix="/api/v1")

    return app


app = create_app()
