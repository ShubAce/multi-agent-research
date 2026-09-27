"""
app/main.py

FastAPI application factory.
Run with:  uvicorn app.main:app --reload --port 8000
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from starlette.concurrency import run_in_threadpool
from starlette.types import ASGIApp, Receive, Scope, Send

from app.api.v1 import health, knowledge, papers, research
from app.core.config import get_settings
from app.core.logging import get_logger, setup_logging

# Configure logging before anything else
setup_logging()
log = get_logger(__name__)
settings = get_settings()

_SSE_PREFIX = "/api/v1/stream/"


class GZipExceptSSE:
    """
    GZip everything except the SSE stream. Starlette 0.37's GZipMiddleware
    compresses text/event-stream, and the compressor buffers small writes —
    so agent progress reached the browser in one burst at the end instead of live.
    """

    def __init__(self, app: ASGIApp, minimum_size: int = 1000) -> None:
        self.app = app
        self.gzip = GZipMiddleware(app, minimum_size=minimum_size)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and scope["path"].startswith(_SSE_PREFIX):
            await self.app(scope, receive, send)
        else:
            await self.gzip(scope, receive, send)


async def _prewarm() -> None:
    """Load the embedding model in the background so the API is responsive immediately."""
    try:
        from app.ingestion.pipeline import get_pipeline

        await run_in_threadpool(get_pipeline)
        log.info("embedding_model_ready")
    except Exception as exc:
        log.warning("embedding_model_prewarm_failed", error=str(exc))


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info(
        "app_starting",
        environment=settings.environment,
        groq_model=settings.groq_model,
        chroma_collection=settings.chroma_collection,
    )
    prewarm = asyncio.create_task(_prewarm())

    yield  # app is running

    prewarm.cancel()
    log.info("app_shutting_down")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Research Assistant API",
        description=(
            "Multi-agent RAG system: submit a research query, stream live agent "
            "events, receive a grounded answer with citations."
        ),
        version="0.2.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # ── Middleware ────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )
    app.add_middleware(GZipExceptSSE, minimum_size=1000)

    # ── Routers ───────────────────────────────────────────────────────────────
    app.include_router(research.router,  prefix="/api/v1")
    app.include_router(health.router,    prefix="/api/v1")
    app.include_router(papers.router,    prefix="/api/v1")
    app.include_router(knowledge.router, prefix="/api/v1")

    return app


app = create_app()
