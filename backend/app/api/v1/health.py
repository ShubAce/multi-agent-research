"""
app/api/v1/health.py

  GET /api/v1/health   — liveness probe (is the process alive?)
  GET /api/v1/ready    — readiness probe (Redis, ChromaDB, Celery worker, configured features)
  GET /api/v1/metrics  — Prometheus scrape endpoint
"""

from __future__ import annotations

import time

import httpx
import redis
from fastapi import APIRouter
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.concurrency import run_in_threadpool

from app.core.config import get_settings
from app.core.logging import get_logger
from app.models.response import HealthResponse

log = get_logger(__name__)
router = APIRouter(tags=["ops"])
settings = get_settings()


@router.get("/health", response_model=HealthResponse, summary="Liveness probe")
async def health() -> HealthResponse:
    """Returns 200 if the process is running. No dependency checks."""
    return HealthResponse(status="ok", services={})


_LLM_CHECK_TTL_S = 300
_llm_check: tuple[float, str] | None = None


def _check_llm() -> str:
    """
    Confirm the configured Groq models exist for this key (cached for 5 min).
    Catches retired models, which otherwise surface only as failed agent steps.
    """
    global _llm_check
    if _llm_check and time.monotonic() - _llm_check[0] < _LLM_CHECK_TTL_S:
        return _llm_check[1]

    if not settings.groq_api_key:
        result = "error: GROQ_API_KEY is not set"
    else:
        try:
            response = httpx.get(
                "https://api.groq.com/openai/v1/models",
                headers={"Authorization": f"Bearer {settings.groq_api_key}"},
                timeout=5,
            )
            if response.status_code == 401:
                result = "error: GROQ_API_KEY was rejected"
            else:
                response.raise_for_status()
                available = {m["id"] for m in response.json().get("data", [])}
                missing = [m for m in (settings.groq_model, settings.groq_model_fast) if m not in available]
                result = f"error: model not available: {', '.join(missing)}" if missing else "ok"
        except Exception as exc:
            result = f"error: could not reach Groq ({exc})"

    _llm_check = (time.monotonic(), result)
    return result


def _check_dependencies() -> tuple[dict[str, str], int | None]:
    services: dict[str, str] = {"llm": _check_llm()}
    chunks: int | None = None

    try:
        from app.workers.celery_app import WORKER_HEARTBEAT_KEY

        r = redis.Redis.from_url(settings.redis_url, socket_connect_timeout=2)
        r.ping()
        services["redis"] = "ok"
        services["worker"] = "ok" if r.exists(WORKER_HEARTBEAT_KEY) else "error: no worker running"
    except Exception as exc:
        services["redis"] = f"error: {exc}"
        services["worker"] = "error: unknown (Redis unreachable)"

    try:
        from app.ingestion.pipeline import get_corpus_collection

        chunks = get_corpus_collection().count()
        services["chromadb"] = "ok"
    except Exception as exc:
        services["chromadb"] = f"error: {exc}"

    return services, chunks


@router.get("/ready", response_model=HealthResponse, summary="Readiness probe")
async def ready() -> HealthResponse:
    """
    Checks all critical dependencies and reports which optional features are configured.
    Status is "ok" only when Redis, ChromaDB and at least one worker are up.
    """
    services, chunks = await run_in_threadpool(_check_dependencies)
    overall = "ok" if all(v == "ok" for v in services.values()) else "degraded"
    return HealthResponse(
        status=overall,  # type: ignore[arg-type]
        services=services,
        features={
            "llm": services.get("llm") == "ok",
            "web_search": bool(settings.tavily_api_key),
        },
        knowledge_chunks=chunks,
    )


@router.get("/metrics", summary="Prometheus metrics")
async def metrics() -> Response:
    """Prometheus scrape endpoint. Wire this to your Grafana datasource."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
