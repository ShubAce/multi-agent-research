"""
app/api/v1/health.py

Two endpoints:
  GET /api/v1/health   — liveness probe (is the process alive?)
  GET /api/v1/ready    — readiness probe (are dependencies reachable?)
  GET /api/v1/metrics  — Prometheus scrape endpoint
"""

from __future__ import annotations

import chromadb
import redis
from fastapi import APIRouter
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

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


@router.get("/ready", response_model=HealthResponse, summary="Readiness probe")
async def ready() -> HealthResponse:
    """
    Checks all critical dependencies.
    Returns 200 only when Redis and ChromaDB are reachable.
    Used by Docker healthcheck and Kubernetes readiness probe.
    """
    services: dict[str, str] = {}
    overall = "ok"

    # Check Redis
    try:
        r = redis.Redis.from_url(settings.redis_url, socket_connect_timeout=2)
        r.ping()
        services["redis"] = "ok"
    except Exception as exc:
        services["redis"] = f"error: {exc}"
        overall = "degraded"

    # Check ChromaDB
    try:
        from chromadb.config import Settings as ChromaSettings
        client = chromadb.HttpClient(
            host=settings.chroma_host,
            port=settings.chroma_port,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        client.heartbeat()
        services["chromadb"] = "ok"
    except Exception as exc:
        services["chromadb"] = f"error: {exc}"
        overall = "degraded"

    return HealthResponse(status=overall, services=services)  # type: ignore[arg-type]


@router.get("/metrics", summary="Prometheus metrics")
async def metrics() -> Response:
    """Prometheus scrape endpoint. Wire this to your Grafana datasource."""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )
