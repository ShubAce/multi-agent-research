"""
app/api/v1/research.py

Three endpoints:

  POST /api/v1/research
      Submit a research query. Returns a job_id immediately (202 Accepted).
      The agent pipeline runs asynchronously in a Celery worker.

  GET  /api/v1/stream/{job_id}
      Server-Sent Events stream. Connect here after submitting a job to
      receive live agent-progress events and the final answer.

  POST /api/v1/ingest
      Trigger ArXiv ingestion for a given query. Returns the number of
      nodes indexed. Useful for seeding the corpus from the UI or CLI.
"""

from __future__ import annotations

import json
import uuid

import redis.asyncio as aioredis
from fastapi import APIRouter, Depends, HTTPException, status
from sse_starlette.sse import EventSourceResponse

from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.metrics import active_sessions, research_requests_total
from app.core.security import verify_api_key
from app.models.request import IngestRequest, ResearchRequest
from app.models.response import IngestResponse, ResearchJobResponse

log = get_logger(__name__)
router = APIRouter(tags=["research"])
settings = get_settings()

# Async Redis client shared across requests
_redis: aioredis.Redis | None = None


def get_redis() -> aioredis.Redis:
    global _redis
    if _redis is None:
        _redis = aioredis.from_url(settings.redis_url, decode_responses=True)
    return _redis


# ── Submit research job ───────────────────────────────────────────────────────

@router.post(
    "/research",
    response_model=ResearchJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit a research query",
)
async def submit_research(
    request: ResearchRequest,
    _: str = Depends(verify_api_key),
) -> ResearchJobResponse:
    """
    Submit a research query for asynchronous processing.
    Returns a job_id — use GET /stream/{job_id} to receive live updates.
    """
    from app.workers.tasks import run_agent_pipeline

    job_id = str(uuid.uuid4())
    session_id = request.session_id or str(uuid.uuid4())

    log.info(
        "research_job_submitted",
        job_id=job_id,
        session_id=session_id,
        query=request.query[:80],
    )

    run_agent_pipeline.delay(
        job_id=job_id,
        query=request.query,
        session_id=session_id,
    )

    research_requests_total.labels(status="queued").inc()
    active_sessions.inc()

    return ResearchJobResponse(
        job_id=job_id,
        session_id=session_id,
        status="queued",
    )


# ── SSE stream ────────────────────────────────────────────────────────────────

@router.get(
    "/stream/{job_id}",
    summary="Stream live agent events for a job (SSE)",
)
async def stream_job_events(job_id: str) -> EventSourceResponse:
    """
    Server-Sent Events endpoint.

    Event types emitted:
      job_started   — job picked up by worker
      agent_done    — one agent node finished (includes agent name)
      done          — pipeline complete (includes final_answer + citations)
      error         — unrecoverable failure

    The connection closes automatically after a "done" or "error" event.
    """
    r = get_redis()

    async def event_generator():
        async with r.pubsub() as pubsub:
            await pubsub.subscribe(f"job:{job_id}:events")
            log.debug("sse_client_connected", job_id=job_id)

            async for message in pubsub.listen():
                if message["type"] != "message":
                    continue

                try:
                    data = json.loads(message["data"])
                except json.JSONDecodeError:
                    continue

                event_type = data.get("type", "update")
                yield {
                    "event": event_type,
                    "data": json.dumps(data),
                }

                # Close the stream when the job is finished
                if event_type in ("done", "error"):
                    active_sessions.dec()
                    log.debug("sse_stream_closed", job_id=job_id, event_type=event_type)
                    break

    return EventSourceResponse(event_generator())


# ── Ingest endpoint ───────────────────────────────────────────────────────────

@router.post(
    "/ingest",
    response_model=IngestResponse,
    summary="Trigger ArXiv ingestion",
)
async def ingest_arxiv(
    request: IngestRequest,
    _: str = Depends(verify_api_key),
) -> IngestResponse:
    """
    Fetch papers from ArXiv and index them into ChromaDB.
    Runs synchronously — expect 10–30 seconds for 20 papers.
    """
    from app.ingestion.pipeline import IngestionPipeline

    log.info("ingest_requested", query=request.arxiv_query, max=request.max_papers)

    try:
        pipeline = IngestionPipeline()
        nodes_indexed = pipeline.ingest_arxiv(
            query=request.arxiv_query,
            max_papers=request.max_papers,
        )
        stats = pipeline.collection_stats()

        return IngestResponse(
            status="ok",
            papers_indexed=nodes_indexed,
            collection=stats["collection"],
        )
    except Exception as exc:
        log.error("ingest_error", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion failed: {exc}",
        )
