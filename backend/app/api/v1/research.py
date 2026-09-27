"""
app/api/v1/research.py

  POST   /api/v1/research                 Submit a query → job_id (202). Runs in a Celery worker.
  GET    /api/v1/stream/{job_id}          SSE stream of the job's events (replayable, resumable).
  POST   /api/v1/research/{job_id}/cancel Stop a running job at the next agent boundary.
  DELETE /api/v1/sessions/{session_id}    Forget a conversation's short-term memory.
  POST   /api/v1/ingest                   Fetch ArXiv papers into the knowledge base.
"""

from __future__ import annotations

import json
import time
import uuid

import redis.asyncio as aioredis
from fastapi import APIRouter, Depends, HTTPException, Request, status
from sse_starlette.sse import EventSourceResponse
from starlette.concurrency import run_in_threadpool

from app.core.config import get_settings
from app.core.events import TERMINAL_EVENTS, append_event_async, cancel_key, events_key
from app.core.logging import get_logger
from app.core.metrics import active_sessions, research_requests_total
from app.core.security import verify_api_key
from app.models.request import IngestRequest, ResearchRequest
from app.models.response import IngestResponse, ResearchJobResponse

log = get_logger(__name__)
router = APIRouter(tags=["research"])
settings = get_settings()

# Give up on a stream that has produced nothing for this long
_STREAM_IDLE_TIMEOUT_S = 600

# Async Redis client shared across requests
_redis: aioredis.Redis | None = None


def get_redis() -> aioredis.Redis:
    global _redis
    if _redis is None:
        _redis = aioredis.from_url(
            settings.redis_url, decode_responses=True, socket_connect_timeout=3
        )
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

    log.info("research_job_submitted", job_id=job_id, session_id=session_id, query=request.query[:80])

    # Seed the event log first so the stream exists before the worker starts
    try:
        await append_event_async(
            get_redis(), job_id, "queued",
            {"job_id": job_id, "query": request.query},
            settings.job_events_ttl_seconds,
        )
    except Exception as exc:
        log.warning("queued_event_failed", error=str(exc))

    try:
        run_agent_pipeline.delay(
            job_id=job_id,
            query=request.query,
            session_id=session_id,
            use_web_search=request.use_web_search,
        )
    except Exception as exc:
        log.error("enqueue_failed", error=str(exc))
        research_requests_total.labels(status="error").inc()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Job queue unavailable — is Redis running?",
        ) from exc

    research_requests_total.labels(status="queued").inc()
    return ResearchJobResponse(job_id=job_id, session_id=session_id, status="queued")


# ── SSE stream ────────────────────────────────────────────────────────────────

@router.get("/stream/{job_id}", summary="Stream live agent events for a job (SSE)")
async def stream_job_events(job_id: str, request: Request) -> EventSourceResponse:
    """
    Server-Sent Events endpoint. Replays the job's event log from the start,
    then follows it live. Each event's `id` is its stream position, so a
    reconnecting EventSource resumes via Last-Event-ID without duplicates.

    Event types: queued · job_started · agent_done · done · failed · cancelled
    The connection closes after a terminal event (done / failed / cancelled).
    """
    r = get_redis()
    key = events_key(job_id)
    start_id = request.headers.get("last-event-id") or "0"

    async def event_generator():
        last_id = start_id
        if not await r.exists(key):
            yield {
                "event": "failed",
                "data": json.dumps({"type": "failed", "message": "Unknown or expired job."}),
            }
            return

        active_sessions.inc()
        idle_since = time.monotonic()
        try:
            while True:
                entries = await r.xread({key: last_id}, block=5000, count=100)
                if not entries:
                    if time.monotonic() - idle_since > _STREAM_IDLE_TIMEOUT_S:
                        yield {
                            "event": "failed",
                            "data": json.dumps({
                                "type": "failed",
                                "message": "No progress for 10 minutes — is the Celery worker running?",
                            }),
                        }
                        return
                    continue

                idle_since = time.monotonic()
                for _stream, items in entries:
                    for entry_id, fields in items:
                        last_id = entry_id
                        raw = fields.get("data", "{}")
                        try:
                            event_type = json.loads(raw).get("type", "update")
                        except json.JSONDecodeError:
                            continue
                        yield {"id": entry_id, "event": event_type, "data": raw}
                        if event_type in TERMINAL_EVENTS:
                            log.debug("sse_stream_closed", job_id=job_id, event_type=event_type)
                            return
        finally:
            active_sessions.dec()

    return EventSourceResponse(
        event_generator(),
        ping=15,
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Cancel ────────────────────────────────────────────────────────────────────

@router.post(
    "/research/{job_id}/cancel",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Cancel a running research job",
)
async def cancel_research(job_id: str, _: str = Depends(verify_api_key)) -> dict:
    """
    Flags the job as cancelled and closes its stream immediately. The worker
    stops at the next agent boundary (or skips the job if it has not started).
    """
    r = get_redis()
    ttl = settings.job_events_ttl_seconds
    await r.set(cancel_key(job_id), "1", ex=ttl)
    await append_event_async(r, job_id, "cancelled", {"message": "Research cancelled."}, ttl)
    log.info("research_job_cancelled", job_id=job_id)
    return {"job_id": job_id, "status": "cancelled"}


# ── Session memory ────────────────────────────────────────────────────────────

@router.delete("/sessions/{session_id}", summary="Forget a conversation's short-term memory")
async def delete_session(session_id: str, _: str = Depends(verify_api_key)) -> dict:
    from app.memory.short_term import ConversationMemory

    deleted = await get_redis().delete(ConversationMemory._key(session_id))
    log.info("session_memory_deleted", session_id=session_id)
    return {"session_id": session_id, "deleted": bool(deleted)}


# ── Ingest endpoint ───────────────────────────────────────────────────────────

@router.post("/ingest", response_model=IngestResponse, summary="Trigger ArXiv ingestion")
async def ingest_arxiv(
    request: IngestRequest,
    _: str = Depends(verify_api_key),
) -> IngestResponse:
    """
    Fetch papers from ArXiv and index them into ChromaDB. Papers already in the
    knowledge base are skipped. Expect 10–60 seconds depending on max_papers.
    """
    from app.ingestion.pipeline import get_pipeline

    log.info("ingest_requested", query=request.arxiv_query, max=request.max_papers)

    def _run() -> tuple[dict, dict]:
        pipeline = get_pipeline()
        result = pipeline.ingest_arxiv(query=request.arxiv_query, max_papers=request.max_papers)
        return result, pipeline.collection_stats()

    try:
        result, stats = await run_in_threadpool(_run)
    except Exception as exc:
        log.error("ingest_error", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion failed: {exc}",
        ) from exc

    return IngestResponse(
        status="ok",
        papers_indexed=result["papers_added"],
        papers_found=result["papers_found"],
        papers_skipped=result["papers_skipped"],
        chunks_indexed=result["chunks_indexed"],
        total_chunks=stats["document_count"],
        collection=stats["collection"],
    )
