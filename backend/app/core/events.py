"""
app/core/events.py

Job event log backed by a Redis Stream (one stream per job).

Why a stream and not pub/sub:
  Pub/sub drops every message published before the SSE client subscribes, so
  early events (or the whole run, if the client connects late) were lost and
  the UI could hang forever waiting for "done". A stream is an append-only log:
  the SSE endpoint replays it from the start (or from Last-Event-ID after a
  reconnect) and then blocks for new entries.

Event types:
  queued       API accepted the job
  job_started  a worker picked it up
  agent_done   one graph node finished (agent, duration_ms, next_agent, detail)
  done         final answer + citations + quality metrics
  failed       unrecoverable error (message)
  cancelled    user cancelled the job
"""

from __future__ import annotations

import json
import time
from typing import Any

TERMINAL_EVENTS = frozenset({"done", "failed", "cancelled"})


def events_key(job_id: str) -> str:
    return f"job:{job_id}:events"


def cancel_key(job_id: str) -> str:
    return f"job:{job_id}:cancel"


class _JSONEncoder(json.JSONEncoder):
    """Handles numpy scalars that leak out of rerankers / BM25."""

    def default(self, obj: Any) -> Any:
        try:
            import numpy as np

            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
        except ImportError:
            pass
        return super().default(obj)


def encode_event(event_type: str, data: dict) -> str:
    return json.dumps({"type": event_type, "timestamp": time.time(), **data}, cls=_JSONEncoder)


def append_event(redis_client, job_id: str, event_type: str, data: dict, ttl: int) -> None:
    """Append an event to the job's stream (sync Redis client)."""
    key = events_key(job_id)
    redis_client.xadd(key, {"data": encode_event(event_type, data)}, maxlen=500)
    redis_client.expire(key, ttl)


async def append_event_async(redis_client, job_id: str, event_type: str, data: dict, ttl: int) -> None:
    """Append an event to the job's stream (async Redis client)."""
    key = events_key(job_id)
    await redis_client.xadd(key, {"data": encode_event(event_type, data)}, maxlen=500)
    await redis_client.expire(key, ttl)
