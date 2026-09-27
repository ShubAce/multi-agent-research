"""
app/workers/celery_app.py
Celery application factory. Workers are started separately from the API server.
"""

import threading

import redis
from celery import Celery
from celery.signals import worker_ready, worker_shutdown

from app.core.config import get_settings
from app.core.logging import setup_logging

settings = get_settings()
setup_logging()  # the worker never imports app.main, so configure structlog here too

# Workers refresh this key while alive; /api/v1/ready reports it.
# (celery's control.ping can't be used: a solo-pool worker is deaf to it mid-task.)
WORKER_HEARTBEAT_KEY = "research:worker:heartbeat"
_HEARTBEAT_EVERY_S = 10

celery_app = Celery(
    "research_worker",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=["app.workers.tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
    task_acks_late=True,               # re-queue on unexpected worker crash
    worker_prefetch_multiplier=1,      # fair dispatch across workers
    task_soft_time_limit=settings.agent_timeout_seconds,       # raises SoftTimeLimitExceeded
    task_time_limit=settings.agent_timeout_seconds + 30,       # hard kill
    result_expires=3600,               # keep results in Redis for 1 hour
    broker_connection_retry_on_startup=True,
)

_stop_heartbeat = threading.Event()


def _heartbeat_loop() -> None:
    client = redis.Redis.from_url(settings.redis_url, socket_connect_timeout=3)
    while not _stop_heartbeat.is_set():
        try:
            client.set(WORKER_HEARTBEAT_KEY, "1", ex=_HEARTBEAT_EVERY_S * 3)
        except Exception:
            pass
        _stop_heartbeat.wait(_HEARTBEAT_EVERY_S)


def _prewarm_models() -> None:
    """Load the embedding model + reranker now so the first job doesn't wait ~20s."""
    try:
        from app.agents.rag_agent import _get_reranker
        from app.ingestion.pipeline import get_pipeline

        get_pipeline()
        _get_reranker()
    except Exception:
        pass  # the first job retries lazily and reports any real error


@worker_ready.connect
def _start_heartbeat(**_: object) -> None:
    threading.Thread(target=_heartbeat_loop, name="worker-heartbeat", daemon=True).start()
    threading.Thread(target=_prewarm_models, name="worker-prewarm", daemon=True).start()


@worker_shutdown.connect
def _stop_heartbeat_loop(**_: object) -> None:
    _stop_heartbeat.set()
