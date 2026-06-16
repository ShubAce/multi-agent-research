"""
app/workers/celery_app.py
Celery application factory. Workers are started separately from the API server.
"""

from celery import Celery
from app.core.config import get_settings

settings = get_settings()

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
    task_soft_time_limit=120,          # soft limit: raises SoftTimeLimitExceeded
    task_time_limit=150,               # hard kill after 150s
    result_expires=3600,               # keep results in Redis for 1 hour
    broker_connection_retry_on_startup=True,
)
