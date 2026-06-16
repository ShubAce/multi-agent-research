"""
app/core/logging.py
Configures structlog for structured JSON logging in production,
pretty console logging in development.
"""

import logging
import sys
import structlog
from app.core.config import get_settings


class NamedPrintLoggerFactory:
    def __init__(self, file=None):
        self.file = file or sys.stdout

    def __call__(self, *args, **kwargs) -> structlog.PrintLogger:
        name = args[0] if args else "root"
        logger = structlog.PrintLogger(self.file)
        logger.name = name
        return logger


def setup_logging() -> None:
    settings = get_settings()
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)

    # Shared processors for all environments
    shared_processors: list = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    if settings.is_production:
        # JSON output — easy to forward to Loki / CloudWatch
        processors = shared_processors + [
            structlog.processors.dict_tracebacks,
            structlog.processors.JSONRenderer(),
        ]
        renderer = structlog.processors.JSONRenderer()
    else:
        # Human-readable coloured output for local dev
        processors = shared_processors + [
            structlog.dev.ConsoleRenderer(colors=True),
        ]
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=NamedPrintLoggerFactory(file=sys.stdout),
        wrapper_class=structlog.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Also configure stdlib logging so uvicorn / celery logs go through structlog
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
    )
    logging.getLogger("uvicorn.access").setLevel(log_level)
    logging.getLogger("celery").setLevel(log_level)


def get_logger(name: str = __name__) -> structlog.BoundLogger:
    return structlog.get_logger(name)
