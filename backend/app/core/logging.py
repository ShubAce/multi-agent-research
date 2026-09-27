"""
app/core/logging.py
Configures structlog for structured JSON logging in production,
pretty console logging in development.
"""

import logging
import sys
import structlog
from app.core.config import get_settings


class _SafeStream:
    """
    Never let a log line crash the caller. Windows consoles and pipes default
    to cp1252, and LLM output is full of characters it can't encode (e.g. the
    non-breaking hyphen U+2011) — an unguarded print raised UnicodeEncodeError
    inside agent nodes and replaced good answers with the fallback.
    """

    def __init__(self, stream):
        self._stream = stream

    def write(self, text: str) -> int:
        try:
            return self._stream.write(text)
        except UnicodeEncodeError:
            encoding = getattr(self._stream, "encoding", None) or "ascii"
            return self._stream.write(text.encode(encoding, "backslashreplace").decode(encoding))
        except Exception:
            return 0

    def flush(self) -> None:
        try:
            self._stream.flush()
        except Exception:
            pass


class NamedPrintLoggerFactory:
    def __init__(self, file=None):
        self.file = _SafeStream(file or sys.stdout)

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
        wrapper_class=structlog.make_filtering_bound_logger(log_level),  # honour LOG_LEVEL
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
