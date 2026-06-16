"""
app/memory/short_term.py

Redis-backed conversation buffer.
Stores the last N turns of a session with a TTL so stale sessions
are automatically cleaned up without manual housekeeping.
"""

from __future__ import annotations

import json
import time

import redis.asyncio as aioredis

from app.core.config import get_settings
from app.core.logging import get_logger

log = get_logger(__name__)


class ConversationMemory:
    """
    Async Redis list — each element is a JSON-serialised conversation turn.

    Key schema:  session:{session_id}:history
    TTL:         1 hour by default (reset on every write)
    """

    def __init__(
        self,
        redis_url: str | None = None,
        ttl_seconds: int = 3600,
    ) -> None:
        settings = get_settings()
        self._redis = aioredis.from_url(
            redis_url or settings.redis_url,
            decode_responses=True,
        )
        self._ttl = ttl_seconds

    # ── Write ─────────────────────────────────────────────────────────────────

    async def add_turn(
        self,
        session_id: str,
        role: str,                  # "user" | "assistant"
        content: str,
        metadata: dict | None = None,
    ) -> None:
        key = self._key(session_id)
        turn = json.dumps({
            "role": role,
            "content": content,
            "ts": time.time(),
            **(metadata or {}),
        })
        await self._redis.rpush(key, turn)
        await self._redis.expire(key, self._ttl)
        log.debug("conversation_turn_saved", session_id=session_id, role=role)

    # ── Read ──────────────────────────────────────────────────────────────────

    async def get_history(
        self,
        session_id: str,
        last_n: int = 8,
    ) -> list[dict]:
        """Return the last `last_n` turns for a session, oldest first."""
        key = self._key(session_id)
        raw = await self._redis.lrange(key, -last_n, -1)
        return [json.loads(r) for r in raw]

    async def format_for_prompt(
        self,
        session_id: str,
        last_n: int = 6,
    ) -> str:
        """Return history formatted as a prompt-ready string."""
        history = await self.get_history(session_id, last_n)
        if not history:
            return ""
        lines = [f"{t['role'].capitalize()}: {t['content']}" for t in history]
        return "Conversation history:\n" + "\n".join(lines)

    # ── Delete ────────────────────────────────────────────────────────────────

    async def clear_session(self, session_id: str) -> None:
        await self._redis.delete(self._key(session_id))
        log.info("session_cleared", session_id=session_id)

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _key(session_id: str) -> str:
        return f"session:{session_id}:history"
