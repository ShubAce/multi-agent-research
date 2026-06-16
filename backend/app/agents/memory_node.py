"""
app/agents/memory_node.py

Memory Injection Node — runs BEFORE the Planner.

Reads the last N conversation turns from Redis and adds them to the state
as a formatted string. The Planner's prompt then includes this history so
it can decompose follow-up questions correctly.

Example:
  Previous turns:
    User: What is Flash Attention?
    Assistant: Flash Attention is an IO-aware attention algorithm...

  New query: "Who invented it?"
  Planner sees history → understands "it" = Flash Attention
  → routes to RAG with sub-task "Who invented Flash Attention?"

Without memory: "Who invented it?" → meaningless retrieval → bad answer
With memory:    "Who invented it?" → correct retrieval → good answer
"""

from __future__ import annotations

import asyncio

from app.agents.state import AgentState
from app.core.logging import get_logger

log = get_logger(__name__)


def memory_injection_node(state: AgentState) -> dict:
    """
    LangGraph node: Memory Injection.

    Fetches conversation history from Redis and adds it to the state.
    Runs synchronously (Celery context) by running the async call in an
    event loop.

    Input state keys used:  session_id
    Output state keys set:  conversation_history
    """
    session_id = state.get("session_id", "")
    if not session_id:
        return {"conversation_history": ""}

    try:
        history_str = asyncio.run(_fetch_history(session_id))
        log.info(
            "memory_injected",
            session_id=session_id,
            has_history=bool(history_str),
        )
        return {"conversation_history": history_str}
    except Exception as exc:
        log.warning("memory_injection_failed", error=str(exc))
        return {"conversation_history": ""}


async def _fetch_history(session_id: str) -> str:
    from app.memory.short_term import ConversationMemory
    memory = ConversationMemory()
    return await memory.format_for_prompt(session_id, last_n=6)


async def save_turn_to_memory(
    session_id: str,
    user_query: str,
    assistant_answer: str,
) -> None:
    """
    Called after a successful pipeline run to persist the turn.
    Saves both user and assistant messages to Redis.
    """
    from app.memory.short_term import ConversationMemory
    memory = ConversationMemory()
    await memory.add_turn(session_id, role="user", content=user_query)
    await memory.add_turn(session_id, role="assistant", content=assistant_answer[:500])
    log.info("turn_saved_to_memory", session_id=session_id)
