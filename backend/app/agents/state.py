"""
app/agents/state.py

Single source of truth for what flows between LangGraph nodes.
v3 additions:
  - conversation_history  : injected by memory layer before Planner runs
  - critic_score          : 0–10 quality score from Critic node
  - critic_feedback       : what the Critic said was weak
  - retry_count           : how many times we have re-retrieved (max 1)
v4 additions:
  - use_web_search        : user toggle — when False the planner never routes to the web
  - fact_check            : faithfulness-filter report surfaced in the UI
"""

from __future__ import annotations

import operator
from collections.abc import Sequence
from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage


class AgentState(TypedDict, total=False):
    # ── Core ──────────────────────────────────────────────────────────────────
    query: str
    session_id: str
    messages: Annotated[Sequence[BaseMessage], operator.add]
    use_web_search: bool

    # ── Session memory (injected before Planner) ──────────────────────────────
    conversation_history: str          # formatted last-N turns from Redis

    # ── Planner output ────────────────────────────────────────────────────────
    sub_tasks: list[str]
    routing: dict[str, str]

    # ── Agent results ─────────────────────────────────────────────────────────
    web_results: list[dict]
    rag_results: list[dict]
    source_contexts: list[str]
    web_fallback: bool                 # RAG found nothing, so the web was searched instead

    # ── Synthesiser output ────────────────────────────────────────────────────
    final_answer: str | None
    citations: list[dict]
    agents_used: list[str]
    confidence_score: float | None
    fact_check: dict | None

    # ── Critic output ─────────────────────────────────────────────────────────
    critic_score: int | None           # 0–10; None means critic hasn't run yet
    critic_feedback: str | None        # what was weak
    retry_count: int                   # incremented each time we re-retrieve

    # ── Control ───────────────────────────────────────────────────────────────
    error: str | None
    iteration_count: int
