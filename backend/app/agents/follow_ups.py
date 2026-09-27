"""
app/agents/follow_ups.py

Suggests follow-up research questions after an answer is finished.
One fast-LLM call; failures return an empty list so they never block a result.
"""

from __future__ import annotations

import re

from langchain_core.messages import HumanMessage, SystemMessage

from app.agents.llm import get_fast_llm
from app.core.config import get_settings
from app.core.logging import get_logger
from app.retrieval.query_rewriter import _parse_variants

log = get_logger(__name__)

_PROMPT = """You suggest follow-up questions for a research assistant.
Given a question and the answer it received, propose 3 specific follow-up questions a
researcher would naturally ask next: go deeper, compare with alternatives, or probe limitations.
Each must be self-contained (no "it"/"this"), under 15 words, and end with "?".
Return ONLY a JSON array of 3 strings."""


def suggest_follow_ups(query: str, answer: str, n: int = 3) -> list[str]:
    if not get_settings().follow_ups_enabled:
        return []
    try:
        response = get_fast_llm().invoke([
            SystemMessage(content=_PROMPT),
            HumanMessage(content=f"Question: {query}\n\nAnswer:\n{answer[:1500]}"),
        ])
        suggestions = []
        for s in _parse_variants(str(response.content)):
            s = re.sub(r"^\d+[.)]\s*", "", s).strip()
            if 10 <= len(s) <= 160:
                suggestions.append(s if s.endswith("?") else f"{s}?")
        return suggestions[:n]
    except Exception as exc:
        log.warning("follow_ups_failed", error=str(exc))
        return []
