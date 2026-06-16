"""
app/agents/llm.py

Centralised LLM factory so every agent uses the same configured model.
Swap `get_llm()` to change the provider for the whole system.

We use Groq with LLaMA 3 70B for main reasoning (free tier, fast)
and LLaMA 3 8B for cheaper classification tasks (planner routing).
"""

from __future__ import annotations
from functools import lru_cache

from langchain_groq import ChatGroq
from app.core.config import get_settings


@lru_cache(maxsize=1)
def get_llm() -> ChatGroq:
    """Main LLM — LLaMA 3 70B via Groq."""
    settings = get_settings()
    return ChatGroq(
        model=settings.groq_model,
        groq_api_key=settings.groq_api_key,
        temperature=0.1,           # low temp for factual research tasks
        max_tokens=2048,
    )


@lru_cache(maxsize=1)
def get_fast_llm() -> ChatGroq:
    """
    Cheaper / faster LLM — LLaMA 3 8B via Groq.
    Used for the Planner's routing classification — no need for 70B there.
    """
    settings = get_settings()
    return ChatGroq(
        model=settings.groq_model_fast,
        groq_api_key=settings.groq_api_key,
        temperature=0.0,
        max_tokens=512,
    )
