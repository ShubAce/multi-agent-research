"""
app/agents/llm.py

Centralised LLM factory so every agent uses the same configured model.
Swap `get_llm()` to change the provider for the whole system.

Groq retired the Llama 3 models this project started with; the defaults are
now OpenAI's open-weight gpt-oss models on Groq (120B for synthesis, 20B for
planning / critique / fact-checking). Both are reasoning models: their hidden
reasoning tokens count toward max_tokens, so budgets are larger than before
and reasoning effort is kept low for latency.
"""

from __future__ import annotations

from functools import lru_cache

from langchain_groq import ChatGroq

from app.core.config import get_settings


def _reasoning_kwargs(model: str) -> dict:
    """Provider extras for reasoning models (sent as extra request-body fields)."""
    settings = get_settings()
    if "gpt-oss" in model:
        return {"model_kwargs": {"extra_body": {"reasoning_effort": settings.groq_reasoning_effort}}}
    if "qwen3" in model:
        # Keep <think> blocks out of the answer text
        return {"model_kwargs": {"extra_body": {"reasoning_format": "hidden"}}}
    return {}


@lru_cache(maxsize=1)
def get_llm() -> ChatGroq:
    """Main LLM — used by the Synthesiser and the Paper Summariser."""
    settings = get_settings()
    return ChatGroq(
        model=settings.groq_model,
        groq_api_key=settings.groq_api_key,
        temperature=0.1,           # low temp for factual research tasks
        max_tokens=4096,
        **_reasoning_kwargs(settings.groq_model),
    )


@lru_cache(maxsize=1)
def get_fast_llm() -> ChatGroq:
    """
    Cheaper / faster LLM — planner routing, query rewriting, critic,
    faithfulness checks and follow-up suggestions.
    """
    settings = get_settings()
    return ChatGroq(
        model=settings.groq_model_fast,
        groq_api_key=settings.groq_api_key,
        temperature=0.0,
        max_tokens=1536,
        **_reasoning_kwargs(settings.groq_model_fast),
    )
