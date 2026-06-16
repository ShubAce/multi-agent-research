"""
app/retrieval/query_rewriter.py

Query Rewriting: before hitting the vector DB, rewrite the user's query
into N semantically diverse variants. Each variant hits the retriever
independently; results are fused via RRF.

Why this helps:
  - Users ask vague questions ("how does attention work?")
  - The vector DB returns different (complementary) chunks for each variant
  - Fusing N result sets gives much better recall than a single query

This is called Multi-Query Retrieval in the LlamaIndex/LangChain ecosystem.
It consistently improves context_recall by 10–20% on benchmarks.

Example:
  Input:  "how does attention work"
  Output: [
    "self-attention mechanism transformer architecture explanation",
    "scaled dot-product attention query key value computation",
    "multi-head attention neural network sequence modelling"
  ]
"""

from __future__ import annotations

import json
import re

from langchain_core.messages import HumanMessage, SystemMessage

from app.agents.llm import get_fast_llm
from app.core.config import get_settings
from app.core.logging import get_logger

log = get_logger(__name__)

_REWRITE_PROMPT = """You are a search query optimisation expert for an academic paper retrieval system.

Given a research question, generate {n} different search queries that will together retrieve
the most comprehensive set of relevant academic paper chunks.

Rules:
- Each query should target a different aspect or framing of the topic
- Use precise technical terminology that would appear in academic papers
- Vary the specificity: some broad, some narrow
- Do NOT add questions or explanations — just the queries
- Return ONLY a JSON array of strings, no markdown, no extra text

Example input: "how does flash attention work"
Example output: ["Flash Attention IO-aware algorithm implementation", "memory efficient attention mechanism GPU", "FlashAttention tiling computation HBM bandwidth"]
"""


def rewrite_query(query: str, n: int | None = None) -> list[str]:
    """
    Rewrite a single query into N retrieval-optimised variants.

    Args:
        query: The original user query.
        n:     Number of variants to generate. Defaults to settings value.

    Returns:
        List of rewritten queries. Always includes the original as the first
        element so retrieval never regresses if rewriting fails.
    """
    settings = get_settings()
    n = n or settings.query_rewrite_n

    if not settings.query_rewriting_enabled:
        return [query]

    log.info("query_rewriting_start", query=query[:80], n=n)

    try:
        llm = get_fast_llm()   # 8B is enough for this task
        messages = [
            SystemMessage(content=_REWRITE_PROMPT.format(n=n)),
            HumanMessage(content=f"Research question: {query}"),
        ]

        response = llm.invoke(messages)
        content = response.content
        if not isinstance(content, str):
            raise ValueError(f"Expected string response content from LLM, got {type(content)}")
        variants = _parse_variants(content)

        # Always prepend the original query — guarantees we never regress
        all_queries = [query] + [v for v in variants if v != query]

        log.info(
            "query_rewriting_done",
            original=query[:60],
            variants=all_queries[1:],
        )
        return all_queries[:n + 1]   # original + n variants

    except Exception as exc:
        log.warning("query_rewriting_failed", error=str(exc), fallback="original query")
        return [query]   # graceful fallback — never crash the pipeline


def _parse_variants(text: str) -> list[str]:
    """Extract a list of strings from the LLM's JSON response."""
    text = text.strip()

    # Try direct JSON parse
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(v).strip() for v in parsed if v]
    except json.JSONDecodeError:
        pass

    # Try extracting JSON array from markdown block
    match = re.search(r"```(?:json)?\s*(\[[\s\S]+?\])\s*```", text)
    if match:
        try:
            parsed = json.loads(match.group(1))
            if isinstance(parsed, list):
                return [str(v).strip() for v in parsed if v]
        except json.JSONDecodeError:
            pass

    # Try finding the first [...] block
    match = re.search(r"\[[\s\S]+?\]", text)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, list):
                return [str(v).strip() for v in parsed if v]
        except json.JSONDecodeError:
            pass

    # Last resort: split by newlines and treat each as a query
    lines = [line.strip().strip('"').strip("'").strip("-").strip()
             for line in text.split("\n") if line.strip()]
    return [l for l in lines if l and len(l) > 5]
