"""
app/agents/synthesiser.py

The Synthesiser is the final node in the LangGraph.
It receives all sub-results (RAG answers + web results) and produces:
  - A single, coherent, well-cited answer
  - A deduplicated citation list
  - A brief reasoning trace (which agents contributed what)

Uses the main 70B model — this is the step that needs the most reasoning.
"""

from __future__ import annotations
import time
import json

from langchain_core.messages import HumanMessage, SystemMessage

from app.agents.llm import get_llm
from app.agents.state import AgentState
from app.core.logging import get_logger
from app.core.metrics import agent_runs_total, agent_duration_seconds

log = get_logger(__name__)

_SYSTEM_PROMPT = """You are a research synthesiser. You receive outputs from multiple
specialised agents (web search results and RAG retrieval from academic papers) and
produce a single, accurate, well-structured answer.

Rules:
1. Prioritise information from academic papers (RAG results) for technical depth.
2. Use web search results for recency and current context.
3. Every factual claim must reference a source. Use [1], [2], ... notation.
4. Be concise but complete — aim for 3–5 paragraphs.
5. If sources conflict, note the disagreement and explain which you trust more.
6. Do NOT invent facts. If you cannot answer from the provided sources, say so.
7. End with a "Key Takeaways" bullet list (3–5 points).
"""


def _format_rag_context(rag_results: list[dict]) -> str:
    if not rag_results:
        return "No RAG results available."
    lines = []
    for i, r in enumerate(rag_results, 1):
        lines.append(f"[RAG-{i}] Task: {r['task']}\nAnswer: {r['answer']}\n")
        for j, c in enumerate(r.get("citations", [])[:2], 1):
            lines.append(f"  Source {j}: {c['title']} ({c.get('published', '')})")
    return "\n".join(lines)


def _format_web_context(web_results: list[dict]) -> str:
    if not web_results:
        return "No web search results available."
    lines = []
    for i, r in enumerate(web_results[:6], 1):
        lines.append(f"[WEB-{i}] {r['title']}\n{r['content'][:400]}\nURL: {r['url']}\n")
    return "\n".join(lines)


def _deduplicate_citations(citations: list[dict]) -> list[dict]:
    """Remove duplicate citations by arxiv_id or URL."""
    seen: set[str] = set()
    deduped: list[dict] = []
    for c in citations:
        key = c.get("arxiv_id") or c.get("url") or c.get("title", "")
        if key and key not in seen:
            seen.add(key)
            deduped.append(c)
    return deduped


def synthesiser_node(state: AgentState) -> dict:
    """
    LangGraph node: Synthesiser (final node).

    Input state keys used:  query, rag_results, web_results, citations
    Output state keys set:  final_answer, citations, agents_used
    """
    t0 = time.perf_counter()
    log.info("synthesiser_start", query=state["query"][:80])

    llm = get_llm()

    rag_context = _format_rag_context(state.get("rag_results", []))
    web_context = _format_web_context(state.get("web_results", []))

    user_message = f"""Original query: {state['query']}

--- ACADEMIC PAPER RESULTS (RAG) ---
{rag_context}

--- WEB SEARCH RESULTS ---
{web_context}

Please synthesise a comprehensive answer to the original query using the above sources."""

    try:
        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=user_message),
        ]

        response = llm.invoke(messages)
        content = response.content
        if not isinstance(content, str):
            raise ValueError(f"Expected string response content from LLM, got {type(content)}")
        raw_answer: str = content

        # ── Faithfulness Filter ───────────────────────────────────────────────
        # Check each sentence against the retrieved source contexts.
        # Removes hallucinated claims and computes a confidence score.
        from app.retrieval.faithfulness_filter import apply_faithfulness_filter

        source_contexts = state.get("source_contexts", [])
        filter_result = apply_faithfulness_filter(
            answer=raw_answer,
            source_contexts=source_contexts,
        )

        final_answer = filter_result.filtered_answer
        confidence_score = filter_result.confidence_score

        if filter_result.removed_sentences > 0:
            log.info(
                "faithfulness_filter_applied",
                removed=filter_result.removed_sentences,
                flagged=filter_result.flagged_sentences,
                confidence=confidence_score,
            )

        # Build clean citation list
        all_citations = _deduplicate_citations(state.get("citations", []))

        # Append web citations
        for r in state.get("web_results", []):
            if r.get("url"):
                all_citations.append({
                    "title": r.get("title", "Web Result"),
                    "authors": [],
                    "url": r["url"],
                    "arxiv_id": "",
                    "published": "",
                    "relevance_score": r.get("score", 0.5),
                    "excerpt": r.get("content", "")[:200],
                })

        elapsed = (time.perf_counter() - t0) * 1000
        log.info(
            "synthesiser_done",
            answer_length=len(final_answer),
            citations=len(all_citations),
            confidence=confidence_score,
            duration_ms=round(elapsed),
        )
        agent_runs_total.labels(agent_name="synthesiser", status="success").inc()
        agent_duration_seconds.labels(agent_name="synthesiser").observe(elapsed / 1000)

        return {
            "final_answer":    final_answer,
            "citations":       all_citations[:10],
            "confidence_score": confidence_score,
            "agents_used":     state.get("agents_used", []) + ["synthesiser"],
            "messages":        [*state.get("messages", []), *messages, response],
        }

    except Exception as exc:
        log.error("synthesiser_error", error=str(exc))
        agent_runs_total.labels(agent_name="synthesiser", status="error").inc()

        # Graceful degradation — concatenate raw RAG answers
        fallback = "\n\n".join(
            r.get("answer", "") for r in state.get("rag_results", [])
        ) or "Unable to generate an answer due to an internal error."

        return {
            "final_answer": fallback,
            "citations": state.get("citations", []),
            "agents_used": state.get("agents_used", []) + ["synthesiser"],
            "error": f"Synthesiser failed: {exc}",
        }
