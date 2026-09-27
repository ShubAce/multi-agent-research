"""
app/agents/critic.py

Critic Agent — evaluates the synthesised answer and triggers a retry
if quality is too low.

Scoring rubric (0–10):
  10  Perfect: complete, fully grounded, directly answers the question
  7–9 Good: answers the question, minor gaps or weak citations
  4–6 Mediocre: partially answers, noticeable gaps, some unsupported claims
  0–3 Poor: does not answer the question, mostly hallucinated

If score < 6 AND retry_count == 0:
  - The Critic rewrites the retrieval queries to be more specific
  - The graph loops back to parallel_agents with the improved queries
    (earlier evidence is kept and the new results are added to it)
  - This happens at most ONCE (retry_count guards against infinite loops)

If score >= 6 OR retry_count >= 1:
  - Accept the answer and move to END (the retried answer is still scored)

This self-reflection loop is what separates a basic RAG pipeline from an
agentic system — the key interview talking point for this feature.
"""

from __future__ import annotations

import re
import time

from langchain_core.messages import HumanMessage, SystemMessage

from app.agents.llm import get_fast_llm
from app.agents.state import AgentState
from app.core.logging import get_logger
from app.core.metrics import agent_duration_seconds, agent_runs_total

log = get_logger(__name__)

_MAX_RETRIES = 1   # retry at most once — prevents runaway loops

_CRITIC_PROMPT = """You are a quality critic for an AI research assistant.

Evaluate the answer below on a scale of 0–10 using this rubric:
  10  Perfect: complete, fully grounded, directly answers the question
  7–9 Good: answers well, minor gaps or weak citations
  4–6 Mediocre: partial answer, noticeable gaps, some unsupported claims
  0–3 Poor: does not answer the question or is mostly hallucinated

Also provide:
1. A one-sentence reason for your score
2. If score < 6: 1–2 improved retrieval queries that would fix the gaps

Return ONLY this JSON (no markdown):
{
  "score": <integer 0-10>,
  "reason": "<one specific sentence about THIS answer's main strength or gap>",
  "improved_queries": ["<query>", ...]
}

If score >= 6, improved_queries should be an empty list [].
"""


def _parse_critic_response(text: str) -> dict:
    """Extract JSON from critic response, with fallback."""
    import json
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{[\s\S]+\}", text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    # Fallback — accept only an explicit score ("score": 7 / 7/10 / 7 out of 10);
    # never guess one from an arbitrary digit, since the score is shown to users
    score_match = re.search(r'"score"\s*:\s*(\d{1,2})|\b(\d{1,2})\s*(?:/|out of)\s*10\b', text)
    score = int(score_match.group(1) or score_match.group(2)) if score_match else None
    return {"score": score, "reason": "", "improved_queries": []}


def critic_node(state: AgentState) -> dict:
    """
    LangGraph node: Critic.

    Scores the current final_answer and decides:
      - score >= 6  → accept answer (graph proceeds to END)
      - score < 6 and retries left → rewrite queries, loop back to parallel_agents
      - score < 6 and no retries left → accept (prevents infinite loops)

    The score is always a real evaluation — it is shown to the user, so it is
    None (not a made-up default) when the critic cannot run.

    Input state keys used:  query, final_answer, citations, retry_count
    Output state keys set:  critic_score, critic_feedback, sub_tasks, routing (on retry)
    """
    t0 = time.perf_counter()

    answer = state.get("final_answer", "")
    query = state.get("query", "")
    retry_count = state.get("retry_count", 0)
    agents_used = state.get("agents_used", []) + ["critic"]

    if not answer:
        log.warning("critic_no_answer")
        return {"critic_score": 0, "critic_feedback": "No answer was generated.", "agents_used": agents_used}

    log.info("critic_start", query=query[:60], answer_length=len(answer))

    citations = state.get("citations", [])
    n_papers = sum(1 for c in citations if c.get("type") == "paper")
    source_summary = (
        f"\n\nSources available to the writer: {len(citations)} "
        f"({n_papers} papers, {len(citations) - n_papers} web pages)"
    )

    messages = [
        SystemMessage(content=_CRITIC_PROMPT),
        HumanMessage(content=(
            f"Original question: {query}\n\n"
            f"Answer to evaluate:\n{answer[:3000]}"
            f"{source_summary}"
        )),
    ]

    try:
        response = get_fast_llm().invoke(messages, response_format={"type": "json_object"})
        result = _parse_critic_response(str(response.content))
        if result.get("score") is None:
            raise ValueError(f"no score in critic response: {str(response.content)[:120]!r}")

        score = max(0, min(10, int(result["score"])))
        reason: str = result.get("reason", "")
        improved_queries: list[str] = [
            str(q).strip() for q in result.get("improved_queries", []) if str(q).strip()
        ][:2]
        will_retry = score < 6 and bool(improved_queries) and retry_count < _MAX_RETRIES

        elapsed = (time.perf_counter() - t0) * 1000
        log.info(
            "critic_done",
            score=score,
            reason=reason,
            will_retry=will_retry,
            duration_ms=round(elapsed),
        )
        agent_runs_total.labels(agent_name="critic", status="success").inc()
        agent_duration_seconds.labels(agent_name="critic").observe(elapsed / 1000)

        update: dict = {
            "critic_score":    score,
            "critic_feedback": reason,
            "agents_used":     agents_used,
        }

        if will_retry:
            log.info("critic_triggering_retry", improved_queries=improved_queries)
            # Knowledge base came up empty last time → look on the web instead
            from app.agents.web_search import web_search_available

            use_web = (
                not state.get("rag_results")
                and state.get("use_web_search", True)
                and web_search_available()
            )
            agent = "web_search" if use_web else "rag"
            update["sub_tasks"]    = improved_queries
            update["routing"]      = {q: agent for q in improved_queries}
            update["retry_count"]  = retry_count + 1
            # Earlier evidence is kept; parallel_agents appends the new results.
            # Clearing the answer is what signals should_retry to loop.
            update["final_answer"] = None

        return update

    except Exception as exc:
        log.error("critic_error", error=str(exc))
        agent_runs_total.labels(agent_name="critic", status="error").inc()
        return {
            "critic_score":    None,
            "critic_feedback": "Quality review unavailable for this answer.",
            "agents_used":     agents_used,
        }


def should_retry(state: AgentState) -> str:
    """
    Conditional edge after Critic node.

    Returns:
      "retry"  → loop back to parallel_agents with improved queries
      "end"    → accept answer and finish
    """
    score = state.get("critic_score")
    retry_count = state.get("retry_count", 0)
    has_new_tasks = bool(state.get("sub_tasks")) and state.get("final_answer") is None

    if score is not None and score < 6 and retry_count <= _MAX_RETRIES and has_new_tasks:
        log.info("critic_routing_retry", score=score, retry_count=retry_count)
        return "retry"

    log.info("critic_routing_end", score=score)
    return "end"
