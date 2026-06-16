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
  - This happens at most ONCE (retry_count guards against infinite loops)

If score >= 6 OR retry_count >= 1:
  - Accept the answer and move to END

This self-reflection loop is what separates a basic RAG pipeline from an
agentic system — the key interview talking point for this feature.
"""

from __future__ import annotations
import re
import time

from langchain_core.messages import HumanMessage, SystemMessage

from app.agents.llm import get_fast_llm
from app.agents.state import AgentState
from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.metrics import agent_runs_total, agent_duration_seconds

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
  "score": 8,
  "reason": "The answer covers the main points but misses the complexity analysis.",
  "improved_queries": []
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
    # Fallback — extract score from text if JSON fails
    score_match = re.search(r"\b([0-9]|10)\b", text)
    score = int(score_match.group(1)) if score_match else 7
    return {"score": score, "reason": "Could not parse critic response.", "improved_queries": []}


def critic_node(state: AgentState) -> dict:
    """
    LangGraph node: Critic.

    Evaluates the current final_answer and decides:
      - score >= 6  → accept answer (graph proceeds to END)
      - score < 6 and retry_count == 0 → rewrite queries, loop back to parallel_agents
      - score < 6 and retry_count >= 1 → accept anyway (prevent infinite loop)

    Input state keys used:  query, final_answer, source_contexts, retry_count
    Output state keys set:  critic_score, critic_feedback, sub_tasks, routing (on retry)
    """
    t0 = time.perf_counter()

    answer = state.get("final_answer", "")
    query = state.get("query", "")
    retry_count = state.get("retry_count", 0)

    if not answer:
        log.warning("critic_no_answer")
        return {"critic_score": 0, "critic_feedback": "No answer was generated."}

    # Don't retry if we've already retried once
    if retry_count >= _MAX_RETRIES:
        log.info("critic_skipped_max_retries", retry_count=retry_count)
        return {
            "critic_score":    7,   # assume acceptable after retry
            "critic_feedback": "Max retries reached — accepting answer.",
        }

    log.info("critic_start", query=query[:60], answer_length=len(answer))

    llm = get_fast_llm()

    # Build a concise context summary for the critic
    source_summary = ""
    contexts = state.get("source_contexts", [])
    if contexts:
        source_summary = f"\n\nSource contexts available ({len(contexts)} passages)"

    messages = [
        SystemMessage(content=_CRITIC_PROMPT),
        HumanMessage(content=(
            f"Original question: {query}\n\n"
            f"Answer to evaluate:\n{answer[:1500]}"
            f"{source_summary}"
        )),
    ]

    try:
        response = llm.invoke(messages)
        result = _parse_critic_response(response.content)

        score: int = int(result.get("score", 7))
        reason: str = result.get("reason", "")
        improved_queries: list[str] = result.get("improved_queries", [])

        elapsed = (time.perf_counter() - t0) * 1000
        log.info(
            "critic_done",
            score=score,
            reason=reason,
            will_retry=(score < 6 and bool(improved_queries)),
            duration_ms=round(elapsed),
        )
        agent_runs_total.labels(agent_name="critic", status="success").inc()
        agent_duration_seconds.labels(agent_name="critic").observe(elapsed / 1000)

        update: dict = {
            "critic_score":    score,
            "critic_feedback": reason,
            "agents_used":     state.get("agents_used", []) + ["critic"],
        }

        # If score is poor AND we have improved queries AND haven't retried yet
        if score < 6 and improved_queries and retry_count == 0:
            log.info(
                "critic_triggering_retry",
                improved_queries=improved_queries,
            )
            # Rewrite sub_tasks and routing so parallel_agents re-runs
            new_routing = {q: "rag" for q in improved_queries}
            update["sub_tasks"]    = improved_queries
            update["routing"]      = new_routing
            update["retry_count"]  = retry_count + 1
            # Clear previous results so they don't pollute the retry
            update["rag_results"]     = []
            update["web_results"]     = []
            update["source_contexts"] = []
            update["final_answer"]    = None
            update["citations"]       = []

        return update

    except Exception as exc:
        log.error("critic_error", error=str(exc))
        agent_runs_total.labels(agent_name="critic", status="error").inc()
        return {
            "critic_score":    7,
            "critic_feedback": f"Critic failed: {exc}",
            "agents_used":     state.get("agents_used", []) + ["critic"],
        }


def should_retry(state: AgentState) -> str:
    """
    Conditional edge after Critic node.

    Returns:
      "retry"  → loop back to parallel_agents with improved queries
      "end"    → accept answer and finish
    """
    score = state.get("critic_score", 10)
    retry_count = state.get("retry_count", 0)
    has_new_tasks = bool(state.get("sub_tasks")) and state.get("final_answer") is None

    if score < 6 and retry_count <= _MAX_RETRIES and has_new_tasks:
        log.info("critic_routing_retry", score=score, retry_count=retry_count)
        return "retry"

    log.info("critic_routing_end", score=score)
    return "end"
