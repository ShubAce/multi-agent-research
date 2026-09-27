"""
app/workers/tasks.py  (v4)

Runs the agent graph for one research job and appends progress events to the
job's Redis stream (see app/core/events.py). After a successful run the turn
is saved to short-term memory so follow-up questions work.
"""

from __future__ import annotations

import time

import redis as sync_redis
from celery import Task
from celery.exceptions import SoftTimeLimitExceeded
from celery.utils.log import get_task_logger

from app.core.config import get_settings
from app.core.events import append_event, cancel_key
from app.workers.celery_app import celery_app

logger = get_task_logger(__name__)
settings = get_settings()

_redis = sync_redis.Redis.from_url(settings.redis_url, decode_responses=True)


def _publish(job_id: str, event_type: str, data: dict) -> None:
    append_event(_redis, job_id, event_type, data, settings.job_events_ttl_seconds)


def _is_cancelled(job_id: str) -> bool:
    try:
        return bool(_redis.exists(cancel_key(job_id)))
    except Exception:
        return False


def _next_node(node: str, state: dict) -> str | None:
    """Which node runs next — lets the UI show what is in progress."""
    from app.agents.critic import should_retry
    from app.agents.graph import _route_after_planner

    if node == "memory_injection":
        return "planner"
    if node == "planner":
        return _route_after_planner(state)
    if node == "parallel_agents":
        return "synthesiser"
    if node == "synthesiser":
        return "critic"
    if node == "critic":
        return "parallel_agents" if should_retry(state) == "retry" else None
    return None


def _detail(node: str, update: dict, state: dict) -> dict:
    """Per-node summary shown in the run inspector."""
    if node == "memory_injection":
        history = state.get("conversation_history") or ""
        turns = sum(1 for line in history.splitlines() if line.startswith("User: "))
        return {"turns": turns, "has_history": bool(history)}
    if node == "planner":
        return {
            "plan": [{"task": t, "agent": a} for t, a in (state.get("routing") or {}).items()],
            "error": update.get("error"),
        }
    if node == "parallel_agents":
        passages = [p for r in state.get("rag_results", []) for p in r.get("passages", [])]
        papers = {p.get("arxiv_id") or p.get("title") for p in passages}
        return {
            "papers": len(papers),
            "passages": len(passages),
            "web_results": len(state.get("web_results", [])),
            "web_fallback": bool(update.get("web_fallback")),
            "retry": state.get("retry_count", 0) > 0,
            "error": update.get("error"),
        }
    if node == "synthesiser":
        return {
            "sources": len(state.get("citations", [])),
            "confidence": state.get("confidence_score"),
            "fact_check": state.get("fact_check"),
            "error": update.get("error"),
        }
    if node == "critic":
        return {
            "score": state.get("critic_score"),
            "feedback": state.get("critic_feedback"),
            "improved_queries": update.get("sub_tasks", []) if "retry_count" in update else [],
        }
    return {}


def _unique(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))


class ResearchCeleryTask(Task):
    _graph = None

    @property
    def graph(self):
        if self._graph is None:
            from app.agents.graph import build_graph
            self._graph = build_graph()
        return self._graph


@celery_app.task(
    bind=True,
    base=ResearchCeleryTask,
    max_retries=0,   # nodes degrade gracefully; re-running would duplicate LLM spend
    name="research_assistant.run_pipeline",
)
def run_agent_pipeline(
    self,
    *,
    job_id: str,
    query: str,
    session_id: str,
    use_web_search: bool = True,
) -> dict:
    logger.info(f"Starting job_id={job_id}")

    if _is_cancelled(job_id):
        _publish(job_id, "cancelled", {"message": "Cancelled before it started."})
        return {"status": "cancelled", "job_id": job_id}

    _publish(job_id, "job_started", {"job_id": job_id, "query": query})
    t_start = time.perf_counter()

    try:
        state: dict = {
            "query":                query,
            "session_id":           session_id,
            "use_web_search":       use_web_search,
            "messages":             [],
            "conversation_history": "",   # filled by memory_injection_node
            "sub_tasks":            [],
            "routing":              {},
            "web_results":          [],
            "rag_results":          [],
            "source_contexts":      [],
            "web_fallback":         False,
            "final_answer":         None,
            "citations":            [],
            "agents_used":          [],
            "confidence_score":     None,
            "fact_check":           None,
            "critic_score":         None,
            "critic_feedback":      None,
            "retry_count":          0,
            "error":                None,
            "iteration_count":      0,
        }

        t_prev = t_start
        for step in self.graph.stream(state.copy(), config={"configurable": {"thread_id": session_id}}):
            node_name, update = next(iter(step.items()))
            update = update or {}
            state.update(update)

            now = time.perf_counter()
            _publish(job_id, "agent_done", {
                "agent":       node_name,
                "status":      "completed",
                "duration_ms": round((now - t_prev) * 1000),
                "next_agent":  _next_node(node_name, state),
                "detail":      _detail(node_name, update, state),
            })
            t_prev = now

            if _is_cancelled(job_id):
                _publish(job_id, "cancelled", {"message": "Research cancelled."})
                return {"status": "cancelled", "job_id": job_id}

        final_answer = state.get("final_answer")
        if not (isinstance(final_answer, str) and final_answer):
            _publish(job_id, "failed", {"message": "The pipeline finished without producing an answer."})
            return {"status": "error", "job_id": job_id}

        from app.agents.follow_ups import suggest_follow_ups

        follow_ups = suggest_follow_ups(query, final_answer)

        try:
            from app.agents.memory_node import save_turn_to_memory
            save_turn_to_memory(session_id, query, final_answer)
        except Exception as mem_exc:
            logger.warning(f"Memory save failed: {mem_exc}")

        _publish(job_id, "done", {
            "final_answer":     final_answer,
            "citations":        state.get("citations", []),
            "agents_used":      _unique(state.get("agents_used", [])),
            "confidence_score": state.get("confidence_score"),
            "critic_score":     state.get("critic_score"),
            "critic_feedback":  state.get("critic_feedback"),
            "fact_check":       state.get("fact_check"),
            "follow_ups":       follow_ups,
            "sub_tasks":        state.get("sub_tasks", []),
            "retries":          state.get("retry_count", 0),
            "web_fallback":     state.get("web_fallback", False),
            "duration_ms":      round((time.perf_counter() - t_start) * 1000),
        })
        return {"status": "done", "job_id": job_id}

    except SoftTimeLimitExceeded:
        logger.error(f"Pipeline timed out job_id={job_id}")
        _publish(job_id, "failed", {"message": "Research took too long and was stopped. Try a narrower question."})
        return {"status": "error", "job_id": job_id}
    except Exception as exc:
        logger.exception(f"Pipeline failed job_id={job_id}: {exc}")
        _publish(job_id, "failed", {"message": f"Pipeline error: {exc}"})
        return {"status": "error", "job_id": job_id}
