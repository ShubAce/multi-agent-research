"""
app/workers/tasks.py  (v3)

After a successful pipeline run, saves the user query + assistant answer
to Redis short-term memory so follow-up questions work.
"""

from __future__ import annotations
import asyncio
import json
import time
from typing import Any, cast

import redis as sync_redis
from celery.utils.log import get_task_logger

from app.workers.celery_app import celery_app
from app.core.config import get_settings

logger = get_task_logger(__name__)
settings = get_settings()

_redis = sync_redis.Redis.from_url(settings.redis_url, decode_responses=True)


import logging

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        import numpy as np
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

def _publish(job_id: str, event_type: str, data: dict) -> None:
    payload = json.dumps({"type": event_type, "timestamp": time.time(), **data}, cls=NumpyEncoder)
    _redis.publish(f"job:{job_id}:events", payload)


class ResearchTask:
    _graph = None

    @property
    def graph(self):
        if self._graph is None:
            from app.agents.graph import build_graph
            self._graph = build_graph()
        return self._graph


from celery import Task

class ResearchCeleryTask(Task, ResearchTask):
    pass


@celery_app.task(
    bind=True,
    base=ResearchCeleryTask,
    max_retries=2,
    default_retry_delay=10,
    name="research_assistant.run_pipeline",
)
def run_agent_pipeline(self, *, job_id: str, query: str, session_id: str) -> dict:
    logger.info(f"Starting job_id={job_id}")
    _publish(job_id, "job_started", {"job_id": job_id, "query": query})

    try:
        initial_state = {
            "query":                query,
            "session_id":           session_id,
            "messages":             [],
            "conversation_history": "",   # filled by memory_injection_node
            "sub_tasks":            [],
            "routing":              {},
            "web_results":          [],
            "rag_results":          [],
            "source_contexts":      [],
            "final_answer":         None,
            "citations":            [],
            "agents_used":          [],
            "confidence_score":     None,
            "critic_score":         None,
            "critic_feedback":      None,
            "retry_count":          0,
            "error":                None,
            "iteration_count":      0,
        }

        state = initial_state.copy()

        for step in self.graph.stream(
            initial_state,
            config={"configurable": {"thread_id": session_id}},
        ):
            node_name = next(iter(step))
            state_snapshot = step[node_name]

            # Publish agent progress event (skip memory_injection — not interesting to UI)
            if node_name not in ("memory_injection",):
                _publish(job_id, "agent_done", {
                    "agent":  node_name,
                    "status": "completed",
                    # Include critic score if available
                    **({"critic_score": state_snapshot.get("critic_score")}
                       if node_name == "critic" and state_snapshot.get("critic_score") is not None
                       else {}),
                })

            # Update our accumulated state
            for k, v in state_snapshot.items():
                if k == "messages" and isinstance(v, (list, tuple)):
                    state["messages"] = list(cast(Any, state.get("messages") or [])) + list(cast(Any, v))
                else:
                    state[k] = v

        final_answer = state.get("final_answer")
        if isinstance(final_answer, str) and final_answer:
            # ── Save turn to Redis memory for follow-up questions ─────────────
            try:
                asyncio.run(_save_memory(session_id, query, final_answer))
            except Exception as mem_exc:
                logger.warning(f"Memory save failed: {mem_exc}")

            _publish(job_id, "done", {
                "final_answer":    final_answer,
                "citations":       state.get("citations", []),
                "agents_used":     state.get("agents_used", []),
                "confidence_score": state.get("confidence_score"),
                "critic_score":    state.get("critic_score"),
            })
            return {"status": "done", "job_id": job_id}

        _publish(job_id, "error", {"message": "Pipeline completed without a final answer."})
        return {"status": "error", "job_id": job_id}

    except Exception as exc:
        logger.exception(f"Pipeline failed job_id={job_id}: {exc}")
        _publish(job_id, "error", {"message": str(exc)})
        raise self.retry(exc=exc)


async def _save_memory(session_id: str, query: str, answer: str) -> None:
    from app.agents.memory_node import save_turn_to_memory
    await save_turn_to_memory(session_id, query, answer)
