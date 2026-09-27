"""
app/agents/planner.py  (v2 — with session memory)

The Planner receives conversation history so it can correctly
decompose follow-up questions like "Who invented it?" or "Tell me more".
"""

from __future__ import annotations

import json
import re
import time

from langchain_core.messages import HumanMessage, SystemMessage

from app.agents.llm import get_fast_llm
from app.agents.state import AgentState
from app.agents.web_search import web_search_available
from app.core.logging import get_logger
from app.core.metrics import agent_duration_seconds, agent_runs_total

log = get_logger(__name__)

_VALID_AGENTS = {"rag", "web_search", "skip"}
_MAX_SUB_TASKS = 4

_SYSTEM_PROMPT = """You are a research planning agent. Your job is to decompose a
research question into 2–4 focused sub-tasks and decide which tool handles each one.

Available tools:
  web_search  — recent news, current events, or facts not in our database
  rag         — in-depth technical content from our ArXiv paper database
  skip        — if a sub-task is not needed

IMPORTANT: If conversation history is provided, use it to resolve vague references.
For example if history mentions "Flash Attention" and the new query is "who invented it?",
your sub-tasks should reference "Flash Attention" explicitly, not "it".

Return ONLY valid JSON, no markdown, no explanation:
{
  "sub_tasks": ["sub-task 1", "sub-task 2"],
  "routing": {
    "sub-task 1": "rag",
    "sub-task 2": "web_search"
  }
}"""

_RAG_ONLY_NOTE = "\n\nWeb search is DISABLED for this request — route every sub-task to rag."


def _extract_json(text: str) -> dict:
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass
    match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    match = re.search(r"\{[\s\S]+\}", text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    raise ValueError(f"Could not extract JSON from planner response: {text[:200]!r}")


def _normalise_plan(
    sub_tasks: list,
    routing: dict,
    query: str,
    allow_web: bool,
) -> tuple[list[str], dict[str, str]]:
    """
    Make the LLM's plan safe to execute:
      - every sub-task gets exactly one valid agent (unknown → rag)
      - routing keys that were paraphrased are matched back by position
      - web_search is rewritten to rag when the web is unavailable
      - skipped tasks are dropped; an empty plan falls back to the raw query
    """
    routing = {str(k).strip(): str(v).strip().lower() for k, v in (routing or {}).items()}

    tasks = []
    for t in sub_tasks:
        # Some models return {"task": ..., "routing": ...} objects instead of strings
        if isinstance(t, dict):
            text = next((t[k] for k in ("task", "description", "query", "text") if t.get(k)), "")
            agent = next((t[k] for k in ("routing", "agent", "tool") if t.get(k)), None)
            t = str(text).strip()
            if t and agent and t not in routing:
                routing[t] = str(agent).strip().lower()
        t = str(t).strip()
        if t and t not in tasks:
            tasks.append(t)
    tasks = tasks[:_MAX_SUB_TASKS]
    positional = list(routing.values())

    plan: dict[str, str] = {}
    for i, task in enumerate(tasks):
        agent = routing.get(task)
        if agent is None and len(positional) == len(tasks):
            agent = positional[i]
        if agent not in _VALID_AGENTS:
            agent = "rag"
        if agent == "web_search" and not allow_web:
            agent = "rag"
        if agent != "skip":
            plan[task] = agent

    if not plan:
        plan = {query: "rag"}
    return list(plan), plan


def planner_node(state: AgentState) -> dict:
    """
    LangGraph node: Planner.
    Includes conversation_history in the prompt for follow-up resolution.
    """
    t0 = time.perf_counter()
    query = state["query"]
    history = state.get("conversation_history", "")
    allow_web = state.get("use_web_search", True) and web_search_available()

    log.info("planner_start", query=query[:80], has_history=bool(history), allow_web=allow_web)

    llm = get_fast_llm()

    user_content = f"Research query: {query}"
    if history:
        user_content = f"{history}\n\nNew research query: {query}"

    messages = [
        SystemMessage(content=_SYSTEM_PROMPT + ("" if allow_web else _RAG_ONLY_NOTE)),
        HumanMessage(content=user_content),
    ]

    try:
        response = llm.invoke(messages, response_format={"type": "json_object"})
        plan = _extract_json(str(response.content))
        sub_tasks, routing = _normalise_plan(
            plan.get("sub_tasks", []), plan.get("routing", {}), query, allow_web
        )

        elapsed = (time.perf_counter() - t0) * 1000
        log.info("planner_done", sub_tasks=sub_tasks, duration_ms=round(elapsed))
        agent_runs_total.labels(agent_name="planner", status="success").inc()
        agent_duration_seconds.labels(agent_name="planner").observe(elapsed / 1000)

        return {
            "sub_tasks":       sub_tasks,
            "routing":         routing,
            "agents_used":     state.get("agents_used", []) + ["planner"],
            "iteration_count": state.get("iteration_count", 0) + 1,
        }

    except Exception as exc:
        log.error("planner_error", error=str(exc))
        agent_runs_total.labels(agent_name="planner", status="error").inc()
        return {
            "sub_tasks":       [query],
            "routing":         {query: "rag"},
            "agents_used":     state.get("agents_used", []) + ["planner"],
            "error":           f"Planner failed: {exc}",
            "iteration_count": state.get("iteration_count", 0) + 1,
        }
