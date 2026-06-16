"""
app/agents/planner.py  (v2 — with session memory)

The Planner now receives conversation history so it can correctly
decompose follow-up questions like "Who invented it?" or "Tell me more".
"""

from __future__ import annotations
import json
import re
import time

from langchain_core.messages import HumanMessage, SystemMessage

from app.agents.llm import get_fast_llm
from app.agents.state import AgentState
from app.core.logging import get_logger
from app.core.metrics import agent_runs_total, agent_duration_seconds

log = get_logger(__name__)

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


def planner_node(state: AgentState) -> dict:
    """
    LangGraph node: Planner.
    Now includes conversation_history in the prompt for follow-up resolution.
    """
    t0 = time.perf_counter()
    query = state["query"]
    history = state.get("conversation_history", "")

    log.info("planner_start", query=query[:80], has_history=bool(history))

    llm = get_fast_llm()

    # Build user message — include history if present
    user_content = f"Research query: {query}"
    if history:
        user_content = f"{history}\n\nNew research query: {query}"

    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=user_content),
    ]

    try:
        response = llm.invoke(messages)
        plan = _extract_json(response.content)

        sub_tasks: list[str] = plan.get("sub_tasks", [])
        routing: dict[str, str] = plan.get("routing", {})

        if not sub_tasks:
            sub_tasks = [query]
            routing = {query: "rag"}

        elapsed = (time.perf_counter() - t0) * 1000
        log.info("planner_done", sub_tasks=sub_tasks, duration_ms=round(elapsed))
        agent_runs_total.labels(agent_name="planner", status="success").inc()
        agent_duration_seconds.labels(agent_name="planner").observe(elapsed / 1000)

        return {
            "sub_tasks":      sub_tasks,
            "routing":        routing,
            "agents_used":    state.get("agents_used", []) + ["planner"],
            "messages":       [*state.get("messages", []), *messages, response],
            "iteration_count": state.get("iteration_count", 0) + 1,
        }

    except Exception as exc:
        log.error("planner_error", error=str(exc))
        agent_runs_total.labels(agent_name="planner", status="error").inc()
        return {
            "sub_tasks":      [query],
            "routing":        {query: "rag"},
            "agents_used":    state.get("agents_used", []) + ["planner"],
            "error":          f"Planner failed: {exc}",
            "iteration_count": state.get("iteration_count", 0) + 1,
        }
