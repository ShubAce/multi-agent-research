"""
app/agents/web_search.py

Web Search Agent — uses the Tavily API to retrieve current, external information.
Runs only for sub-tasks the Planner routed to "web_search".

Tavily is purpose-built for LLM pipelines: it returns clean, structured results
without the HTML parsing overhead of a raw Google/Bing API call.
"""

from __future__ import annotations
import time

from tavily import TavilyClient

from app.agents.state import AgentState
from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.metrics import agent_runs_total, agent_duration_seconds

log = get_logger(__name__)


def _get_tavily_client() -> TavilyClient:
    settings = get_settings()
    if not settings.tavily_api_key:
        raise RuntimeError(
            "TAVILY_API_KEY is not set. "
            "Either set the key or disable web search in your request."
        )
    return TavilyClient(api_key=settings.tavily_api_key)


def web_search_node(state: AgentState) -> dict:
    """
    LangGraph node: Web Search Agent.

    Input state keys used:  sub_tasks, routing
    Output state keys set:  web_results, agents_used
    """
    t0 = time.perf_counter()

    # Only run tasks explicitly routed to web_search
    tasks_for_web = [
        task for task, agent in state.get("routing", {}).items()
        if agent == "web_search"
    ]

    if not tasks_for_web:
        log.debug("web_search_skipped", reason="no tasks routed to web_search")
        return {}

    log.info("web_search_start", tasks=tasks_for_web)

    try:
        client = _get_tavily_client()
        results: list[dict] = []

        for task in tasks_for_web:
            search_result = client.search(
                query=task,
                search_depth="basic",   # "advanced" uses more API credits
                max_results=4,
                include_answer=True,    # Tavily's own one-line summary
            )

            for item in search_result.get("results", []):
                results.append({
                    "task": task,
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "content": item.get("content", "")[:800],  # cap length
                    "score": item.get("score", 0.0),
                    "source": "web",
                })

            # If Tavily produced a direct answer, include it too
            if search_result.get("answer"):
                results.append({
                    "task": task,
                    "title": "Tavily Direct Answer",
                    "url": "",
                    "content": search_result["answer"],
                    "score": 1.0,
                    "source": "web_answer",
                })

        elapsed = (time.perf_counter() - t0) * 1000
        log.info("web_search_done", results=len(results), duration_ms=round(elapsed))
        agent_runs_total.labels(agent_name="web_search", status="success").inc()
        agent_duration_seconds.labels(agent_name="web_search").observe(elapsed / 1000)

        return {
            "web_results": results,
            "agents_used": state.get("agents_used", []) + ["web_search"],
        }

    except Exception as exc:
        log.error("web_search_error", error=str(exc))
        agent_runs_total.labels(agent_name="web_search", status="error").inc()
        return {
            "web_results": [],
            "agents_used": state.get("agents_used", []) + ["web_search"],
            "error": f"Web search failed: {exc}",
        }
