"""
app/agents/web_search.py

Web Search Agent — uses the Tavily API to retrieve current, external information.
Runs only for sub-tasks the Planner routed to "web_search".

Tavily is purpose-built for LLM pipelines: it returns clean, structured results
without the HTML parsing overhead of a raw Google/Bing API call.
"""

from __future__ import annotations

import concurrent.futures
import time

from tavily import TavilyClient

from app.agents.state import AgentState
from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.metrics import agent_duration_seconds, agent_runs_total

log = get_logger(__name__)


def web_search_available() -> bool:
    return bool(get_settings().tavily_api_key)


def _get_tavily_client() -> TavilyClient:
    settings = get_settings()
    if not settings.tavily_api_key:
        raise RuntimeError(
            "TAVILY_API_KEY is not set. "
            "Either set the key or disable web search in your request."
        )
    return TavilyClient(api_key=settings.tavily_api_key)


def _search(client: TavilyClient, task: str) -> list[dict]:
    response = client.search(
        query=task,
        search_depth="basic",   # "advanced" uses more API credits
        max_results=4,
    )
    return [
        {
            "task": task,
            "title": item.get("title", "") or item.get("url", ""),
            "url": item.get("url", ""),
            "content": item.get("content", "")[:800],  # cap length
            "score": item.get("score", 0.0),
            "source": "web",
        }
        for item in response.get("results", [])
        if item.get("url")
    ]


def web_search_node(state: AgentState) -> dict:
    """
    LangGraph node: Web Search Agent.

    Input state keys used:  routing
    Output state keys set:  web_results, agents_used
    """
    t0 = time.perf_counter()

    tasks_for_web = [
        task for task, agent in state.get("routing", {}).items()
        if agent == "web_search"
    ]

    if not tasks_for_web:
        log.debug("web_search_skipped", reason="no tasks routed to web_search")
        return {}

    log.info("web_search_start", tasks=tasks_for_web)
    agents_used = state.get("agents_used", []) + ["web_search"]

    try:
        client = _get_tavily_client()
        results: list[dict] = []
        errors: list[str] = []

        # Sub-tasks are independent, so search them concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futures = {pool.submit(_search, client, task): task for task in tasks_for_web}
            for future in concurrent.futures.as_completed(futures):
                try:
                    results.extend(future.result())
                except Exception as exc:
                    errors.append(str(exc))
                    log.warning("web_search_task_failed", task=futures[future], error=str(exc))

        if errors and not results:
            raise RuntimeError(errors[0])

        elapsed = (time.perf_counter() - t0) * 1000
        log.info("web_search_done", results=len(results), duration_ms=round(elapsed))
        agent_runs_total.labels(agent_name="web_search", status="success").inc()
        agent_duration_seconds.labels(agent_name="web_search").observe(elapsed / 1000)

        return {"web_results": results, "agents_used": agents_used}

    except Exception as exc:
        log.error("web_search_error", error=str(exc))
        agent_runs_total.labels(agent_name="web_search", status="error").inc()
        return {
            "web_results": [],
            "agents_used": agents_used,
            "error": f"Web search failed: {exc}",
        }
