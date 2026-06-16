"""
app/agents/graph.py  (v3)

Full pipeline:
  [memory_injection]
        ↓
  [planner]  ← sees conversation history
        ↓
  [parallel_agents]  ← web + RAG run concurrently
        ↓
  [synthesiser]
        ↓
  [critic]  ← scores answer
        ↓
  score >= 6 → END
  score < 6  → loop back to [parallel_agents] with improved queries (max 1 retry)
"""

from __future__ import annotations
import concurrent.futures
from typing import Any

from langgraph.graph import StateGraph, END

from app.agents.state import AgentState
from app.agents.memory_node import memory_injection_node
from app.agents.planner import planner_node
from app.agents.web_search import web_search_node
from app.agents.rag_agent import rag_agent_node
from app.agents.synthesiser import synthesiser_node
from app.agents.critic import critic_node, should_retry
from app.core.logging import get_logger

log = get_logger(__name__)


def _needs_web(state: AgentState) -> bool:
    return "web_search" in state.get("routing", {}).values()


def _needs_rag(state: AgentState) -> bool:
    return "rag" in state.get("routing", {}).values()


def parallel_agents_node(state: AgentState) -> dict:
    """
    Runs web_search and rag_agent concurrently with ThreadPoolExecutor.
    If only one is needed, runs directly without threading overhead.
    """
    run_web = _needs_web(state)
    run_rag = _needs_rag(state)

    log.info("parallel_agents_start", run_web=run_web, run_rag=run_rag)

    if run_web and not run_rag:
        return web_search_node(state)
    if run_rag and not run_web:
        return rag_agent_node(state)
    if not run_web and not run_rag:
        return {}

    # Both needed — run concurrently
    web_result: dict[str, Any] = {}
    rag_result: dict[str, Any] = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        web_future = executor.submit(lambda: web_search_node(state))
        rag_future = executor.submit(lambda: rag_agent_node(state))

        done, _ = concurrent.futures.wait([web_future, rag_future], timeout=100)

        web_result = web_future.result() if web_future in done else {}
        rag_result = rag_future.result() if rag_future in done else {}

    merged: dict[str, Any] = {
        "web_results":     web_result.get("web_results", []),
        "rag_results":     rag_result.get("rag_results", []),
        "citations":       rag_result.get("citations", []),
        "source_contexts": rag_result.get("source_contexts", []),
        "agents_used": (
            state.get("agents_used", [])
            + web_result.get("agents_used", [])
            + rag_result.get("agents_used", [])
        ),
    }

    log.info(
        "parallel_agents_done",
        web=len(merged["web_results"]),
        rag=len(merged["rag_results"]),
    )
    return merged


def _route_after_planner(state: AgentState) -> str:
    if state.get("error") and not state.get("sub_tasks"):
        return "synthesiser"
    return "parallel_agents"


def build_graph() -> StateGraph:
    workflow = StateGraph(AgentState)

    # ── Nodes ─────────────────────────────────────────────────────────────────
    workflow.add_node("memory_injection", memory_injection_node)
    workflow.add_node("planner",          planner_node)
    workflow.add_node("parallel_agents",  parallel_agents_node)
    workflow.add_node("synthesiser",      synthesiser_node)
    workflow.add_node("critic",           critic_node)

    # ── Entry ─────────────────────────────────────────────────────────────────
    workflow.set_entry_point("memory_injection")

    # ── Edges ─────────────────────────────────────────────────────────────────
    workflow.add_edge("memory_injection", "planner")

    workflow.add_conditional_edges(
        "planner",
        _route_after_planner,
        {
            "parallel_agents": "parallel_agents",
            "synthesiser":     "synthesiser",
        },
    )

    workflow.add_edge("parallel_agents", "synthesiser")
    workflow.add_edge("synthesiser",     "critic")

    # Critic either retries (loop back) or ends
    workflow.add_conditional_edges(
        "critic",
        should_retry,
        {
            "retry": "parallel_agents",   # re-run with improved queries
            "end":   END,
        },
    )

    log.info("graph_compiled_v3")
    return workflow.compile()
