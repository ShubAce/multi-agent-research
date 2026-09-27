"""
app/agents/rag_agent.py

RAG Agent — Hybrid Search + Query Rewriting + cross-encoder reranking.

Full pipeline per sub-task:
  1. Query Rewriting   → original + N variants
  2. Dense retrieval   → top-K candidates per variant
  3. BM25 retrieval    → top-K candidates per variant (fused with dense via RRF)
  4. RRF Fusion        → merge all per-variant lists into one
  5. Sentence-window   → restore surrounding context
  6. Cross-encoder     → rerank fused candidates, keep top-N

The reranked passages go straight to the Synthesiser, which writes one answer
with numbered citations. (Earlier versions also ran a LlamaIndex query engine
per sub-task — a second retrieval plus a 70B call whose output was mostly
discarded.)
"""

from __future__ import annotations

import math
import threading
import time

from llama_index.core.postprocessor import (
    MetadataReplacementPostProcessor,
    SentenceTransformerRerank,
)
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle

from app.agents.state import AgentState
from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.metrics import (
    agent_duration_seconds,
    agent_runs_total,
    documents_retrieved_total,
    vector_db_query_seconds,
)
from app.retrieval.hybrid import HybridRetriever, reciprocal_rank_fusion
from app.retrieval.query_rewriter import rewrite_query

log = get_logger(__name__)

_reranker: SentenceTransformerRerank | None = None
_reranker_lock = threading.Lock()


def _get_reranker() -> SentenceTransformerRerank:
    """Load the cross-encoder once per process."""
    global _reranker
    with _reranker_lock:
        if _reranker is None:
            settings = get_settings()
            _reranker = SentenceTransformerRerank(
                model=settings.rerank_model,
                top_n=settings.rerank_top_n,
            )
        return _reranker


def _build_retrievers(pipeline) -> tuple[VectorIndexRetriever, HybridRetriever | None]:
    settings = get_settings()
    dense = VectorIndexRetriever(
        index=pipeline.get_index(),
        similarity_top_k=settings.retrieval_top_k,
    )

    hybrid: HybridRetriever | None = None
    if settings.hybrid_search_enabled:
        try:
            all_nodes = pipeline.corpus_nodes()
            if all_nodes:
                hybrid = HybridRetriever(
                    all_nodes=all_nodes,
                    dense_top_k=settings.retrieval_top_k,
                    bm25_top_k=settings.bm25_top_k,
                    rrf_k=settings.rrf_k,
                    final_top_n=settings.retrieval_top_k + settings.bm25_top_k,
                )
        except Exception as exc:
            log.warning("hybrid_retriever_init_failed", error=str(exc), fallback="dense only")

    return dense, hybrid


def _retrieve_with_query_variants(
    query: str,
    dense: VectorIndexRetriever,
    hybrid: HybridRetriever | None,
) -> list[NodeWithScore]:
    """
    Multi-query retrieval:
      1. Rewrite query into N variants
      2. For each variant: dense retrieve → optionally fuse with BM25
      3. RRF-fuse all per-variant result sets into one final list
    """
    settings = get_settings()
    queries = rewrite_query(query)
    log.info("multi_query_retrieval", num_variants=len(queries))

    result_lists: list[list[NodeWithScore]] = []
    for q in queries:
        dense_nodes = dense.retrieve(q)
        result_lists.append(hybrid.fuse(dense_nodes, q) if hybrid else dense_nodes)

    if len(result_lists) == 1:
        return result_lists[0]

    return reciprocal_rank_fusion(
        result_lists,
        k=settings.rrf_k,
        top_n=settings.retrieval_top_k * 2,  # give the reranker plenty to work with
    )


def _rerank(task: str, nodes: list[NodeWithScore]) -> list[NodeWithScore]:
    settings = get_settings()
    bundle = QueryBundle(query_str=task)
    windowed = MetadataReplacementPostProcessor(target_metadata_key="window").postprocess_nodes(
        nodes, query_bundle=bundle
    )
    try:
        reranked = _get_reranker().postprocess_nodes(windowed, query_bundle=bundle)
        # The cross-encoder returns raw logits; squash them to a 0–1 relevance
        # and drop passages it considers off-topic (the corpus may simply not
        # cover the question — better no paper than an irrelevant one)
        for nws in reranked:
            nws.score = 1.0 / (1.0 + math.exp(-float(nws.score or 0.0)))
        return [nws for nws in reranked if nws.score >= settings.rerank_min_score]
    except Exception as exc:
        log.warning("reranker_failed", error=str(exc), fallback="using fused order")
        return windowed[: settings.rerank_top_n]


def _passage(nws: NodeWithScore) -> dict:
    meta = nws.node.metadata or {}
    authors = meta.get("authors") or []
    if isinstance(authors, str):
        authors = [a.strip() for a in authors.split(",") if a.strip()]
    score = float(nws.score or 0.0)
    return {
        "node_id": nws.node.node_id,
        "text": nws.node.get_content(),
        "title": meta.get("title", "Untitled paper"),
        "authors": authors,
        "url": meta.get("url", ""),
        "arxiv_id": meta.get("arxiv_id", ""),
        "published": meta.get("published", ""),
        "relevance_score": round(min(max(score, 0.0), 1.0), 4),
    }


def rag_agent_node(state: AgentState) -> dict:
    """
    LangGraph node: RAG Agent.

    Input state keys used:  routing
    Output state keys set:  rag_results, source_contexts, agents_used
      rag_results = [{"task": str, "passages": [passage, ...]}]  (tasks with no hits omitted)
    """
    t0 = time.perf_counter()

    tasks_for_rag = [task for task, agent in state.get("routing", {}).items() if agent == "rag"]
    if not tasks_for_rag:
        log.debug("rag_agent_skipped", reason="no tasks routed to rag")
        return {}

    log.info("rag_agent_start", tasks=tasks_for_rag)
    agents_used = state.get("agents_used", []) + ["rag_agent"]

    try:
        from app.ingestion.pipeline import get_pipeline

        pipeline = get_pipeline()
        if pipeline.collection.count() == 0:
            log.warning("rag_agent_empty_corpus")
            return {"rag_results": [], "source_contexts": [], "agents_used": agents_used}

        dense, hybrid = _build_retrievers(pipeline)

        rag_results: list[dict] = []
        source_contexts: list[str] = []
        seen_nodes: set[str] = set()

        for task in tasks_for_rag:
            vdb_t0 = time.perf_counter()
            fused = _retrieve_with_query_variants(task, dense, hybrid)
            reranked = _rerank(task, fused) if fused else []
            vector_db_query_seconds.observe(time.perf_counter() - vdb_t0)
            documents_retrieved_total.inc(len(reranked))

            passages = []
            for nws in reranked:
                if nws.node.node_id in seen_nodes:
                    continue
                seen_nodes.add(nws.node.node_id)
                passage = _passage(nws)
                passages.append(passage)
                source_contexts.append(passage["text"])

            if passages:
                rag_results.append({"task": task, "passages": passages})

        elapsed = (time.perf_counter() - t0) * 1000
        log.info(
            "rag_agent_done",
            tasks=len(tasks_for_rag),
            passages=len(source_contexts),
            duration_ms=round(elapsed),
        )
        agent_runs_total.labels(agent_name="rag_agent", status="success").inc()
        agent_duration_seconds.labels(agent_name="rag_agent").observe(elapsed / 1000)

        return {
            "rag_results": rag_results,
            "source_contexts": source_contexts,
            "agents_used": agents_used,
        }

    except Exception as exc:
        log.error("rag_agent_error", error=str(exc))
        agent_runs_total.labels(agent_name="rag_agent", status="error").inc()
        return {
            "rag_results": [],
            "agents_used": agents_used,
            "error": f"RAG agent failed: {exc}",
        }
