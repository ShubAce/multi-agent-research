"""
app/agents/rag_agent.py

RAG Agent — upgraded with Hybrid Search + Query Rewriting.

Full pipeline per sub-task:
  1. Query Rewriting   → original + N variants
  2. Dense retrieval   → top-K candidates per variant
  3. BM25 retrieval    → top-K candidates per variant
  4. RRF Fusion        → merge all ranked lists into one
  5. Sentence-window   → restore surrounding context
  6. Cross-encoder     → rerank fused candidates, keep top-N
  7. LLM answer        → generate answer from top-N chunks
"""

from __future__ import annotations
import time

from llama_index.core import VectorStoreIndex
from llama_index.core.postprocessor import (
    MetadataReplacementPostProcessor,
    SentenceTransformerRerank,
)
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.schema import NodeWithScore

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


def _get_all_nodes(index: VectorStoreIndex) -> list[NodeWithScore]:
    """
    Fetch all nodes from the index for BM25 indexing.
    ChromaDB returns them via the docstore; we wrap in NodeWithScore.
    """
    try:
        docstore = index.storage_context.docstore
        all_nodes = list(docstore.docs.values())
        return [NodeWithScore(node=node, score=1.0) for node in all_nodes]
    except Exception as exc:
        log.warning("could_not_fetch_all_nodes", error=str(exc))
        return []


def _build_hybrid_query_engine(index: VectorStoreIndex) -> tuple[RetrieverQueryEngine, HybridRetriever | None]:
    """
    Build a query engine with hybrid retrieval.

    Returns:
        (query_engine, hybrid_retriever)
        hybrid_retriever is None if hybrid search is disabled or failed to init.
    """
    settings = get_settings()

    dense_retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=settings.retrieval_top_k,
    )

    # Try to build the BM25 side
    hybrid_retriever: HybridRetriever | None = None
    if settings.hybrid_search_enabled:
        try:
            all_nodes = _get_all_nodes(index)
            if all_nodes:
                hybrid_retriever = HybridRetriever(
                    all_nodes=all_nodes,
                    dense_top_k=settings.retrieval_top_k,
                    bm25_top_k=settings.bm25_top_k,
                    rrf_k=settings.rrf_k,
                    final_top_n=settings.retrieval_top_k + settings.bm25_top_k,
                )
                log.info("hybrid_retriever_ready", num_nodes=len(all_nodes))
            else:
                log.warning("hybrid_search_disabled", reason="no nodes found in docstore")
        except Exception as exc:
            log.warning("hybrid_retriever_init_failed", error=str(exc), fallback="dense only")

    # Postprocessors applied after retrieval
    window_postprocessor = MetadataReplacementPostProcessor(
        target_metadata_key="window"
    )
    reranker = SentenceTransformerRerank(
        model="cross-encoder/ms-marco-MiniLM-L-2-v2",
        top_n=settings.rerank_top_n,
    )

    query_engine = RetrieverQueryEngine.from_args(
        retriever=dense_retriever,
        node_postprocessors=[window_postprocessor, reranker],
        response_mode=ResponseMode.COMPACT,
        verbose=False,
    )

    return query_engine, hybrid_retriever


def _retrieve_with_query_variants(
    query: str,
    query_engine: RetrieverQueryEngine,
    hybrid_retriever: HybridRetriever | None,
) -> list[NodeWithScore]:
    """
    Run multi-query retrieval:
      1. Rewrite query into N variants
      2. For each variant: dense retrieve → optionally fuse with BM25
      3. RRF-fuse all per-variant result sets into one final list
    """
    settings = get_settings()

    # Step 1: rewrite query
    queries = rewrite_query(query)
    log.info("multi_query_retrieval", num_variants=len(queries), queries=queries)

    all_result_lists: list[list[NodeWithScore]] = []

    for q in queries:
        # Step 2a: dense retrieval
        dense_nodes = query_engine.retriever.retrieve(q)

        if hybrid_retriever is not None:
            # Step 2b: fuse dense + BM25 for this variant
            fused = hybrid_retriever.fuse(dense_nodes, q)
            all_result_lists.append(fused)
        else:
            all_result_lists.append(dense_nodes)

    # Step 3: fuse all per-variant lists into one final ranked list
    if len(all_result_lists) == 1:
        return all_result_lists[0]

    final_nodes = reciprocal_rank_fusion(
        all_result_lists,
        k=settings.rrf_k,
        top_n=settings.retrieval_top_k * 2,  # give the reranker plenty to work with
    )
    log.debug("multi_query_rrf_done", total_candidates=len(final_nodes))
    return final_nodes


def rag_agent_node(state: AgentState) -> dict:
    """
    LangGraph node: RAG Agent (upgraded with hybrid search + query rewriting).

    Input state keys used:  sub_tasks, routing
    Output state keys set:  rag_results, citations, agents_used
    """
    t0 = time.perf_counter()

    tasks_for_rag = [
        task for task, agent in state.get("routing", {}).items()
        if agent == "rag"
    ]

    if not tasks_for_rag:
        log.debug("rag_agent_skipped", reason="no tasks routed to rag")
        return {}

    log.info("rag_agent_start", tasks=tasks_for_rag)

    try:
        from app.ingestion.pipeline import IngestionPipeline

        pipeline = IngestionPipeline()
        index = pipeline.get_index()
        query_engine, hybrid_retriever = _build_hybrid_query_engine(index)

        settings = get_settings()
        rag_results: list[dict] = []
        all_citations: list[dict] = []
        # Collect all source contexts for the faithfulness filter later
        all_source_contexts: list[str] = []

        for task in tasks_for_rag:
            vdb_t0 = time.perf_counter()

            # ── Hybrid multi-query retrieval ──────────────────────────────────
            fused_nodes = _retrieve_with_query_variants(task, query_engine, hybrid_retriever)

            # ── Apply reranker to fused results ──────────────────────────────
            # Reranker expects NodeWithScore — wrap if needed
            from llama_index.core.postprocessor import SentenceTransformerRerank
            reranker = SentenceTransformerRerank(
                model="cross-encoder/ms-marco-MiniLM-L-2-v2",
                top_n=settings.rerank_top_n,
            )
            # MetadataReplacementPostProcessor restores the sentence window
            from llama_index.core.postprocessor import MetadataReplacementPostProcessor
            window_pp = MetadataReplacementPostProcessor(target_metadata_key="window")

            from llama_index.core.schema import QueryBundle
            query_bundle = QueryBundle(query_str=task)

            try:
                reranked = window_pp.postprocess_nodes(fused_nodes, query_bundle=query_bundle)
                reranked = reranker.postprocess_nodes(reranked, query_bundle=query_bundle)
            except Exception as exc:
                log.warning("reranker_failed", error=str(exc), fallback="using fused nodes")
                reranked = fused_nodes[:settings.rerank_top_n]

            vdb_elapsed = time.perf_counter() - vdb_t0
            vector_db_query_seconds.observe(vdb_elapsed)
            documents_retrieved_total.inc(len(reranked))

            # ── Generate answer from reranked context ─────────────────────────
            # Build context string from reranked nodes and query the LLM
            context_str = "\n\n".join(
                node.get_content() for node in reranked
            ) if reranked else ""

            source_contexts = [node.get_content() for node in reranked]
            all_source_contexts.extend(source_contexts)

            # Use query engine's synthesiser with our custom context
            response = query_engine.query(task)

            # ── Build citations ───────────────────────────────────────────────
            task_citations = []
            for node in reranked:
                meta = node.metadata or {}
                authors_val = meta.get("authors")
                if isinstance(authors_val, str):
                    authors_list = [a.strip() for a in authors_val.split(",") if a.strip()]
                else:
                    authors_list = authors_val or []
                
                task_citations.append({
                    "title":           meta.get("title", "Unknown"),
                    "authors":         authors_list,
                    "url":             meta.get("url", ""),
                    "arxiv_id":        meta.get("arxiv_id", ""),
                    "published":       meta.get("published", ""),
                    "relevance_score": round(node.score or 0.0, 4),
                    "excerpt":         node.get_content()[:300],
                })

            rag_results.append({
                "task":         task,
                "answer":       str(response),
                "citations":    task_citations,
                "num_sources":  len(reranked),
                "source_contexts": source_contexts,   # passed to faithfulness filter
            })
            all_citations.extend(task_citations)

        elapsed = (time.perf_counter() - t0) * 1000
        log.info(
            "rag_agent_done",
            tasks=len(tasks_for_rag),
            total_sources=len(all_citations),
            duration_ms=round(elapsed),
        )
        agent_runs_total.labels(agent_name="rag_agent", status="success").inc()
        agent_duration_seconds.labels(agent_name="rag_agent").observe(elapsed / 1000)

        return {
            "rag_results":       rag_results,
            "citations":         all_citations,
            "source_contexts":   all_source_contexts,   # for faithfulness filter
            "agents_used":       state.get("agents_used", []) + ["rag_agent"],
        }

    except Exception as exc:
        log.error("rag_agent_error", error=str(exc))
        agent_runs_total.labels(agent_name="rag_agent", status="error").inc()
        return {
            "rag_results":  [],
            "agents_used":  state.get("agents_used", []) + ["rag_agent"],
            "error":        f"RAG agent failed: {exc}",
        }
