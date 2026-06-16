"""
app/retrieval/hybrid.py

Hybrid retrieval: BM25 + dense vector search fused with RRF.

v2 change: BM25 index is now CACHED in a module-level singleton.
Old behaviour: rebuild BM25 index from ChromaDB on every request → ~1–2s overhead
New behaviour: build once at first request, reuse for all subsequent → ~0ms overhead

The cache is invalidated when `invalidate_bm25_cache()` is called —
do this after ingesting new documents.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

from rank_bm25 import BM25Okapi
from llama_index.core.schema import NodeWithScore, TextNode

from app.core.logging import get_logger

log = get_logger(__name__)


# ── Module-level BM25 cache ───────────────────────────────────────────────────
# Thread-safe: protected by _cache_lock
_bm25_cache: "BM25Cache | None" = None
_cache_lock = threading.Lock()


@dataclass
class BM25Cache:
    """Holds the BM25 index and the nodes it was built from."""
    bm25: BM25Okapi
    nodes: list[NodeWithScore]
    num_docs: int


def get_bm25_cache(nodes: list[NodeWithScore]) -> BM25Cache:
    """
    Return the cached BM25 index, building it if necessary.
    Thread-safe via a lock — only one thread builds the index at a time.
    """
    global _bm25_cache

    with _cache_lock:
        # Rebuild if: no cache, or document count has changed (new ingestion)
        if _bm25_cache is None or _bm25_cache.num_docs != len(nodes):
            log.info("building_bm25_cache", num_nodes=len(nodes))
            tokenised = [n.node.get_content().lower().split() for n in nodes]
            _bm25_cache = BM25Cache(
                bm25=BM25Okapi(tokenised),
                nodes=nodes,
                num_docs=len(nodes),
            )
            log.info("bm25_cache_ready", num_docs=len(nodes))
        else:
            log.debug("bm25_cache_hit", num_docs=_bm25_cache.num_docs)

    return _bm25_cache


def invalidate_bm25_cache() -> None:
    """
    Clear the BM25 cache. Call this after ingesting new documents
    so the next request rebuilds the index with fresh data.
    """
    global _bm25_cache
    with _cache_lock:
        _bm25_cache = None
    log.info("bm25_cache_invalidated")


# ── HybridRetriever ───────────────────────────────────────────────────────────

class HybridRetriever:
    """
    Fuses dense vector retrieval with BM25 keyword search via RRF.

    The BM25 index is shared across all instances via the module cache,
    so the expensive tokenisation + index build only happens once per
    process lifetime (or after a cache invalidation).
    """

    def __init__(
        self,
        all_nodes: list[NodeWithScore],
        dense_top_k: int = 10,
        bm25_top_k: int = 10,
        rrf_k: int = 60,
        final_top_n: int = 15,
    ) -> None:
        self.all_nodes = all_nodes
        self.dense_top_k = dense_top_k
        self.bm25_top_k = bm25_top_k
        self.rrf_k = rrf_k
        self.final_top_n = final_top_n
        # Trigger cache build now (not lazily) so first request is fast
        self._cache = get_bm25_cache(all_nodes)

    def _bm25_retrieve(self, query: str) -> list[tuple[int, float]]:
        """BM25 search — uses the cached index."""
        tokenised_query = query.lower().split()
        scores = self._cache.bm25.get_scores(tokenised_query)
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[: self.bm25_top_k]
        return [(idx, float(scores[idx])) for idx in top_indices]

    @staticmethod
    def _rrf_score(rank: int, k: int = 60) -> float:
        return 1.0 / (k + rank)

    def fuse(
        self,
        dense_results: list[NodeWithScore],
        query: str,
    ) -> list[NodeWithScore]:
        """
        Fuse dense results with BM25 results using RRF.
        Dense results come from the caller; BM25 is computed here.
        """
        rrf_scores: dict[str, float] = {}
        node_map: dict[str, NodeWithScore] = {}

        # Score dense results
        for rank, nws in enumerate(dense_results, start=1):
            nid = nws.node.node_id
            rrf_scores[nid] = rrf_scores.get(nid, 0.0) + self._rrf_score(rank, self.rrf_k)
            node_map[nid] = nws

        # Score BM25 results
        for rank, (node_idx, _) in enumerate(self._bm25_retrieve(query), start=1):
            if node_idx >= len(self.all_nodes):
                continue
            bm25_nws = self.all_nodes[node_idx]
            nid = bm25_nws.node.node_id
            rrf_scores[nid] = rrf_scores.get(nid, 0.0) + self._rrf_score(rank, self.rrf_k)
            if nid not in node_map:
                node_map[nid] = bm25_nws

        # Sort by cumulative RRF score
        sorted_ids = sorted(rrf_scores, key=lambda n: rrf_scores[n], reverse=True)

        fused = [
            NodeWithScore(node=node_map[nid].node, score=rrf_scores[nid])
            for nid in sorted_ids[: self.final_top_n]
        ]

        log.debug(
            "rrf_fusion_done",
            dense=len(dense_results),
            fused=len(fused),
        )
        return fused


def reciprocal_rank_fusion(
    ranked_lists: list[list[NodeWithScore]],
    k: int = 60,
    top_n: int = 15,
) -> list[NodeWithScore]:
    """
    General-purpose RRF for N ranked lists.
    Used to merge per-query-variant result sets in the multi-query pipeline.
    """
    rrf_scores: dict[str, float] = {}
    node_map: dict[str, NodeWithScore] = {}

    for ranked_list in ranked_lists:
        for rank, nws in enumerate(ranked_list, start=1):
            nid = nws.node.node_id
            rrf_scores[nid] = rrf_scores.get(nid, 0.0) + (1.0 / (k + rank))
            node_map[nid] = nws

    sorted_ids = sorted(rrf_scores, key=lambda n: rrf_scores[n], reverse=True)
    return [
        NodeWithScore(node=node_map[nid].node, score=rrf_scores[nid])
        for nid in sorted_ids[:top_n]
    ]
