"""
app/api/v1/knowledge.py

Browse and manage the paper knowledge base (the ChromaDB corpus).

  GET    /api/v1/knowledge/stats              chunk + paper counts
  GET    /api/v1/knowledge/papers             papers in the corpus (newest first)
  DELETE /api/v1/knowledge/papers/{arxiv_id}  remove a paper and all its chunks

These only touch Chroma — they never load the embedding model.
"""

from __future__ import annotations

import threading

from fastapi import APIRouter, Depends, HTTPException, Query, status
from starlette.concurrency import run_in_threadpool

from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.security import verify_api_key
from app.ingestion.arxiv_loader import base_arxiv_id
from app.ingestion.pipeline import get_corpus_collection
from app.models.response import KnowledgePaper, KnowledgePapersResponse, KnowledgeStats

log = get_logger(__name__)
router = APIRouter(tags=["knowledge"])

# Grouping every chunk's metadata is O(corpus); cache it until the count changes
_cache_lock = threading.Lock()
_papers_cache: tuple[int, list[KnowledgePaper], dict[str, list[str]]] | None = None


def _unavailable(exc: Exception) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail=f"Knowledge base unavailable — is ChromaDB running? ({exc})",
    )


def _load_papers() -> tuple[int, list[KnowledgePaper], dict[str, list[str]]]:
    """Returns (chunk_count, papers, base_id → stored arxiv_id variants)."""
    global _papers_cache
    collection = get_corpus_collection()
    count = collection.count()

    with _cache_lock:
        if _papers_cache and _papers_cache[0] == count:
            return _papers_cache

        metas = collection.get(include=["metadatas"])["metadatas"] or []
        papers: dict[str, KnowledgePaper] = {}
        variants: dict[str, set[str]] = {}
        for m in metas:
            raw_id = str(m.get("arxiv_id") or "")
            key = base_arxiv_id(raw_id) or str(m.get("title", ""))
            if not key:
                continue
            variants.setdefault(key, set()).add(raw_id)
            paper = papers.get(key)
            if paper is None:
                authors = m.get("authors") or ""
                papers[key] = KnowledgePaper(
                    arxiv_id=key,
                    title=str(m.get("title", "Untitled")),
                    authors=[a.strip() for a in str(authors).split(",") if a.strip()],
                    published=str(m.get("published", "")),
                    url=str(m.get("url", "")),
                    categories=str(m.get("categories", "")),
                    chunks=1,
                )
            else:
                paper.chunks += 1

        ordered = sorted(papers.values(), key=lambda p: p.published, reverse=True)
        _papers_cache = (count, ordered, {k: sorted(v) for k, v in variants.items()})
        return _papers_cache


@router.get("/knowledge/stats", response_model=KnowledgeStats, summary="Knowledge base size")
async def knowledge_stats() -> KnowledgeStats:
    try:
        count, papers, _ = await run_in_threadpool(_load_papers)
    except Exception as exc:
        raise _unavailable(exc) from exc
    return KnowledgeStats(collection=get_settings().chroma_collection, chunks=count, papers=len(papers))


@router.get(
    "/knowledge/papers",
    response_model=KnowledgePapersResponse,
    summary="List papers in the knowledge base",
)
async def knowledge_papers(
    q: str = Query(default="", max_length=200, description="Filter by title or author"),
    limit: int = Query(default=100, ge=1, le=500),
) -> KnowledgePapersResponse:
    try:
        _, papers, _ = await run_in_threadpool(_load_papers)
    except Exception as exc:
        raise _unavailable(exc) from exc

    if q:
        needle = q.lower()
        papers = [
            p for p in papers
            if needle in p.title.lower() or any(needle in a.lower() for a in p.authors)
        ]
    return KnowledgePapersResponse(total=len(papers), papers=papers[:limit])


@router.delete(
    "/knowledge/papers/{arxiv_id}",
    summary="Remove a paper from the knowledge base",
)
async def delete_paper(arxiv_id: str, _: str = Depends(verify_api_key)) -> dict:
    key = base_arxiv_id(arxiv_id)

    def _delete() -> int:
        _, _, variants = _load_papers()
        stored_ids = [v for v in variants.get(key, []) if v]
        if not stored_ids:
            return 0
        collection = get_corpus_collection()
        before = collection.count()
        collection.delete(where={"arxiv_id": {"$in": stored_ids}})
        return before - collection.count()

    try:
        removed = await run_in_threadpool(_delete)
    except Exception as exc:
        raise _unavailable(exc) from exc

    if removed == 0:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Paper not found.")

    from app.retrieval.hybrid import invalidate_bm25_cache

    invalidate_bm25_cache()
    log.info("paper_removed", arxiv_id=key, chunks=removed)
    return {"arxiv_id": key, "chunks_removed": removed}
