"""
app/api/v1/papers.py

POST /api/v1/papers/summarise
  Body: { "arxiv_input": "https://arxiv.org/abs/2205.14135" }
  Returns: structured PaperSummary with all fields

This endpoint is called from the frontend's Paper Summariser panel.
The paper is also auto-ingested into ChromaDB so it becomes
immediately searchable in future research queries.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from app.core.logging import get_logger
from app.core.security import verify_api_key

log = get_logger(__name__)
router = APIRouter(tags=["papers"])


class SummariseRequest(BaseModel):
    arxiv_input: str = Field(
        ...,
        min_length=5,
        description="ArXiv URL (https://arxiv.org/abs/XXXX.XXXXX) or plain ID (XXXX.XXXXX)",
        examples=["https://arxiv.org/abs/2205.14135", "2205.14135"],
    )


class SummariseResponse(BaseModel):
    arxiv_id: str
    title: str
    authors: list[str]
    published: str
    url: str
    one_liner: str
    problem: str
    method: str
    key_results: str
    limitations: str
    contributions: list[str]
    related_work: list[str]
    nodes_ingested: int
    already_indexed: bool = False
    message: str


@router.post(
    "/papers/summarise",
    response_model=SummariseResponse,
    summary="Summarise an ArXiv paper and ingest it into the knowledge base",
)
async def summarise_paper(
    request: SummariseRequest,
    _: str = Depends(verify_api_key),
) -> SummariseResponse:
    """
    Given an ArXiv URL or ID:
    1. Fetches the paper from ArXiv
    2. Generates a structured summary (problem / method / results / limitations)
    3. Ingests the paper into ChromaDB so it's searchable immediately
    4. Returns the summary

    Typical response time: 8–15 seconds (ArXiv fetch + LLM summarisation).
    """
    log.info("summarise_request", arxiv_input=request.arxiv_input)

    try:
        from app.services.paper_summariser import summarise_paper as _summarise
        # Blocking work (ArXiv + LLM + embeddings) — keep it off the event loop
        summary = await run_in_threadpool(_summarise, request.arxiv_input)

        return SummariseResponse(
            arxiv_id=summary.arxiv_id,
            title=summary.title,
            authors=summary.authors,
            published=summary.published,
            url=summary.url,
            one_liner=summary.one_liner,
            problem=summary.problem,
            method=summary.method,
            key_results=summary.key_results,
            limitations=summary.limitations,
            contributions=summary.contributions,
            related_work=summary.related_work,
            nodes_ingested=summary.nodes_ingested,
            already_indexed=summary.already_indexed,
            message=(
                f"Added to the knowledge base ({summary.nodes_ingested} chunks)."
                if summary.nodes_ingested > 0
                else "Already in the knowledge base."
                if summary.already_indexed
                else "Summary generated, but adding the paper to the knowledge base failed."
            ),
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        log.error("summarise_error", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to summarise paper: {exc}",
        ) from exc
