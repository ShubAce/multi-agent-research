"""
app/models/response.py
Pydantic v2 response schemas.
"""

from typing import Literal
from pydantic import BaseModel, Field


class ResearchJobResponse(BaseModel):
    job_id: str
    session_id: str
    status: Literal["queued", "running", "done", "error"] = "queued"
    message: str = "Job submitted successfully. Connect to /api/v1/stream/{job_id} for live updates."


class CitationModel(BaseModel):
    title: str
    authors: list[str] = []
    source: str          # URL or arXiv ID
    relevance_score: float = Field(ge=0.0, le=1.0)
    excerpt: str = ""    # short snippet from the source


class ResearchResultResponse(BaseModel):
    job_id: str
    session_id: str
    query: str
    final_answer: str
    citations: list[CitationModel] = []
    agents_used: list[str] = []
    reasoning_trace: list[dict] = []
    ragas_faithfulness: float | None = None


class SessionSummary(BaseModel):
    session_id: str
    query: str
    created_at: str
    agents_used: list[str] = []


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    services: dict[str, str]     # e.g. {"redis": "ok", "chromadb": "ok"}
    version: str = "0.1.0"


class IngestResponse(BaseModel):
    status: str
    papers_indexed: int
    collection: str
