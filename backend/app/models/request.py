"""
app/models/request.py
Pydantic v2 request schemas with validation.
"""

from typing import Literal
from pydantic import BaseModel, Field, field_validator
from app.core.security import detect_prompt_injection, sanitise_query


class ResearchRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=5,
        max_length=1000,
        description="The research question or topic to investigate.",
        examples=["What are the latest advances in Flash Attention?"],
    )
    session_id: str | None = Field(
        default=None,
        description="Pass an existing session_id to continue a conversation. "
                    "Omit to start a new session.",
    )
    domain: Literal["arxiv", "general"] = Field(
        default="arxiv",
        description="Which knowledge base to search.",
    )
    use_web_search: bool = Field(
        default=True,
        description="Whether the web search agent should run.",
    )

    @field_validator("query")
    @classmethod
    def validate_query(cls, v: str) -> str:
        v = sanitise_query(v)
        if detect_prompt_injection(v):
            raise ValueError(
                "Query contains disallowed patterns. "
                "Please rephrase your question."
            )
        return v


class IngestRequest(BaseModel):
    arxiv_query: str = Field(
        ...,
        min_length=3,
        max_length=200,
        description="ArXiv search query (e.g. 'attention mechanism transformers').",
        examples=["Flash Attention memory efficient"],
    )
    max_papers: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Maximum number of papers to fetch and index.",
    )
