"""
app/ingestion/arxiv_loader.py

Fetches papers from the ArXiv API and returns them as LlamaIndex Documents.
Each document carries rich metadata (title, authors, published date, arxiv_id)
so citations in the final answer are traceable back to the real paper.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import arxiv
from llama_index.core import Document

from app.core.logging import get_logger

log = get_logger(__name__)


@dataclass
class ArxivPaper:
    arxiv_id: str
    title: str
    authors: list[str]
    abstract: str
    published: str          # ISO date string
    url: str
    categories: list[str] = field(default_factory=list)


def fetch_arxiv_papers(
    query: str,
    max_results: int = 20,
    sort_by: arxiv.SortCriterion = arxiv.SortCriterion.Relevance,
) -> list[ArxivPaper]:
    """
    Search ArXiv and return a list of ArxivPaper dataclasses.

    Args:
        query:       Natural-language or field-specific ArXiv query.
                     e.g. "Flash Attention memory efficient transformers"
        max_results: How many papers to pull (1–100).
        sort_by:     arxiv.SortCriterion.Relevance | .SubmittedDate
    """
    log.info("fetching_arxiv_papers", query=query, max_results=max_results)

    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=sort_by,
    )

    papers: list[ArxivPaper] = []
    client = arxiv.Client()

    for result in client.results(search):
        papers.append(
            ArxivPaper(
                arxiv_id=result.entry_id.split("/")[-1],
                title=result.title,
                authors=[a.name for a in result.authors],
                abstract=result.summary,
                published=result.published.strftime("%Y-%m-%d"),
                url=result.entry_id,
                categories=result.categories,
            )
        )
        # Be polite to the ArXiv API
        time.sleep(0.1)

    log.info("arxiv_papers_fetched", count=len(papers))
    return papers


def papers_to_documents(papers: list[ArxivPaper]) -> list[Document]:
    """
    Convert ArxivPaper objects into LlamaIndex Document objects.

    The document text is: title + abstract (full papers would need PDF parsing,
    but abstracts are rich enough for a strong RAG baseline and avoid the
    complexity of PDF download + OCR in a first iteration).

    Metadata is stored per-node so the synthesiser can build proper citations.
    """
    documents: list[Document] = []

    for paper in papers:
        text = (
            f"Title: {paper.title}\n\n"
            f"Authors: {', '.join(paper.authors)}\n\n"
            f"Published: {paper.published}\n\n"
            f"Abstract:\n{paper.abstract}"
        )

        doc = Document(
            text=text,
            metadata={
                "arxiv_id": paper.arxiv_id,
                "title": paper.title,
                "authors": ", ".join(paper.authors) if paper.authors else "",
                "published": paper.published,
                "url": paper.url,
                "categories": ", ".join(paper.categories) if paper.categories else "",
                "source": "arxiv",
            },
            # Exclude heavy fields from being embedded — keep only the text
            excluded_embed_metadata_keys=["authors", "categories", "url"],
            excluded_llm_metadata_keys=["categories"],
        )
        documents.append(doc)

    return documents
