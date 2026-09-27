"""
app/services/paper_summariser.py

Paper Summariser — given an ArXiv URL or ID:
  1. Fetch the paper metadata + abstract from ArXiv API
  2. Generate a structured summary (problem, method, results, limitations)
  3. Ingest the paper into ChromaDB so it's retrievable in future queries
  4. Return the summary + metadata

Structured summary schema:
  - problem       : what problem does this paper solve?
  - method        : what approach / architecture / algorithm?
  - key_results   : main quantitative or qualitative findings
  - limitations   : what does the paper acknowledge as limitations?
  - contributions : bullet list of key contributions
  - related_work  : 2–3 papers this builds on (from abstract mentions)
  - one_liner     : single sentence summary for quick reference
"""

from __future__ import annotations
import re
from dataclasses import dataclass

import arxiv
from langchain_core.messages import HumanMessage, SystemMessage

from app.agents.llm import get_llm
from app.core.logging import get_logger

log = get_logger(__name__)

_SUMMARISE_PROMPT = """You are an expert at reading academic papers and producing
structured summaries for machine learning researchers.

Given the title and abstract of a paper, produce a structured summary.

Return ONLY this JSON (no markdown, no extra text):
{
  "one_liner": "one sentence describing what this paper does",
  "problem": "what problem or limitation does this paper address?",
  "method": "what is the core approach, architecture, or algorithm?",
  "key_results": "main findings — include numbers if mentioned in the abstract",
  "limitations": "what limitations or future work does the paper mention?",
  "contributions": ["contribution 1", "contribution 2", "contribution 3"],
  "related_work": ["paper or concept 1", "paper or concept 2"]
}"""


@dataclass
class PaperSummary:
    arxiv_id: str
    title: str
    authors: list[str]
    published: str
    url: str
    abstract: str
    one_liner: str
    problem: str
    method: str
    key_results: str
    limitations: str
    contributions: list[str]
    related_work: list[str]
    nodes_ingested: int
    already_indexed: bool = False


def _extract_arxiv_id(input_str: str) -> str:
    """
    Extract ArXiv ID from various input formats:
      - Full URL:  https://arxiv.org/abs/2205.14135
      - PDF URL:   https://arxiv.org/pdf/2205.14135
      - Plain ID:  2205.14135
      - With v:    2205.14135v2
    """
    input_str = input_str.strip()

    # URL pattern
    match = re.search(r"arxiv\.org/(?:abs|pdf)/([0-9]{4}\.[0-9]+(?:v\d+)?)", input_str)
    if match:
        return match.group(1).split("v")[0]   # strip version suffix

    # Plain ID pattern
    match = re.match(r"^([0-9]{4}\.[0-9]+)(?:v\d+)?$", input_str)
    if match:
        return match.group(1)

    raise ValueError(
        f"Could not extract ArXiv ID from: {input_str!r}\n"
        "Expected format: https://arxiv.org/abs/2205.14135 or 2205.14135"
    )


def _fetch_paper(arxiv_id: str) -> arxiv.Result:
    """Fetch paper metadata from ArXiv API."""
    client = arxiv.Client()
    search = arxiv.Search(id_list=[arxiv_id])
    results = list(client.results(search))
    if not results:
        raise ValueError(f"No paper found for ArXiv ID: {arxiv_id}")
    return results[0]


def _generate_summary(title: str, abstract: str) -> dict:
    """Call the LLM to produce a structured summary."""
    llm = get_llm()   # use 70B for quality summaries
    messages = [
        SystemMessage(content=_SUMMARISE_PROMPT),
        HumanMessage(content=f"Title: {title}\n\nAbstract:\n{abstract}"),
    ]
    response = llm.invoke(messages)

    import json
    text = str(response.content).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try extracting JSON block
        match = re.search(r"\{[\s\S]+\}", text)
        if match:
            return json.loads(match.group(0))
        raise ValueError(f"Could not parse summary JSON: {text[:200]}")


def summarise_paper(arxiv_input: str) -> PaperSummary:
    """
    Main entry point — given an ArXiv URL or ID, return a PaperSummary.

    Also ingests the paper into ChromaDB so it's available for future queries.
    """
    log.info("paper_summarise_start", input=arxiv_input)

    # 1. Extract clean ArXiv ID
    arxiv_id = _extract_arxiv_id(arxiv_input)

    # 2. Fetch from ArXiv API
    paper = _fetch_paper(arxiv_id)
    title = paper.title
    abstract = paper.summary
    authors = [a.name for a in paper.authors]
    published = paper.published.strftime("%Y-%m-%d")
    url = paper.entry_id

    log.info("paper_fetched", title=title[:60], arxiv_id=arxiv_id)

    # 3. Generate structured summary
    summary_data = _generate_summary(title, abstract)

    # 4. Ingest into ChromaDB for future retrieval (skipped if already indexed)
    nodes_ingested = 0
    already_indexed = False
    try:
        from app.ingestion.arxiv_loader import ArxivPaper
        from app.ingestion.pipeline import get_pipeline

        result = get_pipeline().ingest_papers([
            ArxivPaper(
                arxiv_id=arxiv_id,
                title=title,
                authors=authors,
                abstract=abstract,
                published=published,
                url=url,
                categories=paper.categories,
            )
        ])
        nodes_ingested = result["chunks_indexed"]
        already_indexed = result["papers_skipped"] > 0
        log.info("paper_ingested", arxiv_id=arxiv_id, nodes=nodes_ingested, already_indexed=already_indexed)

    except Exception as exc:
        log.warning("paper_ingest_failed", error=str(exc))

    return PaperSummary(
        arxiv_id=arxiv_id,
        title=title,
        authors=authors,
        published=published,
        url=url,
        abstract=abstract,
        one_liner=summary_data.get("one_liner", ""),
        problem=summary_data.get("problem", ""),
        method=summary_data.get("method", ""),
        key_results=summary_data.get("key_results", ""),
        limitations=summary_data.get("limitations", ""),
        contributions=summary_data.get("contributions", []),
        related_work=summary_data.get("related_work", []),
        nodes_ingested=nodes_ingested,
        already_indexed=already_indexed,
    )
