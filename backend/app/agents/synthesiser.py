"""
app/agents/synthesiser.py

The Synthesiser receives every retrieved passage (papers + web) and produces:
  - One numbered source list — [n] in the answer is the n-th entry, so the UI
    can link citations to sources
  - A Markdown answer citing those numbers
  - A fact-check report from the faithfulness filter

Uses the main 70B model — this is the step that needs the most reasoning.
"""

from __future__ import annotations

import re
import time

from langchain_core.messages import HumanMessage, SystemMessage

from app.agents.llm import get_llm
from app.agents.state import AgentState
from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.metrics import agent_duration_seconds, agent_runs_total
from app.ingestion.arxiv_loader import base_arxiv_id

log = get_logger(__name__)

_SYSTEM_PROMPT = """You are a research synthesiser writing for a technical reader.
Answer the question using ONLY the numbered sources provided.

Format (Markdown):
- Open with a direct 2–3 sentence answer. No heading before it.
- Follow with 2–4 short sections under "### " headings covering the key aspects.
- Finish with "### Key takeaways": 3–5 concise bullet points.
- Use **bold** sparingly for key terms. Use a table only when comparing several items.
- No LaTeX: write maths in plain text or Unicode, e.g. O(N²), d_k, softmax(QKᵀ/√d).

Citations:
- Support every factual claim with source numbers in square brackets, e.g. "... memory [2]." or "[1][3]".
  Use exactly that plain form — no 【】 brackets, † marks or line ranges.
- Only cite numbers that appear in the source list. Never invent sources.
- Prefer academic papers for technical depth and web sources for recent developments.
- Quote figures (percentages, scores, sizes, dates) only when they appear in a source; never estimate them.
- If sources disagree, say so and explain which is more credible.
- If the sources do not cover part of the question, say that plainly instead of guessing.
- Do not add a references or bibliography section — the interface lists sources separately."""

_NO_SOURCES_PROMPT = """You are a research assistant. No relevant sources were found for
this question in the paper knowledge base or on the web.

In 3–5 sentences of Markdown:
1. Say clearly that no supporting sources were found.
2. Give only a brief, high-level orientation that you are confident about, labelled as general background (not source-backed).
3. Suggest a next step: add relevant papers to the knowledge base, enable web search, or rephrase the question."""


def _paper_key(passage: dict) -> str:
    return base_arxiv_id(passage.get("arxiv_id") or "") or passage.get("title", "")


def build_sources(rag_results: list[dict], web_results: list[dict], max_sources: int) -> list[dict]:
    """
    Merge RAG passages (grouped per paper) and web results into one numbered list.
    Papers come first; web results get up to half the slots when both exist.
    """
    papers: dict[str, dict] = {}
    for result in rag_results:
        for p in result.get("passages", []):
            key = _paper_key(p)
            if not key:
                continue
            src = papers.get(key)
            if src is None:
                papers[key] = {
                    "type": "paper",
                    "title": p.get("title", "Untitled paper"),
                    "authors": p.get("authors", []),
                    "url": p.get("url", ""),
                    "arxiv_id": base_arxiv_id(p.get("arxiv_id", "")),
                    "published": p.get("published", ""),
                    "relevance_score": p.get("relevance_score", 0.0),
                    "passages": [p.get("text", "")],
                }
            else:
                src["relevance_score"] = max(src["relevance_score"], p.get("relevance_score", 0.0))
                if p.get("text") and p["text"] not in src["passages"]:
                    src["passages"].append(p["text"])

    web: dict[str, dict] = {}
    for r in web_results:
        url = r.get("url")
        if not url or url in web:
            continue
        web[url] = {
            "type": "web",
            "title": r.get("title") or url,
            "authors": [],
            "url": url,
            "arxiv_id": "",
            "published": "",
            "relevance_score": float(r.get("score") or 0.0),
            "passages": [r.get("content", "")],
        }

    paper_list = sorted(papers.values(), key=lambda s: s["relevance_score"], reverse=True)
    web_list = sorted(web.values(), key=lambda s: s["relevance_score"], reverse=True)

    n_web = min(len(web_list), max_sources // 2 if paper_list else max_sources)
    n_papers = min(len(paper_list), max_sources - n_web)
    n_web = min(len(web_list), max_sources - n_papers)

    sources = paper_list[:n_papers] + web_list[:n_web]
    for i, s in enumerate(sources, start=1):
        s["id"] = i
        s["relevance_score"] = round(min(max(float(s["relevance_score"]), 0.0), 1.0), 4)
    return sources


def _source_text(source: dict, max_chars: int = 1200) -> str:
    return "\n".join(source["passages"])[:max_chars]


def _format_sources(sources: list[dict]) -> str:
    blocks = []
    for s in sources:
        meta = []
        if s["authors"]:
            meta.append(", ".join(s["authors"][:3]) + (" et al." if len(s["authors"]) > 3 else ""))
        if s["published"]:
            meta.append(s["published"][:4])
        header = f"[{s['id']}] ({s['type']}) {s['title']}"
        if meta:
            header += f" — {' · '.join(meta)}"
        blocks.append(f"{header}\n{_source_text(s)}")
    return "\n\n".join(blocks)


def _public_citation(source: dict) -> dict:
    """What the UI receives — no full passage text."""
    excerpt = " ".join(source["passages"][0].split())[:320] if source["passages"] else ""
    return {k: v for k, v in source.items() if k != "passages"} | {"excerpt": excerpt}


# Optional leading space is captured so a dropped marker leaves no gap
_CITE_GROUP = re.compile(r"(\s*)\[(\d{1,2}(?:\s*[,–-]\s*\d{1,2})*)\](?!\()")


# gpt-oss was trained with a browsing tool and often cites as 【3†L1-L4】 or [3†source]
_TOOL_MARKER = re.compile(r"【(\d{1,2})(?:†[^】]*)?】|\[(\d{1,2})†[^\]]*\]")


def normalise_citation_markers(text: str) -> str:
    """Rewrite tool-style citation markers to the plain [n] form."""
    return _TOOL_MARKER.sub(lambda m: f"[{m.group(1) or m.group(2)}]", text)


def _expand_group(inner: str) -> list[int]:
    nums: list[int] = []
    for part in inner.split(","):
        bounds = [int(x) for x in re.split(r"[–-]", part) if x.strip().isdigit()]
        if len(bounds) == 2 and 0 < bounds[1] - bounds[0] <= 10:
            nums.extend(range(bounds[0], bounds[1] + 1))
        elif bounds:
            nums.append(bounds[0])
    return nums


def renumber_citations(
    answer: str, sources: list[dict], extra_texts: list[str] | None = None
) -> tuple[str, list[dict], list[str]]:
    """
    Keep only the sources the answer actually cites, numbered in reading order
    ([1] is the first citation a reader meets). Markers pointing at no source
    are dropped. `extra_texts` (e.g. fact-check sentences) get the same mapping.
    """
    valid = {s["id"] for s in sources}
    mapping: dict[int, int] = {}
    for match in _CITE_GROUP.finditer(answer):
        for n in _expand_group(match.group(2)):
            if n in valid and n not in mapping:
                mapping[n] = len(mapping) + 1
    if not mapping:
        return answer, sources, extra_texts or []

    def repl(match: re.Match) -> str:
        nums = [mapping[n] for n in _expand_group(match.group(2)) if n in mapping]
        return match.group(1) + "".join(f"[{n}]" for n in dict.fromkeys(nums)) if nums else ""

    by_id = {s["id"]: s for s in sources}
    cited = [{**by_id[old], "id": new} for old, new in sorted(mapping.items(), key=lambda kv: kv[1])]
    return (
        _CITE_GROUP.sub(repl, answer),
        cited,
        [_CITE_GROUP.sub(repl, t) for t in extra_texts or []],
    )


def _fallback_answer(sources: list[dict]) -> str:
    if not sources:
        return "I couldn't generate an answer and found no relevant sources. Please try again."
    lines = [
        "I couldn't write a full synthesis this time, but these sources look relevant:",
        "",
        *[f"- **{s['title']}** [{s['id']}]" for s in sources],
    ]
    return "\n".join(lines)


def synthesiser_node(state: AgentState) -> dict:
    """
    LangGraph node: Synthesiser.

    Input state keys used:  query, conversation_history, sub_tasks, rag_results, web_results
    Output state keys set:  final_answer, citations, confidence_score, fact_check, agents_used
    """
    t0 = time.perf_counter()
    settings = get_settings()
    query = state["query"]
    log.info("synthesiser_start", query=query[:80])

    sources = build_sources(
        state.get("rag_results", []), state.get("web_results", []), settings.max_sources
    )
    agents_used = state.get("agents_used", []) + ["synthesiser"]

    parts = []
    history = state.get("conversation_history", "")
    if history:
        parts.append(f"{history}\n(Use the conversation only to resolve references in the question.)")
    parts.append(f"Question: {query}")
    sub_tasks = state.get("sub_tasks", [])
    if sub_tasks:
        parts.append("Aspects investigated:\n" + "\n".join(f"- {t}" for t in sub_tasks))
    if sources:
        parts.append(f"SOURCES:\n\n{_format_sources(sources)}")

    messages = [
        SystemMessage(content=_SYSTEM_PROMPT if sources else _NO_SOURCES_PROMPT),
        HumanMessage(content="\n\n".join(parts)),
    ]

    try:
        response = get_llm().invoke(messages)
        content = response.content
        if not isinstance(content, str):
            raise ValueError(f"Expected string response content from LLM, got {type(content)}")
        raw_answer = normalise_citation_markers(content.strip())

        confidence: float | None = None
        fact_check: dict | None = None
        final_answer = raw_answer

        if sources:
            # Check each sentence against exactly the passages the model saw
            from app.retrieval.faithfulness_filter import apply_faithfulness_filter

            result = apply_faithfulness_filter(
                answer=raw_answer,
                source_contexts=[f"[{s['id']}] {s['title']}\n{_source_text(s)}" for s in sources],
            )
            final_answer = result.filtered_answer
            confidence = result.confidence_score
            if result.checked_sentences:
                fact_check = result.report()

            # Show only cited sources, numbered in reading order
            extra = (fact_check or {}).get("inferred_sentences", []) + (fact_check or {}).get("removed_sentences", [])
            final_answer, sources, extra = renumber_citations(final_answer, sources, extra)
            if fact_check:
                n_inferred = len(fact_check["inferred_sentences"])
                fact_check["inferred_sentences"] = extra[:n_inferred]
                fact_check["removed_sentences"] = extra[n_inferred:]

        elapsed = (time.perf_counter() - t0) * 1000
        log.info(
            "synthesiser_done",
            answer_length=len(final_answer),
            sources=len(sources),
            confidence=confidence,
            duration_ms=round(elapsed),
        )
        agent_runs_total.labels(agent_name="synthesiser", status="success").inc()
        agent_duration_seconds.labels(agent_name="synthesiser").observe(elapsed / 1000)

        return {
            "final_answer":     final_answer,
            "citations":        [_public_citation(s) for s in sources],
            "confidence_score": confidence,
            "fact_check":       fact_check,
            "agents_used":      agents_used,
        }

    except Exception as exc:
        log.error("synthesiser_error", error=str(exc))
        agent_runs_total.labels(agent_name="synthesiser", status="error").inc()
        return {
            "final_answer":     _fallback_answer(sources),
            "citations":        [_public_citation(s) for s in sources],
            "confidence_score": None,
            "fact_check":       None,
            "agents_used":      agents_used,
            "error":            f"Synthesiser failed: {exc}",
        }
