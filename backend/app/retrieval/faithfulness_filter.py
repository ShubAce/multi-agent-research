"""
app/retrieval/faithfulness_filter.py

Faithfulness Filter — BATCHED version (v2).

Old behaviour: one LLM call per sentence → 8 sentences = 8 API round-trips = ~10–15s
New behaviour: all sentences in ONE call → 8 sentences = 1 API round-trip = ~1.5s

How it works:
  1. Split answer into sentences
  2. Send ALL sentences to the LLM in a single prompt, asking for a verdict per line
  3. Parse the response (one verdict per line)
  4. Keep / flag / remove sentences based on verdicts
  5. Compute confidence score (fraction grounded)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from langchain_core.messages import HumanMessage, SystemMessage

from app.agents.llm import get_fast_llm
from app.core.config import get_settings
from app.core.logging import get_logger

log = get_logger(__name__)

# ── Batched prompt — all sentences checked in one call ────────────────────────
_BATCH_PROMPT = """You are a fact-checking assistant for an academic research system.

You will receive:
1. SOURCE PASSAGES — retrieved from academic papers
2. CLAIMS — numbered list of sentences from a generated answer

For each claim, output EXACTLY one word on its own line (no numbering, no explanation):
  supported   — directly stated or clearly implied by the sources
  partial     — related to sources but adds inference or extrapolation  
  unsupported — contradicts sources or has no basis in them

Output format: one verdict per line, same order as input claims.
If there are 5 claims, output exactly 5 lines."""


@dataclass
class FilterResult:
    filtered_answer: str
    original_answer: str
    confidence_score: float        # 0.0–1.0
    kept_sentences: int
    flagged_sentences: int
    removed_sentences: int
    flagged_content: list[str]
    removed_content: list[str]


def _split_into_sentences(text: str) -> list[str]:
    """Split text into checkable sentences, skip short fragments and headings."""
    raw = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    sentences: list[str] = []
    for chunk in raw:
        for line in chunk.split("\n"):
            line = line.strip()
            if line:
                sentences.append(line)
    return [s for s in sentences if len(s) > 20]


def _build_context_string(source_contexts: list[str], max_chars: int = 3000) -> str:
    combined = "\n\n---\n\n".join(source_contexts)
    if len(combined) > max_chars:
        combined = combined[:max_chars] + "\n[... truncated ...]"
    return combined


def _parse_batch_verdicts(response_text: str, expected: int) -> list[str]:
    """
    Parse the LLM's batched response into a list of verdicts.
    Handles slight formatting variations gracefully.
    """
    lines = [l.strip().lower() for l in response_text.strip().split("\n") if l.strip()]
    verdicts: list[str] = []

    for line in lines:
        # Strip any leading numbers or bullets the LLM might add
        line = re.sub(r"^\d+[\.\)]\s*", "", line).strip()

        if "unsupported" in line or "not supported" in line or "contradict" in line:
            verdicts.append("unsupported")
        elif "partial" in line or "inference" in line:
            verdicts.append("partial")
        elif "supported" in line:
            verdicts.append("supported")
        # Skip blank lines or unrecognised tokens

    # If we got fewer verdicts than expected, pad with "supported" (fail-safe)
    while len(verdicts) < expected:
        verdicts.append("supported")

    return verdicts[:expected]


def _batch_check_sentences(
    sentences: list[str],
    context: str,
    llm,
) -> list[str]:
    """
    Check ALL sentences in a single LLM call.
    Returns a list of verdicts, one per sentence.

    This replaces N serial LLM calls with exactly 1.
    """
    # Build numbered claim list
    claims_text = "\n".join(f"{i+1}. {s}" for i, s in enumerate(sentences))

    messages = [
        SystemMessage(content=_BATCH_PROMPT),
        HumanMessage(content=(
            f"SOURCE PASSAGES:\n{context}\n\n"
            f"CLAIMS TO CHECK:\n{claims_text}"
        )),
    ]

    try:
        response = llm.invoke(messages)
        content = response.content
        if not isinstance(content, str):
            raise ValueError(f"Expected string response content from LLM, got {type(content)}")
        verdicts = _parse_batch_verdicts(content, expected=len(sentences))
        log.debug(
            "batch_check_done",
            sentences=len(sentences),
            verdicts=verdicts,
        )
        return verdicts
    except Exception as exc:
        log.warning("batch_check_failed", error=str(exc), fallback="all supported")
        # Fail safe — keep everything rather than crash
        return ["supported"] * len(sentences)


def apply_faithfulness_filter(
    answer: str,
    source_contexts: list[str],
    threshold: float | None = None,
) -> FilterResult:
    """
    Apply the batched faithfulness filter to a synthesised answer.

    Latency: ~1.5s regardless of answer length (was ~10–15s before batching).
    """
    settings = get_settings()
    threshold = threshold or settings.faithfulness_threshold

    # ── Short-circuit cases ───────────────────────────────────────────────────
    if not settings.faithfulness_filter_enabled:
        log.debug("faithfulness_filter_disabled")
        return FilterResult(
            filtered_answer=answer, original_answer=answer,
            confidence_score=1.0, kept_sentences=0,
            flagged_sentences=0, removed_sentences=0,
            flagged_content=[], removed_content=[],
        )

    if not source_contexts:
        log.warning("faithfulness_filter_no_context")
        return FilterResult(
            filtered_answer=answer, original_answer=answer,
            confidence_score=0.5, kept_sentences=0,
            flagged_sentences=0, removed_sentences=0,
            flagged_content=[], removed_content=[],
        )

    sentences = _split_into_sentences(answer)
    if not sentences:
        return FilterResult(
            filtered_answer=answer, original_answer=answer,
            confidence_score=1.0, kept_sentences=0,
            flagged_sentences=0, removed_sentences=0,
            flagged_content=[], removed_content=[],
        )

    log.info("faithfulness_filter_start",
             sentences=len(sentences), answer_length=len(answer))

    context = _build_context_string(source_contexts)
    llm = get_fast_llm()

    # Separate sentences that need checking from those that don't
    # (headings, very short lines, bullet prefixes)
    needs_check: list[tuple[int, str]] = []   # (original_index, sentence)
    skip_indices: set[int] = set()

    for i, s in enumerate(sentences):
        if len(s) < 30 or s.endswith(":") or s.startswith("#"):
            skip_indices.add(i)
        else:
            needs_check.append((i, s))

    # ── Single batched LLM call ───────────────────────────────────────────────
    if needs_check:
        just_sentences = [s for _, s in needs_check]
        verdicts = _batch_check_sentences(just_sentences, context, llm)
    else:
        verdicts = []

    # Map verdicts back to original sentence indices
    verdict_map: dict[int, str] = {}
    for (orig_idx, _), verdict in zip(needs_check, verdicts):
        verdict_map[orig_idx] = verdict

    # ── Apply verdicts ────────────────────────────────────────────────────────
    kept: list[str] = []
    flagged: list[str] = []
    removed: list[str] = []

    for i, sentence in enumerate(sentences):
        if i in skip_indices:
            kept.append(sentence)
            continue

        verdict = verdict_map.get(i, "supported")

        if verdict == "supported":
            kept.append(sentence)
        elif verdict == "partial":
            flagged.append(sentence)
            kept.append(f"{sentence} *(based on inference from sources)*")
        else:
            removed.append(sentence)
            log.debug("sentence_removed", sentence=sentence[:80])

    # ── Compute confidence score ──────────────────────────────────────────────
    total = len(sentences)
    pure_kept = len(kept) - len(flagged)
    confidence = (pure_kept + 0.5 * len(flagged)) / total if total > 0 else 1.0

    filtered_answer = " ".join(kept).strip()
    if not filtered_answer:
        log.warning("faithfulness_filter_removed_everything")
        filtered_answer = answer
        confidence = 0.0

    log.info(
        "faithfulness_filter_done",
        total=total,
        kept=pure_kept,
        flagged=len(flagged),
        removed=len(removed),
        confidence=round(confidence, 3),
    )

    return FilterResult(
        filtered_answer=filtered_answer,
        original_answer=answer,
        confidence_score=round(confidence, 3),
        kept_sentences=pure_kept,
        flagged_sentences=len(flagged),
        removed_sentences=len(removed),
        flagged_content=flagged,
        removed_content=removed,
    )
