"""
app/retrieval/faithfulness_filter.py

Faithfulness Filter — BATCHED, structure-preserving (v3).

How it works:
  1. Walk the answer line by line (Markdown headings, lists and paragraphs)
  2. Split prose lines into sentences; collect the ones worth checking
  3. Send ALL of them to the LLM in a single prompt, one verdict per claim
  4. Drop unsupported sentences in place — headings, bullets and paragraph
     breaks survive (v2 joined everything into one line, destroying Markdown)
  5. Compute confidence = (supported + 0.5 × partial) / checked
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from langchain_core.messages import HumanMessage, SystemMessage

from app.agents.llm import get_fast_llm
from app.core.config import get_settings
from app.core.logging import get_logger

log = get_logger(__name__)

# ── Batched prompt — all sentences checked in one call ────────────────────────
_BATCH_PROMPT = """You are a fact-checking assistant for an academic research system.

You will receive:
1. SOURCE PASSAGES — numbered sources retrieved from papers and the web
2. CLAIMS — numbered list of sentences from a generated answer

For each claim, output EXACTLY one word on its own line (no numbering, no explanation):
  supported   — directly stated or clearly implied by the sources
  partial     — related to sources but adds inference or extrapolation
  unsupported — contradicts sources or has no basis in them

Output format: one verdict per line, same order as input claims.
If there are 5 claims, output exactly 5 lines."""

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\[(\"*])")
_LIST_PREFIX = re.compile(r"^(\s*(?:>\s*)?(?:(?:[-*+]|\d+[.)])\s+)?)")
_MIN_CHECK_LEN = 30


@dataclass
class FilterResult:
    filtered_answer: str
    original_answer: str
    confidence_score: float        # 0.0–1.0
    kept_sentences: int
    flagged_sentences: int
    removed_sentences: int
    flagged_content: list[str] = field(default_factory=list)
    removed_content: list[str] = field(default_factory=list)
    checked_sentences: int = 0
    unverified: bool = False       # nothing could be verified; answer returned unchanged

    def report(self) -> dict:
        """Compact summary for the UI's fact-check panel."""
        return {
            "checked": self.checked_sentences,
            "supported": self.kept_sentences,
            "inferred": self.flagged_sentences,
            "removed": self.removed_sentences,
            "unverified": self.unverified,
            "inferred_sentences": self.flagged_content[:10],
            "removed_sentences": self.removed_content[:10],
        }


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


def _build_context_string(source_contexts: list[str], max_chars: int = 8000) -> str:
    combined = "\n\n---\n\n".join(source_contexts)
    if len(combined) > max_chars:
        combined = combined[:max_chars] + "\n[... truncated ...]"
    return combined


def _parse_batch_verdicts(response_text: str, expected: int) -> list[str]:
    """
    Parse the LLM's batched response into a list of verdicts.
    Handles slight formatting variations gracefully.
    """
    lines = [ln.strip().lower() for ln in response_text.strip().split("\n") if ln.strip()]
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


def _batch_check_sentences(sentences: list[str], context: str, llm) -> list[str]:
    """
    Check ALL sentences in a single LLM call.
    Returns a list of verdicts, one per sentence.
    """
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
        log.debug("batch_check_done", sentences=len(sentences), verdicts=verdicts)
        return verdicts
    except Exception as exc:
        log.warning("batch_check_failed", error=str(exc), fallback="all supported")
        # Fail safe — keep everything rather than crash
        return ["supported"] * len(sentences)


def _is_checkable(sentence: str) -> bool:
    plain = re.sub(r"\[\d+\]|[*_`]", "", sentence).strip()
    return len(plain) >= _MIN_CHECK_LEN and not plain.endswith(":")


def _parse_answer(answer: str) -> list[dict]:
    """
    Break the answer into lines:
      {"raw": line}                       structure (headings, code, table header) — kept as is
      {"row": line, "claim": text}        table data row — checked as one claim
      {"prefix": "- ", "sentences": [..]} prose — checked sentence by sentence
    Table rows are checked because they are where models tend to put invented figures.
    """
    parsed: list[dict] = []
    in_code = False
    table_row = 0
    for line in answer.split("\n"):
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code = not in_code
            parsed.append({"raw": line})
            continue
        if not in_code and stripped.startswith("|"):
            is_separator = set(stripped) <= set("|-: ")
            if table_row == 0 or is_separator:
                parsed.append({"raw": line})
            else:
                cells = [c.strip() for c in stripped.strip("|").split("|") if c.strip()]
                parsed.append({"row": line, "claim": " — ".join(cells)})
            table_row += 1
            continue
        table_row = 0
        if in_code or not stripped or stripped.startswith("#") or set(stripped) <= set("-*_ "):
            parsed.append({"raw": line})
            continue
        prefix = _LIST_PREFIX.match(line).group(1)  # type: ignore[union-attr]
        body = line[len(prefix):]
        parsed.append({"prefix": prefix, "sentences": [s for s in _SENTENCE_SPLIT.split(body) if s]})
    return parsed


def _drop_empty_tables(lines: list[str]) -> list[str]:
    """Remove tables whose data rows were all filtered out (header + separator left)."""
    out: list[str] = []
    block: list[str] = []
    for line in [*lines, ""]:
        if line.strip().startswith("|"):
            block.append(line)
            continue
        if len(block) > 2:
            out.extend(block)
        block = []
        out.append(line)
    return out[:-1]


def _tidy(lines: list[str]) -> str:
    """Drop empty tables and headings left without content; collapse blank-line runs."""
    lines = _drop_empty_tables(lines)
    out: list[str] = []
    for i, line in enumerate(lines):
        if line.strip().startswith("#"):
            nxt = next((ln for ln in lines[i + 1:] if ln.strip()), None)
            if nxt is None or nxt.strip().startswith("#"):
                continue
        if not line.strip() and out and not out[-1].strip():
            continue
        out.append(line)
    return "\n".join(out).strip()


def _passthrough(answer: str, confidence: float) -> FilterResult:
    return FilterResult(
        filtered_answer=answer, original_answer=answer,
        confidence_score=confidence, kept_sentences=0,
        flagged_sentences=0, removed_sentences=0,
    )


def apply_faithfulness_filter(
    answer: str,
    source_contexts: list[str],
    threshold: float | None = None,
) -> FilterResult:
    """
    Apply the batched faithfulness filter to a synthesised answer.
    One LLM call regardless of answer length.
    """
    settings = get_settings()

    # ── Short-circuit cases ───────────────────────────────────────────────────
    if not settings.faithfulness_filter_enabled:
        log.debug("faithfulness_filter_disabled")
        return _passthrough(answer, 1.0)

    if not source_contexts:
        log.warning("faithfulness_filter_no_context")
        return _passthrough(answer, 0.5)

    parsed = _parse_answer(answer)
    to_check: list[tuple[int, int, str]] = []
    for li, line in enumerate(parsed):
        if "claim" in line and _is_checkable(line["claim"]):
            to_check.append((li, 0, line["claim"]))
        for si, s in enumerate(line.get("sentences", [])):
            if _is_checkable(s):
                to_check.append((li, si, s))
    if not to_check:
        return _passthrough(answer, 1.0)

    log.info("faithfulness_filter_start", sentences=len(to_check), answer_length=len(answer))

    verdicts = _batch_check_sentences(
        [s for _, _, s in to_check],
        _build_context_string(source_contexts),
        get_fast_llm(),
    )
    verdict_map = {(li, si): v for (li, si, _), v in zip(to_check, verdicts, strict=False)}

    # ── Rebuild the answer without unsupported sentences ──────────────────────
    supported: list[str] = []
    flagged: list[str] = []
    removed: list[str] = []
    out_lines: list[str] = []

    def _record(verdict: str | None, text: str) -> bool:
        """Tally a verdict; returns False when the text must be dropped."""
        if verdict == "unsupported":
            removed.append(text.strip())
            log.debug("sentence_removed", sentence=text[:80])
            return False
        if verdict == "partial":
            flagged.append(text.strip())
        elif verdict == "supported":
            supported.append(text.strip())
        return True

    for li, line in enumerate(parsed):
        if "raw" in line:
            out_lines.append(line["raw"])
            continue
        if "row" in line:
            if _record(verdict_map.get((li, 0)), line["claim"]):
                out_lines.append(line["row"])
            continue
        kept = [
            sentence
            for si, sentence in enumerate(line["sentences"])
            if _record(verdict_map.get((li, si)), sentence)
        ]
        if kept:
            out_lines.append(line["prefix"] + " ".join(kept))

    checked = len(to_check)

    if removed and not supported and not flagged:
        # Deleting every claim would leave no answer. Return it unchanged but
        # flag it as unverified, so the UI warns instead of claiming removals.
        log.warning("faithfulness_filter_nothing_verified", checked=checked)
        return FilterResult(
            filtered_answer=answer, original_answer=answer,
            confidence_score=0.0, kept_sentences=0,
            flagged_sentences=0, removed_sentences=0,
            checked_sentences=checked, unverified=True,
        )

    confidence = (len(supported) + 0.5 * len(flagged)) / checked
    filtered_answer = _tidy(out_lines)

    if not filtered_answer:
        log.warning("faithfulness_filter_removed_everything")
        filtered_answer = answer
        confidence = 0.0

    log.info(
        "faithfulness_filter_done",
        checked=checked,
        supported=len(supported),
        flagged=len(flagged),
        removed=len(removed),
        confidence=round(confidence, 3),
    )

    return FilterResult(
        filtered_answer=filtered_answer,
        original_answer=answer,
        confidence_score=round(confidence, 3),
        kept_sentences=len(supported),
        flagged_sentences=len(flagged),
        removed_sentences=len(removed),
        flagged_content=flagged,
        removed_content=removed,
        checked_sentences=checked,
    )
