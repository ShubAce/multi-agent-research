"""
tests/unit/test_faithfulness_filter.py
Tests for the faithfulness filter — mocks the LLM call.
"""

import pytest
from unittest.mock import patch, MagicMock

from app.retrieval.faithfulness_filter import (
    apply_faithfulness_filter,
    _split_into_sentences,
    _build_context_string,
)


class TestSplitSentences:
    def test_basic_split(self):
        text = "Flash Attention is fast. It uses tiling techniques to optimize memory. Memory footprint is reduced."
        sentences = _split_into_sentences(text)
        assert len(sentences) == 3

    def test_short_fragments_excluded(self):
        text = "Yes. Flash Attention is an IO-aware algorithm for transformers."
        sentences = _split_into_sentences(text)
        # "Yes." is too short (<20 chars) and should be excluded
        assert all(len(s) >= 20 for s in sentences)

    def test_empty_string(self):
        assert _split_into_sentences("") == []


class TestBuildContext:
    def test_concatenates_contexts(self):
        contexts = ["context one", "context two"]
        result = _build_context_string(contexts)
        assert "context one" in result
        assert "context two" in result

    def test_truncates_long_context(self):
        long_context: list[str] = ["x" * 4000]
        result = _build_context_string(long_context, max_chars=100)
        assert len(result) <= 130  # truncated + "... truncated" tag


class TestApplyFaithfulnessFilter:
    def test_disabled_returns_original(self, monkeypatch):
        from app.core.config import get_settings
        settings = get_settings()
        monkeypatch.setattr(settings, "faithfulness_filter_enabled", False)

        result = apply_faithfulness_filter(
            answer="Any answer here.",
            source_contexts=["Some source context."],
        )
        assert result.filtered_answer == "Any answer here."
        assert result.confidence_score == 1.0

    def test_no_context_returns_original(self):
        result = apply_faithfulness_filter(
            answer="Some answer about Flash Attention.",
            source_contexts=[],
        )
        # With no context, filter can't check — returns original
        assert result.filtered_answer == "Some answer about Flash Attention."

    @patch("app.retrieval.faithfulness_filter._batch_check_sentences")
    def test_supported_sentences_kept(self, mock_check):
        mock_check.return_value = ["supported", "supported"]
        result = apply_faithfulness_filter(
            answer="Flash Attention uses tiling to reduce memory. It achieves linear memory complexity.",
            source_contexts=["Flash Attention uses tiling algorithms."],
        )
        assert result.removed_sentences == 0
        assert result.confidence_score > 0.8

    @patch("app.retrieval.faithfulness_filter._batch_check_sentences")
    def test_unsupported_sentences_removed(self, mock_check):
        mock_check.return_value = ["unsupported", "unsupported"]
        result = apply_faithfulness_filter(
            answer="Flash Attention was invented in 1995. It uses quantum computing.",
            source_contexts=["Flash Attention is a modern GPU algorithm from 2022."],
        )
        assert result.removed_sentences > 0

    @patch("app.retrieval.faithfulness_filter._batch_check_sentences")
    def test_partial_sentences_flagged(self, mock_check):
        mock_check.return_value = ["partial"]
        result = apply_faithfulness_filter(
            answer="Flash Attention improves training speed significantly.",
            source_contexts=["Flash Attention reduces memory usage."],
        )
        assert result.flagged_sentences > 0
        assert "*(based on inference from sources)*" in result.filtered_answer

    @patch("app.retrieval.faithfulness_filter._batch_check_sentences")
    def test_fallback_when_all_removed(self, mock_check):
        """If everything is removed, return the original answer as fallback."""
        mock_check.return_value = ["unsupported"]
        result = apply_faithfulness_filter(
            answer="This answer has content that will be removed by the filter.",
            source_contexts=["Completely unrelated source."],
        )
        # Should fall back to original, not return empty string
        assert len(result.filtered_answer) > 0
