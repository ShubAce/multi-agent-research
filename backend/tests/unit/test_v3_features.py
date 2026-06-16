"""
tests/unit/test_v3_features.py
Tests for session memory, critic, and paper summariser.
"""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock


# ── ArXiv ID extraction ───────────────────────────────────────────────────────

class TestArxivIdExtraction:
    def test_full_abs_url(self):
        from app.services.paper_summariser import _extract_arxiv_id
        result = _extract_arxiv_id("https://arxiv.org/abs/2205.14135")
        assert result == "2205.14135"

    def test_pdf_url(self):
        from app.services.paper_summariser import _extract_arxiv_id
        result = _extract_arxiv_id("https://arxiv.org/pdf/2205.14135")
        assert result == "2205.14135"

    def test_plain_id(self):
        from app.services.paper_summariser import _extract_arxiv_id
        result = _extract_arxiv_id("2205.14135")
        assert result == "2205.14135"

    def test_versioned_id_stripped(self):
        from app.services.paper_summariser import _extract_arxiv_id
        result = _extract_arxiv_id("2205.14135v2")
        assert result == "2205.14135"

    def test_invalid_raises(self):
        from app.services.paper_summariser import _extract_arxiv_id
        with pytest.raises(ValueError):
            _extract_arxiv_id("not a url or id")


# ── Critic scoring ────────────────────────────────────────────────────────────

class TestCriticParsing:
    def test_clean_json(self):
        from app.agents.critic import _parse_critic_response
        text = '{"score": 8, "reason": "Good answer.", "improved_queries": []}'
        result = _parse_critic_response(text)
        assert result["score"] == 8
        assert result["improved_queries"] == []

    def test_low_score_with_queries(self):
        from app.agents.critic import _parse_critic_response
        text = '{"score": 4, "reason": "Missing key details.", "improved_queries": ["Flash Attention algorithm details", "FlashAttention complexity analysis"]}'
        result = _parse_critic_response(text)
        assert result["score"] == 4
        assert len(result["improved_queries"]) == 2

    def test_fallback_extracts_score(self):
        from app.agents.critic import _parse_critic_response
        text = "I would rate this answer a 7 out of 10."
        result = _parse_critic_response(text)
        assert result["score"] == 7


class TestCriticNode:
    def _make_state(self, answer: str, retry_count: int = 0) -> dict:
        return {
            "query": "What is Flash Attention?",
            "session_id": "test",
            "messages": [],
            "conversation_history": "",
            "sub_tasks": ["What is Flash Attention?"],
            "routing": {"What is Flash Attention?": "rag"},
            "web_results": [],
            "rag_results": [],
            "source_contexts": ["Flash Attention uses tiling."],
            "final_answer": answer,
            "citations": [],
            "agents_used": ["planner", "rag_agent", "synthesiser"],
            "confidence_score": 0.9,
            "critic_score": None,
            "critic_feedback": None,
            "retry_count": retry_count,
            "error": None,
            "iteration_count": 1,
        }

    @patch("app.agents.critic.get_fast_llm")
    def test_good_answer_no_retry(self, mock_llm):
        from app.agents.critic import critic_node, should_retry
        mock_llm.return_value.invoke.return_value.content = \
            '{"score": 8, "reason": "Great answer.", "improved_queries": []}'

        state = self._make_state("Flash Attention is an IO-aware algorithm.")
        result = critic_node(state)
        assert result["critic_score"] == 8

        updated = {**state, **result}
        assert should_retry(updated) == "end"

    @patch("app.agents.critic.get_fast_llm")
    def test_weak_answer_triggers_retry(self, mock_llm):
        from app.agents.critic import critic_node, should_retry
        mock_llm.return_value.invoke.return_value.content = \
            '{"score": 3, "reason": "Too vague.", "improved_queries": ["Flash Attention tiling algorithm GPU"]}'

        state = self._make_state("Flash Attention is fast.", retry_count=0)
        result = critic_node(state)
        assert result["critic_score"] == 3
        assert len(result["sub_tasks"]) == 1

        updated = {**state, **result}
        assert should_retry(updated) == "retry"

    @patch("app.agents.critic.get_fast_llm")
    def test_max_retries_prevents_loop(self, mock_llm):
        from app.agents.critic import critic_node, should_retry
        # Even with a bad score, retry_count=1 should prevent another retry
        state = self._make_state("Weak answer.", retry_count=1)
        result = critic_node(state)

        updated = {**state, **result}
        assert should_retry(updated) == "end"


# ── Memory node ───────────────────────────────────────────────────────────────

class TestMemoryNode:
    def test_empty_session_returns_empty_history(self):
        from app.agents.memory_node import memory_injection_node
        state = {
            "query": "test",
            "session_id": "",
            "messages": [],
            "conversation_history": "",
            "sub_tasks": [], "routing": {}, "web_results": [],
            "rag_results": [], "source_contexts": [], "final_answer": None,
            "citations": [], "agents_used": [], "confidence_score": None,
            "critic_score": None, "critic_feedback": None,
            "retry_count": 0, "error": None, "iteration_count": 0,
        }
        result = memory_injection_node(state)
        assert result["conversation_history"] == ""

    @patch("app.agents.memory_node.asyncio.run")
    def test_memory_failure_returns_empty(self, mock_run):
        from app.agents.memory_node import memory_injection_node
        mock_run.side_effect = Exception("Redis connection failed")
        state = {"session_id": "test-session",
                 "query": "test", "messages": [],
                 "conversation_history": "", "sub_tasks": [],
                 "routing": {}, "web_results": [], "rag_results": [],
                 "source_contexts": [], "final_answer": None, "citations": [],
                 "agents_used": [], "confidence_score": None,
                 "critic_score": None, "critic_feedback": None,
                 "retry_count": 0, "error": None, "iteration_count": 0}
        result = memory_injection_node(state)
        # Should not crash — graceful fallback
        assert result["conversation_history"] == ""
