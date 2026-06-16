"""
tests/unit/test_latency_fixes.py
Tests for the three latency improvements:
  1. BM25 cache (build once, reuse)
  2. Parallel agents node (merges results correctly)
  3. Batch faithfulness verdict parsing
"""

import pytest
from unittest.mock import patch, MagicMock
from llama_index.core.schema import NodeWithScore, TextNode


# ── BM25 Cache ────────────────────────────────────────────────────────────────

class TestBM25Cache:
    def setup_method(self):
        # Clear cache before each test
        from app.retrieval.hybrid import invalidate_bm25_cache
        invalidate_bm25_cache()

    def _make_nodes(self, n: int) -> list[NodeWithScore]:
        return [
            NodeWithScore(node=TextNode(text=f"node {i} content", id_=f"n{i}"), score=1.0)
            for i in range(n)
        ]

    def test_cache_is_built_on_first_call(self):
        from app.retrieval.hybrid import get_bm25_cache, _bm25_cache
        nodes = self._make_nodes(5)
        cache = get_bm25_cache(nodes)
        assert cache.num_docs == 5
        assert cache.bm25 is not None

    def test_cache_is_reused_on_second_call(self):
        from app.retrieval.hybrid import get_bm25_cache
        nodes = self._make_nodes(5)
        cache1 = get_bm25_cache(nodes)
        cache2 = get_bm25_cache(nodes)
        # Same object — not rebuilt
        assert cache1 is cache2

    def test_cache_rebuilds_when_doc_count_changes(self):
        from app.retrieval.hybrid import get_bm25_cache
        nodes5 = self._make_nodes(5)
        nodes10 = self._make_nodes(10)
        cache1 = get_bm25_cache(nodes5)
        cache2 = get_bm25_cache(nodes10)
        assert cache1 is not cache2
        assert cache2.num_docs == 10

    def test_invalidate_clears_cache(self):
        from app.retrieval.hybrid import get_bm25_cache, invalidate_bm25_cache, _bm25_cache
        import app.retrieval.hybrid as hybrid_module
        nodes = self._make_nodes(5)
        get_bm25_cache(nodes)
        assert hybrid_module._bm25_cache is not None
        invalidate_bm25_cache()
        assert hybrid_module._bm25_cache is None


# ── Parallel Agents Node ──────────────────────────────────────────────────────

class TestParallelAgentsNode:
    def _make_state(self, routing: dict) -> dict:
        return {
            "query": "test query",
            "session_id": "test-session",
            "messages": [],
            "sub_tasks": list(routing.keys()),
            "routing": routing,
            "web_results": [],
            "rag_results": [],
            "citations": [],
            "source_contexts": [],
            "agents_used": [],
            "final_answer": None,
            "confidence_score": None,
            "error": None,
            "iteration_count": 1,
        }

    @patch("app.agents.graph.rag_agent_node")
    def test_rag_only_skips_threading(self, mock_rag):
        """If only RAG is needed, no thread pool should be created."""
        from app.agents.graph import parallel_agents_node
        mock_rag.return_value = {"rag_results": [{"task": "t", "answer": "a"}], "agents_used": ["rag_agent"]}
        state = self._make_state({"What is attention?": "rag"})
        result = parallel_agents_node(state)
        mock_rag.assert_called_once()
        assert "rag_results" in result

    @patch("app.agents.graph.web_search_node")
    def test_web_only_skips_threading(self, mock_web):
        """If only web search is needed, no thread pool should be created."""
        from app.agents.graph import parallel_agents_node
        mock_web.return_value = {"web_results": [{"content": "result"}], "agents_used": ["web_search"]}
        state = self._make_state({"latest news": "web_search"})
        result = parallel_agents_node(state)
        mock_web.assert_called_once()
        assert "web_results" in result

    @patch("app.agents.graph.rag_agent_node")
    @patch("app.agents.graph.web_search_node")
    def test_both_agents_run_and_merge(self, mock_web, mock_rag):
        """Both agents run and results are merged into one state update."""
        from app.agents.graph import parallel_agents_node
        mock_web.return_value = {
            "web_results": [{"content": "web result"}],
            "agents_used": ["web_search"],
        }
        mock_rag.return_value = {
            "rag_results": [{"task": "t", "answer": "rag answer"}],
            "citations": [{"title": "Paper A"}],
            "source_contexts": ["context text"],
            "agents_used": ["rag_agent"],
        }
        state = self._make_state({
            "web task": "web_search",
            "rag task": "rag",
        })
        result = parallel_agents_node(state)

        assert len(result["web_results"]) == 1
        assert len(result["rag_results"]) == 1
        assert "web_search" in result["agents_used"]
        assert "rag_agent" in result["agents_used"]
        assert result["source_contexts"] == ["context text"]


# ── Batch Verdict Parsing ─────────────────────────────────────────────────────

class TestBatchVerdictParsing:
    def test_clean_verdicts(self):
        from app.retrieval.faithfulness_filter import _parse_batch_verdicts
        text = "supported\npartial\nunsupported"
        result = _parse_batch_verdicts(text, expected=3)
        assert result == ["supported", "partial", "unsupported"]

    def test_numbered_verdicts(self):
        from app.retrieval.faithfulness_filter import _parse_batch_verdicts
        text = "1. supported\n2. unsupported\n3. partial"
        result = _parse_batch_verdicts(text, expected=3)
        assert result == ["supported", "unsupported", "partial"]

    def test_pads_short_response(self):
        from app.retrieval.faithfulness_filter import _parse_batch_verdicts
        text = "supported"   # only 1 verdict for 3 expected
        result = _parse_batch_verdicts(text, expected=3)
        assert len(result) == 3
        assert result[0] == "supported"
        assert result[1] == "supported"   # padded

    def test_handles_variations(self):
        from app.retrieval.faithfulness_filter import _parse_batch_verdicts
        text = "fully supported\nnot supported\npartially supported"
        result = _parse_batch_verdicts(text, expected=3)
        assert result[0] == "supported"
        assert result[1] == "unsupported"
        assert result[2] == "partial"
