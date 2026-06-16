"""
tests/unit/test_hybrid_retrieval.py
Tests for RRF fusion logic — no ChromaDB needed.
"""

import pytest
from llama_index.core.schema import NodeWithScore, TextNode

from app.retrieval.hybrid import reciprocal_rank_fusion, HybridRetriever


def make_node(node_id: str, text: str = "sample text", score: float = 1.0) -> NodeWithScore:
    return NodeWithScore(
        node=TextNode(text=text, id_=node_id),
        score=score,
    )


class TestRRFFusion:
    def test_single_list_passthrough(self):
        nodes = [make_node(f"n{i}") for i in range(5)]
        result = reciprocal_rank_fusion([nodes], k=60, top_n=5)
        assert len(result) == 5

    def test_overlapping_lists_boosts_shared(self):
        """A node appearing in both lists should rank higher than one in only one list."""
        list_a = [make_node("shared"), make_node("only_a")]
        list_b = [make_node("shared"), make_node("only_b")]

        result = reciprocal_rank_fusion([list_a, list_b], k=60, top_n=3)
        ids = [n.node.node_id for n in result]

        # "shared" appears in both lists → should be first
        assert ids[0] == "shared"

    def test_top_n_respected(self):
        nodes = [make_node(f"n{i}") for i in range(10)]
        result = reciprocal_rank_fusion([nodes], k=60, top_n=3)
        assert len(result) == 3

    def test_rrf_scores_are_positive(self):
        nodes = [make_node("a"), make_node("b")]
        result = reciprocal_rank_fusion([nodes], k=60, top_n=2)
        assert all(n.score > 0 for n in result)

    def test_empty_lists_handled(self):
        result = reciprocal_rank_fusion([[], []], k=60, top_n=5)
        assert result == []

    def test_higher_rank_gets_lower_rrf_score(self):
        """Node at rank 1 should have higher RRF score than node at rank 2."""
        nodes = [make_node("first"), make_node("second")]
        result = reciprocal_rank_fusion([nodes], k=60, top_n=2)
        assert result[0].score > result[1].score


class TestQueryRewriter:
    def test_returns_original_on_disabled(self, monkeypatch):
        """If query_rewriting_enabled=False, return only the original."""
        from app.retrieval.query_rewriter import rewrite_query
        from app.core.config import get_settings

        settings = get_settings()
        monkeypatch.setattr(settings, "query_rewriting_enabled", False)

        result = rewrite_query("What is Flash Attention?")
        assert result == ["What is Flash Attention?"]

    def test_parse_variants_clean_json(self):
        from app.retrieval.query_rewriter import _parse_variants
        text = '["query one", "query two", "query three"]'
        result = _parse_variants(text)
        assert len(result) == 3
        assert "query one" in result

    def test_parse_variants_markdown_block(self):
        from app.retrieval.query_rewriter import _parse_variants
        text = '```json\n["a", "b"]\n```'
        result = _parse_variants(text)
        assert result == ["a", "b"]
