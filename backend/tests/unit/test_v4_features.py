"""
tests/unit/test_v4_features.py
Plan normalisation, numbered sources, structure-preserving fact-check,
web fallback, honest critic scores, SSE compression, follow-ups.
"""

from unittest.mock import patch

from httpx import ASGITransport, AsyncClient
from starlette.applications import Starlette
from starlette.responses import PlainTextResponse
from starlette.routing import Route

# ── Planner ───────────────────────────────────────────────────────────────────

class TestNormalisePlan:
    def test_paraphrased_routing_keys_matched_by_position(self):
        from app.agents.planner import _normalise_plan
        tasks, routing = _normalise_plan(
            ["What is X?", "Recent news on X"],
            {"what is x": "rag", "news about X": "web_search"},
            "query", allow_web=True,
        )
        assert routing == {"What is X?": "rag", "Recent news on X": "web_search"}

    def test_unknown_agent_becomes_rag(self):
        from app.agents.planner import _normalise_plan
        _, routing = _normalise_plan(["a task"], {"a task": "calculator"}, "q", allow_web=True)
        assert routing == {"a task": "rag"}

    def test_web_disabled_rewrites_to_rag(self):
        from app.agents.planner import _normalise_plan
        _, routing = _normalise_plan(["news"], {"news": "web_search"}, "q", allow_web=False)
        assert routing == {"news": "rag"}

    def test_all_skipped_falls_back_to_query(self):
        from app.agents.planner import _normalise_plan
        tasks, routing = _normalise_plan(["x"], {"x": "skip"}, "original query", allow_web=True)
        assert tasks == ["original query"]
        assert routing == {"original query": "rag"}

    def test_caps_and_dedupes_tasks(self):
        from app.agents.planner import _normalise_plan
        tasks, _ = _normalise_plan(["a", "a", "b", "c", "d", "e"], {}, "q", allow_web=True)
        assert tasks == ["a", "b", "c", "d"]


# ── Synthesiser sources ───────────────────────────────────────────────────────

def _passage(arxiv_id, title, score, text):
    return {"arxiv_id": arxiv_id, "title": title, "relevance_score": score, "text": text,
            "authors": ["A. Author"], "url": f"http://arxiv.org/abs/{arxiv_id}", "published": "2023-01-01"}


class TestBuildSources:
    def test_groups_passages_by_paper_and_numbers_them(self):
        from app.agents.synthesiser import build_sources
        rag = [{"task": "t", "passages": [
            _passage("2205.14135v1", "FlashAttention", 0.9, "tiling"),
            _passage("2205.14135v2", "FlashAttention", 0.7, "recompute"),
            _passage("2307.08691", "FlashAttention-2", 0.95, "parallelism"),
        ]}]
        sources = build_sources(rag, [], max_sources=10)
        assert [s["id"] for s in sources] == [1, 2]
        assert sources[0]["title"] == "FlashAttention-2"          # highest score first
        assert sources[1]["passages"] == ["tiling", "recompute"]  # versions merged

    def test_web_gets_at_most_half_when_papers_exist(self):
        from app.agents.synthesiser import build_sources
        rag = [{"task": "t", "passages": [_passage(f"2301.0000{i}", f"P{i}", 0.5, "x") for i in range(8)]}]
        web = [{"url": f"https://site{i}.com", "title": f"W{i}", "content": "c", "score": 0.9} for i in range(8)]
        sources = build_sources(rag, web, max_sources=6)
        assert len(sources) == 6
        assert sum(s["type"] == "web" for s in sources) == 3

    def test_web_fills_all_slots_without_papers(self):
        from app.agents.synthesiser import build_sources
        web = [{"url": f"https://s{i}.com", "title": "W", "content": "c", "score": 0.5} for i in range(4)]
        sources = build_sources([], web, max_sources=10)
        assert len(sources) == 4 and all(s["type"] == "web" for s in sources)


class TestRenumberCitations:
    def test_cited_sources_only_in_reading_order(self):
        from app.agents.synthesiser import renumber_citations
        sources = [{"id": i, "title": f"S{i}"} for i in range(1, 6)]
        text, cited, extra = renumber_citations(
            "First [3][4]. Then [1, 3]. Bogus [9]. Year [2021].", sources, ["dropped claim [4]"]
        )
        assert text == "First [1][2]. Then [3][1]. Bogus. Year [2021]."
        assert [(c["id"], c["title"]) for c in cited] == [(1, "S3"), (2, "S4"), (3, "S1")]
        assert extra == ["dropped claim [2]"]

    def test_tool_style_markers_normalised(self):
        from app.agents.synthesiser import normalise_citation_markers
        text = "Grounded claim【6†L1-L4】【8†L1-L4】. Other [3†source] and plain [2]."
        assert normalise_citation_markers(text) == "Grounded claim[6][8]. Other [3] and plain [2]."

    def test_no_citations_keeps_sources(self):
        from app.agents.synthesiser import renumber_citations
        sources = [{"id": 1, "title": "S1"}]
        assert renumber_citations("No markers here.", sources) == ("No markers here.", sources, [])


# ── Faithfulness filter keeps Markdown structure ──────────────────────────────

class TestFilterPreservesStructure:
    @patch("app.retrieval.faithfulness_filter._batch_check_sentences")
    def test_headings_and_bullets_survive(self, mock_check):
        from app.retrieval.faithfulness_filter import apply_faithfulness_filter
        answer = (
            "FlashAttention reduces memory traffic with tiling [1]. It was invented in 1995 by aliens.\n"
            "\n"
            "### Key takeaways\n"
            "- Tiling keeps attention blocks inside fast on-chip SRAM [1].\n"
            "- The method also cures every known disease in humans."
        )
        mock_check.return_value = ["supported", "unsupported", "supported", "unsupported"]
        result = apply_faithfulness_filter(answer, ["FlashAttention uses tiling in SRAM."])

        assert result.filtered_answer == (
            "FlashAttention reduces memory traffic with tiling [1].\n"
            "\n"
            "### Key takeaways\n"
            "- Tiling keeps attention blocks inside fast on-chip SRAM [1]."
        )
        assert result.removed_sentences == 2
        assert result.confidence_score == 0.5

    @patch("app.retrieval.faithfulness_filter._batch_check_sentences")
    def test_empty_section_heading_is_dropped(self, mock_check):
        from app.retrieval.faithfulness_filter import apply_faithfulness_filter
        answer = (
            "The core idea is well supported by the retrieved sources [1].\n\n"
            "### Speculation\n"
            "This paragraph is completely made up and unsupported by anything."
        )
        mock_check.return_value = ["supported", "unsupported"]
        result = apply_faithfulness_filter(answer, ["context"])
        assert "### Speculation" not in result.filtered_answer

    @patch("app.retrieval.faithfulness_filter._batch_check_sentences")
    def test_unsupported_table_rows_removed(self, mock_check):
        from app.retrieval.faithfulness_filter import apply_faithfulness_filter
        answer = (
            "| Method | Result |\n"
            "|---|---|\n"
            "| FlashAttention | 3x faster training on GPT-2 [1] |\n"
            "| MagicAttention | 900% accuracy gain everywhere [2] |"
        )
        mock_check.return_value = ["supported", "unsupported"]
        result = apply_faithfulness_filter(answer, ["FlashAttention trains GPT-2 3x faster."])
        assert "MagicAttention" not in result.filtered_answer
        assert "| FlashAttention |" in result.filtered_answer
        assert result.removed_content == ["MagicAttention — 900% accuracy gain everywhere [2]"]

    @patch("app.retrieval.faithfulness_filter._batch_check_sentences")
    def test_table_with_no_rows_left_is_dropped(self, mock_check):
        from app.retrieval.faithfulness_filter import apply_faithfulness_filter
        answer = (
            "Grounded opening sentence about attention mechanisms [1].\n\n"
            "| Method | Result |\n|---|---|\n| MagicAttention | 900% accuracy gain everywhere [2] |"
        )
        mock_check.return_value = ["supported", "unsupported"]
        result = apply_faithfulness_filter(answer, ["context"])
        assert "|" not in result.filtered_answer

    @patch("app.retrieval.faithfulness_filter._batch_check_sentences")
    def test_bold_line_is_not_treated_as_bullet(self, mock_check):
        from app.retrieval.faithfulness_filter import apply_faithfulness_filter
        mock_check.return_value = ["supported"]
        answer = "**FlashAttention** is an IO-aware exact attention algorithm [1]."
        result = apply_faithfulness_filter(answer, ["context"])
        assert result.filtered_answer == answer


# ── Web fallback when the knowledge base has nothing ──────────────────────────

class TestWebFallback:
    def _state(self, **overrides):
        state = {"query": "q", "routing": {"resolved task": "rag"}, "web_results": [],
                 "rag_results": [], "source_contexts": [], "agents_used": ["planner"],
                 "use_web_search": True}
        return {**state, **overrides}

    @patch("app.agents.graph.web_search_available", return_value=True)
    @patch("app.agents.graph.web_search_node")
    @patch("app.agents.graph.rag_agent_node")
    def test_empty_rag_falls_back_to_web(self, mock_rag, mock_web, _avail):
        from app.agents.graph import parallel_agents_node
        mock_rag.return_value = {"rag_results": [], "agents_used": ["planner", "rag_agent"]}
        mock_web.return_value = {"web_results": [{"url": "https://x.com"}]}

        result = parallel_agents_node(self._state())

        fallback_state = mock_web.call_args.args[0]
        assert fallback_state["routing"] == {"resolved task": "web_search"}
        assert result["web_fallback"] is True
        assert result["agents_used"] == ["planner", "rag_agent", "web_search"]

    @patch("app.agents.graph.web_search_node")
    @patch("app.agents.graph.rag_agent_node")
    def test_no_fallback_when_web_disabled(self, mock_rag, mock_web):
        from app.agents.graph import parallel_agents_node
        mock_rag.return_value = {"rag_results": []}
        result = parallel_agents_node(self._state(use_web_search=False))
        mock_web.assert_not_called()
        assert result["web_fallback"] is False

    @patch("app.agents.graph.rag_agent_node")
    def test_retry_appends_to_previous_evidence(self, mock_rag):
        from app.agents.graph import parallel_agents_node
        mock_rag.return_value = {"rag_results": [{"task": "new", "passages": [{}]}]}
        state = self._state(rag_results=[{"task": "old", "passages": [{}]}])
        result = parallel_agents_node(state)
        assert [r["task"] for r in result["rag_results"]] == ["old", "new"]


# ── Critic reports real scores only ───────────────────────────────────────────

class TestCriticHonesty:
    def _state(self, retry_count=0):
        return {"query": "q", "final_answer": "An answer.", "citations": [],
                "agents_used": [], "retry_count": retry_count, "sub_tasks": ["q"]}

    @patch("app.agents.critic.get_fast_llm")
    def test_llm_failure_gives_no_score(self, mock_llm):
        from app.agents.critic import critic_node, should_retry
        mock_llm.return_value.invoke.side_effect = RuntimeError("rate limited")
        result = critic_node(self._state())
        assert result["critic_score"] is None
        assert should_retry({**self._state(), **result}) == "end"

    @patch("app.agents.critic.get_fast_llm")
    def test_retried_answer_is_still_scored(self, mock_llm):
        from app.agents.critic import critic_node
        mock_llm.return_value.invoke.return_value.content = \
            '{"score": 4, "reason": "Thin.", "improved_queries": ["better query"]}'
        result = critic_node(self._state(retry_count=1))
        assert result["critic_score"] == 4
        assert "retry_count" not in result   # no second retry


# ── SSE responses bypass gzip ─────────────────────────────────────────────────

class TestGzipExceptSSE:
    async def _get(self, path):
        from app.main import GZipExceptSSE

        async def big(_request):
            return PlainTextResponse("x" * 5000)

        inner = Starlette(routes=[Route("/{path:path}", big)])
        async with AsyncClient(transport=ASGITransport(app=GZipExceptSSE(inner)), base_url="http://t") as c:
            return await c.get(path, headers={"Accept-Encoding": "gzip"})

    async def test_stream_path_not_compressed(self):
        response = await self._get("/api/v1/stream/job-1")
        assert "content-encoding" not in response.headers

    async def test_other_paths_compressed(self):
        response = await self._get("/api/v1/knowledge/papers")
        assert response.headers.get("content-encoding") == "gzip"


# ── Worker helpers ────────────────────────────────────────────────────────────

class TestWorkerHelpers:
    def test_next_node_sequence(self):
        from app.workers.tasks import _next_node
        assert _next_node("memory_injection", {}) == "planner"
        assert _next_node("planner", {"sub_tasks": ["a"]}) == "parallel_agents"
        assert _next_node("synthesiser", {}) == "critic"
        assert _next_node("critic", {"critic_score": 9, "final_answer": "x"}) is None
        assert _next_node(
            "critic", {"critic_score": 3, "retry_count": 1, "sub_tasks": ["b"], "final_answer": None}
        ) == "parallel_agents"

    def test_parallel_detail_counts_unique_papers(self):
        from app.workers.tasks import _detail
        state = {"rag_results": [{"passages": [{"arxiv_id": "1"}, {"arxiv_id": "1"}, {"arxiv_id": "2"}]}],
                 "web_results": [{}, {}], "retry_count": 0}
        detail = _detail("parallel_agents", {}, state)
        assert detail["papers"] == 2 and detail["passages"] == 3 and detail["web_results"] == 2


# ── Follow-up suggestions ─────────────────────────────────────────────────────

class TestFollowUps:
    @patch("app.agents.follow_ups.get_fast_llm")
    def test_parses_and_normalises(self, mock_llm):
        from app.agents.follow_ups import suggest_follow_ups
        mock_llm.return_value.invoke.return_value.content = \
            '["How does FlashAttention-2 differ", "What are the limits of tiling?", "Why?", "Extra one?"]'
        result = suggest_follow_ups("q", "answer")
        assert result == ["How does FlashAttention-2 differ?", "What are the limits of tiling?", "Extra one?"]

    @patch("app.agents.follow_ups.get_fast_llm")
    def test_failure_returns_empty(self, mock_llm):
        from app.agents.follow_ups import suggest_follow_ups
        mock_llm.return_value.invoke.side_effect = RuntimeError("boom")
        assert suggest_follow_ups("q", "answer") == []


class TestArxivIds:
    def test_base_id_strips_version(self):
        from app.ingestion.arxiv_loader import base_arxiv_id
        assert base_arxiv_id("2205.14135v3") == "2205.14135"
        assert base_arxiv_id("2205.14135") == "2205.14135"
