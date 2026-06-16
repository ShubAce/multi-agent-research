"""
tests/unit/test_planner.py
Tests for the planner's JSON extraction logic.
"""

import pytest
from app.agents.planner import _extract_json


class TestExtractJson:
    def test_clean_json(self):
        text = '{"sub_tasks": ["a", "b"], "routing": {"a": "rag"}}'
        result = _extract_json(text)
        assert result["sub_tasks"] == ["a", "b"]

    def test_json_in_markdown_block(self):
        text = '```json\n{"sub_tasks": ["x"], "routing": {"x": "web_search"}}\n```'
        result = _extract_json(text)
        assert result["routing"]["x"] == "web_search"

    def test_json_with_surrounding_text(self):
        text = 'Here is my plan:\n{"sub_tasks": ["q1"], "routing": {"q1": "rag"}}\nDone.'
        result = _extract_json(text)
        assert result["sub_tasks"] == ["q1"]

    def test_invalid_json_raises(self):
        with pytest.raises(ValueError):
            _extract_json("this is not json at all")
