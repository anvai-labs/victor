"""Tests for KeywordMatcher."""

from victor.tools.keyword_matcher import KeywordMatcher


class TestKeywordMatcher:

    def test_index_and_match(self):
        matcher = KeywordMatcher()
        matcher.index_tool("graph", ["graph", "analyze", "dependency"])
        matcher.index_tool("search", ["search", "find", "query"])

        results = matcher.match("analyze the dependency graph")
        assert len(results) > 0
        # graph tool should match
        tool_names = [r[0] for r in results]
        assert "graph" in tool_names

    def test_mandatory_keywords(self):
        matcher = KeywordMatcher()
        matcher.index_tool("graph", ["graph"], mandatory_keywords=["analyze codebase"])

        matches = matcher.get_mandatory_matches("please analyze codebase")
        assert "graph" in matches

    def test_no_match(self):
        matcher = KeywordMatcher()
        matcher.index_tool("graph", ["graph"])
        results = matcher.match("hello world")
        assert len(results) == 0

    def test_top_k_limit(self):
        matcher = KeywordMatcher()
        for i in range(10):
            matcher.index_tool(f"tool_{i}", [f"keyword_{i}", "common"])
        results = matcher.match("common keyword_1 keyword_2", top_k=3)
        assert len(results) <= 3
