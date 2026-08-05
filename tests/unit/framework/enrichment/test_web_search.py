# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for web search result formatting utilities."""

import pytest
from victor.framework.enrichment.web_search import (
    WebSearchFormatter,
    format_web_results,
    truncate_snippet,
)


class TestTruncateSnippet:
    """Tests for truncate_snippet function."""

    def test_empty_snippet_returns_empty(self):
        """Test empty snippet returns empty string."""
        result = truncate_snippet("")
        assert result == ""

    def test_none_returns_none(self):
        """Test None snippet is handled."""
        result = truncate_snippet(None)
        assert result is None

    def test_short_snippet_unchanged(self):
        """Test short snippet is not truncated."""
        snippet = "This is a short snippet."
        result = truncate_snippet(snippet, max_length=200)
        assert result == snippet

    def test_snippet_at_max_length_unchanged(self):
        """Test snippet at exactly max_length is not truncated."""
        snippet = "x" * 200
        result = truncate_snippet(snippet, max_length=200)
        assert result == snippet
        assert "..." not in result

    def test_long_snippet_truncated(self):
        """Test long snippet is truncated."""
        snippet = "x" * 300
        result = truncate_snippet(snippet, max_length=200)
        assert len(result) < 300
        assert result.endswith("...")

    def test_truncates_at_word_boundary(self):
        """Test truncation prefers word boundaries."""
        snippet = "This is a test sentence with many words to check word boundary"
        result = truncate_snippet(snippet, max_length=50)
        # Should not end mid-word (unless no space found)
        assert result.endswith("...") or result == snippet

    def test_custom_suffix(self):
        """Test custom truncation suffix."""
        snippet = "x" * 300
        result = truncate_snippet(snippet, max_length=200, suffix="[more]")
        assert result.endswith("[more]")

    def test_empty_suffix(self):
        """Test empty suffix."""
        snippet = "x" * 300
        result = truncate_snippet(snippet, max_length=200, suffix="")
        assert not result.endswith("...")

    def test_very_small_max_length(self):
        """Test very small max_length."""
        snippet = "This is a test"
        result = truncate_snippet(snippet, max_length=5)
        assert len(result) <= 5 + len("...")

    def test_word_boundary_not_too_aggressive(self):
        """Test word boundary truncation doesn't cut too much."""
        # If last space is too early (< 60% of max_length), don't use it
        snippet = "A " + "x" * 200
        result = truncate_snippet(snippet, max_length=100)
        # Should truncate within the x's, not go back to "A"
        assert len(result) > 60

    def test_no_spaces_in_snippet(self):
        """Test snippet with no spaces."""
        snippet = "x" * 300
        result = truncate_snippet(snippet, max_length=100)
        # Should still truncate
        assert len(result) <= 103  # 100 + "..."

    def test_preserves_content_within_limit(self):
        """Test content is preserved within limit."""
        snippet = "Hello world this is content"
        result = truncate_snippet(snippet, max_length=1000)
        assert result == snippet

    def test_strips_trailing_whitespace_before_suffix(self):
        """Test trailing whitespace is stripped before suffix."""
        snippet = "Hello world   " + "x" * 200
        result = truncate_snippet(snippet, max_length=14)
        # Should strip trailing spaces before adding ...
        if "..." in result:
            assert not result.endswith(" ...")


class TestFormatWebResults:
    """Tests for format_web_results function."""

    def test_empty_results_returns_empty_string(self):
        """Test empty results returns empty string."""
        result = format_web_results([])
        assert result == ""

    def test_single_result_formats_correctly(self):
        """Test single result is formatted correctly."""
        results = [
            {
                "title": "Test Title",
                "snippet": "Test snippet",
                "url": "https://test.com",
            }
        ]
        result = format_web_results(results)
        assert "Test Title" in result
        assert "Test snippet" in result
        assert "https://test.com" in result

    def test_multiple_results_formatted(self):
        """Test multiple results are formatted."""
        results = [
            {"title": "Title 1", "snippet": "Snippet 1", "url": "https://one.com"},
            {"title": "Title 2", "snippet": "Snippet 2", "url": "https://two.com"},
        ]
        result = format_web_results(results)
        assert "Title 1" in result
        assert "Title 2" in result
        assert "1." in result
        assert "2." in result

    def test_max_results_limits_output(self):
        """Test max_results limits number of results."""
        results = [
            {
                "title": f"Title {i}",
                "snippet": f"Snippet {i}",
                "url": f"https://{i}.com",
            }
            for i in range(10)
        ]
        result = format_web_results(results, max_results=3)
        assert "Title 0" in result
        assert "Title 1" in result
        assert "Title 2" in result
        assert "Title 3" not in result

    def test_default_header(self):
        """Test default header is included."""
        results = [{"title": "Test", "snippet": "Test", "url": "https://test.com"}]
        result = format_web_results(results)
        assert "Relevant web search results:" in result

    def test_custom_header(self):
        """Test custom header is used."""
        results = [{"title": "Test", "snippet": "Test", "url": "https://test.com"}]
        result = format_web_results(results, header="Custom Header:")
        assert "Custom Header:" in result
        assert "Relevant web search results:" not in result

    def test_include_urls_true(self):
        """Test URLs are included when include_urls=True."""
        results = [{"title": "Test", "snippet": "Test", "url": "https://test.com"}]
        result = format_web_results(results, include_urls=True)
        assert "Source: https://test.com" in result

    def test_include_urls_false(self):
        """Test URLs are excluded when include_urls=False."""
        results = [{"title": "Test", "snippet": "Test", "url": "https://test.com"}]
        result = format_web_results(results, include_urls=False)
        assert "Source:" not in result
        assert "https://test.com" not in result

    def test_truncates_long_snippets(self):
        """Test long snippets are truncated."""
        long_snippet = "x" * 500
        results = [{"title": "Test", "snippet": long_snippet, "url": "https://test.com"}]
        result = format_web_results(results, max_snippet_length=100)
        assert "..." in result
        assert len(result) < 600

    def test_handles_missing_title(self):
        """Test handles missing title."""
        results = [{"snippet": "Test snippet", "url": "https://test.com"}]
        result = format_web_results(results)
        assert "Untitled" in result

    def test_handles_missing_snippet(self):
        """Test handles missing snippet."""
        results = [{"title": "Test", "url": "https://test.com"}]
        result = format_web_results(results)
        assert "Test" in result

    def test_handles_missing_url(self):
        """Test handles missing URL."""
        results = [{"title": "Test", "snippet": "Test snippet"}]
        result = format_web_results(results, include_urls=True)
        assert "Test" in result
        # Should not have Source: line for empty URL
        assert "Source: " not in result or "Source: \n" in result or result.count("Source:") == 0

    def test_handles_empty_result_dict(self):
        """Test handles empty result dict."""
        results = [{}]
        result = format_web_results(results)
        assert "Untitled" in result

    def test_format_structure(self):
        """Test format structure matches expected pattern."""
        results = [{"title": "Title", "snippet": "Snippet", "url": "https://url.com"}]
        result = format_web_results(results)
        # Check structure
        lines = result.split("\n")
        assert lines[0] == "Relevant web search results:"
        assert "**Title**" in result

    def test_numbering_format(self):
        """Test results are numbered correctly."""
        results = [
            {
                "title": f"Title {i}",
                "snippet": f"Snippet {i}",
                "url": f"https://{i}.com",
            }
            for i in range(5)
        ]
        result = format_web_results(results, max_results=5)
        assert "1. **Title 0**" in result
        assert "2. **Title 1**" in result
        assert "3. **Title 2**" in result
        assert "4. **Title 3**" in result
        assert "5. **Title 4**" in result


class TestWebSearchFormatter:
    """Tests for WebSearchFormatter class."""

    def test_init_default_values(self):
        """Test default initialization."""
        formatter = WebSearchFormatter()
        assert formatter.max_results == 3
        assert formatter._max_snippet_length == 200
        assert formatter._include_urls is True
        assert formatter._header is None

    def test_init_custom_max_results(self):
        """Test custom max_results."""
        formatter = WebSearchFormatter(max_results=10)
        assert formatter.max_results == 10

    def test_init_custom_max_snippet_length(self):
        """Test custom max_snippet_length."""
        formatter = WebSearchFormatter(max_snippet_length=500)
        assert formatter._max_snippet_length == 500

    def test_init_custom_include_urls(self):
        """Test custom include_urls."""
        formatter = WebSearchFormatter(include_urls=False)
        assert formatter._include_urls is False

    def test_init_custom_header(self):
        """Test custom header."""
        formatter = WebSearchFormatter(header="Custom:")
        assert formatter._header == "Custom:"

    def test_format_empty_results(self):
        """Test format with empty results."""
        formatter = WebSearchFormatter()
        result = formatter.format([])
        assert result == ""

    def test_format_with_results(self):
        """Test format with results."""
        formatter = WebSearchFormatter()
        results = [{"title": "Test", "snippet": "Snippet", "url": "https://test.com"}]
        result = formatter.format(results)
        assert "Test" in result
        assert "Snippet" in result

    def test_format_respects_max_results(self):
        """Test format respects max_results setting."""
        formatter = WebSearchFormatter(max_results=2)
        results = [
            {
                "title": f"Title {i}",
                "snippet": f"Snippet {i}",
                "url": f"https://{i}.com",
            }
            for i in range(5)
        ]
        result = formatter.format(results)
        assert "Title 0" in result
        assert "Title 1" in result
        assert "Title 2" not in result

    def test_format_with_query_no_header(self):
        """Test format with query when no header set."""
        formatter = WebSearchFormatter()
        results = [{"title": "Test", "snippet": "Snippet", "url": "https://test.com"}]
        result = formatter.format(results, query="python tutorial")
        assert "Web search results for: python tutorial" in result

    def test_format_with_query_and_header(self):
        """Test format with query when header is set."""
        formatter = WebSearchFormatter(header="Custom Header:")
        results = [{"title": "Test", "snippet": "Snippet", "url": "https://test.com"}]
        result = formatter.format(results, query="python tutorial")
        # Should use custom header, not query-based header
        assert "Custom Header:" in result
        assert "Web search results for:" not in result

    def test_format_without_query(self):
        """Test format without query uses default header."""
        formatter = WebSearchFormatter()
        results = [{"title": "Test", "snippet": "Snippet", "url": "https://test.com"}]
        result = formatter.format(results)
        assert "Relevant web search results:" in result

    def test_format_for_citation_empty(self):
        """Test format_for_citation with empty results."""
        formatter = WebSearchFormatter()
        result = formatter.format_for_citation([])
        assert result == []

    def test_format_for_citation_single_result(self):
        """Test format_for_citation with single result."""
        formatter = WebSearchFormatter()
        results = [{"title": "Test", "snippet": "Snippet", "url": "https://test.com"}]
        result = formatter.format_for_citation(results)
        assert len(result) == 1
        assert result[0]["title"] == "Test"
        assert result[0]["url"] == "https://test.com"
        assert result[0]["snippet"] == "Snippet"

    def test_format_for_citation_respects_max_results(self):
        """Test format_for_citation respects max_results."""
        formatter = WebSearchFormatter(max_results=2)
        results = [
            {
                "title": f"Title {i}",
                "snippet": f"Snippet {i}",
                "url": f"https://{i}.com",
            }
            for i in range(5)
        ]
        result = formatter.format_for_citation(results)
        assert len(result) == 2

    def test_format_for_citation_truncates_snippets(self):
        """Test format_for_citation truncates long snippets."""
        formatter = WebSearchFormatter(max_snippet_length=50)
        results = [{"title": "Test", "snippet": "x" * 100, "url": "https://test.com"}]
        result = formatter.format_for_citation(results)
        assert len(result[0]["snippet"]) < 100
        assert "..." in result[0]["snippet"]

    def test_format_for_citation_handles_missing_fields(self):
        """Test format_for_citation handles missing fields."""
        formatter = WebSearchFormatter()
        results = [{}]
        result = formatter.format_for_citation(results)
        assert result[0]["title"] == "Untitled"
        assert result[0]["url"] == ""
        assert result[0]["snippet"] == ""

    def test_max_results_property_getter(self):
        """Test max_results property getter."""
        formatter = WebSearchFormatter(max_results=7)
        assert formatter.max_results == 7

    def test_max_results_property_setter(self):
        """Test max_results property setter."""
        formatter = WebSearchFormatter(max_results=5)
        formatter.max_results = 10
        assert formatter.max_results == 10

    def test_max_results_setter_enforces_minimum(self):
        """Test max_results setter enforces minimum of 1."""
        formatter = WebSearchFormatter()
        formatter.max_results = 0
        assert formatter.max_results >= 1
        formatter.max_results = -5
        assert formatter.max_results >= 1


class TestEdgeCases:
    """Tests for edge cases."""

    def test_unicode_content(self):
        """Test handling of unicode content."""
        results = [
            {
                "title": "Unicode Title",
                "snippet": "Unicode snippet with special chars",
                "url": "https://test.com",
            }
        ]
        result = format_web_results(results)
        assert "Unicode Title" in result

    def test_special_characters_in_url(self):
        """Test handling of special characters in URL."""
        results = [
            {
                "title": "Test",
                "snippet": "Test",
                "url": "https://test.com/path?query=value&other=1",
            }
        ]
        result = format_web_results(results)
        assert "query=value" in result

    def test_html_entities_in_snippet(self):
        """Test handling of HTML entities in snippet."""
        results = [
            {
                "title": "Test",
                "snippet": "Test &amp; more &lt;content&gt;",
                "url": "https://test.com",
            }
        ]
        result = format_web_results(results)
        assert "&amp;" in result or "& more" in result

    def test_newlines_in_snippet(self):
        """Test handling of newlines in snippet."""
        results = [
            {
                "title": "Test",
                "snippet": "Line 1\nLine 2\nLine 3",
                "url": "https://test.com",
            }
        ]
        result = format_web_results(results)
        assert "Line 1" in result

    def test_very_long_title(self):
        """Test handling of very long title."""
        results = [{"title": "x" * 500, "snippet": "Test", "url": "https://test.com"}]
        result = format_web_results(results)
        # Should still format without error
        assert "x" * 100 in result

    def test_empty_snippet_with_url(self):
        """Test result with empty snippet but valid URL."""
        results = [{"title": "Test", "snippet": "", "url": "https://test.com"}]
        result = format_web_results(results)
        assert "Test" in result
        assert "https://test.com" in result

    def test_all_fields_empty(self):
        """Test result with all empty fields."""
        results = [{"title": "", "snippet": "", "url": ""}]
        result = format_web_results(results)
        # Should have Untitled as fallback
        assert "Untitled" in result or result != ""

    def test_mixed_valid_invalid_results(self):
        """Test mix of valid and empty results."""
        results = [
            {"title": "Valid", "snippet": "Valid snippet", "url": "https://valid.com"},
            {},
            {
                "title": "Also Valid",
                "snippet": "Also snippet",
                "url": "https://also.com",
            },
        ]
        result = format_web_results(results)
        assert "Valid" in result
        assert "Also Valid" in result

    def test_result_with_none_values(self):
        """Test result with None values."""
        results = [{"title": None, "snippet": None, "url": None}]
        result = format_web_results(results)
        # Should handle None gracefully
        assert "Untitled" in result or result != ""

    def test_formatter_format_for_citation_with_none_snippet(self):
        """Test format_for_citation handles None snippet."""
        formatter = WebSearchFormatter()
        results = [{"title": "Test", "snippet": None, "url": "https://test.com"}]
        result = formatter.format_for_citation(results)
        # truncate_snippet(None) returns None
        assert result[0]["snippet"] is None or result[0]["snippet"] == ""


class TestIntegration:
    """Integration tests for web search formatting."""

    def test_full_workflow(self):
        """Test complete formatting workflow."""
        formatter = WebSearchFormatter(
            max_results=3,
            max_snippet_length=100,
            include_urls=True,
            header="Search Results:",
        )

        results = [
            {
                "title": "Python Documentation",
                "snippet": "Python is a programming language that lets you work quickly and integrate systems more effectively. "
                * 5,
                "url": "https://python.org",
            },
            {
                "title": "Python Tutorial",
                "snippet": "Learn Python programming from scratch with our comprehensive tutorial.",
                "url": "https://tutorial.python.org",
            },
        ]

        formatted = formatter.format(results, query="python")
        citations = formatter.format_for_citation(results)

        assert "Search Results:" in formatted
        assert "Python Documentation" in formatted
        assert "Python Tutorial" in formatted
        assert "..." in formatted  # Long snippet should be truncated

        assert len(citations) == 2
        assert citations[0]["title"] == "Python Documentation"

    def test_citation_roundtrip(self):
        """Test that format_for_citation produces usable data."""
        formatter = WebSearchFormatter()
        results = [{"title": "Test", "snippet": "Test content", "url": "https://test.com"}]

        citations = formatter.format_for_citation(results)

        # Citations should be usable as results for another format call
        formatted = format_web_results(citations)
        assert "Test" in formatted
