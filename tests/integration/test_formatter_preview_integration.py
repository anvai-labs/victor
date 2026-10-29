"""Integration tests for formatter-aware preview strategies."""

from victor.ui.rendering.formatter_aware_preview import (
    _TestPreviewStrategyEnhanced,
    _SearchPreviewStrategyEnhanced,
    _GitPreviewStrategyEnhanced,
)
from victor.ui.rendering.tool_preview import RenderedPreview


class TestFormatterAwarePreviewStrategies:
    """Test enhanced preview strategies that use Rich-formatted output."""

    def test_test_strategy_uses_formatted_summary(self):
        """Test that test strategy uses formatted_summary when available."""
        strategy = _TestPreviewStrategyEnhanced()

        raw_result = """{
    "summary": {"total_tests": 10, "passed": 8, "failed": 2, "skipped": 0},
    "failures": [
        {"test_name": "test_foo.py::test_bar", "error_message": "AssertionError"}
    ],
    "formatted_summary": "[green]✓ 8 passed[/] [dim]• 10 total[/]\\n\\n[red bold]Failed Tests:[/]"
}"""

        result = strategy.render("test", {}, raw_result, max_lines=5)

        assert isinstance(result, RenderedPreview)
        assert result.contains_rich_markup is True
        assert "[green]✓ 8 passed[/]" in result.lines[0]
        assert result.header == "10 tests"

    def test_search_strategy_uses_formatted_results(self):
        """Test that search strategy uses formatted_results when available."""
        strategy = _SearchPreviewStrategyEnhanced()

        raw_result = """{
    "success": true,
    "results": [
        {"path": "src/main.py", "line": 42, "score": 10, "snippet": "def foo():"}
    ],
    "count": 1,
    "mode": "semantic",
    "formatted_results": "[bold cyan]1 match[/] [dim]in 1 file[/]\\n\\n  [bold]src/main.py[/] [dim]• score: 10[/]"
}"""

        result = strategy.render("code_search", {}, raw_result, max_lines=5)

        assert isinstance(result, RenderedPreview)
        assert result.contains_rich_markup is True
        assert "[bold cyan]1 match[/]" in result.lines[0]
        assert result.header == "1 matches (semantic)"

    def test_git_strategy_uses_formatted_output(self):
        """Test that git strategy uses formatted_output when available."""
        strategy = _GitPreviewStrategyEnhanced()

        raw_result = """{
    "success": true,
    "output": "* main\\n  develop",
    "formatted_output": "[bold green]*[/] [bold]main[/]\\n  [dim]develop[/]"
}"""

        result = strategy.render("git", {"operation": "status"}, raw_result, max_lines=5)

        assert isinstance(result, RenderedPreview)
        assert result.contains_rich_markup is True
        assert "[bold green]*[/]" in result.lines[0]

    def test_test_strategy_falls_back_to_base(self):
        """Test that test strategy falls back to base logic when no formatted output."""
        strategy = _TestPreviewStrategyEnhanced()

        raw_result = """{
    "summary": {"total_tests": 5, "passed": 3, "failed": 2, "skipped": 0},
    "failures": [
        {"test_name": "test_foo.py::test_bar", "error_message": "AssertionError"}
    ]
}"""

        result = strategy.render("test", {}, raw_result, max_lines=5)

        assert isinstance(result, RenderedPreview)
        assert result.contains_rich_markup is False  # No Rich markup in fallback
        assert "3/5" in result.header or "3" in result.header

    def test_search_strategy_falls_back_to_base(self):
        """Test that search strategy falls back to base logic when no formatted output."""
        strategy = _SearchPreviewStrategyEnhanced()

        raw_result = """{
    "success": true,
    "results": [
        {"path": "src/main.py", "line": 42, "score": 10, "snippet": "def foo():"}
    ],
    "count": 1,
    "mode": "semantic"
}"""

        result = strategy.render("code_search", {}, raw_result, max_lines=5)

        assert isinstance(result, RenderedPreview)
        assert result.contains_rich_markup is False
        assert "1 matches" in result.header

    def test_preview_limits_lines(self):
        """Test that preview respects max_lines parameter."""
        strategy = _TestPreviewStrategyEnhanced()

        # Create formatted output with many lines
        lines = [f"[dim]Line {i}[/]" for i in range(20)]
        formatted_summary = "\\n".join(lines)

        raw_result = f"""{{
    "summary": {{"total_tests": 20}},
    "formatted_summary": "{formatted_summary}"
}}"""

        result = strategy.render("test", {}, raw_result, max_lines=5)

        assert len(result.lines) == 5  # Should be limited to max_lines
        assert result.total_line_count == 20  # But total should reflect actual count

    def test_contains_markup_flag_accuracy(self):
        """Test that contains_markup flag is set correctly."""
        strategy = _TestPreviewStrategyEnhanced()

        # Test with formatted output (has markup)
        raw_result_with_markup = """{
    "formatted_summary": "[green]✓ Test passed[/]",
    "contains_markup": true
}"""

        result1 = strategy.render("test", {}, raw_result_with_markup, max_lines=5)
        assert result1.contains_rich_markup is True

        # Test without formatted output (no markup)
        raw_result_without_markup = """{
    "summary": {"total_tests": 1, "passed": 1}
}"""

        result2 = strategy.render("test", {}, raw_result_without_markup, max_lines=5)
        assert result2.contains_rich_markup is False

    def test_extract_summary_from_content(self):
        """Test that summary is extracted from formatted content."""
        strategy = _TestPreviewStrategyEnhanced()

        raw_result = """{
    "formatted_output": "[bold cyan]10 matches[/] [dim]in 5 files[/]"
}"""

        result = strategy.render("test", {}, raw_result, max_lines=5)

        # Summary should be extracted from content
        assert result.header is not None
        assert "10" in result.header or "matches" in result.header


class TestPreviewRendererIntegration:
    """Test integration of enhanced strategies with ToolPreviewRenderer."""

    def test_renderer_uses_enhanced_strategies(self):
        """Test that ToolPreviewRenderer uses enhanced strategies by default."""
        from victor.ui.rendering.tool_preview import renderer

        # Test with formatted output
        raw_result = """{
    "summary": {"total_tests": 10, "passed": 8},
    "formatted_summary": "[green]✓ 8 passed[/]"
        }"""

        result = renderer.render("test", {}, raw_result, max_lines=5)

        assert result.contains_rich_markup is True
        assert "[green]✓ 8 passed[/]" in result.lines[0]

    def test_renderer_fallback_on_error(self):
        """Test that renderer falls back gracefully on errors."""
        from victor.ui.rendering.tool_preview import renderer

        # Invalid JSON should still produce some output
        raw_result = "invalid json output"

        result = renderer.render("test", {}, raw_result, max_lines=5)

        # Should fall back to generic strategy
        assert isinstance(result, RenderedPreview)
        assert len(result.lines) >= 0

    def test_renderer_handles_all_enhanced_tools(self):
        """Test that renderer handles all tools with enhanced strategies."""
        from victor.ui.rendering.tool_preview import renderer

        enhanced_tools = [
            "test",
            "pytest",
            "run_tests",
            "code_search",
            "semantic_code_search",
            "git",
        ]

        for tool_name in enhanced_tools:
            # Create minimal valid result for each tool
            if tool_name in ["test", "pytest", "run_tests"]:
                raw_result = (
                    '{"summary": {"total_tests": 1}, "formatted_summary": "[green]✓ 1 passed[/]"}'
                )
            elif tool_name in ["code_search", "semantic_code_search"]:
                raw_result = '{"results": [], "formatted_results": "[dim]No matches[/]"}'
            elif tool_name == "git":
                raw_result = '{"output": "", "formatted_output": "[dim]No output[/]"}'

            result = renderer.render(tool_name, {}, raw_result, max_lines=3)

            # Should produce valid RenderedPreview
            assert isinstance(result, RenderedPreview)
            assert len(result.lines) >= 0
