"""Integration tests for Rich formatter system.

Tests the full integration between tools, formatters, and preview strategies.
"""

from victor.tools.formatters import (
    format_test_results,
    format_search_results,
    format_git_output,
    format_http_response,
    format_database_results,
)
from victor.tools.formatters.registry import get_formatter_registry


class TestFormatterIntegration:
    """Test formatter system integration."""

    def test_registry_auto_registers_formatters(self):
        """Test that all formatters are auto-registered on import."""
        registry = get_formatter_registry()

        # Check that all formatters are registered
        formatters = registry.list_formatters()

        assert "test" in formatters
        assert "pytest" in formatters
        assert "code_search" in formatters
        assert "git" in formatters
        assert "http" in formatters
        assert "database" in formatters

    def test_convenience_functions_use_registry(self):
        """Test that convenience functions use the registry."""
        # This tests the integration between convenience functions and registry
        result = format_test_results(
            {
                "summary": {"total_tests": 10, "passed": 10, "failed": 0, "skipped": 0},
                "failures": [],
            }
        )

        assert result.contains_markup is True
        assert "[green]✓ 10 passed[/]" in result.content

    def test_tool_to_formatter_workflow(self):
        """Test complete workflow from tool data to formatted output."""
        # Simulate tool output
        tool_data = {
            "summary": {
                "total_tests": 5,
                "passed": 3,
                "failed": 2,
                "skipped": 0,
            },
            "failures": [
                {
                    "test_name": "test_foo.py::test_bar",
                    "error_message": "AssertionError",
                },
                {
                    "test_name": "test_baz.py::test_qux",
                    "error_message": "ValueError",
                },
            ],
        }

        # Format using formatter system
        formatted = format_test_results(tool_data)

        # Verify formatted output
        assert formatted.contains_markup is True
        assert formatted.format_type == "rich"
        assert formatted.summary == "5 tests"
        assert "3 passed" in formatted.content
        assert "2 failed" in formatted.content
        assert "test_bar" in formatted.content

    def test_formatter_fallback_chain(self):
        """Test that fallback chain works correctly."""
        registry = get_formatter_registry()

        # Get formatter for unregistered tool
        formatter = registry.get_formatter("nonexistent_tool")

        # Should fall back to GenericFormatter
        assert formatter.__class__.__name__ == "GenericFormatter"

        # Should still produce output
        result = formatter.format({"key": "value"})
        assert result.format_type == "plain"
        assert "key: value" in result.content

    def test_multiple_tools_same_formatter(self):
        """Test that multiple tools can use the same formatter."""
        registry = get_formatter_registry()

        # test, pytest, and run_tests should all use TestResultsFormatter
        test_formatter = registry.get_formatter("test")
        pytest_formatter = registry.get_formatter("pytest")
        run_tests_formatter = registry.get_formatter("run_tests")

        # All should be the same formatter instance
        assert test_formatter.__class__ == pytest_formatter.__class__
        assert test_formatter.__class__ == run_tests_formatter.__class__

    def test_formatter_error_isolation(self):
        """Test that formatter errors don't break the tool workflow."""
        # Even if formatter fails, tool should still return data
        tool_data = {"invalid": "data"}

        # Generic formatter should handle this gracefully
        from victor.tools.formatters.generic import GenericFormatter

        formatter = GenericFormatter()

        result = formatter.format(tool_data)

        # Should produce some output even with invalid data
        assert result.format_type == "plain"
        assert len(result.content) > 0

    def test_formatter_parameter_passing(self):
        """Test that parameters are correctly passed through the system."""
        tool_data = {
            "results": [
                {"path": f"file_{i}.py", "line": i, "score": 10, "snippet": "code"}
                for i in range(15)
            ],
            "mode": "semantic",
        }

        # Format with custom max_files parameter
        result = format_search_results(tool_data, max_files=5)

        assert result.contains_markup is True
        # Should truncate output
        assert "... and 10 more files" in result.content

    def test_cross_formatter_consistency(self):
        """Test that all formatters follow the same pattern."""
        registry = get_formatter_registry()

        # Get a sample of formatters
        test_formatter = registry.get_formatter("test")
        search_formatter = registry.get_formatter("code_search")
        git_formatter = registry.get_formatter("git")

        # All should have the same interface
        for formatter in [test_formatter, search_formatter, git_formatter]:
            assert hasattr(formatter, "format")
            assert hasattr(formatter, "validate_input")
            assert hasattr(formatter, "get_fallback")

            # All should return FormattedOutput
            result = formatter.format({})
            assert hasattr(result, "content")
            assert hasattr(result, "format_type")
            assert hasattr(result, "summary")
            assert hasattr(result, "contains_markup")

    def test_formatter_metadata_preservation(self):
        """Test that formatters preserve metadata correctly."""
        tool_data = {
            "summary": {"total_tests": 1, "passed": 1, "failed": 0, "skipped": 0},
            "failures": [],
        }

        result = format_test_results(tool_data)

        # FormattedOutput should have metadata field
        assert hasattr(result, "metadata")
        assert isinstance(result.metadata, dict)

    def test_line_count_accuracy(self):
        """Test that line_count is accurately calculated."""
        # Multi-line content
        tool_data = {
            "results": [
                {"path": "file1.py", "line": 1, "score": 10, "snippet": "code1"},
                {"path": "file2.py", "line": 2, "score": 10, "snippet": "code2"},
                {"path": "file3.py", "line": 3, "score": 10, "snippet": "code3"},
            ],
            "mode": "search",
        }

        result = format_search_results(tool_data)

        # Line count should match actual content lines
        assert result.line_count == len(result.content.splitlines())

    def test_contains_markup_flag_accuracy(self):
        """Test that contains_markup flag is set correctly."""
        # Test formatter that produces markup
        result1 = format_test_results(
            {
                "summary": {"total_tests": 1, "passed": 1, "failed": 0, "skipped": 0},
                "failures": [],
            }
        )
        assert result1.contains_markup is True
        assert "[" in result1.content  # Has markup tags

        # Test generic formatter (no markup)
        from victor.tools.formatters.generic import GenericFormatter

        generic = GenericFormatter()
        result2 = generic.format({"key": "value"})
        assert result2.contains_markup is False
        assert "[" not in result2.content  # No markup tags

    def test_concurrent_formatter_access(self):
        """Test that formatters are thread-safe."""
        import concurrent.futures

        def format_in_thread(tool_name, data):
            formatter = get_formatter_registry().get_formatter(tool_name)
            return formatter.format(data)

        # Access multiple formatters concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [
                executor.submit(
                    format_in_thread,
                    "test",
                    {
                        "summary": {
                            "total_tests": 1,
                            "passed": 1,
                            "failed": 0,
                            "skipped": 0,
                        },
                        "failures": [],
                    },
                ),
                executor.submit(format_in_thread, "code_search", {"results": [], "mode": "search"}),
                executor.submit(format_in_thread, "git", {"output": "", "operation": "status"}),
            ]

            # All should complete without errors
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

            assert len(results) == 3
            for result in results:
                assert isinstance(result, type(results[0]))


class TestFormatterToolCompatibility:
    """Test compatibility between tools and formatters."""

    def test_testing_tool_compatibility(self):
        """Test that test formatter works with actual tool output structure."""
        # Simulate actual testing tool output
        tool_output = {
            "summary": {
                "total_tests": 10,
                "passed": 8,
                "failed": 2,
                "skipped": 0,
                "xfailed": 0,
                "xpassed": 0,
            },
            "failures": [
                {
                    "test_name": "tests/test_api.py::test_get_user",
                    "error_message": "AssertionError: Expected 200 but got 404",
                    "full_error": "Traceback...",
                },
                {
                    "test_name": "tests/test_db.py::test_database_connection",
                    "error_message": "ConnectionError: Unable to connect",
                    "full_error": "Traceback...",
                },
            ],
        }

        # Format should work correctly
        result = format_test_results(tool_output)

        assert result.contains_markup is True
        assert "8 passed" in result.content
        assert "2 failed" in result.content
        assert "test_get_user" in result.content
        assert "test_database_connection" in result.content

    def test_code_search_tool_compatibility(self):
        """Test that search formatter works with actual tool output structure."""
        # Simulate actual code search tool output
        tool_output = {
            "success": True,
            "results": [
                {
                    "path": "src/api/users.py",
                    "line": 42,
                    "score": 9.5,
                    "snippet": "def get_user(user_id):",
                },
                {
                    "path": "src/models/user.py",
                    "line": 15,
                    "score": 8.2,
                    "snippet": "class User:",
                },
            ],
            "count": 2,
            "mode": "semantic",
        }

        # Format should work correctly
        result = format_search_results(tool_output)

        assert result.contains_markup is True
        assert "2 matches" in result.content
        assert "src/api/users.py" in result.content
        assert "src/models/user.py" in result.content

    def test_git_tool_compatibility(self):
        """Test that git formatter works with actual tool output structure."""
        # Simulate actual git tool output
        tool_output = {
            "success": True,
            "output": "* main\n  develop\nM  modified_file.py\nA  new_file.py\n",
            "error": "",
        }

        # Format should work correctly
        result = format_git_output(tool_output, operation="status")

        assert result.contains_markup is True
        assert "[bold green]*[/]" in result.content
        assert "[bold]main[/]" in result.content
        assert "[yellow]M [/]" in result.content
        assert "[green]A [/]" in result.content

    def test_http_tool_compatibility(self):
        """Test that HTTP formatter works with actual tool output structure."""
        # Simulate actual HTTP tool output
        tool_output = {
            "status_code": 200,
            "status": "OK",
            "duration_ms": 125,
            "headers": {
                "Content-Type": "application/json",
                "Content-Length": "1234",
            },
            "body": {"user": "alice", "id": 123},
        }

        # Format should work correctly
        result = format_http_response(tool_output)

        assert result.contains_markup is True
        assert "[green bold]200 OK[/]" in result.content
        assert "125ms" in result.content
        assert "application/json" in result.content

    def test_database_tool_compatibility(self):
        """Test that database formatter works with actual tool output structure."""
        # Simulate actual database tool output
        tool_output = {
            "success": True,
            "columns": ["id", "name", "email"],
            "rows": [
                [1, "Alice", "alice@example.com"],
                [2, "Bob", "bob@example.com"],
                [3, "Charlie", "charlie@example.com"],
            ],
            "count": 3,
        }

        # Format should work correctly
        result = format_database_results(tool_output)

        assert result.contains_markup is True
        assert "3 rows" in result.summary
        assert "id" in result.content or "Alice" in result.content


class TestFormatterErrorHandling:
    """Test error handling in the formatter system."""

    def test_formatter_handles_missing_fields(self):
        """Test that formatters handle missing optional fields gracefully."""
        # Test with minimal data
        result = format_test_results(
            {
                "summary": {"total_tests": 0},
                "failures": [],
            }
        )

        # Should still produce output
        assert result.contains_markup is True
        assert len(result.content) > 0

    def test_formatter_handles_invalid_data_types(self):
        """Test that formatters handle invalid data types gracefully."""
        from victor.tools.formatters.generic import GenericFormatter

        formatter = GenericFormatter()

        # Should handle various data types
        result1 = formatter.format("string data")
        assert result1.content == "string data"

        result2 = formatter.format(["list", "data"])
        assert "list" in result2.content

        result3 = formatter.format(12345)
        assert result3.content == "12345"

    def test_formatter_validation_errors(self):
        """Test that validation errors are handled gracefully."""
        from victor.tools.formatters.registry import format_tool_output

        # Invalid data for test formatter
        result = format_tool_output("test", "invalid_data")

        # Should fall back to plain text
        assert result.format_type == "plain"
        assert "Invalid test data" in result.summary

    def test_formatter_exception_in_format_method(self):
        """Test that exceptions in format() are caught and handled."""
        from victor.tools.formatters.base import ToolFormatter, FormattedOutput

        class BrokenFormatter(ToolFormatter):
            def validate_input(self, data):
                return True

            def format(self, data, **kwargs):
                raise ValueError("Intentional error")

        # Register broken formatter
        registry = get_formatter_registry()
        registry.register("broken", BrokenFormatter())

        # Use the format_tool_output function which has error handling
        from victor.tools.formatters.registry import format_tool_output

        result = format_tool_output("broken", {})

        # Should fall back to plain text on error
        assert isinstance(result, FormattedOutput)
        assert result.format_type == "plain"

    def test_empty_data_handling(self):
        """Test that formatters handle empty data correctly."""
        result1 = format_test_results(
            {
                "summary": {},
                "failures": [],
            }
        )

        result2 = format_search_results(
            {
                "results": [],
            }
        )

        result3 = format_git_output(
            {
                "output": "",
            }
        )

        # All should produce some output
        for result in [result1, result2, result3]:
            assert len(result.content) > 0
            assert result.contains_markup is True
