"""Unit tests for TestResultsFormatter."""

from victor.tools.formatters.test_results import TestResultsFormatter


class TestTestResultsFormatter:
    """Test TestResultsFormatter."""

    def test_validate_input_valid(self):
        """Test validate_input() with valid data."""
        formatter = TestResultsFormatter()

        assert formatter.validate_input({"summary": {}}) is True
        assert formatter.validate_input({"summary": {}, "failures": []}) is True

    def test_validate_input_invalid(self):
        """Test validate_input() with invalid data."""
        formatter = TestResultsFormatter()

        assert formatter.validate_input({}) is False
        assert formatter.validate_input({"failures": []}) is False
        assert formatter.validate_input(None) is False

    def test_format_all_passed(self):
        """Test formatting with all tests passed."""
        formatter = TestResultsFormatter()
        data = {
            "summary": {
                "total_tests": 10,
                "passed": 10,
                "failed": 0,
                "skipped": 0,
            },
            "failures": [],
        }

        result = formatter.format(data)

        assert result.contains_markup is True
        assert "[green]✓ 10 passed[/]" in result.content
        assert "10 total" in result.content
        assert result.summary == "10 tests"

    def test_format_with_failures(self):
        """Test formatting with test failures."""
        formatter = TestResultsFormatter()
        data = {
            "summary": {
                "total_tests": 10,
                "passed": 8,
                "failed": 2,
                "skipped": 0,
            },
            "failures": [
                {
                    "test_name": "test_module.py::TestClass::test_method",
                    "error_message": "AssertionError: Expected 5 but got 3",
                },
                {
                    "test_name": "test_another.py::test_other",
                    "error_message": "ValueError: Invalid input",
                },
            ],
        }

        result = formatter.format(data)

        assert result.contains_markup is True
        assert "[green]✓ 8 passed[/]" in result.content
        assert "[red]✗ 2 failed[/]" in result.content
        assert "[red bold]Failed Tests:[/]" in result.content
        assert "test_method" in result.content
        assert "test_module.py" in result.content
        assert "AssertionError: Expected 5 but got 3" in result.content

    def test_format_with_skipped(self):
        """Test formatting with skipped tests."""
        formatter = TestResultsFormatter()
        data = {
            "summary": {
                "total_tests": 10,
                "passed": 7,
                "failed": 1,
                "skipped": 2,
            },
            "failures": [
                {
                    "test_name": "test_failing.py::test_fail",
                    "error_message": "AssertionError",
                }
            ],
        }

        result = formatter.format(data)

        assert result.contains_markup is True
        assert "[green]✓ 7 passed[/]" in result.content
        assert "[red]✗ 1 failed[/]" in result.content
        assert "[yellow]○ 2 skipped[/]" in result.content

    def test_format_max_failures(self):
        """Test max_failures parameter truncates output."""
        formatter = TestResultsFormatter()
        data = {
            "summary": {
                "total_tests": 20,
                "passed": 10,
                "failed": 10,
                "skipped": 0,
            },
            "failures": [
                {"test_name": f"test_{i}.py::test_{i}", "error_message": f"Error {i}"}
                for i in range(10)
            ],
        }

        result = formatter.format(data, max_failures=3)

        assert result.contains_markup is True
        # Should show first 3 failures
        assert "test_0" in result.content
        assert "test_1" in result.content
        assert "test_2" in result.content
        # Should indicate more failures
        assert "... and 7 more failures" in result.content

    def test_format_truncates_long_error_messages(self):
        """Test that long error messages are truncated."""
        formatter = TestResultsFormatter()
        data = {
            "summary": {
                "total_tests": 1,
                "passed": 0,
                "failed": 1,
                "skipped": 0,
            },
            "failures": [
                {
                    "test_name": "test.py::test_long_error",
                    "error_message": "A" * 150,  # Long error message
                }
            ],
        }

        result = formatter.format(data)

        assert result.contains_markup is True
        # Error message is truncated but overall content is longer due to markup
        assert "..." in result.content  # Truncation indicator present

    def test_format_no_tests(self):
        """Test formatting when no tests were run."""
        formatter = TestResultsFormatter()
        data = {
            "summary": {
                "total_tests": 0,
                "passed": 0,
                "failed": 0,
                "skipped": 0,
            },
            "failures": [],
        }

        result = formatter.format(data)

        assert result.contains_markup is True
        # When total=0, shows "No tests run" instead of status line
        assert "[dim]No tests run[/]" in result.content

    def test_format_test_name_without_separator(self):
        """Test formatting test name without :: separator."""
        formatter = TestResultsFormatter()
        data = {
            "summary": {
                "total_tests": 1,
                "passed": 0,
                "failed": 1,
                "skipped": 0,
            },
            "failures": [
                {
                    "test_name": "simple_test_name",
                    "error_message": "Error",
                }
            ],
        }

        result = formatter.format(data)

        assert result.contains_markup is True
        assert "simple_test_name" in result.content

    def test_format_missing_test_name(self):
        """Test formatting when test_name is missing."""
        formatter = TestResultsFormatter()
        data = {
            "summary": {
                "total_tests": 1,
                "passed": 0,
                "failed": 1,
                "skipped": 0,
            },
            "failures": [
                {
                    "test_name": "",
                    "error_message": "Error",
                }
            ],
        }

        result = formatter.format(data)

        assert result.contains_markup is True
        assert "[red]✗[/]" in result.content
