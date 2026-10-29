"""Unit tests for SearchResultsFormatter."""

from victor.tools.formatters.search import SearchResultsFormatter


class TestSearchResultsFormatter:
    """Test SearchResultsFormatter."""

    def test_validate_input_valid(self):
        """Test validate_input() with valid data."""
        formatter = SearchResultsFormatter()

        assert formatter.validate_input({"results": []}) is True
        assert formatter.validate_input({"matches": []}) is True
        assert formatter.validate_input({"results": [{}]}) is True

    def test_validate_input_invalid(self):
        """Test validate_input() with invalid data."""
        formatter = SearchResultsFormatter()

        assert formatter.validate_input({}) is False
        assert formatter.validate_input({"other": []}) is False
        assert formatter.validate_input(None) is False

    def test_format_no_matches(self):
        """Test formatting with no matches."""
        formatter = SearchResultsFormatter()
        data = {"results": [], "mode": "semantic"}

        result = formatter.format(data)

        assert result.contains_markup is True
        assert "[dim]No matches found[/]" in result.content
        # Summary doesn't include mode when there are no results
        assert result.summary == "0 matches"

    def test_format_with_matches(self):
        """Test formatting with search results."""
        formatter = SearchResultsFormatter()
        data = {
            "results": [
                {
                    "path": "src/main.py",
                    "line": 42,
                    "score": 10,
                    "snippet": "def hello_world():",
                },
                {
                    "path": "src/utils.py",
                    "line": 15,
                    "score": 8,
                    "snippet": "def utility_func():",
                },
            ],
            "mode": "semantic",
        }

        result = formatter.format(data)

        assert result.contains_markup is True
        assert "[bold cyan]2 matches[/]" in result.content
        assert "[dim]in 2 files[/]" in result.content
        assert "src/main.py" in result.content
        assert "src/utils.py" in result.content
        assert "def hello_world():" in result.content
        assert result.summary == "2 matches (semantic)"

    def test_format_groups_by_file(self):
        """Test that results are grouped by file."""
        formatter = SearchResultsFormatter()
        data = {
            "results": [
                {
                    "path": "src/main.py",
                    "line": 42,
                    "score": 10,
                    "snippet": "def func1():",
                },
                {
                    "path": "src/main.py",
                    "line": 55,
                    "score": 9,
                    "snippet": "def func2():",
                },
                {
                    "path": "src/utils.py",
                    "line": 15,
                    "score": 8,
                    "snippet": "def helper():",
                },
            ],
            "mode": "semantic",
        }

        result = formatter.format(data)

        assert result.contains_markup is True
        # Should show 2 files, not 3 matches
        assert "[dim]in 2 files[/]" in result.content

    def test_format_calculates_average_score(self):
        """Test that average score is calculated per file."""
        formatter = SearchResultsFormatter()
        data = {
            "results": [
                {
                    "path": "src/main.py",
                    "line": 42,
                    "score": 10,
                    "snippet": "def func1():",
                },
                {
                    "path": "src/main.py",
                    "line": 55,
                    "score": 6,
                    "snippet": "def func2():",
                },
            ],
            "mode": "semantic",
        }

        result = formatter.format(data)

        assert result.contains_markup is True
        # Average of 10 and 6 is 8
        assert "score: 8" in result.content

    def test_format_max_files(self):
        """Test max_files parameter limits output."""
        formatter = SearchResultsFormatter()
        data = {
            "results": [
                {
                    "path": f"file_{i}.py",
                    "line": 1,
                    "score": 10,
                    "snippet": f"def func_{i}():",
                }
                for i in range(15)
            ],
            "mode": "semantic",
        }

        result = formatter.format(data, max_files=10)

        assert result.contains_markup is True
        assert "... and 5 more files" in result.content

    def test_format_max_matches_per_file(self):
        """Test max_matches_per_file parameter limits matches per file."""
        formatter = SearchResultsFormatter()
        data = {
            "results": [
                {
                    "path": "src/main.py",
                    "line": i,
                    "score": 10,
                    "snippet": f"def func_{i}():",
                }
                for i in range(10)
            ],
            "mode": "semantic",
        }

        result = formatter.format(data, max_matches_per_file=3)

        assert result.contains_markup is True
        assert "... and 7 more matches in this file" in result.content

    def test_format_truncates_long_snippets(self):
        """Test that long snippets are truncated."""
        formatter = SearchResultsFormatter()
        data = {
            "results": [
                {
                    "path": "src/main.py",
                    "line": 42,
                    "score": 10,
                    "snippet": "a" * 150,  # Long snippet
                }
            ],
            "mode": "semantic",
        }

        result = formatter.format(data)

        assert result.contains_markup is True
        # Should be truncated to ~100 chars
        assert "..." in result.content

    def test_format_handles_missing_fields(self):
        """Test formatting with missing optional fields."""
        formatter = SearchResultsFormatter()
        data = {
            "results": [
                {
                    "path": "src/main.py",
                    # Missing line, score, snippet
                }
            ],
            "mode": "semantic",
        }

        result = formatter.format(data)

        assert result.contains_markup is True
        assert "src/main.py" in result.content

    def test_format_matches_key_fallback(self):
        """Test that 'matches' key is supported as fallback."""
        formatter = SearchResultsFormatter()
        data = {
            "matches": [
                {
                    "path": "src/main.py",
                    "line": 42,
                    "score": 10,
                    "snippet": "def test():",
                }
            ],
            "mode": "literal",
        }

        result = formatter.format(data)

        assert result.contains_markup is True
        assert "src/main.py" in result.content
        assert result.summary == "1 matches (literal)"

    def test_format_default_mode(self):
        """Test default mode when not specified."""
        formatter = SearchResultsFormatter()
        data = {
            "results": [
                {
                    "path": "src/main.py",
                    "line": 42,
                    "score": 10,
                    "snippet": "def test():",
                }
            ]
        }

        result = formatter.format(data)

        assert result.contains_markup is True
        assert result.summary == "1 matches (search)"
