"""Unit tests for DatabaseFormatter."""

from victor.tools.formatters.database import DatabaseFormatter


class TestDatabaseFormatter:
    """Test DatabaseFormatter."""

    def test_validate_input_valid(self):
        """Test validate_input() with valid data."""
        formatter = DatabaseFormatter()

        assert formatter.validate_input({"rows": []}) is True
        assert formatter.validate_input({"error": "test"}) is True
        assert formatter.validate_input({"success": True}) is True

    def test_validate_input_invalid(self):
        """Test validate_input() with invalid data."""
        formatter = DatabaseFormatter()

        assert formatter.validate_input({}) is False
        assert formatter.validate_input(None) is False

    def test_format_error(self):
        """Test formatting error response."""
        formatter = DatabaseFormatter()
        data = {
            "error": "Connection failed",
        }

        result = formatter.format(data)

        assert result.contains_markup is True
        assert "[red bold]Error:[/]" in result.content
        assert "Connection failed" in result.content
        assert result.summary == "Query failed"

    def test_format_no_success(self):
        """Test formatting failed success flag."""
        formatter = DatabaseFormatter()
        data = {
            "success": False,
        }

        result = formatter.format(data)

        assert result.contains_markup is True
        assert "[red bold]Query failed[/]" in result.content

    def test_format_no_results(self):
        """Test formatting empty result set."""
        formatter = DatabaseFormatter()
        data = {
            "rows": [],
            "columns": [],
            "count": 0,
        }

        result = formatter.format(data)

        assert result.contains_markup is True
        assert "[dim]No results[/]" in result.content
        # Summary shows count, but content just says "No results"
        assert result.summary == "0 rows"

    def test_format_with_results(self):
        """Test formatting query results."""
        formatter = DatabaseFormatter()
        data = {
            "columns": ["id", "name", "value"],
            "rows": [
                [1, "Alice", 100],
                [2, "Bob", 200],
                [3, "Charlie", 300],
            ],
            "count": 3,
        }

        result = formatter.format(data)

        assert result.contains_markup is True
        assert "id" in result.content
        assert "Alice" in result.content
        assert "3 rows returned" in result.content
        assert result.summary == "3 rows"

    def test_format_max_rows(self):
        """Test max_rows parameter limits output."""
        formatter = DatabaseFormatter()
        data = {
            "columns": ["id"],
            "rows": [[i] for i in range(100)],
            "count": 100,
        }

        result = formatter.format(data, max_rows=50)

        assert result.contains_markup is True
        assert "... and 50 more rows truncated" in result.content

    def test_format_max_columns(self):
        """Test max_columns parameter limits columns."""
        formatter = DatabaseFormatter()
        data = {
            "columns": [f"col_{i}" for i in range(15)],
            "rows": [list(range(15))],
            "count": 1,
        }

        result = formatter.format(data, max_columns=10)

        assert result.contains_markup is True
        assert "... and 5 more columns truncated" in result.content

    def test_format_singular_row(self):
        """Test formatting singular row count."""
        formatter = DatabaseFormatter()
        data = {
            "columns": ["id"],
            "rows": [[1]],
            "count": 1,
        }

        result = formatter.format(data)

        assert result.contains_markup is True
        assert "1 row returned" in result.content  # Not "rows"

    def test_format_plaintext_fallback(self):
        """Test plain text fallback when Rich fails."""
        formatter = DatabaseFormatter()
        data = {
            "columns": ["id", "name"],
            "rows": [[1, "Alice"]],
            "count": 1,
        }

        result = formatter.format(data)

        assert result.contains_markup is True
        assert "id" in result.content or "Alice" in result.content

    def test_format_missing_optional_fields(self):
        """Test formatting with missing optional fields."""
        formatter = DatabaseFormatter()
        data = {
            "rows": [[1, 2, 3]],
            # Missing columns, count
        }

        result = formatter.format(data)

        assert result.contains_markup is True
        # Should still format something
        assert len(result.content) > 0

    def test_format_with_empty_columns(self):
        """Test formatting results with no columns."""
        formatter = DatabaseFormatter()
        data = {
            "columns": [],
            "rows": [],
            "count": 0,
        }

        result = formatter.format(data)

        assert result.contains_markup is True
        # Empty columns/rows shows "No results" in content
        assert "[dim]No results[/]" in result.content
        assert result.summary == "0 rows"
