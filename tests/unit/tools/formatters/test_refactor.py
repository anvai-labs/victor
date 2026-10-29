"""Unit tests for RefactorFormatter."""

from victor.tools.formatters.refactor import RefactorFormatter


class TestRefactorFormatter:
    """Test RefactorFormatter."""

    def test_validate_input_valid(self):
        """Test validate_input() with valid data."""
        formatter = RefactorFormatter()

        assert formatter.validate_input({"operations": []}) is True
        assert formatter.validate_input({"plan": []}) is True
        assert formatter.validate_input({"changes": []}) is True
        assert formatter.validate_input({"operations": [{}]}) is True

    def test_validate_input_invalid(self):
        """Test validate_input() with invalid data."""
        formatter = RefactorFormatter()

        assert formatter.validate_input({}) is False
        assert formatter.validate_input(None) is False

    def test_format_no_operations(self):
        """Test formatting with no operations."""
        formatter = RefactorFormatter()
        data = {
            "operations": [],
        }

        result = formatter.format(data)

        assert result.contains_markup is True
        assert "[dim]No refactoring operations[/]" in result.content
        assert result.summary == "No operations"

    def test_format_rename_operation(self):
        """Test formatting rename operation."""
        formatter = RefactorFormatter()
        data = {
            "operations": [
                {
                    "type": "rename",
                    "description": "Rename function for clarity",
                    "from": "old_function",
                    "to": "new_function",
                }
            ]
        }

        result = formatter.format(data)

        assert result.contains_markup is True
        assert "[cyan]↝[/]" in result.content
        assert "[bold]Rename:[/]" in result.content
        assert "old_function" in result.content
        assert "new_function" in result.content
        assert result.summary == "1 operations"

    def test_format_extract_operation(self):
        """Test formatting extract operation."""
        formatter = RefactorFormatter()
        data = {
            "operations": [
                {
                    "type": "extract",
                    "description": "Extract method to reduce complexity",
                    "from": "long_function",
                    "to": "extracted_helper",
                }
            ]
        }

        result = formatter.format(data)

        assert result.contains_markup is True
        assert "[green]♢[/]" in result.content
        assert "[bold]Extract:[/]" in result.content

    def test_format_inline_operation(self):
        """Test formatting inline operation."""
        formatter = RefactorFormatter()
        data = {
            "operations": [
                {
                    "type": "inline",
                    "description": "Inline simple function",
                    "from": "simple_func",
                    "to": "caller",
                }
            ]
        }

        result = formatter.format(data)

        assert result.contains_markup is True
        assert "[yellow]♦[/]" in result.content
        assert "[bold]Inline:[/]" in result.content

    def test_format_move_operation(self):
        """Test formatting move operation."""
        formatter = RefactorFormatter()
        data = {
            "operations": [
                {
                    "type": "move",
                    "description": "Move class to appropriate module",
                }
            ]
        }

        result = formatter.format(data)

        assert result.contains_markup is True
        assert "[blue]→[/]" in result.content
        assert "[bold]Move:[/]" in result.content

    def test_format_delete_operation(self):
        """Test formatting delete operation."""
        formatter = RefactorFormatter()
        data = {
            "operations": [
                {
                    "type": "delete",
                    "description": "Remove unused code",
                }
            ]
        }

        result = formatter.format(data)

        assert result.contains_markup is True
        assert "[red]✗[/]" in result.content
        assert "[bold]Delete:[/]" in result.content

    def test_format_unknown_operation(self):
        """Test formatting unknown operation type."""
        formatter = RefactorFormatter()
        data = {
            "operations": [
                {
                    "type": "unknown",
                    "description": "Unknown operation",
                }
            ]
        }

        result = formatter.format(data)

        assert result.contains_markup is True
        assert "[white]•[/]" in result.content
        assert "[bold]Unknown:[/]" in result.content

    def test_format_with_file_path(self):
        """Test formatting operation with file path."""
        formatter = RefactorFormatter()
        data = {
            "operations": [
                {
                    "type": "rename",
                    "description": "Rename function",
                    "file": "src/module.py",
                }
            ]
        }

        result = formatter.format(data)

        assert result.contains_markup is True
        assert "[cyan]src/module.py[/]" in result.content

    def test_format_with_line_number(self):
        """Test formatting operation with line number."""
        formatter = RefactorFormatter()
        data = {
            "operations": [
                {
                    "type": "extract",
                    "description": "Extract method",
                    "line": 42,
                }
            ]
        }

        result = formatter.format(data)

        assert result.contains_markup is True
        assert "[dim]Line 42[/]" in result.content

    def test_format_multiple_operations(self):
        """Test formatting multiple operations."""
        formatter = RefactorFormatter()
        data = {
            "operations": [
                {
                    "type": "rename",
                    "description": "First rename",
                },
                {
                    "type": "extract",
                    "description": "Extract method",
                },
                {
                    "type": "inline",
                    "description": "Inline function",
                },
            ]
        }

        result = formatter.format(data)

        assert result.contains_markup is True
        assert "[cyan]↝[/]" in result.content
        assert "[green]♢[/]" in result.content
        assert "[yellow]♦[/]" in result.content
        assert result.summary == "3 operations"

    def test_format_max_operations(self):
        """Test max_operations parameter limits output."""
        formatter = RefactorFormatter()
        data = {
            "operations": [{"type": "rename", "description": f"Operation {i}"} for i in range(25)]
        }

        result = formatter.format(data, max_operations=20)

        assert result.contains_markup is True
        assert "... and 5 more operations" in result.content

    def test_format_plan_key(self):
        """Test that 'plan' key is supported."""
        formatter = RefactorFormatter()
        data = {
            "plan": [
                {
                    "type": "rename",
                    "description": "Test plan",
                }
            ]
        }

        result = formatter.format(data)

        assert result.contains_markup is True
        assert "[cyan]↝[/]" in result.content

    def test_format_changes_key(self):
        """Test that 'changes' key is supported."""
        formatter = RefactorFormatter()
        data = {
            "changes": [
                {
                    "type": "extract",
                    "description": "Test changes",
                }
            ]
        }

        result = formatter.format(data)

        assert result.contains_markup is True
        assert "[green]♢[/]" in result.content

    def test_format_missing_optional_fields(self):
        """Test formatting with missing optional fields."""
        formatter = RefactorFormatter()
        data = {
            "operations": [
                {
                    "type": "rename",
                    # Missing description, from, to, file, line
                }
            ]
        }

        result = formatter.format(data)

        assert result.contains_markup is True
        assert "[bold]Rename:[/]" in result.content
