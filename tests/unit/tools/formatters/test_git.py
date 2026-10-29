"""Unit tests for GitFormatter."""

from victor.tools.formatters.git import GitFormatter


class TestGitFormatter:
    """Test GitFormatter."""

    def test_validate_input_valid(self):
        """Test validate_input() with valid data."""
        formatter = GitFormatter()

        assert formatter.validate_input({"output": ""}) is True
        assert formatter.validate_input({"formatted_output": "test"}) is True

    def test_validate_input_invalid(self):
        """Test validate_input() with invalid data."""
        formatter = GitFormatter()

        assert formatter.validate_input({}) is False
        assert formatter.validate_input(None) is False

    def test_format_status_branch(self):
        """Test formatting git status output."""
        formatter = GitFormatter()
        data = {
            "output": "* main\n  develop\n",
        }

        result = formatter.format(data, operation="status")

        assert result.contains_markup is True
        assert "[bold green]*[/]" in result.content
        assert "[bold]main[/]" in result.content
        assert "[dim]develop[/]" in result.content
        assert result.summary == "git status"

    def test_format_status_file_changes(self):
        """Test formatting file status in git output."""
        formatter = GitFormatter()
        data = {
            "output": "M  modified_file.py\nA  new_file.py\nD  deleted_file.py\n",
        }

        result = formatter.format(data, operation="status")

        assert result.contains_markup is True
        assert "[yellow]M [/]" in result.content
        assert "[green]A [/]" in result.content
        assert "[red]D [/]" in result.content
        assert "modified_file.py" in result.content

    def test_format_status_renamed_file(self):
        """Test formatting renamed file in git output."""
        formatter = GitFormatter()
        data = {
            "output": "R  old_name.py -> new_name.py\n",
        }

        result = formatter.format(data, operation="status")

        assert result.contains_markup is True
        assert "[cyan]R [/]" in result.content

    def test_format_log(self):
        """Test formatting git log output."""
        formatter = GitFormatter()
        data = {
            "output": "abc123 Fix bug\ndef456 Add feature\n",
        }

        result = formatter.format(data, operation="log")

        assert result.contains_markup is True
        assert "[cyan]abc123[/]" in result.content
        assert "[dim]Fix bug[/]" in result.content
        assert "[cyan]def456[/]" in result.content

    def test_format_diff(self):
        """Test formatting git diff output."""
        formatter = GitFormatter()
        data = {
            "output": "-old line\n+new line\n",
        }

        result = formatter.format(data, operation="diff")

        assert result.contains_markup is True
        # Diff formatter should add colors
        assert result.format_type == "rich"

    def test_format_branch(self):
        """Test formatting git branch output."""
        formatter = GitFormatter()
        data = {
            "output": "* main\n  develop\n  feature-branch\n",
        }

        result = formatter.format(data, operation="branch")

        assert result.contains_markup is True
        assert "[bold green]*[/]" in result.content
        assert "[bold]main[/]" in result.content
        assert "[dim]develop[/]" in result.content

    def test_format_with_preformatted_output(self):
        """Test that pre-formatted output is passed through."""
        formatter = GitFormatter()
        preformatted = "[bold]Already formatted[/]"
        data = {
            "formatted_output": preformatted,
        }

        result = formatter.format(data, operation="status")

        assert result.content == preformatted
        assert result.contains_markup is True

    def test_format_empty_output(self):
        """Test formatting empty output."""
        formatter = GitFormatter()
        data = {
            "output": "",
        }

        result = formatter.format(data, operation="status")

        assert result.contains_markup is True
        assert "[dim]No output[/]" in result.content

    def test_format_unknown_operation(self):
        """Test formatting unknown operation passes through."""
        formatter = GitFormatter()
        data = {
            "output": "Some raw output\n",
        }

        result = formatter.format(data, operation="unknown")

        assert result.contains_markup is True
        assert "Some raw output" in result.content

    def test_format_status_mixed_output(self):
        """Test formatting mixed status output."""
        formatter = GitFormatter()
        data = {
            "output": "* main\nM  file1.py\nA  file2.py\n  feature\n",
        }

        result = formatter.format(data, operation="status")

        assert result.contains_markup is True
        assert "[bold green]*[/]" in result.content
        assert "[yellow]M [/]" in result.content
        assert "[green]A [/]" in result.content

    def test_format_log_without_message(self):
        """Test formatting log with hash only."""
        formatter = GitFormatter()
        data = {
            "output": "abc123\n",
        }

        result = formatter.format(data, operation="log")

        assert result.contains_markup is True
        assert "[cyan]abc123[/]" in result.content
