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

"""Tests for Build formatter."""

import pytest

from victor.tools.formatters.build import BuildFormatter
from victor.tools.formatters.base import FormattedOutput


class TestBuildFormatter:
    """Test BuildFormatter for build tool operations."""

    def test_validate_input_valid_with_operation(self):
        """Test validation with operation field."""
        formatter = BuildFormatter()
        data = {"operation": "build"}
        assert formatter.validate_input(data) is True

    def test_validate_input_valid_with_target(self):
        """Test validation with target field."""
        formatter = BuildFormatter()
        data = {"target": "myapp"}
        assert formatter.validate_input(data) is True

    def test_validate_input_valid_with_success(self):
        """Test validation with success field."""
        formatter = BuildFormatter()
        data = {"success": True}
        assert formatter.validate_input(data) is True

    def test_validate_input_invalid(self):
        """Test validation with invalid data."""
        formatter = BuildFormatter()
        data = {"invalid": "data"}
        assert formatter.validate_input(data) is False

    def test_format_successful_build(self):
        """Test formatting successful build."""
        formatter = BuildFormatter()
        data = {
            "tool": "make",
            "operation": "build",
            "target": "myapp",
            "success": True,
            "duration_ms": 1250,
        }

        result = formatter.format(data)

        assert isinstance(result, FormattedOutput)
        assert result.contains_markup is True
        assert result.format_type == "rich"
        assert "[green]✓ Build succeeded[/]" in result.content
        assert "1250" in result.content  # Duration in ms, not converted to seconds

    def test_format_failed_build(self):
        """Test formatting failed build."""
        formatter = BuildFormatter()
        data = {
            "tool": "cargo",
            "operation": "build",
            "success": False,
            "errors": [{"file": "src/main.rs", "line": 42, "message": "expected identifier"}],
        }

        result = formatter.format(data)

        assert isinstance(result, FormattedOutput)
        assert result.contains_markup is True
        assert "[red]✗ Build failed[/]" in result.content
        assert "[red bold]Errors (1):[/]" in result.content  # Includes count

    def test_format_with_warnings(self):
        """Test formatting build with warnings."""
        formatter = BuildFormatter()
        warnings = [{"message": "unused variable: x"}, {"message": "dead code"}]
        data = {
            "tool": "cmake",
            "operation": "configure",
            "success": True,
            "warnings": warnings,
        }

        result = formatter.format(data)

        assert isinstance(result, FormattedOutput)
        assert result.contains_markup is True
        assert "[yellow]⚠ Warnings (2):[/]" in result.content  # Includes count
        assert "unused variable: x" in result.content

    def test_format_with_artifacts(self):
        """Test formatting build with artifacts."""
        formatter = BuildFormatter()
        artifacts = [
            {"name": "myapp", "size": "2.5 MB"},
            {"name": "libmyapp.so", "size": "1.2 MB"},
        ]
        data = {
            "tool": "cargo",
            "operation": "build",
            "success": True,
            "artifacts": artifacts,
        }

        result = formatter.format(data)

        assert isinstance(result, FormattedOutput)
        assert result.contains_markup is True
        assert "[bold]Artifacts (2):[/]" in result.content  # Includes count
        assert "[green]✓[/] myapp" in result.content

    def test_format_with_build_steps(self):
        """Test formatting build with steps."""
        formatter = BuildFormatter()
        steps = [
            {"name": "Compiling", "success": True, "duration_ms": 500},
            {"name": "Linking", "success": True, "duration_ms": 200},
            {"name": "Packaging", "success": False, "duration_ms": 100},
        ]
        data = {"tool": "make", "operation": "all", "success": False, "steps": steps}

        result = formatter.format(data)

        assert isinstance(result, FormattedOutput)
        assert result.contains_markup is True
        assert "[bold]Steps:[/]" in result.content
        assert "[green]✓[/] Compiling" in result.content
        assert "[red]✗[/] Packaging" in result.content

    def test_format_with_max_errors(self):
        """Test formatting with max_errors limit."""
        formatter = BuildFormatter()
        errors = [{"file": f"file{i}.py", "line": i, "message": f"error {i}"} for i in range(20)]
        data = {
            "tool": "python",
            "operation": "compile",
            "success": False,
            "errors": errors,
        }

        result = formatter.format(data, max_errors=5)

        assert isinstance(result, FormattedOutput)
        assert result.contains_markup is True
        assert "more errors" in result.content

    def test_format_with_max_warnings(self):
        """Test formatting with max_warnings limit."""
        formatter = BuildFormatter()
        warnings = [{"message": f"warning {i}"} for i in range(30)]
        data = {
            "tool": "gcc",
            "operation": "compile",
            "success": True,
            "warnings": warnings,
        }

        result = formatter.format(data, max_warnings=10)

        assert isinstance(result, FormattedOutput)
        assert result.contains_markup is True
        assert "more warnings" in result.content

    def test_summary_extraction_with_target(self):
        """Test summary extraction with target."""
        formatter = BuildFormatter()
        data = {
            "tool": "make",
            "operation": "build",
            "target": "myapp",
            "success": True,
        }

        result = formatter.format(data)

        assert "✓" in result.summary
        assert "Make" in result.summary
        assert "build myapp" in result.summary

    def test_summary_extraction_without_target(self):
        """Test summary extraction without target."""
        formatter = BuildFormatter()
        data = {"tool": "cargo", "operation": "test", "success": False}

        result = formatter.format(data)

        assert "✗" in result.summary
        assert "Cargo" in result.summary
        assert "test" in result.summary

    def test_fallback_formatter(self):
        """Test fallback formatter is returned."""
        formatter = BuildFormatter()
        fallback = formatter.get_fallback()

        assert fallback is not None
        assert hasattr(fallback, "format")
