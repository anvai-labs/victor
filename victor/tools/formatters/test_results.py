"""Test results formatter with Rich markup."""

from typing import Dict, Any

from .base import ToolFormatter, FormattedOutput


class TestResultsFormatter(ToolFormatter):
    """Format test results with Rich markup.

    Produces color-coded output with:
    - Green checkmarks for passed tests
    - Red X marks for failed tests
    - Yellow circles for skipped tests
    - Cyan file paths for test locations
    - Dimmed error messages
    """

    def validate_input(self, data: Dict) -> bool:
        """Validate test result has required fields."""
        return isinstance(data, dict) and "summary" in data

    def format(self, data: Dict[str, Any], max_failures: int = 5, **kwargs) -> FormattedOutput:
        """Format test results with color-coded status.

        Args:
            data: Test result dict with 'summary' and 'failures' keys
            max_failures: Maximum number of failures to show (default: 5)

        Returns:
            FormattedOutput with Rich markup
        """
        summary = data.get("summary", {})
        failures = data.get("failures", [])

        lines = []

        # Extract summary statistics
        total = summary.get("total_tests", 0)
        passed = summary.get("passed", 0)
        failed = summary.get("failed", 0)
        skipped = summary.get("skipped", 0)

        # Build status line with color-coded indicators
        status_parts = []
        if passed > 0:
            status_parts.append(f"[green]✓ {passed} passed[/]")
        if failed > 0:
            status_parts.append(f"[red]✗ {failed} failed[/]")
        if skipped > 0:
            status_parts.append(f"[yellow]○ {skipped} skipped[/]")

        # Add total test count
        if total > 0:
            lines.append(" ".join(status_parts) + f" [dim]• {total} total[/]")
        else:
            lines.append(" ".join(status_parts) if status_parts else "[dim]No tests run[/]")

        lines.append("")  # Blank line for readability

        # Add failures section if any
        if failures:
            lines.append("[red bold]Failed Tests:[/]")
            lines.append("")

            # Show up to max_failures failures with details
            for i, failure in enumerate(failures[:max_failures], 1):
                test_name = failure.get("test_name", "unknown")
                error_msg = failure.get("error_message", "No error message")

                # Parse test_name to extract file path and function
                # Format: path/to/test.py::TestClass::test_method
                if "::" in test_name:
                    parts = test_name.split("::")
                    file_path = parts[0]
                    test_func = "::".join(parts[1:])  # Keep class::method structure

                    lines.append(f"  [red]✗[/] [bold]{test_func}[/]")
                    lines.append(f"    [dim cyan]{file_path}[/]")
                else:
                    lines.append(f"  [red]✗[/] [bold]{test_name}[/]")

                # Add error message (truncated if too long)
                if len(error_msg) > 100:
                    error_msg = error_msg[:97] + "..."
                lines.append(f"    [dim]{error_msg}[/]")

                # Add blank line between failures (except after last one)
                if i < len(failures) and i < min(max_failures, len(failures)):
                    lines.append("")

            # Add indicator if there are more failures
            if len(failures) > max_failures:
                lines.append(
                    f"  [dim]... and {len(failures) - max_failures} more failure{'s' if len(failures) - max_failures > 1 else ''}[/]"
                )

        content = "\n".join(lines)

        return FormattedOutput(
            content=content,
            format_type="rich",
            summary=f"{total} tests",
            line_count=len(lines),
            contains_markup=True,
        )
