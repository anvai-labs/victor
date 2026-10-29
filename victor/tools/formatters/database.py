"""Database query results formatter with Rich markup."""

from typing import Dict, Any

from .base import ToolFormatter, FormattedOutput


class DatabaseFormatter(ToolFormatter):
    """Format database query results with Rich markup.

    Produces formatted output for:
    - Query results as Rich table (if Rich is available)
    - Row counts and statistics
    - Error messages with color coding
    """

    def validate_input(self, data: Dict) -> bool:
        """Validate database result has required fields."""
        return isinstance(data, dict) and ("rows" in data or "error" in data or "success" in data)

    def format(
        self, data: Dict[str, Any], max_rows: int = 50, max_columns: int = 10, **kwargs
    ) -> FormattedOutput:
        """Format database query results with Rich markup.

        Args:
            data: Database result dict with rows, columns, count, error, etc.
            max_rows: Maximum rows to display (default: 50)
            max_columns: Maximum columns to display (default: 10)

        Returns:
            FormattedOutput with Rich markup or plain text
        """
        # Check for error
        if "error" in data:
            error_msg = data.get("error", "Unknown error")
            return FormattedOutput(
                content=f"[red bold]Error:[/] [dim]{error_msg}[/]",
                format_type="rich",
                summary="Query failed",
                line_count=1,
                contains_markup=True,
            )

        # Check for success flag
        if not data.get("success", True):
            return FormattedOutput(
                content="[red bold]Query failed[/]",
                format_type="rich",
                summary="Query failed",
                line_count=1,
                contains_markup=True,
            )

        rows = data.get("rows", [])
        columns = data.get("columns", [])
        count = data.get("count", len(rows))

        lines = []

        if not rows:
            lines.append("[dim]No results[/]")
        else:
            # Try to use Rich table if available
            try:
                from rich.table import Table
                from rich.console import Console
                from io import StringIO

                console = Console(file=StringIO(), force_terminal=True)
                table = Table(show_header=True, header_style="bold magenta")

                # Add columns (limited to max_columns)
                for col in columns[:max_columns]:
                    table.add_column(str(col))

                # Add rows (limited to max_rows)
                for row in rows[:max_rows]:
                    # Ensure row values are strings
                    str_row = [str(cell) for cell in row[:max_columns]]
                    table.add_row(*str_row)

                # Render table to string
                console.print(table)
                table_output = console.file.getvalue()

                # Add table output to lines
                table_lines = table_output.strip().split("\n")
                lines.extend(table_lines)

            except Exception:
                # Fallback to plain text formatting if Rich table fails
                lines.append("[bold]Query Results:[/]")
                lines.append("")

                # Header row
                if columns:
                    header = "  |  ".join(str(col)[:15] for col in columns[:max_columns])
                    lines.append(f"[cyan]{header}[/]")
                    lines.append("-" * len(header))

                # Data rows
                for row in rows[:max_rows]:
                    row_str = "  |  ".join(str(cell)[:15] for cell in row[:max_columns])
                    lines.append(f"  {row_str}")

            # Add row count summary
            lines.append("")
            lines.append(f"[dim]{count} row{'s' if count != 1 else ''} returned[/]")

            # Add indicator if truncated
            if len(rows) > max_rows:
                lines.append(f"[dim]... and {len(rows) - max_rows} more rows truncated[/]")
            if len(columns) > max_columns:
                lines.append(f"[dim]... and {len(columns) - max_columns} more columns truncated[/]")

        content = "\n".join(lines)

        return FormattedOutput(
            content=content,
            format_type="rich",
            summary=f"{count} rows",
            line_count=len(lines),
            contains_markup=True,
        )
