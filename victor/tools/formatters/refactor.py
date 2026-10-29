"""Refactor operations formatter with Rich markup."""

from typing import Dict, Any

from .base import ToolFormatter, FormattedOutput


class RefactorFormatter(ToolFormatter):
    """Format refactor operations with Rich markup.

    Produces formatted output for:
    - Refactoring plans with operation types
    - Rename operations (cyan icon)
    - Extract operations (green icon)
    - Inline operations (yellow icon)
    - From/To paths for operations
    """

    def validate_input(self, data: Dict) -> bool:
        """Validate refactor result has required fields."""
        return isinstance(data, dict) and (
            "operations" in data or "plan" in data or "changes" in data
        )

    def format(self, data: Dict[str, Any], max_operations: int = 20, **kwargs) -> FormattedOutput:
        """Format refactor operations with Rich markup.

        Args:
            data: Refactor result dict with operations, plan, or changes
            max_operations: Maximum operations to display (default: 20)

        Returns:
            FormattedOutput with Rich markup
        """
        # Support multiple result structures
        operations = data.get("operations") or data.get("plan") or data.get("changes") or []

        if not operations:
            return FormattedOutput(
                content="[dim]No refactoring operations[/]",
                format_type="rich",
                summary="No operations",
                line_count=1,
                contains_markup=True,
            )

        lines = []
        lines.append("[bold cyan]Refactoring Plan:[/]")
        lines.append("")

        # Display operations
        for i, op in enumerate(list(operations)[:max_operations], 1):
            op_type = op.get("type", "unknown")

            # Color-code operation type
            if op_type == "rename":
                color = "cyan"
                icon = "↝"
            elif op_type == "extract":
                color = "green"
                icon = "♢"
            elif op_type == "inline":
                color = "yellow"
                icon = "♦"
            elif op_type == "move":
                color = "blue"
                icon = "→"
            elif op_type == "delete":
                color = "red"
                icon = "✗"
            else:
                color = "white"
                icon = "•"

            # Operation header
            description = op.get("description", "")
            lines.append(f"  [{color}]{icon}[/] [bold]{op_type.title()}:[/] [dim]{description}[/]")

            # Details
            if "from" in op and "to" in op:
                from_path = op["from"]
                to_path = op["to"]
                lines.append(f"    [dim]{from_path}[/] [dim]→[/] [dim]{to_path}[/]")

            # File path if available
            if "file" in op:
                file_path = op["file"]
                lines.append(f"    [cyan]{file_path}[/]")

            # Line numbers if available
            if "line" in op:
                line_num = op["line"]
                lines.append(f"    [dim]Line {line_num}[/]")

        # Add indicator if truncated
        if len(operations) > max_operations:
            lines.append("")
            lines.append(f"[dim]... and {len(operations) - max_operations} more operations[/]")

        content = "\n".join(lines)

        return FormattedOutput(
            content=content,
            format_type="rich",
            summary=f"{len(operations)} operations",
            line_count=len(lines),
            contains_markup=True,
        )
