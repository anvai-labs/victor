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

"""Common table builders for CLI commands.

This module provides factory functions for creating consistently styled
Rich tables across all CLI commands, reducing duplication and ensuring
uniform visual appearance.

Usage:
    from victor.ui.rendering.table_builder import (
        create_table,
        create_name_status_table,
        create_name_description_table,
    )

    table = create_table(title="My Items")
    table.add_column("Name", style="cyan")
    table.add_column("Status")

    # Or use presets
    table = create_name_status_table(title="Items")
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from rich.table import Table


def create_table(
    title: Optional[str] = None,
    show_header: bool = True,
    show_lines: bool = False,
) -> Table:
    """Create a standard Rich table with Victor styling.

    Args:
        title: Optional table title
        show_header: Whether to show column headers
        show_lines: Whether to show lines between rows

    Returns:
        Configured Rich Table instance
    """
    return Table(
        title=title,
        show_header=show_header,
        show_lines=show_lines,
        header_style="bold cyan",
        title_style="bold",
    )


def create_name_status_table(
    title: Optional[str] = None,
    show_lines: bool = False,
) -> Table:
    """Create a table with Name and Status columns.

    Common pattern for listing items with their status.

    Args:
        title: Optional table title
        show_lines: Whether to show lines between rows

    Returns:
        Configured Rich Table with Name and Status columns
    """
    table = create_table(title=title, show_lines=show_lines)
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("Status")
    return table


def create_name_description_table(
    title: Optional[str] = None,
    show_lines: bool = True,
) -> Table:
    """Create a table with Name and Description columns.

    Common pattern for listing items with descriptions.

    Args:
        title: Optional table title
        show_lines: Whether to show lines between rows

    Returns:
        Configured Rich Table with Name and Description columns
    """
    table = create_table(title=title, show_lines=show_lines)
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("Description")
    return table


def create_three_column_table(
    title: Optional[str] = None,
    col1_name: str = "Name",
    col2_name: str = "Value",
    col3_name: str = "Description",
    col1_style: str = "cyan",
    show_lines: bool = False,
) -> Table:
    """Create a table with three columns.

    Common pattern for name/value/description displays.

    Args:
        title: Optional table title
        col1_name: First column name
        col2_name: Second column name
        col3_name: Third column name
        col1_style: Style for first column
        show_lines: Whether to show lines between rows

    Returns:
        Configured Rich Table with three columns
    """
    table = create_table(title=title, show_lines=show_lines)
    table.add_column(col1_name, style=col1_style)
    table.add_column(col2_name)
    table.add_column(col3_name, style="dim")
    return table


def create_provider_table(
    title: Optional[str] = None,
    show_aliases: bool = True,
) -> Table:
    """Create a table for listing providers.

    Args:
        title: Optional table title
        show_aliases: Whether to include Aliases column

    Returns:
        Configured Rich Table with provider columns
    """
    table = create_table(title=title, show_header=True)
    table.add_column("Provider", style="cyan", no_wrap=True)
    table.add_column("Status", style="green")
    table.add_column("Features")
    if show_aliases:
        table.add_column("Aliases", style="dim")
    return table


def create_skill_table(
    title: Optional[str] = None,
    show_tools: bool = True,
) -> Table:
    """Create a table for listing skills.

    Args:
        title: Optional table title
        show_tools: Whether to include Tools column

    Returns:
        Configured Rich Table with skill columns
    """
    table = create_table(title=title, show_lines=False)
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("Category", style="green")
    table.add_column("Description")
    if show_tools:
        table.add_column("Tools", style="dim")
    return table


def render_table_from_dict(
    data: Dict[str, Any],
    title: Optional[str] = None,
    key_column: str = "Key",
    value_column: str = "Value",
) -> Table:
    """Create and populate a table from a dictionary.

    Args:
        data: Dictionary to display
        title: Optional table title
        key_column: Name for key column
        value_column: Name for value column

    Returns:
        Populated Rich Table
    """
    table = create_table(title=title, show_lines=True)
    table.add_column(key_column, style="cyan")
    table.add_column(value_column)

    for key, value in sorted(data.items()):
        table.add_row(str(key), str(value))

    return table


def render_table_from_list(
    data: List[Dict[str, Any]],
    columns: List[str],
    title: Optional[str] = None,
    column_styles: Optional[Dict[str, str]] = None,
) -> Table:
    """Create and populate a table from a list of dictionaries.

    Args:
        data: List of dictionaries with consistent keys
        columns: Column names (keys from dicts)
        title: Optional table title
        column_styles: Optional mapping of column name to Rich style

    Returns:
        Populated Rich Table
    """
    table = create_table(title=title, show_lines=True)

    # Add columns with optional styles
    for col in columns:
        style = column_styles.get(col) if column_styles else None
        table.add_column(col, style=style)

    # Populate rows
    for row_data in data:
        row = [str(row_data.get(col, "")) for col in columns]
        table.add_row(*row)

    return table


def format_status(status: bool) -> str:
    """Format a boolean status for display in a table.

    Args:
        status: Boolean status value

    Returns:
        Formatted status string with Rich markup
    """
    return "[green]✓[/]" if status else "[red]✗[/]"


def format_configured_status(configured: bool) -> str:
    """Format a configured status for display.

    Args:
        configured: Whether item is configured

    Returns:
        Formatted status string with Rich markup
    """
    return "[green]✓ Configured[/]" if configured else "[dim]Not set[/]"


def format_enabled_status(enabled: bool) -> str:
    """Format an enabled status for display.

    Args:
        enabled: Whether item is enabled

    Returns:
        Formatted status string with Rich markup
    """
    return "[green]enabled[/]" if enabled else "[dim]disabled[/]"


def create_plan_task_table(title: Optional[str] = None) -> Table:
    """Create an optimized table for plan tasks with status icons and progress.

    Args:
        title: Optional table title

    Returns:
        Configured Rich Table with plan task columns
    """
    table = Table(
        title=title,
        show_header=True,
        show_lines=False,
        header_style="bold cyan",
        title_style="bold",
        border_style="dim cyan",
        padding=(0, 1),
    )
    table.add_column("#", style="dim cyan", no_wrap=True, width=3)
    table.add_column("Task", style="white", no_wrap=False)
    table.add_column("Status", style="cyan", no_wrap=True, width=12)
    table.add_column("Details", style="dim", no_wrap=False)
    return table


def create_plan_list_table(title: Optional[str] = None) -> Table:
    """Create a table for listing saved plans with enhanced styling.

    Args:
        title: Optional table title

    Returns:
        Configured Rich Table for plan listing
    """
    table = Table(
        title=title,
        show_header=True,
        show_lines=True,
        header_style="bold cyan",
        title_style="bold",
        border_style="cyan",
        padding=(0, 1),
    )
    table.add_column("ID", style="cyan", no_wrap=True, width=12)
    table.add_column("Goal", style="white", no_wrap=False)
    table.add_column("Created", style="dim", no_wrap=True, width=16)
    table.add_column("Status", style="green", no_wrap=True, width=10)
    table.add_column("Tasks", style="dim", no_wrap=True, width=6)
    return table


def format_plan_status(status: str) -> str:
    """Format plan status with icon and color.

    Args:
        status: Plan status string

    Returns:
        Formatted status with Rich markup
    """
    status_map = {
        "pending": ("○", "dim"),
        "draft": ("○", "dim"),
        "in_progress": ("◷", "yellow"),
        "active": ("◷", "yellow"),
        "completed": ("✓", "green"),
        "done": ("✓", "green"),
        "blocked": ("✗", "red"),
        "failed": ("✗", "red"),
        "archived": ("▣", "dim"),
    }
    icon, color = status_map.get(status.lower(), ("?", "white"))
    return f"[{color}]{icon}[/]"


def format_task_status(status: str) -> str:
    """Format task status with icon and color for plan tasks.

    Args:
        status: Task status string

    Returns:
        Formatted status with Rich markup
    """
    status_map = {
        "pending": ("○", "dim"),
        "in_progress": ("◷", "yellow"),
        "completed": ("✓", "green"),
        "done": ("✓", "green"),
        "blocked": ("✗", "red"),
        "skipped": ("⊘", "dim"),
        "failed": ("✗", "red"),
    }
    icon, color = status_map.get(status.lower(), ("?", "white"))
    return f"[{color}]{icon}[/] {status.capitalize()}"
