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

"""Lightweight overlay panels for CLI contextual information.

This module provides overlay panel utilities that work with prompt-toolkit's
run_in_terminal pattern to show contextual information without disrupting
the prompt.

Usage:
    from victor.ui.rendering.overlay import show_shortcuts_overlay, show_status_overlay

    # In a key binding handler:
    event.app.run_in_terminal(lambda: show_shortcuts_overlay(console))

    # Show active tools status:
    show_status_overlay(console, active_tools=[...])
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


def show_shortcuts_overlay(
    console: Console,
    title: str = "Keyboard Shortcuts",
    border_style: str = "cyan",
) -> None:
    """Show a lightweight shortcuts overlay panel.

    Args:
        console: Rich console instance
        title: Panel title
        border_style: Border style string
    """
    from victor.ui.commands.chat import _build_cli_shortcuts_panel

    panel = _build_cli_shortcuts_panel()
    # Update border style if different
    if border_style != "cyan":
        panel = Panel(
            panel.renderable,
            title=title,
            border_style=border_style,
        )
    console.print(panel)


def show_status_overlay(
    console: Console,
    active_tools: List[Dict[str, Any]],
    title: str = "Active Tools",
    border_style: str = "cyan",
) -> None:
    """Show current execution status overlay with active tools.

    Args:
        console: Rich console instance
        active_tools: List of active tool dicts with keys: name, status, elapsed
        title: Panel title
        border_style: Border style string
    """
    if not active_tools:
        console.print("[dim]No active tools[/]")
        return

    table = Table.grid(padding=(0, 2))
    table.add_column("Tool", style="cyan", no_wrap=True)
    table.add_column("Status", style="white")

    for tool in active_tools:
        tool_name = tool.get("name", "unknown")
        status = tool.get("status", "running")
        elapsed = tool.get("elapsed", 0)

        if status == "running":
            status_text = f"[yellow]▸[/] Running [dim]({elapsed:.1f}s)[/]"
        elif status == "completed":
            status_text = "[green]✓[/] Completed"
        elif status == "failed":
            status_text = "[red]✗[/] Failed"
        else:
            status_text = f"[dim]○[/] {status}"

        table.add_row(tool_name, status_text)

    panel = Panel(
        table,
        title=title,
        border_style=border_style,
        padding=(0, 1),
    )
    console.print(panel)


def show_plan_progress_overlay(
    console: Console,
    plan_id: str,
    goal: str,
    steps: List[Dict[str, Any]],
    title: Optional[str] = None,
    border_style: str = "blue",
) -> None:
    """Show plan progress overlay with task status.

    Args:
        console: Rich console instance
        plan_id: Plan identifier
        goal: Plan goal description
        steps: List of plan step dicts with keys: id, description, status
        title: Optional panel title (defaults to "Plan Progress")
        border_style: Border style string
    """
    from victor.ui.rendering.table_builder import format_task_status

    if title is None:
        title = f"Plan Progress: {plan_id[:12]}..."

    # Calculate progress
    total = len(steps)
    completed = sum(1 for s in steps if s.get("status") == "completed")
    progress_pct = (completed / total * 100) if total > 0 else 0

    # Build header with progress
    header = Text()
    header.append(f"{goal[:60]}...\n\n", style="bold white")
    header.append("Progress: ", style="dim")
    header.append(f"{completed}/{total}", style="cyan")
    header.append(f" ({progress_pct:.0f}%)", style="cyan")

    # Build steps table
    if steps:
        table = Table.grid(padding=(0, 2))
        table.add_column("#", style="dim cyan", no_wrap=True, width=3)
        table.add_column("Task", style="white")
        table.add_column("Status", no_wrap=True, width=12)

        for idx, step in enumerate(steps[:10], 1):  # Show max 10 steps
            status_str = step.get("status", "pending")
            status_formatted = format_task_status(status_str)

            description = step.get("description", "")
            if len(description) > 50:
                description = description[:50] + "..."

            table.add_row(str(idx), description, status_formatted)

        if len(steps) > 10:
            table.add_row("", f"[dim]... and {len(steps) - 10} more[/]", "")
    else:
        table = Text("[dim]No tasks in this plan[/]", style="dim")

    # Combine header and table
    from rich.console import Group

    panel = Panel(
        Group(header, table),
        title=title,
        border_style=border_style,
        padding=(0, 1),
    )
    console.print(panel)


def show_context_overlay(
    console: Console,
    context_info: Dict[str, Any],
    title: str = "Current Context",
    border_style: str = "cyan",
) -> None:
    """Show context information overlay.

    Args:
        console: Rich console instance
        context_info: Dict with context keys and values
        title: Panel title
        border_style: Border style string
    """
    table = Table.grid(padding=(0, 2))
    table.add_column("Key", style="cyan", no_wrap=True)
    table.add_column("Value", style="white")

    for key, value in context_info.items():
        value_str = str(value)
        if len(value_str) > 60:
            value_str = value_str[:60] + "..."
        table.add_row(key, value_str)

    panel = Panel(
        table,
        title=title,
        border_style=border_style,
        padding=(0, 1),
    )
    console.print(panel)


def show_mini_help_overlay(
    console: Console,
    message: str = "Press Ctrl+O to expand last output • F1 for shortcuts",
    border_style: str = "dim",
) -> None:
    """Show a minimal help hint overlay.

    Args:
        console: Rich console instance
        message: Help message to display
        border_style: Border style string
    """
    panel = Panel(
        message,
        border_style=border_style,
        padding=(0, 1),
    )
    console.print(panel)


def build_compact_shortcuts_table() -> Table:
    """Build a compact shortcuts table for overlays.

    Returns:
        Configured Table with common shortcuts
    """
    table = Table.grid(padding=(0, 2))
    table.add_column("Key", style="cyan", no_wrap=True, width=12)
    table.add_column("Action", style="white")

    shortcuts = [
        ("Enter", "send message"),
        ("Alt+Enter", "insert newline"),
        ("Tab", "complete commands"),
        ("Esc", "clear input"),
        ("Ctrl+O", "expand tool output"),
        ("F1", "show shortcuts"),
        ("Ctrl+D", "exit"),
        ("/help", "command help"),
        ("/plan", "plan mode"),
        ("/mode", "switch mode"),
    ]

    for key, action in shortcuts:
        table.add_row(key, action)

    return table


def show_plan_status_overlay(
    console: Console,
    plan: Any,
    border_style: str = "blue",
) -> None:
    """Show plan status overlay for the current plan.

    Args:
        console: Rich console instance
        plan: ExecutionPlan instance
        border_style: Border style string
    """
    from victor.ui.rendering.table_builder import format_task_status

    # Calculate progress
    total = len(plan.steps)
    completed = len(plan.get_completed_steps())
    progress_pct = plan.progress_percentage()

    # Build header
    header = f"[bold]{plan.goal[:60]}...[/]\n\n"
    header += f"Progress: [cyan]{completed}/{total}[/] ([cyan]{progress_pct:.0f}%[/])"

    # Build steps table
    if plan.steps:
        table = Table.grid(padding=(0, 2))
        table.add_column("#", style="dim cyan", no_wrap=True, width=3)
        table.add_column("Task", style="white")
        table.add_column("Status", no_wrap=True, width=12)

        for idx, step in enumerate(plan.steps[:10], 1):  # Show max 10 steps
            status_str = step.status.value if hasattr(step.status, "value") else str(step.status)
            status_formatted = format_task_status(status_str)

            description = (
                step.description[:50] + "..." if len(step.description) > 50 else step.description
            )

            table.add_row(str(idx), description, status_formatted)

        if len(plan.steps) > 10:
            table.add_row("", f"[dim]... and {len(plan.steps) - 10} more[/]", "")

        content = f"{header}\n\n{table}"
    else:
        content = f"{header}\n\n[dim]No tasks in this plan[/]"

    panel = Panel(
        content,
        title=f"Plan: {plan.id[:12]}...",
        border_style=border_style,
        padding=(0, 1),
    )
    console.print(panel)
