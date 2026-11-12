# Copyright 2025 Vijaykumar Singh
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Documentation access CLI command for Victor.

This module provides the `victor docs` command for accessing Victor's
documentation guides directly from the terminal.
"""

import os
import subprocess
import sys
import webbrowser
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from victor.ui.console import get_console

docs_app = typer.Typer(
    name="docs",
    help="Access Victor's documentation guides.",
)

console = get_console()

# Mapping of short names/aliases to documentation files
DOCS_MAPPING = {
    # EventBus / Observability
    "eventbus": "OBSERVABILITY.md",
    "observability": "OBSERVABILITY.md",
    "events": "OBSERVABILITY.md",
    # Multi-Agent Teams
    "teams": "MULTI_AGENT_TEAMS.md",
    "multi-agent": "MULTI_AGENT_TEAMS.md",
    "agents": "MULTI_AGENT_TEAMS.md",
    # Resilience
    "resilience": "RESILIENCE.md",
    "circuit-breaker": "RESILIENCE.md",
    "retry": "RESILIENCE.md",
    # MCP Integration
    "mcp": "MCP_INTEGRATION.md",
    "mcp-integration": "MCP_INTEGRATION.md",
    # HITL Workflows
    "hitl": "HITL_WORKFLOWS.md",
    "human-in-the-loop": "HITL_WORKFLOWS.md",
    "approval": "HITL_WORKFLOWS.md",
    # Workflow DSL
    "workflow": "WORKFLOW_DSL.md",
    "dsl": "WORKFLOW_DSL.md",
    "stategraph": "WORKFLOW_DSL.md",
    # Workflow Scheduler
    "scheduler": "WORKFLOW_SCHEDULER.md",
    "scheduling": "WORKFLOW_SCHEDULER.md",
    "cron": "WORKFLOW_SCHEDULER.md",
    # Additional guides
    "quickstart": "QUICKSTART.md",
    "installation": "INSTALLATION.md",
    "first-run": "FIRST_RUN.md",
    "providers": "PROVIDER_SETUP.md",
    "provider-setup": "PROVIDER_SETUP.md",
    "local": "LOCAL_MODELS.md",
    "local-models": "LOCAL_MODELS.md",
    "ollama": "LOCAL_MODELS.md",
    "verticals": "VERTICAL_DEVELOPMENT.md",
    "vertical": "VERTICAL_DEVELOPMENT.md",
    "plugins": "PLUGIN_GUIDE.md",
    "plugin": "PLUGIN_GUIDE.md",
    "benchmark": "BENCHMARKING.md",
    "benchmarking": "BENCHMARKING.md",
}

# Human-readable descriptions for each doc file
DOCS_DESCRIPTIONS = {
    "OBSERVABILITY.md": "EventBus, metrics, tracing, and monitoring infrastructure",
    "MULTI_AGENT_TEAMS.md": "Multi-agent team orchestration and collaboration",
    "RESILIENCE.md": "Circuit breakers, retry strategies, and fault tolerance",
    "MCP_INTEGRATION.md": "Model Context Protocol integration for tool interoperability",
    "HITL_WORKFLOWS.md": "Human-in-the-loop approval and oversight workflows",
    "WORKFLOW_DSL.md": "StateGraph DSL for building stateful agent workflows",
    "WORKFLOW_SCHEDULER.md": "Workflow scheduling, versioning, and cron jobs",
    "QUICKSTART.md": "Quick start guide for getting started with Victor",
    "INSTALLATION.md": "Installation instructions and requirements",
    "FIRST_RUN.md": "First run setup and configuration",
    "PROVIDER_SETUP.md": "LLM provider setup and configuration",
    "LOCAL_MODELS.md": "Local model setup with Ollama, LMStudio, etc.",
    "VERTICAL_DEVELOPMENT.md": "Creating custom domain verticals",
    "PLUGIN_GUIDE.md": "Plugin development and extension guide",
    "BENCHMARKING.md": "Benchmarking and evaluation guide",
}


def _get_docs_dir() -> Path:
    """Get the docs/guides directory path."""
    # Navigate from victor/ui/commands/ to docs/guides/
    return Path(__file__).parent.parent.parent.parent / "docs" / "guides"


def _get_doc_path(doc_name: str) -> Optional[Path]:
    """Get the full path to a documentation file.

    Args:
        doc_name: Short name or alias for the documentation

    Returns:
        Path to the doc file if found, None otherwise
    """
    docs_dir = _get_docs_dir()

    # Check if it's a direct alias
    if doc_name.lower() in DOCS_MAPPING:
        doc_file = docs_dir / DOCS_MAPPING[doc_name.lower()]
        if doc_file.exists():
            return doc_file

    # Check if it's a direct file reference
    if doc_name.upper().endswith(".md"):
        doc_file = docs_dir / doc_name.upper()
    else:
        doc_file = docs_dir / f"{doc_name.upper()}.md"

    if doc_file.exists():
        return doc_file

    return None


def _list_available_docs() -> list[tuple[str, str, str]]:
    """List all available documentation files with their aliases.

    Returns:
        List of (primary_alias, file_name, description) tuples
    """
    docs_dir = _get_docs_dir()
    available = []

    # Get unique doc files and their primary aliases
    file_to_aliases: dict[str, list[str]] = {}
    for alias, filename in DOCS_MAPPING.items():
        if filename not in file_to_aliases:
            file_to_aliases[filename] = []
        file_to_aliases[filename].append(alias)

    # Check which files exist
    for filename, aliases in sorted(file_to_aliases.items()):
        doc_path = docs_dir / filename
        if doc_path.exists():
            primary_alias = aliases[0]
            description = DOCS_DESCRIPTIONS.get(filename, "No description available")
            available.append((primary_alias, filename, description))

    return available


@docs_app.callback(invoke_without_command=True)
def docs_main(
    ctx: typer.Context,
    doc_name: Optional[str] = typer.Argument(
        None,
        help="Documentation to display (e.g., eventbus, teams, resilience, mcp, hitl)",
    ),
    list_docs: bool = typer.Option(
        False,
        "--list",
        "-l",
        help="List all available documentation with descriptions",
    ),
    open_browser: bool = typer.Option(
        False,
        "--open",
        "-o",
        help="Open documentation in system browser or editor",
    ),
    show_path: bool = typer.Option(
        False,
        "--path",
        "-p",
        help="Show file path instead of content",
    ),
) -> None:
    """Access Victor's documentation guides.

    Examples:
        victor docs                    # List all available docs
        victor docs eventbus           # Display EventBus guide
        victor docs teams              # Display Multi-Agent Teams guide
        victor docs resilience         # Display Resilience guide
        victor docs mcp                # Display MCP Integration guide
        victor docs hitl               # Display HITL Workflows guide
        victor docs observability      # Same as eventbus
        victor docs --list             # List all docs with descriptions
        victor docs eventbus --open    # Open in browser/editor
        victor docs --path             # Show docs directory path
        victor docs eventbus --path    # Show path to specific doc
    """
    # If --path flag without doc name, show docs directory
    if show_path and doc_name is None:
        console.print(str(_get_docs_dir()))
        return

    # If --list flag is provided, show list regardless of doc_name
    if list_docs or doc_name is None:
        _display_docs_list()
        return

    # If a doc name is provided, find it
    doc_path = _get_doc_path(doc_name)
    if doc_path is None:
        console.print(f"[red]Documentation not found:[/red] {doc_name}")
        console.print()
        console.print("[dim]Available documentation:[/dim]")
        _display_docs_list(compact=True)
        raise typer.Exit(1)

    # Handle the doc based on flags
    if show_path:
        console.print(str(doc_path))
    elif open_browser:
        _open_doc(doc_path)
    else:
        _display_doc(doc_path)


def _display_docs_list(compact: bool = False) -> None:
    """Display list of available documentation."""
    available_docs = _list_available_docs()

    if not available_docs:
        console.print("[yellow]No documentation found in docs/guides/[/yellow]")
        return

    if compact:
        # Simple list format
        for alias, _, _ in available_docs:
            console.print(f"  [cyan]{alias}[/cyan]")
        return

    console.print()
    console.print(
        Panel.fit(
            "[bold cyan]Victor Documentation[/bold cyan]\n"
            "[dim]Access guides directly from the terminal[/dim]",
            border_style="cyan",
        )
    )
    console.print()

    table = Table(title="Available Documentation", show_header=True)
    table.add_column("Command", style="cyan", no_wrap=True)
    table.add_column("File", style="dim")
    table.add_column("Description", style="green")

    for alias, filename, description in available_docs:
        table.add_row(f"victor docs {alias}", filename, description)

    console.print(table)
    console.print()

    # Show aliases hint
    console.print("[dim]Some guides have multiple aliases:[/dim]")
    console.print(
        "  [cyan]eventbus[/cyan], [cyan]observability[/cyan], [cyan]events[/cyan] -> OBSERVABILITY.md"
    )
    console.print(
        "  [cyan]teams[/cyan], [cyan]multi-agent[/cyan], [cyan]agents[/cyan] -> MULTI_AGENT_TEAMS.md"
    )
    console.print(
        "  [cyan]workflow[/cyan], [cyan]dsl[/cyan], [cyan]stategraph[/cyan] -> WORKFLOW_DSL.md"
    )
    console.print()
    console.print("[dim]Use --open flag to open in browser/editor:[/dim]")
    console.print("  [cyan]victor docs eventbus --open[/cyan]")
    console.print()


def _display_doc(doc_path: Path) -> None:
    """Display documentation content with Rich Markdown rendering."""
    try:
        content = doc_path.read_text(encoding="utf-8")
    except Exception as e:
        console.print(f"[red]Error reading documentation:[/red] {e}")
        raise typer.Exit(1)

    # Extract title from first line if it's a markdown heading
    lines = content.split("\n")
    title = doc_path.stem.replace("_", " ").title()
    if lines and lines[0].startswith("# "):
        title = lines[0][2:].strip()

    console.print()
    console.print(
        Panel.fit(
            f"[bold cyan]{title}[/bold cyan]\n" f"[dim]{doc_path.name}[/dim]",
            border_style="cyan",
        )
    )
    console.print()

    # Render markdown content
    markdown = Markdown(content)
    console.print(markdown)
    console.print()


def _open_doc(doc_path: Path) -> None:
    """Open documentation in system browser or editor."""
    try:
        # Try to open as file URL in browser
        file_url = doc_path.as_uri()

        # Platform-specific opening
        if sys.platform == "darwin":
            # macOS: try to open with default markdown viewer/browser
            subprocess.run(["open", str(doc_path)], check=True)
            console.print(f"[green]Opened:[/green] {doc_path}")
        elif sys.platform == "win32":
            # Windows
            os.startfile(str(doc_path))
            console.print(f"[green]Opened:[/green] {doc_path}")
        else:
            # Linux and others: try xdg-open
            try:
                subprocess.run(["xdg-open", str(doc_path)], check=True)
                console.print(f"[green]Opened:[/green] {doc_path}")
            except FileNotFoundError:
                # Fallback to webbrowser
                webbrowser.open(file_url)
                console.print(f"[green]Opened in browser:[/green] {file_url}")
    except Exception as e:
        console.print(f"[red]Error opening documentation:[/red] {e}")
        console.print(f"[dim]File location:[/dim] {doc_path}")
        raise typer.Exit(1)
