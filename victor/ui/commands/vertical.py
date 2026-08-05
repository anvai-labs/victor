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

"""Vertical management commands."""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from victor.core.verticals.registry_manager import (
    PackageSpec,
    VerticalRegistryManager,
)
from victor.ui.json_utils import create_json_option, print_json_data

# Initialize components
vertical_app = typer.Typer(
    name="vertical",
    help="Manage verticals (DEPRECATED: use 'victor plugin' instead)",
    deprecated=True,
)
console = Console()

_DEPRECATION_MSG = (
    "[yellow]Warning:[/] 'victor vertical' is deprecated and will be removed in v0.8.0. "
    "Use 'victor plugin' instead — it shows both verticals and plugins in a unified view."
)

_VALID_SOURCES = {"all", "installed", "builtin", "available"}


def _deprecation_notice() -> None:
    """Print deprecation notice."""
    console.print(_DEPRECATION_MSG)
    console.print()


def _handle_error(message: str, detail: Optional[str] = None) -> None:
    """Handle and display errors."""
    console.print(f"[red]Error:[/] {message}")
    if detail:
        console.print(f"[dim]{detail}[/]")
        console.print()
    raise typer.Exit(1)


def _post_mutation_notice() -> None:
    """Print post-install/uninstall notices about cache refresh and session restart."""
    console.print("[dim]Successfully refreshed package caches.[/]")
    console.print("[yellow]Restart other Victor sessions to pick up the change.[/]")


@vertical_app.command("list")
def list_verticals(
    source: str = typer.Option(
        "all",
        "--source",
        "-s",
        help="Filter by source: 'all', 'installed', 'builtin', 'available'",
    ),
    category: str = typer.Option(
        None,
        "--category",
        "-c",
        help="Filter by category (e.g., 'security', 'data', 'devops')",
    ),
    tags: str = typer.Option(
        None,
        "--tags",
        "-t",
        help="Filter by tags (comma-separated)",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show detailed information including tools and workflows",
    ),
) -> None:
    """List vertical packages with optional filtering.

    .. deprecated:: 0.7.0
        Use ``victor plugin list --type vertical`` instead.

    Shows built-in, installed, and available verticals with filtering options.

    Examples:
        # List all verticals
        victor vertical list

        # List only installed
        victor vertical list --source installed

        # List by category
        victor vertical list --category security

        # List by tags
        victor vertical list --tags "security,scanning"
    """
    _deprecation_notice()

    if source not in _VALID_SOURCES:
        _handle_error(f"Invalid source '{source}'. Must be one of: {', '.join(_VALID_SOURCES)}")
        return

    manager = VerticalRegistryManager()

    with console.status("[bold blue]Loading vertical list..."):
        try:
            verticals = manager.list_verticals(source=source)
        except Exception as e:
            _handle_error("Failed to list verticals", str(e))
            return

    # Filter by category if provided
    if category:
        verticals = [v for v in verticals if v.metadata and v.metadata.category == category]

    # Filter by tags if provided
    if tags:
        tag_list = [t.strip().lower() for t in tags.split(",")]
        verticals = [
            v
            for v in verticals
            if v.metadata and any(tg in [mt.lower() for mt in v.metadata.tags] for tg in tag_list)
        ]

    if not verticals:
        console.print(f"[yellow]No verticals found matching source='{source}'[/]")
        return

    table = Table(title="Vertical Ecosystem")
    table.add_column("Name", style="cyan")
    table.add_column("Version", style="magenta")
    table.add_column("Source", style="green")
    table.add_column("Description")

    if verbose:
        table.add_column("Category")
        table.add_column("Tools")

    for v in verticals:
        row = [
            v.name,
            v.version,
            "Builtin" if v.is_builtin else "External",
            (v.metadata.description if v.metadata else "[dim]No description available[/]"),
        ]

        if verbose:
            cat = v.metadata.category if v.metadata and v.metadata.category else "-"
            cs = v.metadata.class_spec if v.metadata else None
            tool_list = cs.provides_tools if cs and cs.provides_tools else []
            tools = ", ".join(tool_list[:5]) if tool_list else "-"
            if len(tool_list) > 5:
                tools += f" (+{len(tool_list) - 5} more)"
            row.extend([cat, tools])

        table.add_row(*row)

    console.print(table)
    console.print(f"\n[dim]Found {len(verticals)} vertical(s)[/]")


@vertical_app.command("info")
def vertical_info(name: str) -> None:
    """Show detailed information for a vertical."""
    manager = VerticalRegistryManager()

    try:
        vertical = manager.get_info(name)
        if not vertical:
            _handle_error(f"Vertical '{name}' not found")
            return

        console.print(f"\n[bold cyan]Vertical: {vertical.name}[/]")
        console.print(f"[bold]Version:[/] {vertical.version}")
        console.print(f"[bold]Source:[/] {'Built-in' if vertical.is_builtin else 'External'}")
        console.print(f"[bold]Location:[/] {vertical.location or 'Embedded'}")

        if vertical.metadata:
            m = vertical.metadata
            console.print(f"\n[bold]Description:[/] {m.description}")
            if m.authors:
                author_names = [a.name if hasattr(a, "name") else str(a) for a in m.authors]
                console.print(f"[bold]Authors:[/] {', '.join(author_names)}")
            if m.category:
                console.print(f"[bold]Category:[/] {m.category}")
            if m.tags:
                console.print(f"[bold]Tags:[/] {', '.join(m.tags)}")

            if m.class_spec:
                if m.class_spec.provides_capabilities:
                    console.print("\n[bold]Capabilities:[/]")
                    for cap in m.class_spec.provides_capabilities:
                        console.print(f"  - {cap}")

                if m.class_spec.provides_tools:
                    console.print("\n[bold]Provides Tools:[/]")
                    for tool in m.class_spec.provides_tools:
                        console.print(f"  - {tool}")

                if m.class_spec.provides_workflows:
                    console.print("\n[bold]Provides Workflows:[/]")
                    for wf in m.class_spec.provides_workflows:
                        console.print(f"  - {wf}")
        else:
            console.print("\n[yellow]No detailed metadata available[/]")

    except Exception as e:
        _handle_error(f"Failed to load information for '{name}'", str(e))


@vertical_app.command("install")
def install_vertical(
    package: str = typer.Argument(..., help="Package name, path, or URL"),
    force: bool = typer.Option(False, "--force", "-f", help="Force re-installation"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be installed"),
    no_validate: bool = typer.Option(False, "--no-validate", help="Skip package validation"),
) -> None:
    """Install a new vertical plugin."""
    _deprecation_notice()
    spec = PackageSpec.parse(package)
    manager = VerticalRegistryManager(dry_run=dry_run)

    # Validate unless explicitly skipped
    if not no_validate and not dry_run:
        errors = manager._validate_package(spec)
        if errors:
            _handle_error("Validation failed: " + "; ".join(errors))
            return

    with console.status(f"[bold blue]Installing {package}..."):
        success, message = manager.install(spec, validate=False)

    if success:
        console.print(f"[green]{message}[/]")
        _post_mutation_notice()
    else:
        console.print(f"[red]{message}[/]")
        raise typer.Exit(1)


@vertical_app.command("uninstall")
def uninstall_vertical(
    name: str = typer.Argument(..., help="Package name to uninstall"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be uninstalled"),
) -> None:
    """Uninstall a vertical plugin."""
    _deprecation_notice()
    manager = VerticalRegistryManager(dry_run=dry_run)

    with console.status(f"[bold blue]Uninstalling {name}..."):
        success, message = manager.uninstall(name)

    if success:
        console.print(f"[green]{message}[/]")
        if not dry_run:
            _post_mutation_notice()
    else:
        console.print(f"[red]{message}[/]")
        raise typer.Exit(1)


@vertical_app.command("search")
def search_verticals(
    query: str = typer.Argument(..., help="Search query"),
) -> None:
    """Search for verticals by name, description, or tags."""
    _deprecation_notice()
    manager = VerticalRegistryManager()

    with console.status("[bold blue]Searching verticals..."):
        results = manager.search(query)

    if not results:
        console.print(f"[yellow]No verticals found matching '{query}'[/]")
        return

    table = Table(title=f"Search Results for '{query}'")
    table.add_column("Name", style="cyan")
    table.add_column("Version", style="magenta")
    table.add_column("Description")

    for v in results:
        table.add_row(
            v.name,
            v.version,
            v.metadata.description if v.metadata else "[dim]No description[/]",
        )

    console.print(table)
    console.print(f"\n[dim]Found {len(results)} result(s)[/]")


@vertical_app.command("audit")
def audit_verticals(
    paths: list[str] = typer.Argument(
        None,
        help="One or more extracted vertical repository paths",
    ),
    json_output: bool = create_json_option(),
    workspace: bool = typer.Option(
        False,
        "--workspace",
        help="Audit the default extracted sibling repos next to the core checkout",
    ),
) -> None:
    """Audit extracted vertical repositories against the supported plugin contract."""
    _deprecation_notice()

    from victor.core.verticals.contract_audit import VerticalContractAuditor
    from victor.core.verticals.extracted_repo_paths import (
        discover_default_extracted_repo_paths,
        normalize_extracted_repo_paths,
    )

    if workspace:
        repo_root = Path(__file__).resolve().parents[3]
        resolved_paths = discover_default_extracted_repo_paths(repo_root=repo_root)
        if not resolved_paths:
            console.print("[yellow]No extracted vertical repositories found to audit.[/]")
            raise typer.Exit(0)
    elif paths:
        resolved_paths = normalize_extracted_repo_paths(paths, cwd=Path.cwd())
    else:
        _handle_error("Provide at least one path or pass --workspace")
        return

    reports = VerticalContractAuditor().audit_paths(resolved_paths)

    if json_output:
        print_json_data(
            {"reports": [report.to_dict() for report in reports], "count": len(reports)}
        )
    else:
        table = Table(title="Vertical Contract Audit")
        table.add_column("Repo", style="cyan")
        table.add_column("Result")
        table.add_column("Errors", justify="right")
        table.add_column("Warnings", justify="right")
        table.add_column("Plugin Entry Points")

        for report in reports:
            status = "[green]PASSED[/]" if report.passed else "[red]FAILED[/]"
            plugin_entries = (
                ", ".join(report.plugin_entry_points) if report.plugin_entry_points else "-"
            )
            table.add_row(
                str(report.root_path),
                status,
                str(report.error_count),
                str(report.warning_count),
                plugin_entries,
            )

        console.print(table)

        for report in reports:
            if not report.issues:
                continue
            console.print(f"\n[bold]{report.project_name or report.root_path.name}[/]")
            for issue in report.issues:
                location = ""
                if issue.path:
                    location = f" ({issue.path}"
                    if issue.line is not None:
                        location += f":{issue.line}"
                    location += ")"
                style = "red" if issue.level == "error" else "yellow"
                console.print(
                    f"[{style}]{issue.level.upper()} {issue.code}[/]{location}: {issue.message}"
                )

    if any(not report.passed for report in reports):
        raise typer.Exit(1)


@vertical_app.command("create")
def create_vertical(
    name: str = typer.Argument(..., help="Name of the new vertical"),
    description: str = typer.Option("A new Victor vertical", "--description", "-d"),
    service_provider: bool = typer.Option(
        False, "--service", help="Include service provider boilerplate"
    ),
    force: bool = typer.Option(False, "--force", help="Overwrite if directory exists"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show files that would be created"),
) -> None:
    """Create (scaffold) a new vertical package."""
    from victor.ui.commands.scaffold import scaffold_plugin

    scaffold_plugin(
        name=name,
        description=description,
        service_provider=service_provider,
        force=force,
        dry_run=dry_run,
        label="vertical",
    )


@vertical_app.command("new")
def scaffold_vertical(
    name: str = typer.Argument(..., help="Name of the new vertical"),
    description: str = typer.Option("A new Victor vertical", "--description", "-d"),
    service_provider: bool = typer.Option(
        False, "--service", help="Include service provider boilerplate"
    ),
    force: bool = typer.Option(False, "--force", help="Overwrite if directory exists"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show files that would be created"),
) -> None:
    """Scaffold a new vertical package."""
    from victor.ui.commands.scaffold import scaffold_plugin

    scaffold_plugin(
        name=name,
        description=description,
        service_provider=service_provider,
        force=force,
        dry_run=dry_run,
        label="vertical",
    )


__all__ = ["vertical_app"]
