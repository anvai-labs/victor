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

"""Scaffold command for generating new vertical structures.

This module provides the `victor vertical` command that generates
a complete vertical structure using Jinja2 templates.

Usage:
    victor vertical create security --description "Security analysis assistant"
    victor vertical create analytics --description "Data analytics assistant" --service-provider
    victor vertical list
"""

import re
import typer
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

try:
    from jinja2 import (
        Environment,
        FileSystemLoader,
        TemplateNotFound,
        select_autoescape,
    )
except ImportError:
    Environment = None  # type: ignore
    FileSystemLoader = None  # type: ignore
    TemplateNotFound = Exception  # type: ignore
    select_autoescape = None  # type: ignore

scaffold_app = typer.Typer(
    name="vertical",
    help="Manage and scaffold vertical structures.",
)
console = Console()


def get_template_dir() -> Path:
    """Get the path to the vertical templates directory."""
    return Path(__file__).parent.parent.parent / "templates" / "vertical"


def validate_vertical_name(name: str) -> str:
    """Validate that the vertical name is a valid Python identifier.

    Args:
        name: The proposed vertical name

    Returns:
        The validated name (lowercase)

    Raises:
        typer.BadParameter: If the name is invalid
    """
    name = name.lower().strip()

    if not name:
        raise typer.BadParameter("Vertical name cannot be empty")

    if not re.match(r"^[a-z][a-z0-9_]*$", name):
        raise typer.BadParameter(
            f"Invalid vertical name '{name}'. "
            "Name must start with a letter and contain only lowercase letters, "
            "numbers, and underscores."
        )

    # Check for reserved names
    reserved = {"victor", "core", "tools", "providers", "config", "ui", "tests"}
    if name in reserved:
        raise typer.BadParameter(f"'{name}' is a reserved name. Please choose a different name.")

    return name


def to_class_name(name: str) -> str:
    """Convert a snake_case name to PascalCase class name.

    Args:
        name: Snake case name (e.g., "data_analysis")

    Returns:
        PascalCase name (e.g., "DataAnalysis")
    """
    return "".join(word.capitalize() for word in name.split("_"))


def to_title(name: str) -> str:
    """Convert a snake_case name to a Title Case string.

    Args:
        name: Snake case name (e.g., "data_analysis")

    Returns:
        Title case string (e.g., "Data Analysis")
    """
    return " ".join(word.capitalize() for word in name.split("_"))


def render_template(
    env: "Environment",
    template_name: str,
    context: dict,
) -> str:
    """Render a Jinja2 template with the given context.

    Args:
        env: Jinja2 environment
        template_name: Name of the template file (e.g., "__init__.py.j2")
        context: Template context variables

    Returns:
        Rendered template content

    Raises:
        typer.Exit: If template is not found
    """
    try:
        template = env.get_template(template_name)
        return template.render(**context)
    except TemplateNotFound:
        console.print(f"[red]Template not found: {template_name}[/]")
        raise typer.Exit(1)


def check_existing_vertical(vertical_dir: Path, force: bool) -> bool:
    """Check if the vertical directory already exists.

    Args:
        vertical_dir: Path to the vertical directory
        force: Whether to overwrite existing files

    Returns:
        True if we should proceed, False otherwise
    """
    if vertical_dir.exists():
        if force:
            console.print(f"[yellow]Warning: Overwriting existing vertical at {vertical_dir}[/]")
            return True
        else:
            console.print(f"[red]Error: Vertical directory already exists at {vertical_dir}[/]")
            console.print("Use --force to overwrite existing files.")
            return False
    return True


def scaffold_plugin(
    name: str,
    description: Optional[str] = None,
    service_provider: bool = False,
    force: bool = False,
    dry_run: bool = False,
    label: str = "plugin",
) -> None:
    """Generate a new plugin/vertical structure from templates.

    This is the core scaffold logic, callable from both ``victor plugin init``
    and the deprecated ``victor vertical create``.

    Args:
        name: Plugin name (lowercase, valid Python identifier).
        description: One-line description of the plugin's purpose.
        service_provider: Whether to include service_provider.py for DI registration.
        force: Overwrite existing files if the directory already exists.
        dry_run: Show what would be created without writing files.
        label: User-facing noun for output messages ("plugin" or "vertical").
    """
    # Check for Jinja2
    if Environment is None:
        console.print(
            "[red]Error: Jinja2 is required for scaffolding. "
            "Install it with: pip install jinja2[/]"
        )
        raise typer.Exit(1)

    # Validate the name
    try:
        name = validate_vertical_name(name)
    except typer.BadParameter as e:
        console.print(f"[red]Error: {e}[/]")
        raise typer.Exit(1)

    # Set default description if not provided
    if not description:
        description = f"{to_title(name)} assistant for specialized tasks"

    # Calculate paths
    victor_dir = Path(__file__).parent.parent.parent
    vertical_dir = victor_dir / name
    template_dir = get_template_dir()

    # Check template directory exists
    if not template_dir.exists():
        console.print(f"[red]Error: Template directory not found at {template_dir}[/]")
        raise typer.Exit(1)

    # Check for existing directory
    if not dry_run and not check_existing_vertical(vertical_dir, force):
        raise typer.Exit(1)

    # Setup Jinja2 environment
    env = Environment(
        loader=FileSystemLoader(str(template_dir)),
        keep_trailing_newline=True,
        autoescape=select_autoescape(
            enabled_extensions=("html", "htm", "xml"),
            default_for_string=False,
            default=False,
        ),
    )

    # Template context
    context = {
        "name": name,
        "name_class": to_class_name(name),
        "name_title": to_title(name),
        "name_upper": name.upper(),
        "description": description,
    }

    # Files to generate
    files_to_create = [
        ("__init__.py.j2", "__init__.py"),
        ("assistant.py.j2", "assistant.py"),
        ("safety.py.j2", "safety.py"),
        ("prompts.py.j2", "prompts.py"),
        ("mode_config.py.j2", "mode_config.py"),
    ]

    if service_provider:
        files_to_create.append(("service_provider.py.j2", "service_provider.py"))

    # Display header
    console.print()
    console.print(
        Panel(
            f"[bold blue]Creating new {label}: {name}[/]\n" f"[dim]Description: {description}[/]",
            title="Victor Scaffold",
            border_style="blue",
        )
    )
    console.print()

    if dry_run:
        console.print("[yellow]Dry run mode - no files will be created[/]\n")

    created_files = []

    for template_name, output_name in files_to_create:
        output_path = vertical_dir / output_name

        try:
            content = render_template(env, template_name, context)
        except typer.Exit:
            raise

        if dry_run:
            console.print(f"[dim]Would create:[/] {output_path}")
            # Show a preview of the first file
            if output_name == "__init__.py":
                console.print()
                console.print(
                    Syntax(
                        content[:500] + "..." if len(content) > 500 else content,
                        "python",
                        theme="monokai",
                        line_numbers=True,
                    )
                )
                console.print()
        else:
            # Create directory if it doesn't exist
            vertical_dir.mkdir(parents=True, exist_ok=True)

            # Write the file
            output_path.write_text(content, encoding="utf-8")
            created_files.append(output_path)
            console.print(f"[green]Created:[/] {output_path}")

    if not dry_run:
        console.print()
        console.print(f"[green]Successfully created {len(created_files)} files[/]")
        console.print()
        console.print("[bold]Next steps:[/]")
        console.print(f"  1. Review and customize the files in victor/{name}/")
        console.print("  2. Update tool and capability requirements in assistant.py")
        console.print(
            f"  3. Verify {to_class_name(name)}Assistant.get_definition() works without runtime imports"
        )
        console.print(
            "  4. Use prompts.py, safety.py, and mode_config.py only for optional follow-on metadata"
        )
        if service_provider:
            console.print("  5. Wire any runtime-only integrations through service_provider.py")
        console.print()
        console.print(f"[dim]To use your new {label}:[/]")
        console.print(f"    from victor.{name} import {to_class_name(name)}Assistant")
        console.print()
    else:
        console.print()
        console.print(
            f"[yellow]Dry run complete. "
            f"Run without --dry-run to create {len(files_to_create)} files.[/]"
        )


@scaffold_app.command("create")
def new_vertical(
    name: str = typer.Argument(
        ...,
        help="Name of the new vertical (e.g., 'security', 'analytics')",
    ),
    description: str = typer.Option(
        None,
        "--description",
        "-d",
        help="Description of the vertical's purpose",
    ),
    service_provider: bool = typer.Option(
        False,
        "--service-provider",
        "-s",
        help="Include service_provider.py for DI container registration",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Overwrite existing files if vertical already exists",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        "-n",
        help="Show what would be created without actually creating files",
    ),
) -> None:
    """Generate a new vertical structure from templates.

    Creates a vertical directory with a contract-first definition layer:
    - __init__.py - Package initialization and assistant export
    - assistant.py - Main vertical definition authored against victor_contracts
    - safety.py - Optional runtime-side safety notes placeholder
    - prompts.py - Optional serializable prompt metadata helper
    - mode_config.py - Optional runtime-side mode metadata placeholder
    - service_provider.py - DI container registration (optional runtime integration)

    Examples:
        victor vertical create security --description "Security analysis assistant"
        victor vertical create analytics -d "Data analytics" --service-provider
        victor vertical create ml --dry-run
    """
    scaffold_plugin(
        name=name,
        description=description,
        service_provider=service_provider,
        force=force,
        dry_run=dry_run,
        label="vertical",
    )


@scaffold_app.command("list")
def list_verticals() -> None:
    """List all available verticals in the codebase.

    Shows both built-in and custom verticals that are registered.
    """
    try:
        from victor.core.verticals import VerticalRegistry

        verticals = VerticalRegistry.list_all()

        if not verticals:
            console.print("[yellow]No verticals found.[/]")
            return

        console.print("\n[bold]Available Verticals[/]\n")
        for name, vertical_class in sorted(verticals):
            description = getattr(vertical_class, "description", "No description")
            console.print(f"  [cyan]{name:20}[/] {description}")

        console.print()
    except ImportError as e:
        console.print(f"[red]Error loading verticals: {e}[/]")
        raise typer.Exit(1)


__all__ = ["scaffold_app", "scaffold_plugin"]
