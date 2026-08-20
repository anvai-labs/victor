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

"""CLI command for discovering and viewing Victor examples."""

import ast
import re
import shutil
from pathlib import Path
from typing import List, Optional, Tuple

import typer
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

examples_app = typer.Typer(
    name="examples",
    help="Discover and view Victor examples.",
)
console = Console()

# Categories for examples based on filename patterns
CATEGORY_PATTERNS = {
    "agent": ["agent", "claude", "gpt", "gemini", "grok", "ollama", "simple_chat"],
    "team": ["team", "ensemble", "multi_agent"],
    "mcp": ["mcp"],
    "workflow": ["workflow", "pipeline", "cicd"],
    "provider": ["provider", "anthropic", "openai", "google", "cerebras", "groq"],
    "tool": ["tool", "composition"],
    "rag": ["rag", "embedding", "semantic", "index", "codebase"],
    "config": ["config", "profile", "airgapped"],
}


def _get_examples_dir() -> Path:
    """Get the examples directory path."""
    # Navigate from victor/ui/commands to project root
    current = Path(__file__).parent
    project_root = current.parent.parent.parent
    examples_dir = project_root / "examples"
    return examples_dir


def _extract_python_description(file_path: Path) -> str:
    """Extract description from Python file docstring."""
    try:
        content = file_path.read_text(encoding="utf-8")
        tree = ast.parse(content)
        docstring = ast.get_docstring(tree)
        if docstring:
            # Return first line or sentence of docstring
            first_line = docstring.split("\n")[0].strip()
            return first_line
        return "No description available"
    except Exception:
        return "No description available"


def _extract_yaml_description(file_path: Path) -> str:
    """Extract description from YAML file comments."""
    try:
        content = file_path.read_text(encoding="utf-8")
        lines = content.split("\n")
        description_lines = []

        for line in lines:
            # Look for comment lines at the top (after the first comment line)
            if line.startswith("#"):
                # Skip the first "# Example:" line if present
                comment = line.lstrip("# ").strip()
                if comment and not comment.startswith("Example:"):
                    description_lines.append(comment)
                    break
            elif line.strip() and not line.startswith("#"):
                break

        if description_lines:
            return description_lines[0]

        # Try to extract from 'description' field in YAML
        for line in lines:
            if "description:" in line.lower():
                match = re.search(r'description:\s*["\']?([^"\']+)["\']?', line, re.IGNORECASE)
                if match:
                    return match.group(1).strip()

        return "No description available"
    except Exception:
        return "No description available"


def _categorize_example(filename: str) -> str:
    """Categorize an example based on its filename."""
    filename_lower = filename.lower()
    for category, patterns in CATEGORY_PATTERNS.items():
        for pattern in patterns:
            if pattern in filename_lower:
                return category
    return "general"


def _discover_examples(
    examples_dir: Path, filter_keyword: Optional[str] = None
) -> List[Tuple[str, str, str, Path]]:
    """Discover all examples in the examples directory.

    Returns:
        List of tuples: (name, description, category, full_path)
    """
    examples = []

    if not examples_dir.exists():
        return examples

    # Find Python files
    for py_file in examples_dir.glob("**/*.py"):
        # Skip __init__.py and files in external_vertical
        if py_file.name == "__init__.py":
            continue
        if "external_vertical" in str(py_file):
            continue

        relative_path = py_file.relative_to(examples_dir)
        name = str(relative_path.with_suffix("")).replace("/", "_").replace("\\", "_")
        description = _extract_python_description(py_file)
        category = _categorize_example(py_file.name)

        # Apply filter if provided
        if filter_keyword:
            keyword_lower = filter_keyword.lower()
            if (
                keyword_lower not in name.lower()
                and keyword_lower not in description.lower()
                and keyword_lower not in category.lower()
            ):
                continue

        examples.append((name, description, category, py_file))

    # Find YAML files
    for yaml_file in examples_dir.glob("**/*.yaml"):
        relative_path = yaml_file.relative_to(examples_dir)
        name = str(relative_path.with_suffix("")).replace("/", "_").replace("\\", "_")
        description = _extract_yaml_description(yaml_file)
        category = _categorize_example(yaml_file.name)

        # Apply filter if provided
        if filter_keyword:
            keyword_lower = filter_keyword.lower()
            if (
                keyword_lower not in name.lower()
                and keyword_lower not in description.lower()
                and keyword_lower not in category.lower()
            ):
                continue

        examples.append((name, description, category, yaml_file))

    # Sort by category, then by name
    examples.sort(key=lambda x: (x[2], x[0]))
    return examples


def _find_example_by_name(examples_dir: Path, name: str) -> Optional[Tuple[str, str, str, Path]]:
    """Find a specific example by name (supports partial matching)."""
    examples = _discover_examples(examples_dir)
    name_lower = (
        name.lower().replace("_", "").replace("-", "").replace(".py", "").replace(".yaml", "")
    )

    # First try exact match
    for example in examples:
        example_name_normalized = example[0].lower().replace("_", "")
        if example_name_normalized == name_lower:
            return example

    # Then try partial match
    matches = []
    for example in examples:
        example_name_normalized = example[0].lower().replace("_", "")
        if name_lower in example_name_normalized:
            matches.append(example)

    if len(matches) == 1:
        return matches[0]
    elif len(matches) > 1:
        # Return the shortest match (most likely the intended one)
        return min(matches, key=lambda x: len(x[0]))

    return None


@examples_app.callback(invoke_without_command=True)
def examples_callback(
    ctx: typer.Context,
    filter_keyword: Optional[str] = typer.Option(
        None,
        "--filter",
        "-f",
        help="Filter examples by keyword (matches name, description, or category)",
    ),
) -> None:
    """Discover and view Victor examples.

    Examples:
        victor examples                      # List all examples
        victor examples list                 # Same as above
        victor examples show claude          # Show claude_example.py content
        victor examples show mcp_server      # Show mcp_server_demo.py
        victor examples --filter teams       # Filter examples by keyword
        victor examples --filter mcp         # Filter MCP-related examples
        victor examples copy claude ./       # Copy example to current directory
    """
    if ctx.invoked_subcommand is None:
        # Default to list command
        list_examples(filter_keyword=filter_keyword)


@examples_app.command("list")
def list_examples(
    filter_keyword: Optional[str] = typer.Option(
        None,
        "--filter",
        "-f",
        help="Filter examples by keyword (matches name, description, or category)",
    ),
) -> None:
    """List all available examples with descriptions.

    Examples:
        victor examples list
        victor examples list --filter mcp
        victor examples list --filter agent
    """
    examples_dir = _get_examples_dir()

    if not examples_dir.exists():
        console.print(f"[red]Examples directory not found:[/] {examples_dir}")
        raise typer.Exit(1)

    examples = _discover_examples(examples_dir, filter_keyword)

    if not examples:
        if filter_keyword:
            console.print(f"[yellow]No examples found matching '{filter_keyword}'[/]")
        else:
            console.print("[yellow]No examples found[/]")
        return

    # Group examples by category
    categories: dict = {}
    for name, description, category, path in examples:
        if category not in categories:
            categories[category] = []
        categories[category].append((name, description, path))

    # Create table
    table = Table(
        title="Victor Examples" + (f" (filtered: {filter_keyword})" if filter_keyword else ""),
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("Category", style="yellow")
    table.add_column("Description", style="green")
    table.add_column("Type", style="dim")

    # Add rows grouped by category
    current_category = None
    for name, description, category, path in examples:
        if current_category != category:
            if current_category is not None:
                table.add_row("", "", "", "")  # Empty row between categories
            current_category = category

        file_type = "Python" if path.suffix == ".py" else "YAML"
        # Truncate description if too long
        if len(description) > 60:
            description = description[:57] + "..."
        table.add_row(name, category, description, file_type)

    console.print(table)
    console.print(f"\n[dim]Found {len(examples)} examples[/]")
    console.print("\n[dim]Usage:[/]")
    console.print("  victor examples show <name>    # View example content")
    console.print("  victor examples copy <name> .  # Copy to current directory")


@examples_app.command("show")
def show_example(
    name: str = typer.Argument(..., help="Name of the example to show (supports partial matching)"),
    lines: Optional[int] = typer.Option(
        None,
        "--lines",
        "-n",
        help="Limit output to first N lines",
    ),
) -> None:
    """Show the content of an example with syntax highlighting.

    Examples:
        victor examples show claude
        victor examples show mcp_server
        victor examples show tool_composition
        victor examples show code_analysis
    """
    examples_dir = _get_examples_dir()

    if not examples_dir.exists():
        console.print(f"[red]Examples directory not found:[/] {examples_dir}")
        raise typer.Exit(1)

    example = _find_example_by_name(examples_dir, name)

    if not example:
        console.print(f"[red]Example not found:[/] {name}")
        console.print("\n[dim]Available examples:[/]")

        # Show some suggestions
        examples = _discover_examples(examples_dir)
        for ex_name, _, _, _ in examples[:10]:
            console.print(f"  - {ex_name}")
        if len(examples) > 10:
            console.print(f"  ... and {len(examples) - 10} more")
        raise typer.Exit(1)

    ex_name, description, category, path = example

    try:
        content = path.read_text(encoding="utf-8")

        if lines:
            content_lines = content.split("\n")
            content = "\n".join(content_lines[:lines])
            if len(content_lines) > lines:
                content += f"\n\n... ({len(content_lines) - lines} more lines)"

        # Determine syntax lexer
        lexer = "python" if path.suffix == ".py" else "yaml"

        # Create syntax-highlighted output
        syntax = Syntax(content, lexer, theme="monokai", line_numbers=True)

        # Display with metadata panel
        console.print(
            Panel(
                f"[bold]{ex_name}[/]\n"
                f"[dim]Category:[/] {category}\n"
                f"[dim]Description:[/] {description}\n"
                f"[dim]Path:[/] {path}",
                title="Example Info",
                border_style="cyan",
            )
        )
        console.print()
        console.print(syntax)

    except Exception as e:
        console.print(f"[red]Error reading example:[/] {e}")
        raise typer.Exit(1)


@examples_app.command("copy")
def copy_example(
    name: str = typer.Argument(..., help="Name of the example to copy"),
    destination: Path = typer.Argument(
        ...,
        help="Destination directory or file path",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Overwrite if destination exists",
    ),
) -> None:
    """Copy an example to a destination directory.

    Examples:
        victor examples copy claude ./
        victor examples copy mcp_server ./my_mcp_server.py
        victor examples copy code_analysis ./my_analysis.yaml
    """
    examples_dir = _get_examples_dir()

    if not examples_dir.exists():
        console.print(f"[red]Examples directory not found:[/] {examples_dir}")
        raise typer.Exit(1)

    example = _find_example_by_name(examples_dir, name)

    if not example:
        console.print(f"[red]Example not found:[/] {name}")
        raise typer.Exit(1)

    ex_name, description, category, source_path = example

    # Determine destination path
    dest_path = Path(destination)
    if dest_path.is_dir():
        dest_path = dest_path / source_path.name
    elif not dest_path.suffix:
        # If no extension, add the original extension
        dest_path = dest_path.with_suffix(source_path.suffix)

    # Check if destination exists
    if dest_path.exists() and not force:
        console.print(f"[red]Destination already exists:[/] {dest_path}")
        console.print("[dim]Use --force to overwrite[/]")
        raise typer.Exit(1)

    try:
        # Ensure parent directory exists
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        # Copy the file
        shutil.copy2(source_path, dest_path)

        console.print("[green]Copied successfully![/]")
        console.print(f"  [dim]Source:[/] {source_path}")
        console.print(f"  [dim]Destination:[/] {dest_path}")
        console.print(f"\n[dim]Description:[/] {description}")

    except Exception as e:
        console.print(f"[red]Error copying example:[/] {e}")
        raise typer.Exit(1)


@examples_app.command("categories")
def list_categories() -> None:
    """List available example categories.

    Examples:
        victor examples categories
    """
    examples_dir = _get_examples_dir()
    examples = _discover_examples(examples_dir)

    # Count examples per category
    category_counts: dict = {}
    for _, _, category, _ in examples:
        category_counts[category] = category_counts.get(category, 0) + 1

    table = Table(title="Example Categories", show_header=True)
    table.add_column("Category", style="cyan")
    table.add_column("Count", style="yellow", justify="right")
    table.add_column("Description", style="green")

    category_descriptions = {
        "agent": "Agent and LLM provider examples",
        "team": "Multi-agent team and ensemble examples",
        "mcp": "Model Context Protocol examples",
        "workflow": "Workflow and pipeline examples",
        "provider": "LLM provider-specific examples",
        "tool": "Tool composition and usage examples",
        "rag": "RAG, embedding, and search examples",
        "config": "Configuration and profile examples",
        "general": "General examples",
    }

    for category in sorted(category_counts.keys()):
        count = category_counts[category]
        desc = category_descriptions.get(category, "")
        table.add_row(category, str(count), desc)

    console.print(table)
    console.print("\n[dim]Use 'victor examples list --filter <category>' to filter by category[/]")
