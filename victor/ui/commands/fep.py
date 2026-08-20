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

"""FEP CLI commands for Victor.

Provides commands for creating, validating, submitting, listing, and viewing
Framework Enhancement Proposals (FEPs).

Commands:
    create   - Create a new FEP from template
    validate - Validate FEP structure and content
    submit   - Submit FEP via GitHub PR
    list     - List all FEPs with filtering
    view     - Display specific FEP details

Example:
    victor fep create --title "My Feature" --type standards
    victor fep validate fep-0002-my-feature.md
    victor fep list --status review
    victor fep view 1
"""

import re
import subprocess
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from victor.feps import FEPType, FEPStatus, FEPValidator, parse_fep_metadata
from victor.feps.manager import FEPManager

fep_app = typer.Typer(
    name="fep",
    help="Manage Framework Enhancement Proposals (FEPs).",
)

console = Console()


def _get_feps_dir() -> Path:
    """Get the FEPs directory path.

    Returns:
        Path to feps directory

    Raises:
        typer.Exit: If feps directory not found
    """
    current = Path.cwd()
    for parent in [current] + list(current.parents):
        candidate = parent / "feps"
        if candidate.exists() and candidate.is_dir():
            return candidate

    console.print("[bold red]Error:[/] FEPs directory not found (feps/)")
    console.print("Run this command from the Victor repository root.")
    raise typer.Exit(1)


def _get_template_path() -> Path:
    """Get the FEP template path.

    Returns:
        Path to FEP template

    Raises:
        typer.Exit: If template not found
    """
    feps_dir = _get_feps_dir()
    template_path = feps_dir / "fep-0000-template.md"

    if not template_path.exists():
        console.print(f"[bold red]Error:[/] FEP template not found: {template_path}")
        raise typer.Exit(1)

    return template_path


def _format_authors(authors: list) -> str:
    """Format author list for display.

    Shows first 2 authors with indication of additional authors.

    Args:
        authors: List of author dicts with 'name' key

    Returns:
        Formatted author string
    """
    formatted = ", ".join([a.get("name", "") for a in authors[:2]])
    if len(authors) > 2:
        formatted += f" (+{len(authors) - 2})"
    return formatted


@fep_app.command("create")
def create_fep(
    title: str = typer.Option(..., "--title", "-t", help="FEP title"),
    fep_type: str = typer.Option(
        "standards",
        "--type",
        "-T",
        help="FEP type (standards, informational, process)",
    ),
    author_name: Optional[str] = typer.Option(
        None, "--author", "-a", help="Author name (default: git config user.name)"
    ),
    author_email: Optional[str] = typer.Option(
        None, "--email", "-e", help="Author email (default: git config user.email)"
    ),
    github_username: Optional[str] = typer.Option(
        None, "--github", "-g", help="GitHub username (default: git config github.user)"
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output file path (default: feps/fep-XXXX-title.md)",
    ),
) -> None:
    """Create a new FEP from template.

    Creates a new FEP markdown file from the template with pre-filled metadata.
    The FEP number will be XXXX (to be assigned on submission).
    """
    # Get git config defaults
    try:
        if author_name is None:
            result = subprocess.run(
                ["git", "config", "user.name"],
                capture_output=True,
                text=True,
                check=True,
            )
            author_name = result.stdout.strip()

        if author_email is None:
            result = subprocess.run(
                ["git", "config", "user.email"],
                capture_output=True,
                text=True,
                check=True,
            )
            author_email = result.stdout.strip()

        if github_username is None:
            result = subprocess.run(
                ["git", "config", "github.user"],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0:
                github_username = result.stdout.strip()
    except FileNotFoundError:
        console.print("[yellow]Warning:[/] Git not found. Specify author info manually.\n")

    if not author_name:
        console.print("[bold red]Error:[/] Author name required (use --author)")
        raise typer.Exit(1)

    # Normalize type
    type_map = {
        "standards": "Standards Track",
        "informational": "Informational",
        "process": "Process",
    }
    type_str = type_map.get(fep_type.lower(), "Standards Track")

    # Get template
    template_path = _get_template_path()
    template_content = template_path.read_text(encoding="utf-8")

    # Generate filename. Default into feps/ (where the validator and repo
    # convention expect drafts) rather than the cwd, which previously left
    # scaffold scratch like fep-XXXX-test.md at the repo root.
    title_slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    if output is None:
        output = _get_feps_dir() / f"fep-XXXX-{title_slug}.md"

    # Get current date
    from datetime import datetime

    today = datetime.now().strftime("%Y-%m-%d")

    # Build YAML frontmatter
    yaml_frontmatter = f"""---
fep: XXXX
title: "{title}"
type: {type_str}
status: Draft
created: {today}
modified: {today}
authors:
  - name: "{author_name}\""""

    if author_email:
        yaml_frontmatter += f'\n    email: "{author_email}"'
    if github_username:
        yaml_frontmatter += f'\n    github: "{github_username}"'

    yaml_frontmatter += """
reviewers: []
discussion: null
implementation: null
---
"""

    # Replace template variables
    replacements = {
        r"\{number\}": "XXXX",
        r"\{Title\}": title,
        r"\{Brief, descriptive title\}": title,
        r"\{Name\}": author_name,
        r"\{email\}": author_email or "",
        r"\{github\}": github_username or "",
        r"\{YYYY-MM-DD\}": today,
    }

    for pattern, replacement in replacements.items():
        template_content = re.sub(pattern, replacement, template_content)

    # Build final content with YAML frontmatter at the very top
    final_content = f"{yaml_frontmatter}\n\n# FEP-XXXX: {title}\n\n{template_content}"

    # Write output
    output.write_text(final_content, encoding="utf-8")

    console.print(f"[green]✓[/] Created FEP: [bold]{output}[/]")
    console.print()
    console.print("Next steps:")
    console.print(f"  1. Edit the FEP file: {output}")
    console.print("  2. Fill out all required sections")
    console.print(f"  3. Validate: victor fep validate {output}")
    console.print("  4. Submit: victor fep submit {output}")


@fep_app.command("validate")
def validate_fep_command(
    fep_path: Path = typer.Argument(..., help="Path to FEP markdown file"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed validation output"),
) -> None:
    """Validate FEP structure and content.

    Validates:
    - YAML frontmatter syntax and required fields
    - Required sections exist
    - Section content quality (word counts)

    Returns exit code 0 if valid, 1 if invalid.
    """
    feps_dir = _get_feps_dir()
    validator = FEPValidator(feps_dir=feps_dir)

    console.print(f"Validating FEP: [bold]{fep_path}[/]")
    console.print()

    result = validator.validate_file(fep_path)

    if result.is_valid:
        console.print("[green]✓[/] FEP is valid!")

        if result.has_warnings:
            console.print()
            console.print("[yellow]Warnings:[/]")
            for warning in result.warnings:
                console.print(f"  {warning}")
    else:
        console.print("[bold red]✗[/] FEP validation failed")

    if verbose or not result.is_valid:
        console.print()
        console.print(result.print_summary())

    raise typer.Exit(0 if result.is_valid else 1)


@fep_app.command("submit")
def submit_fep(
    fep_path: Path = typer.Argument(..., help="Path to FEP markdown file"),
    draft: bool = typer.Option(False, "--draft", help="Create draft PR"),
) -> None:
    """Submit FEP via GitHub PR.

    Creates a GitHub pull request for the FEP.
    Requires gh CLI to be installed and authenticated.

    This will:
    1. Validate the FEP
    2. Create a git branch
    3. Commit the FEP
    4. Push to remote
    5. Create PR using gh CLI
    """
    # Validate first
    console.print(f"Validating FEP before submission: [bold]{fep_path}[/]")
    feps_dir = _get_feps_dir()
    validator = FEPValidator(feps_dir=feps_dir)
    result = validator.validate_file(fep_path)

    if not result.is_valid:
        console.print("[bold red]✗[/] FEP validation failed. Fix errors before submitting.")
        console.print()
        console.print("Errors:")
        for error in result.errors:
            console.print(f"  {error}")
        raise typer.Exit(1)

    console.print("[green]✓[/] FEP is valid")
    console.print()

    # Check if gh CLI is available
    try:
        subprocess.run(
            ["gh", "--version"],
            capture_output=True,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        console.print("[bold red]Error:[/] gh CLI not found")
        console.print()
        console.print("To submit FEPs, install the GitHub CLI:")
        console.print("  https://cli.github.com/")
        console.print()
        console.print("Then authenticate:")
        console.print("  gh auth login")
        raise typer.Exit(1)

    # Get metadata for branch name
    metadata = result.metadata
    if not metadata:
        console.print("[bold red]Error:[/] Could not parse FEP metadata")
        raise typer.Exit(1)

    # Create branch name
    title_slug = re.sub(r"[^a-z0-9]+", "-", metadata.title.lower()).strip("-")
    branch_name = f"fep-{metadata.fep:04d}-{title_slug}"

    console.print("Creating pull request...")
    console.print(f"  Branch: {branch_name}")
    console.print(f"  FEP: FEP-{metadata.fep:04d}")
    console.print(f"  Title: {metadata.title}")
    console.print()

    # Check if we want to proceed
    confirm = typer.confirm("Proceed?", default=False)
    if not confirm:
        console.print("Submission cancelled")
        raise typer.Exit(0)

    # Create PR using gh CLI
    try:
        # Create branch
        console.print(f"Creating branch: {branch_name}")
        subprocess.run(
            ["git", "checkout", "-b", branch_name],
            check=True,
            capture_output=True,
        )

        # Stage file
        console.print(f"Staging FEP file: {fep_path}")
        subprocess.run(
            ["git", "add", str(fep_path)],
            check=True,
        )

        # Commit
        commit_msg = f"FEP-{metadata.fep:04d}: {metadata.title}"
        console.print(f"Committing: {commit_msg}")
        subprocess.run(
            ["git", "commit", "-m", commit_msg],
            check=True,
        )

        # Push
        console.print(f"Pushing branch: {branch_name}")
        subprocess.run(
            ["git", "push", "-u", "origin", branch_name],
            check=True,
        )

        # Create PR
        console.print("Creating pull request...")
        pr_title = f"FEP-{metadata.fep:04d}: {metadata.title}"
        pr_body = f"""## FEP Submission

**FEP**: {metadata.fep:04d}
**Title**: {metadata.title}
**Type**: {metadata.type.value}
**Status**: {metadata.status.value}

## Description

This PR proposes {metadata.title.lower()}.

## Checklist

- [ ] FEP follows the template structure
- [ ] All required sections are filled out
- [ ] FEP has been validated with `victor fep validate`
- [ ] Discussion thread created (if applicable)

## Related

- FEP template: `feps/fep-0000-template.md`
- FEP process: `feps/README.md`
"""

        gh_cmd = [
            "gh",
            "pr",
            "create",
            "--title",
            pr_title,
            "--body",
            pr_body,
            "--label",
            "fep",
            "--label",
            f"status:{metadata.status.value.lower()}",
        ]

        if draft:
            gh_cmd.append("--draft")

        subprocess.run(gh_cmd, check=True)

        console.print()
        console.print("[green]✓[/] Pull request created successfully!")
        console.print()
        console.print("Next steps:")
        console.print("  1. FEP will be assigned a number (replace XXXX)")
        console.print("  2. Minimum 14-day review period begins")
        console.print("  3. Address feedback from the community")

    except subprocess.CalledProcessError as e:
        console.print(f"[bold red]Error:[/] Command failed: {e}")
        console.print()
        console.print("You may need to manually complete the submission:")
        console.print(f"  1. Create branch: git checkout -b {branch_name}")
        console.print(f"  2. Stage file: git add {fep_path}")
        console.print(f"  3. Commit: git commit -m '{pr_title}'")
        console.print(f"  4. Push: git push -u origin {branch_name}")
        console.print("  5. Create PR on GitHub")
        raise typer.Exit(1)


@fep_app.command("list")
def list_feps(
    status: Optional[str] = typer.Option(None, "--status", "-s", help="Filter by status"),
    fep_type: Optional[str] = typer.Option(None, "--type", "-t", help="Filter by type"),
    sort_by: str = typer.Option(
        "number", "--sort", help="Sort by field (number, created, modified, title)"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show more details"),
) -> None:
    """List all FEPs with optional filtering and sorting.

    Displays a table of all FEPs with filtering by status and type.

    Example:
        victor fep list
        victor fep list --status accepted
        victor fep list --type standards --sort modified
    """
    feps_dir = _get_feps_dir()
    manager = FEPManager(feps_dir=feps_dir)

    # Parse filters
    status_filter = None
    if status:
        try:
            status_filter = FEPStatus(status.title())
        except ValueError:
            console.print(f"[bold red]Error:[/] Invalid status: {status}")
            console.print(f"Valid statuses: {[s.value for s in FEPStatus]}")
            raise typer.Exit(1)

    type_filter = None
    if fep_type:
        try:
            type_filter = FEPType(
                fep_type.title() + " Track" if fep_type != "process" else "Process"
            )
        except ValueError:
            console.print(f"[bold red]Error:[/] Invalid type: {fep_type}")
            console.print(f"Valid types: {[t.value for t in FEPType]}")
            raise typer.Exit(1)

    # Special case for "standards" -> "Standards Track"
    if fep_type == "standards":
        type_filter = FEPType.STANDARDS
    elif fep_type == "informational":
        type_filter = FEPType.INFORMATIONAL
    elif fep_type == "process":
        type_filter = FEPType.PROCESS

    # List FEPs
    feps = manager.list_feps(
        status_filter=status_filter,
        type_filter=type_filter,
        sort_by=sort_by,
    )

    if not feps:
        console.print("[yellow]No FEPs found[/]")
        raise typer.Exit(0)

    # Create table
    table = Table(title="Framework Enhancement Proposals")
    table.add_column("FEP", style="cyan", no_wrap=True)
    table.add_column("Title", style="white")
    table.add_column("Type", style="magenta")
    table.add_column("Status", style="green")
    table.add_column("Authors", style="blue")

    if verbose:
        table.add_column("Created", style="dim")
        table.add_column("Modified", style="dim")

    for fep in feps:
        authors = _format_authors(fep.authors)

        row = [
            f"FEP-{fep.fep:04d}",
            fep.title,
            fep.type.value,
            fep.status.value,
            authors,
        ]

        if verbose:
            row.extend([fep.created, fep.modified])

        table.add_row(*row)

    console.print(table)
    console.print()
    console.print(f"Total: {len(feps)} FEP(s)")


@fep_app.command("view")
def view_fep(
    fep_number: int = typer.Argument(..., help="FEP number"),
    markdown: bool = typer.Option(False, "--markdown", "-m", help="Render as markdown"),
) -> None:
    """Display specific FEP details.

    Shows FEP metadata and summary section.
    Use --markdown to render the full FEP as markdown.
    """
    feps_dir = _get_feps_dir()
    validator = FEPValidator(feps_dir=feps_dir)

    # Get FEP
    metadata = validator.get_fep(fep_number)

    if not metadata:
        console.print(f"[bold red]Error:[/] FEP-{fep_number:04d} not found")
        console.print()
        console.print("Available FEPs:")
        feps = validator.list_feps()
        for fep in feps:
            console.print(f"  FEP-{fep.fep:04d}: {fep.title}")
        raise typer.Exit(1)

    # Find FEP file
    fep_files = list(feps_dir.glob(f"fep-{fep_number:04d}-*.md"))
    if not fep_files:
        console.print("[bold red]Error:[/] FEP file not found")
        raise typer.Exit(1)

    fep_file = fep_files[0]
    content = fep_file.read_text(encoding="utf-8")

    if markdown:
        # Render full markdown
        console.print()
        md = Markdown(content)
        console.print(md)
    else:
        # Display metadata panel
        panel_content = f"""[bold cyan]FEP-{metadata.fep:04d}:[/] [bold white]{metadata.title}[/]

[bold]Type:[/] {metadata.type.value}
[bold]Status:[/] {metadata.status.value}
[bold]Created:[/] {metadata.created}
[bold]Modified:[/] {metadata.modified}

[bold]Authors:[/]"""
        for author in metadata.authors:
            name = author.get("name", "")
            email = author.get("email", "")
            github = author.get("github", "")

            if name:
                author_str = f"  • {name}"
                if email:
                    author_str += f" <{email}>"
                if github:
                    author_str += f" (@{github})"
                panel_content += f"\n{author_str}"

        if metadata.reviewers:
            panel_content += "\n\n[bold]Reviewers:[/]\n"
            for reviewer in metadata.reviewers:
                panel_content += f"  • {reviewer}"

        if metadata.discussion:
            panel_content += f"\n\n[bold]Discussion:[/] {metadata.discussion}"

        if metadata.implementation:
            panel_content += f"\n[bold]Implementation:[/] {metadata.implementation}"

        console.print()
        console.print(Panel(panel_content, title=f"FEP-{metadata.fep:04d}", border_style="cyan"))
        console.print()

        # Extract and show summary
        sections = {}
        validator._extract_and_validate_sections(
            validator._remove_frontmatter(content),
            sections,
        )

        if "Summary" in sections:
            console.print("[bold]Summary:[/]")
            console.print(sections["Summary"].content)


@fep_app.command("status")
def update_fep_status(
    fep_number: int = typer.Argument(..., help="FEP number"),
    status: str = typer.Option(
        ...,
        "--status",
        "-s",
        help="New status (draft, review, accepted, rejected, deferred, withdrawn, implemented)",
    ),
) -> None:
    """Update FEP status.

    Updates the status of a FEP in its YAML frontmatter.

    Example:
        victor fep status 1 --status review
    """
    feps_dir = _get_feps_dir()

    # Parse status
    try:
        status_map = {
            "draft": FEPStatus.DRAFT,
            "review": FEPStatus.REVIEW,
            "accepted": FEPStatus.ACCEPTED,
            "rejected": FEPStatus.REJECTED,
            "deferred": FEPStatus.DEFERRED,
            "withdrawn": FEPStatus.WITHDRAWN,
            "implemented": FEPStatus.IMPLEMENTED,
        }
        new_status = status_map[status.lower()]
    except KeyError:
        console.print(f"[bold red]Error:[/] Invalid status: {status}")
        console.print(f"Valid statuses: {', '.join(status_map.keys())}")
        raise typer.Exit(1)

    # Update status using manager
    manager = FEPManager(feps_dir=feps_dir)
    success, message = manager.update_fep_status(fep_number, new_status)

    if success:
        console.print(f"[green]✓[/] {message}")
    else:
        console.print(f"[bold red]Error:[/] {message}")
        raise typer.Exit(1)


@fep_app.command("stats")
def show_fep_stats() -> None:
    """Show FEP statistics.

    Displays statistics about all FEPs including counts by status and type.
    """
    feps_dir = _get_feps_dir()
    manager = FEPManager(feps_dir=feps_dir)

    stats = manager.get_statistics()

    console.print()
    console.print(
        Panel(
            f"[bold]Total FEPs:[/] {stats['total']}",
            title="FEP Statistics",
            border_style="cyan",
        )
    )
    console.print()

    # Status breakdown
    console.print("[bold]By Status:[/]")
    for status_value, count in stats["by_status"].items():
        if count > 0:
            console.print(f"  {status_value}: {count}")
    console.print()

    # Type breakdown
    console.print("[bold]By Type:[/]")
    for type_value, count in stats["by_type"].items():
        if count > 0:
            console.print(f"  {type_value}: {count}")
    console.print()

    # Recent FEPs
    if stats["recent"]:
        console.print("[bold]Recently Modified:[/]")
        for fep in stats["recent"]:
            console.print(f"  FEP-{fep.fep:04d}: {fep.title} ({fep.modified})")
    console.print()


@fep_app.command("search")
def search_feps(
    query: str = typer.Argument(..., help="Search query"),
    search_in: Optional[str] = typer.Option(
        None, "--in", "-i", help="Fields to search in (comma-separated: title,authors)"
    ),
) -> None:
    """Search FEPs by query string.

    Searches through FEP titles and authors matching the query.

    Example:
        victor fep search "workflow"
        victor fep search "vijay" --in authors
    """
    feps_dir = _get_feps_dir()
    manager = FEPManager(feps_dir=feps_dir)

    # Parse search fields
    fields = None
    if search_in:
        fields = [f.strip() for f in search_in.split(",")]

    # Search
    results = manager.search_feps(query, search_in=fields)

    if not results:
        console.print(f"[yellow]No FEPs found matching '{query}'[/]")
        raise typer.Exit(0)

    # Display results
    table = Table(title=f"Search Results: '{query}'", show_header=True)
    table.add_column("FEP", style="cyan", no_wrap=True)
    table.add_column("Title", style="white")
    table.add_column("Type", style="magenta")
    table.add_column("Status", style="green")
    table.add_column("Authors", style="blue")

    for fep in results[:10]:  # Limit to 10 results
        authors = _format_authors(fep.authors)

        table.add_row(
            f"FEP-{fep.fep:04d}",
            fep.title,
            fep.type.value,
            fep.status.value,
            authors,
        )

    console.print(table)
    console.print()
    console.print(f"[dim]Found {len(results)} result(s)[/]")
    console.print()
