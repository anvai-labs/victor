"""Database maintenance and management commands.

Provides CLI tools for monitoring, pruning, archiving, and vacuuming
the global and project databases.

Usage:
    victor db stats                              # Show table sizes and DB file size
    victor db prune --group rl --older-than 30d  # Prune RL tables older than 30 days
    victor db prune --table rl_outcome --keep-last 10000
    victor db vacuum                             # Reclaim disk space (global DB)
    victor db maintain                           # Checkpoint WAL + reclaim global & project DB
    victor db archive --group rl --before 2026-03-01
"""

import os
import re
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from victor.config.settings import get_project_paths
from victor.ui.json_utils import create_json_option, print_json_data

db_app = typer.Typer(name="db", help="Database maintenance and management.")
console = Console()


def _get_default_archive_dir() -> Path:
    """Resolve the default archive directory through centralized Victor paths."""
    return get_project_paths().global_victor_dir / "archives"


def _parse_duration(duration_str: str) -> int:
    """Parse duration string like '30d', '6m', '1y' to days."""
    match = re.match(r"^(\d+)([dDmMyY])$", duration_str.strip())
    if not match:
        raise typer.BadParameter(f"Invalid duration '{duration_str}'. Use format: 30d, 6m, 1y")
    value = int(match.group(1))
    unit = match.group(2).lower()
    if unit == "d":
        return value
    if unit == "m":
        return value * 30
    if unit == "y":
        return value * 365
    return value


@db_app.command("stats")
def db_stats(
    json_output: bool = create_json_option(),
) -> None:
    """Show database statistics (table sizes, row counts, file size)."""
    from victor.core.database import get_database

    db = get_database()
    db_path = db.db_path
    file_size_mb = os.path.getsize(db_path) / 1024 / 1024

    stats = db.get_table_stats()

    if json_output:
        output = {
            "db_path": str(db_path),
            "file_size_mb": round(file_size_mb, 1),
            "tables": stats,
        }
        print_json_data(output)
        return

    console.print(f"\n[bold]Database:[/bold] {db_path}")
    console.print(f"[bold]File size:[/bold] {file_size_mb:.1f} MB\n")

    table = Table(title="Table Statistics")
    table.add_column("Table", style="cyan")
    table.add_column("Rows", justify="right")
    table.add_column("Est. Size", justify="right")
    table.add_column("Oldest", justify="center")
    table.add_column("Newest", justify="center")

    total_rows = 0
    total_kb = 0
    for s in stats:
        rows = s["rows"]
        est_kb = s["est_size_kb"]
        total_rows += rows
        total_kb += est_kb

        size_str = f"{est_kb:,} KB" if est_kb < 1024 else f"{est_kb / 1024:.1f} MB"
        table.add_row(
            s["table"],
            f"{rows:,}",
            size_str,
            s["min_date"] or "-",
            s["max_date"] or "-",
        )

    table.add_section()
    total_size = f"{total_kb:,} KB" if total_kb < 1024 else f"{total_kb / 1024:.1f} MB"
    table.add_row(
        "[bold]TOTAL[/bold]",
        f"[bold]{total_rows:,}[/bold]",
        f"[bold]{total_size}[/bold]",
        "",
        "",
    )

    console.print(table)

    # Show prunable groups
    console.print("\n[dim]Prunable groups: rl, agent, all[/dim]")
    console.print("[dim]Example: victor db prune --group rl --older-than 30d[/dim]\n")


@db_app.command("prune")
def db_prune(
    older_than: Optional[str] = typer.Option(
        None, "--older-than", help="Delete rows older than (e.g., '30d', '6m', '1y')"
    ),
    table: Optional[str] = typer.Option(None, "--table", help="Specific table to prune"),
    group: Optional[str] = typer.Option(None, "--group", help="Table group: rl, agent, all"),
    keep_last: Optional[int] = typer.Option(None, "--keep-last", help="Keep only last N rows"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be deleted"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation"),
) -> None:
    """Prune old data from database tables."""
    if not older_than and not keep_last:
        console.print("[red]Must specify --older-than or --keep-last[/red]")
        raise typer.Exit(1)

    from victor.core.database import get_database

    db = get_database()

    # Determine which tables to prune
    if table:
        if table not in db.get_tables_for_group("all"):
            console.print("[red]Unknown or non-prunable table[/red]")
            raise typer.Exit(1)
        tables = [table]
    elif group:
        tables = db.get_tables_for_group(group)
        if not tables:
            console.print(f"[red]Unknown group '{group}'. Use: rl, agent, all[/red]")
            raise typer.Exit(1)
    else:
        console.print("[red]Must specify --table or --group[/red]")
        raise typer.Exit(1)

    older_than_days = _parse_duration(older_than) if older_than else None

    if dry_run:
        console.print("[bold yellow]DRY RUN[/bold yellow] — no data will be deleted\n")

    total_deleted = 0
    for tbl in tables:
        try:
            conn = db.get_connection()
            if older_than_days:
                from datetime import datetime, timedelta, timezone

                date_col = db._get_date_column(tbl)
                cutoff = (datetime.now(timezone.utc) - timedelta(days=older_than_days)).isoformat()
                count = conn.execute(
                    f"SELECT count(*) FROM [{tbl}] WHERE [{date_col}] < ?", (cutoff,)
                ).fetchone()[0]
            elif keep_last:
                total = conn.execute(f"SELECT count(*) FROM [{tbl}]").fetchone()[0]
                count = max(0, total - keep_last)
            else:
                count = 0

            if count == 0:
                console.print(f"  {tbl}: nothing to prune")
                continue

            if dry_run:
                console.print(f"  {tbl}: [yellow]would delete {count:,} rows[/yellow]")
                total_deleted += count
                continue

            if not yes:
                confirm = typer.confirm(f"  Delete {count:,} rows from {tbl}?")
                if not confirm:
                    continue

            deleted = db.prune_table(
                tbl,
                older_than_days=older_than_days,
                keep_last=keep_last,
            )
            total_deleted += deleted
            console.print(f"  {tbl}: [green]deleted {deleted:,} rows[/green]")
        except ValueError as e:
            console.print(f"  {tbl}: [red]{e}[/red]")
        except Exception as e:
            console.print(f"  {tbl}: [red]error: {e}[/red]")

    action = "would delete" if dry_run else "deleted"
    console.print(f"\n[bold]Total {action}: {total_deleted:,} rows[/bold]")

    if not dry_run and total_deleted > 0:
        console.print("[dim]Run 'victor db vacuum' to reclaim disk space[/dim]")


@db_app.command("vacuum")
def db_vacuum() -> None:
    """Reclaim disk space after pruning (runs SQLite VACUUM)."""
    from victor.core.database import get_database

    db = get_database()
    before = os.path.getsize(db.db_path)

    console.print(f"Vacuuming {db.db_path}...")
    db.vacuum()

    after = os.path.getsize(db.db_path)
    saved = before - after
    console.print(
        f"[green]Done.[/green] {before / 1024 / 1024:.1f} MB → "
        f"{after / 1024 / 1024:.1f} MB (saved {saved / 1024 / 1024:.1f} MB)"
    )


@db_app.command("maintain")
def db_maintain(
    project: bool = typer.Option(
        True, "--project/--no-project", help="Also maintain the project graph DB"
    ),
) -> None:
    """Checkpoint the WAL and reclaim space on the global (and project) databases.

    The project graph DB in particular accumulates a large WAL and free pages after
    re-indexing; this truncates the WAL and VACUUMs so the files shrink toward their live
    size. Safe to rerun.
    """
    from victor.core.database import get_database, get_project_database

    managers = [("global", get_database())]
    if project:
        try:
            managers.append(("project", get_project_database()))
        except Exception:
            console.print(
                "[yellow]No project database found; skipping project maintenance.[/yellow]"
            )

    for label, db in managers:
        if not db.db_path.exists():
            console.print(f"[dim]{label}: {db.db_path} does not exist, skipping.[/dim]")
            continue
        console.print(f"Maintaining {label} DB: {db.db_path}...")
        stats = db.maintain()
        before_mb = stats["size_before"] / 1024 / 1024
        after_mb = stats["size_after"] / 1024 / 1024
        console.print(
            f"[green]{label}:[/green] {before_mb:.1f} MB → {after_mb:.1f} MB "
            f"(saved {before_mb - after_mb:.1f} MB)"
        )


@db_app.command("archive")
def db_archive(
    before: str = typer.Option(..., "--before", help="Archive rows before date (YYYY-MM-DD)"),
    output: Optional[Path] = typer.Option(None, "--output", help="Output directory for archives"),
    group: str = typer.Option("all", "--group", help="Table group: rl, agent, all"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation"),
) -> None:
    """Export old data to compressed archive, then delete."""
    from victor.core.database import get_database

    db = get_database()
    tables = db.get_tables_for_group(group)
    if not tables:
        console.print(f"[red]Unknown group '{group}'[/red]")
        raise typer.Exit(1)

    if output is None:
        output = _get_default_archive_dir()
    output.mkdir(parents=True, exist_ok=True)

    if not yes:
        confirm = typer.confirm(
            f"Archive and delete rows before {before} from {len(tables)} tables to {output}?"
        )
        if not confirm:
            raise typer.Abort()

    total = 0
    for tbl in tables:
        try:
            archive_path = output / f"{tbl}_{before}.jsonl.gz"
            count = db.archive_table(tbl, before, archive_path)
            if count > 0:
                console.print(
                    f"  {tbl}: [green]archived {count:,} rows[/green] → {archive_path.name}"
                )
                total += count
            else:
                console.print(f"  {tbl}: nothing to archive")
        except Exception as e:
            console.print(f"  {tbl}: [red]error: {e}[/red]")

    console.print(f"\n[bold]Total archived: {total:,} rows[/bold]")
    if total > 0:
        console.print("[dim]Run 'victor db vacuum' to reclaim disk space[/dim]")
