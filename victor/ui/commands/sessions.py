# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
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

"""Sessions command for managing Victor conversation sessions."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

import typer
from rich.console import Console
from rich.syntax import Syntax
from rich.table import Table

from victor.agent.conversation.store import ConversationStore
from victor.ui.json_utils import create_json_option, print_json_data

sessions_app = typer.Typer(name="session", help="Manage conversation sessions.")
console = Console()


def _use_legacy_session_backend() -> bool:
    """Return whether CLI should use the legacy SQLite session shim."""

    return bool(os.environ.get("VICTOR_TEST_DB_PATH"))


def _get_session_backend() -> Any:
    """Return the active session backend.

    Tests still seed the legacy SQLiteSessionPersistence directly via
    ``VICTOR_TEST_DB_PATH``. Keep the command layer compatible with that
    persisted shape while the default runtime continues to use ConversationStore.
    """

    if _use_legacy_session_backend():
        from victor.agent.sqlite_session_persistence import SQLiteSessionPersistence

        return SQLiteSessionPersistence(db_path=Path(os.environ["VICTOR_TEST_DB_PATH"]))
    return ConversationStore()


def _preview_count_from_session_data(session_data: dict[str, Any]) -> int:
    """Return preview message count from loaded session payload."""

    conversation = session_data.get("conversation", {})
    preview_messages = conversation.get("preview_messages", [])
    return len(preview_messages) if isinstance(preview_messages, list) else 0


def _summary_from_loaded_session(session_data: dict[str, Any]) -> dict[str, Any]:
    """Build flat session summary from a loaded session payload."""

    metadata = session_data.get("metadata", {})
    return {
        "session_id": metadata.get("session_id", ""),
        "title": metadata.get("title", "Untitled"),
        "model": metadata.get("model", "unknown"),
        "provider": metadata.get("provider", "unknown"),
        "profile": metadata.get("profile", "default"),
        "created_at": metadata.get("created_at"),
        "updated_at": metadata.get("updated_at"),
        "message_count": metadata.get("message_count", 0),
        "preview_count": _preview_count_from_session_data(session_data),
        "tags": metadata.get("tags", []),
    }


def _load_session_data(session_id: str) -> Optional[dict[str, Any]]:
    """Load a full session payload using the active backend."""

    backend = _get_session_backend()
    if _use_legacy_session_backend():
        return backend.load_session(session_id)
    return backend.load_session(session_id)


def _list_session_summaries(limit: int) -> list[dict[str, Any]]:
    """List session summaries with a consistent cross-backend schema."""

    backend = _get_session_backend()
    if _use_legacy_session_backend():
        return backend.list_sessions(limit=limit)

    summaries: list[dict[str, Any]] = []
    for session in backend.list_sessions(limit=limit):
        loaded = backend.load_session(session.session_id)
        if loaded:
            summaries.append(_summary_from_loaded_session(loaded))
    return summaries


def _search_session_summaries(query: str, limit: int) -> list[dict[str, Any]]:
    """Search sessions and normalize result summaries across backends."""

    backend = _get_session_backend()
    raw_results = backend.search_sessions(query, limit=limit)
    if _use_legacy_session_backend():
        return raw_results

    summaries: list[dict[str, Any]] = []
    for result in raw_results:
        session_id = result.get("session_id")
        if not session_id:
            continue
        loaded = backend.load_session(session_id)
        if loaded:
            summaries.append(_summary_from_loaded_session(loaded))
    return summaries


def _exportable_sessions(limit: int = 1000) -> list[dict[str, Any]]:
    """Load full session payloads for export."""

    backend = _get_session_backend()
    sessions: list[dict[str, Any]] = []

    if _use_legacy_session_backend():
        for session in backend.list_sessions(limit=limit):
            loaded = backend.load_session(session["session_id"])
            if loaded:
                sessions.append(loaded)
        return sessions

    for session in backend.list_sessions(limit=limit):
        loaded = backend.load_session(session.session_id)
        if loaded:
            sessions.append(loaded)
    return sessions


def _truncate_preview_body(
    preview_body: str,
    *,
    max_lines: int = 12,
    max_chars: int = 1200,
) -> str:
    """Keep preview snippets compact in terminal output."""
    body = preview_body[:max_chars]
    lines = body.splitlines()
    rendered = "\n".join(lines[:max_lines])

    if len(lines) > max_lines or len(preview_body) > len(body):
        return f"{rendered}\n..." if rendered else "..."

    return rendered


def _render_preview_messages(preview_messages: list[dict[str, object]]) -> None:
    """Render replay-only preview sidecars in human-readable session output."""
    if not preview_messages:
        return

    console.print("\n[bold]Preview Messages:[/]")
    for preview in preview_messages[-3:]:
        if not isinstance(preview, dict):
            continue

        content = str(preview.get("content", "")).strip()
        if content:
            console.print(f"[magenta]Preview:[/] {content}")

        metadata = preview.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}

        preview_kind = str(metadata.get("preview_kind", "")).replace("_", " ").strip()
        preview_path = str(metadata.get("preview_path", "")).strip()
        details = []
        if preview_kind:
            details.append(preview_kind.title())
        if preview_path:
            details.append(preview_path)
        if details:
            console.print(f"[dim]{' | '.join(details)}[/]")

        preview_body = metadata.get("preview_body")
        if isinstance(preview_body, str) and preview_body:
            language = str(metadata.get("preview_language") or "text")
            console.print(
                Syntax(
                    _truncate_preview_body(preview_body),
                    language,
                    theme="monokai",
                    line_numbers=False,
                    word_wrap=True,
                )
            )


@sessions_app.command("list")
def sessions_list(
    limit: int = typer.Option(10, "--limit", "-n", help="Maximum number of sessions to list"),
    all: bool = typer.Option(False, "--all", help="List all sessions (no limit)"),
    json_output: bool = create_json_option(),
) -> None:
    """List saved conversation sessions.

    Examples:
        victor sessions list              # List last 10 sessions
        victor sessions list --limit 20   # List last 20 sessions
        victor sessions list --all        # List all sessions
        victor sessions list --json       # Output as JSON
    """
    try:
        actual_limit = 100000 if all else limit
        sessions = _list_session_summaries(limit=actual_limit)

        if not sessions:
            console.print("[dim]No sessions found[/]")
            sys.exit(0)

        if json_output:
            print_json_data(sessions)
        else:
            table = Table(title=f"Saved Sessions (last {len(sessions)})")
            table.add_column("Session ID", style="cyan", no_wrap=True)
            table.add_column("Title", style="white")
            table.add_column("Model", style="yellow")
            table.add_column("Provider", style="blue")
            table.add_column("Messages", justify="right")
            table.add_column("Previews", justify="right")
            table.add_column("Created", style="dim")

            for session in sessions:
                created_at = str(session.get("created_at") or "")
                date_str = created_at.replace("T", " ")[:16] if created_at else "unknown"
                title = str(session.get("title") or "Untitled")

                table.add_row(
                    str(session.get("session_id", "")),
                    title,
                    str(session.get("model") or "unknown"),
                    str(session.get("provider") or "unknown"),
                    str(session.get("message_count", 0)),
                    str(session.get("preview_count", 0)),
                    date_str,
                )

            console.print(table)
            console.print(f"\n[dim]Total: {len(sessions)} session(s)[/]")
            console.print("[dim]Use 'victor sessions show <session_id>' for details[/]")

    except Exception as e:
        console.print(f"[red]Error listing sessions:[/] {e}")
        sys.exit(1)


@sessions_app.command("show")
def sessions_show(
    session_id: str = typer.Argument(..., help="Session ID to show"),
    json_output: bool = create_json_option(),
) -> None:
    """Show details of a specific session.

    Examples:
        victor sessions show myproj-9Kx7Z2
        victor sessions show myproj-9Kx7Z2 --json
    """
    try:
        session = _load_session_data(session_id)

        if not session:
            console.print(f"[red]Session not found:[/] {session_id}")
            sys.exit(1)

        if json_output:
            print_json_data(session)
        else:
            from rich.panel import Panel

            metadata = session.get("metadata", {})
            conversation = session.get("conversation", {})
            messages = conversation.get("messages", [])
            preview_messages = conversation.get("preview_messages", [])
            message_count = int(metadata.get("message_count", len(messages)))
            preview_count = len(preview_messages) if isinstance(preview_messages, list) else 0

            panel_content = (
                f"[bold]Session ID:[/] {metadata.get('session_id', session_id)}\n"
                f"[bold]Title:[/] {metadata.get('title', 'Untitled')}\n"
                f"[bold]Model:[/] {metadata.get('model', 'N/A')}\n"
                f"[bold]Provider:[/] {metadata.get('provider', 'N/A')}\n"
                f"[bold]Profile:[/] {metadata.get('profile', 'N/A')}\n"
                f"[bold]Messages:[/] {message_count}\n"
                f"[bold]Previews:[/] {preview_count}\n"
                f"[bold]Created:[/] {metadata.get('created_at', 'N/A')}\n"
                f"[bold]Updated:[/] {metadata.get('updated_at', 'N/A')}\n"
            )

            console.print(Panel(panel_content, title="Session Details", border_style="cyan"))

            # Show recent messages (last 5)
            if messages:
                console.print("\n[bold]Recent Messages:[/]")
                for msg in messages[-5:]:
                    role_style = {
                        "user": "cyan",
                        "assistant": "green",
                        "system": "dim",
                    }.get(str(msg.get("role", "")), "white")
                    content = str(msg.get("content", ""))
                    # Truncate long messages
                    if len(content) > 200:
                        content = content[:200] + "..."
                    console.print(
                        f"[{role_style}]{str(msg.get('role', '')).capitalize()}:[/] {content}"
                    )

            if isinstance(preview_messages, list) and preview_messages:
                _render_preview_messages(preview_messages)
            else:
                console.print("[dim]No preview messages available[/]")

    except Exception as e:
        console.print(f"[red]Error showing session:[/] {e}")
        sys.exit(1)


@sessions_app.command("search")
def sessions_search(
    query: str = typer.Argument(..., help="Search query string"),
    limit: int = typer.Option(10, "--limit", "-n", help="Maximum results"),
    json_output: bool = create_json_option(),
) -> None:
    """Search sessions by title or content.

    Examples:
        victor sessions search CI/CD
        victor sessions search "authentication" --limit 5
        victor sessions search test --json
    """
    try:
        sessions = _search_session_summaries(query, limit=limit)

        if not sessions:
            console.print(f"[dim]No sessions found matching '{query}'[/]")
            sys.exit(0)

        if json_output:
            print_json_data(sessions)
        else:
            table = Table(title=f"Sessions matching '{query}'")
            table.add_column("Session ID", style="cyan", no_wrap=True)
            table.add_column("Title", style="white")
            table.add_column("Model", style="yellow")
            table.add_column("Provider", style="blue")
            table.add_column("Messages", justify="right")
            table.add_column("Previews", justify="right")

            for session in sessions:
                table.add_row(
                    str(session.get("session_id", "")),
                    str(session.get("title") or "Untitled"),
                    str(session.get("model") or "unknown"),
                    str(session.get("provider") or "unknown"),
                    str(session.get("message_count", 0)),
                    str(session.get("preview_count", 0)),
                )

            console.print(table)
            console.print(f"\n[dim]Found {len(sessions)} session(s)[/]")

    except Exception as e:
        console.print(f"[red]Error searching sessions:[/] {e}")
        sys.exit(1)


@sessions_app.command("delete")
def sessions_delete(
    session_id: str = typer.Argument(..., help="Session ID to delete"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation"),
) -> None:
    """Delete a session.

    Examples:
        victor sessions delete myproj-9Kx7Z2
        victor sessions delete myproj-9Kx7Z2 --yes
    """
    try:
        if not yes:
            from rich.prompt import Confirm

            if not Confirm.ask(f"Delete session {session_id}?"):
                console.print("[dim]Cancelled[/]")
                sys.exit(0)

        backend = _get_session_backend()
        backend.delete_session(session_id)

        console.print(f"[green]✓[/] Deleted session: {session_id}")

    except Exception as e:
        console.print(f"[red]Error deleting session:[/] {e}")
        sys.exit(1)


@sessions_app.command("export")
def sessions_export(
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file path"),
    pretty: bool = typer.Option(True, "--pretty/--no-pretty", help="Pretty print JSON"),
) -> None:
    """Export all sessions to JSON file.

    Examples:
        victor sessions export
        victor sessions export --output sessions.json
        victor sessions export --no-pretty
    """
    try:
        all_sessions = _exportable_sessions(limit=1000)

        if not all_sessions:
            console.print("[dim]No sessions to export[/]")
            sys.exit(0)

        export_data = all_sessions

        # Determine output path
        if not output:
            from datetime import datetime

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output = Path(f"victor_sessions_export_{timestamp}.json")

        # Write to file
        with open(output, "w") as f:
            if pretty:
                json.dump(export_data, f, indent=2)
            else:
                json.dump(export_data, f)

        console.print(f"[green]✓[/] Exported {len(export_data)} session(s) to {output}")

    except Exception as e:
        console.print(f"[red]Error exporting sessions:[/] {e}")
        sys.exit(1)


@sessions_app.command("clear")
def sessions_clear(
    prefix: Optional[str] = typer.Argument(
        None, help="Clear sessions with IDs starting with this prefix (min 6 chars)"
    ),
    all: bool = typer.Option(
        False,
        "--all",
        help="Clear all sessions (default behavior when no prefix specified)",
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation"),
) -> None:
    """Clear sessions from the database.

    This will permanently delete conversation sessions. Use with caution!

    Examples:
        victor sessions clear                    # Clear all sessions (with confirmation)
        victor sessions clear --all --yes        # Clear all sessions (skip confirmation)
        victor sessions clear myproj-9Kx         # Clear sessions starting with prefix
        victor sessions clear myproj-9Kx --yes   # Clear sessions by prefix (skip confirmation)
    """
    try:
        # Validate prefix if provided
        if prefix and len(prefix) < 6:
            console.print("[red]Error:[/] Prefix must be at least 6 characters long.")
            sys.exit(1)

        # Get all sessions to show what will be deleted
        all_sessions = _list_session_summaries(limit=100000)

        # Filter sessions by prefix if specified
        if prefix:
            sessions_to_delete = [
                s for s in all_sessions if str(s.get("session_id", "")).startswith(prefix)
            ]
            if not sessions_to_delete:
                console.print(f"[dim]No sessions found matching prefix '{prefix}'[/]")
                sys.exit(0)
        else:
            sessions_to_delete = all_sessions

        count = len(sessions_to_delete)

        if count == 0:
            console.print("[dim]No sessions found. Database is already empty.[/]")
            sys.exit(0)

        # Show summary of what will be deleted
        if prefix:
            console.print(f"[yellow]⚠[/]  Found {count} session(s) matching prefix '{prefix}'.")
            if not yes:
                from rich.prompt import Confirm

                if not Confirm.ask(
                    f"Are you sure you want to delete {count} session(s) starting with '{prefix}'? This cannot be undone.",
                    default=False,
                ):
                    console.print("[dim]Cancelled[/]")
                    sys.exit(0)
        else:
            console.print(f"[yellow]⚠[/]  Found {count} session(s) in database.")
            if not yes:
                from rich.prompt import Confirm

                if not Confirm.ask(
                    f"Are you sure you want to delete ALL {count} session(s)? This cannot be undone.",
                    default=False,
                ):
                    console.print("[dim]Cancelled[/]")
                    sys.exit(0)

        # Delete sessions
        backend = _get_session_backend()
        deleted_count = 0
        for session in sessions_to_delete:
            backend.delete_session(str(session.get("session_id", "")))
            deleted_count += 1

        if prefix:
            console.print(
                f"[green]✓[/] Cleared {deleted_count} session(s) matching prefix '{prefix}'."
            )
        else:
            console.print(f"[green]✓[/] Cleared {deleted_count} session(s) from database.")

    except Exception as e:
        console.print(f"[red]Error clearing sessions:[/] {e}")
        sys.exit(1)


@sessions_app.command("resume")
def sessions_resume(
    run_id: str = typer.Argument(
        ..., help="The paused run's run_id (from an `awaiting_approval` result)."
    ),
    approve: bool = typer.Option(
        False,
        "--approve/--reject",
        help="Approve (execute the gated tool) or reject (skip it). Default: reject.",
    ),
    note: Optional[str] = typer.Option(
        None, "--note", help="Optional note recorded with the decision."
    ),
    profile: str = typer.Option("default", "--profile", "-p", help="Profile to use."),
    provider: Optional[str] = typer.Option(None, "--provider", "-P", help="Override provider."),
    model: Optional[str] = typer.Option(None, "--model", help="Override model."),
) -> None:
    """Resume a durably-paused chat turn (FEP-0029) with an approval decision.

    When durable approval is configured, a turn that hits an ``ASK`` gate parks with a ``run_id``
    (persisted in the project database) instead of blocking — from any surface (API, TUI). Answer it
    here: approve executes the exact gated tool call and continues; reject skips it with a tool-error
    and continues.

    Examples:
        victor session resume <run_id> --approve
        victor session resume <run_id> --reject --note "too risky"
    """
    import asyncio

    from victor.config.settings import load_settings
    from victor.framework.approval_pause import ApprovalDecision
    from victor.framework.session_config import SessionConfig
    from victor.framework.session_runner import FrameworkSessionRunner

    async def _run() -> None:
        settings = load_settings()
        config = SessionConfig.from_cli_flags(
            agent_profile=None if profile == "default" else profile,
            provider=provider,
            model=model,
        )
        session_runner = FrameworkSessionRunner(settings, config)
        client = session_runner.create_client(config)
        await session_runner.initialize_client(client)
        try:
            result = await client.resume(
                run_id,
                ApprovalDecision(approved=approve, response=note, responder="cli"),
            )
        except ValueError as exc:
            console.print(f"[bold red]Error:[/] {exc}")
            raise typer.Exit(1)
        finally:
            close = getattr(client, "close", None)
            if close is not None:
                await close()

        status = getattr(result, "status", "ok") or "ok"
        if status == "awaiting_approval":
            new_run = getattr(result, "run_id", None)
            console.print(
                f"[yellow]⏸ Still awaiting approval[/] — a further ASK fired; resume with "
                f"run_id [bold]{new_run}[/]."
            )
        else:
            verb = "approved" if approve else "rejected"
            console.print(f"[green]✓[/] Resumed ({verb}).")
            console.print(getattr(result, "content", "") or "[dim](no further output)[/]")

    asyncio.run(_run())
