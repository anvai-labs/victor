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

"""Navigation and file management slash commands: directory, changes, snapshots, commit, undo, redo, history, copy."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from rich.panel import Panel
from rich.table import Table

from victor.ui.slash.protocol import BaseSlashCommand, CommandContext, CommandMetadata
from victor.ui.slash.registry import register_command

logger = logging.getLogger(__name__)


@register_command
class DirectoryCommand(BaseSlashCommand):
    """Show or change working directory."""

    @property
    def metadata(self) -> CommandMetadata:
        return CommandMetadata(
            name="directory",
            description="Show or change working directory",
            usage="/directory [path]",
            aliases=["cd"],
            category="navigation",
        )

    def execute(self, ctx: CommandContext) -> None:
        if not ctx.args:
            # Show current directory
            cwd = Path.cwd()
            ctx.console.print(f"[bold]Working Directory:[/] {cwd}")

            # Show .victor directory status
            victor_dir = cwd / ".victor"
            if victor_dir.exists():
                ctx.console.print(f"[dim]Victor config: {victor_dir}[/]")
            else:
                ctx.console.print("[dim]No .victor directory (use /init to create)[/]")
            return

        # Change directory
        new_path = Path(ctx.args[0]).expanduser().resolve()

        if not new_path.exists():
            ctx.console.print(f"[red]Path not found:[/] {new_path}")
            return

        if not new_path.is_dir():
            ctx.console.print(f"[red]Not a directory:[/] {new_path}")
            return

        try:
            os.chdir(new_path)
            ctx.console.print(f"[green]Changed to:[/] {new_path}")
        except Exception as e:
            ctx.console.print(f"[red]Failed to change directory:[/] {e}")


@register_command
class ChangesCommand(BaseSlashCommand):
    """View, diff, or revert file changes."""

    @property
    def metadata(self) -> CommandMetadata:
        return CommandMetadata(
            name="changes",
            description="View, diff, or revert file changes",
            usage="/changes [show|revert|stash] [file]",
            aliases=["diff"],
            category="navigation",
            requires_agent=True,
        )

    def execute(self, ctx: CommandContext) -> None:
        if not self._require_agent(ctx):
            return

        subcommand = self._get_arg(ctx, 0, "show").lower()
        target_file = self._get_arg(ctx, 1)

        # Get file tracker if available
        tracker = getattr(ctx.agent, "_file_tracker", None)

        if not tracker:
            ctx.console.print("[yellow]File change tracking not available[/]")
            return

        if subcommand == "show":
            changes = tracker.get_changes()
            if not changes:
                ctx.console.print("[dim]No file changes recorded[/]")
                return

            table = Table(title="File Changes")
            table.add_column("File", style="cyan")
            table.add_column("Action", style="green")
            table.add_column("Time", style="dim")

            for change in changes[-20:]:  # Last 20 changes
                table.add_row(
                    str(change.get("file", "?"))[-50:],
                    change.get("action", "modify"),
                    change.get("timestamp", "")[:19],
                )

            ctx.console.print(table)
            ctx.console.print("\n[dim]Use /changes revert <file> to undo[/]")

        elif subcommand == "revert":
            if not target_file:
                ctx.console.print("[yellow]Usage: /changes revert <file>[/]")
                return

            if tracker.revert_file(target_file):
                ctx.console.print(f"[green]Reverted:[/] {target_file}")
            else:
                ctx.console.print(f"[red]Could not revert:[/] {target_file}")

        elif subcommand == "stash":
            if tracker.stash_changes():
                ctx.console.print("[green]Changes stashed[/]")
            else:
                ctx.console.print("[yellow]No changes to stash[/]")

        else:
            ctx.console.print(f"[red]Unknown subcommand:[/] {subcommand}")
            ctx.console.print("[dim]Use: show, revert, stash[/]")


@register_command
class UndoCommand(BaseSlashCommand):
    """Undo the last file change(s)."""

    @property
    def metadata(self) -> CommandMetadata:
        return CommandMetadata(
            name="undo",
            description="Undo the last file change(s)",
            usage="/undo",
            category="navigation",
            requires_agent=True,
        )

    def execute(self, ctx: CommandContext) -> None:
        if not self._require_agent(ctx):
            return

        from victor.agent.change_tracker import get_change_tracker

        tracker = get_change_tracker()
        success, message, files = tracker.undo()
        style = "green" if success else "yellow"
        ctx.console.print(f"[{style}]{message}[/]")
        for f in files:
            ctx.console.print(f"  [dim]•[/] {f}")


@register_command
class RedoCommand(BaseSlashCommand):
    """Redo the last undone change(s)."""

    @property
    def metadata(self) -> CommandMetadata:
        return CommandMetadata(
            name="redo",
            description="Redo the last undone change(s)",
            usage="/redo",
            category="navigation",
            requires_agent=True,
        )

    def execute(self, ctx: CommandContext) -> None:
        if not self._require_agent(ctx):
            return

        from victor.agent.change_tracker import get_change_tracker

        tracker = get_change_tracker()
        success, message, files = tracker.redo()
        style = "green" if success else "yellow"
        ctx.console.print(f"[{style}]{message}[/]")
        for f in files:
            ctx.console.print(f"  [dim]•[/] {f}")


@register_command
class HistoryCommand(BaseSlashCommand):
    """Show file change history."""

    @property
    def metadata(self) -> CommandMetadata:
        return CommandMetadata(
            name="filehistory",
            description="Show file change history",
            usage="/filehistory [limit]",
            aliases=["timeline"],
            category="navigation",
            requires_agent=True,
        )

    def execute(self, ctx: CommandContext) -> None:
        if not self._require_agent(ctx):
            return

        limit = self._parse_int_arg(ctx, 0, default=20)

        from victor.agent.change_tracker import get_change_tracker

        tracker = get_change_tracker()
        history = tracker.get_history(limit=limit)
        if not history:
            ctx.console.print("[dim]No file change history[/]")
            return

        table = Table(title=f"File Change History (last {len(history)})")
        table.add_column("Time", style="dim")
        table.add_column("Files", justify="right")
        table.add_column("Tool", style="cyan")
        table.add_column("Description", style="green")
        table.add_column("Status", justify="center")

        for entry in history:
            files = entry.get("files", [])
            file_count = entry.get("file_count", len(files) if isinstance(files, list) else 0)
            table.add_row(
                str(entry.get("timestamp", ""))[:19],
                str(file_count),
                str(entry.get("tool_name", "?")),
                str(entry.get("description", ""))[:50],
                "undone" if entry.get("undone") else "applied",
            )

        ctx.console.print(table)


@register_command
class SnapshotsCommand(BaseSlashCommand):
    """Manage workspace snapshots for safe rollback."""

    @property
    def metadata(self) -> CommandMetadata:
        return CommandMetadata(
            name="snapshots",
            description="Manage workspace snapshots for safe rollback",
            usage="/snapshots [list|create|restore|diff|clear] [id]",
            aliases=["snap"],
            category="navigation",
        )

    def execute(self, ctx: CommandContext) -> None:
        subcommand = self._get_arg(ctx, 0, "list").lower()

        try:
            from victor.agent.snapshot_store import get_snapshot_store

            store = get_snapshot_store()

            if subcommand == "list":
                snapshots = store.list_snapshots()
                if not snapshots:
                    ctx.console.print("[dim]No snapshots found[/]")
                    ctx.console.print("[dim]Create one: /snapshots create[/]")
                    return

                table = Table(title="Workspace Snapshots")
                table.add_column("ID", style="cyan")
                table.add_column("Description", style="white")
                table.add_column("Files", justify="right")
                table.add_column("Created", style="dim")

                for snap in snapshots[-10:]:
                    table.add_row(
                        snap.get("id", "?")[:8],
                        snap.get("description", "")[:40],
                        str(snap.get("file_count", 0)),
                        snap.get("created_at", "")[:19],
                    )

                ctx.console.print(table)
                ctx.console.print("\n[dim]Use /snapshots restore <id> to restore[/]")

            elif subcommand == "create":
                description = " ".join(ctx.args[1:]) if len(ctx.args) > 1 else "Manual snapshot"
                snapshot_id = store.create_snapshot(description=description)
                ctx.console.print(f"[green]Snapshot created:[/] {snapshot_id[:8]}")

            elif subcommand == "restore":
                snapshot_id = self._get_arg(ctx, 1)
                if not snapshot_id:
                    ctx.console.print("[yellow]Usage: /snapshots restore <id>[/]")
                    return

                if store.restore_snapshot(snapshot_id):
                    ctx.console.print(f"[green]Restored snapshot:[/] {snapshot_id[:8]}")
                else:
                    ctx.console.print(f"[red]Snapshot not found:[/] {snapshot_id}")

            elif subcommand == "diff":
                snapshot_id = self._get_arg(ctx, 1)
                if not snapshot_id:
                    ctx.console.print("[yellow]Usage: /snapshots diff <id>[/]")
                    return

                diff = store.get_diff(snapshot_id)
                if diff:
                    ctx.console.print(
                        Panel(
                            diff[:2000],
                            title=f"Diff: {snapshot_id[:8]}",
                            border_style="yellow",
                        )
                    )
                else:
                    ctx.console.print(f"[red]Snapshot not found:[/] {snapshot_id}")

            elif subcommand == "clear":
                count = store.clear_snapshots()
                ctx.console.print(f"[green]Cleared {count} snapshots[/]")

            else:
                ctx.console.print(f"[red]Unknown subcommand:[/] {subcommand}")
                ctx.console.print("[dim]Use: list, create, restore, diff, clear[/]")

        except ImportError:
            ctx.console.print("[yellow]Snapshot store not available[/]")
        except Exception as e:
            ctx.console.print(f"[red]Snapshot error:[/] {e}")


@register_command
class CommitCommand(BaseSlashCommand):
    """Commit current changes with AI-generated message."""

    @property
    def metadata(self) -> CommandMetadata:
        return CommandMetadata(
            name="commit",
            description="Commit current changes with AI-generated message",
            usage="/commit [message]",
            aliases=["ci"],
            category="navigation",
            requires_agent=True,
            is_async=True,
        )

    async def execute(self, ctx: CommandContext) -> None:
        if not self._require_agent(ctx):
            return

        import subprocess

        from victor.core.attribution import append_victor_commit_attribution

        # Check for staged changes
        try:
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                cwd=Path.cwd(),
            )
            changes = result.stdout.strip()

            if not changes:
                ctx.console.print("[yellow]No changes to commit[/]")
                return

            ctx.console.print(f"[dim]Found {len(changes.splitlines())} changed files[/]")

        except FileNotFoundError:
            ctx.console.print("[red]Git not available[/]")
            return

        # User-provided or AI-generated message
        if ctx.args:
            commit_message = " ".join(ctx.args)
        else:
            ctx.console.print("[dim]Generating commit message...[/]")

            # Get diff for AI
            diff_result = subprocess.run(
                ["git", "diff", "--staged", "--stat"],
                capture_output=True,
                text=True,
                cwd=Path.cwd(),
            )
            diff_stat = diff_result.stdout[:1000]

            prompt = (
                f"Generate a concise git commit message for these changes:\n\n"
                f"```\n{changes}\n```\n\n"
                f"Stats:\n```\n{diff_stat}\n```\n\n"
                f"Format: <type>(<scope>): <description>\n"
                f"Types: feat, fix, docs, style, refactor, test, chore"
            )

            try:
                response = await ctx.agent.chat(prompt)
                commit_message = response.content.strip().split("\n")[0]
                # Clean up any markdown formatting
                commit_message = commit_message.strip("`").strip()
            except Exception as e:
                ctx.console.print(f"[red]Failed to generate message:[/] {e}")
                return

        ctx.console.print(f"[bold]Commit message:[/] {commit_message}")
        commit_message = append_victor_commit_attribution(commit_message)

        # Stage all changes
        subprocess.run(["git", "add", "-A"], cwd=Path.cwd())

        # Commit
        result = subprocess.run(
            ["git", "commit", "-m", commit_message],
            capture_output=True,
            text=True,
            cwd=Path.cwd(),
        )

        if result.returncode == 0:
            ctx.console.print("[green]Committed successfully![/]")
            # Show short commit hash
            hash_result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                cwd=Path.cwd(),
            )
            ctx.console.print(f"[dim]Commit: {hash_result.stdout.strip()}[/]")
        else:
            ctx.console.print(f"[red]Commit failed:[/] {result.stderr}")


@register_command
class CopyCommand(BaseSlashCommand):
    """Copy last assistant response to clipboard."""

    @property
    def metadata(self) -> CommandMetadata:
        return CommandMetadata(
            name="copy",
            description="Copy last assistant response to clipboard",
            usage="/copy",
            category="navigation",
            requires_agent=True,
        )

    def execute(self, ctx: CommandContext) -> None:
        if not self._require_agent(ctx):
            return

        # Get last assistant message
        messages = ctx.agent.conversation.messages
        last_assistant = None
        for msg in reversed(messages):
            if msg.role == "assistant":
                last_assistant = msg.content
                break

        if not last_assistant:
            ctx.console.print("[yellow]No assistant response to copy[/]")
            return

        try:
            import pyperclip

            pyperclip.copy(last_assistant)
            preview = last_assistant[:100] + "..." if len(last_assistant) > 100 else last_assistant
            ctx.console.print(f"[green]Copied to clipboard:[/] {preview}")
        except ImportError:
            ctx.console.print("[yellow]Clipboard not available (install pyperclip)[/]")
            ctx.console.print(f"[dim]Response preview: {last_assistant[:200]}...[/]")
        except Exception as e:
            ctx.console.print(f"[red]Copy failed:[/] {e}")
