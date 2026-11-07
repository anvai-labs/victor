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

"""``/import`` and ``/export``: bidirectional session context interop.

Thin UI over victor.agent.conversation.interop — detection, conversion
reporting, DB persistence (deterministic upsert), and multi-tool export
including direct install into the claude/codex local stores.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from rich.panel import Panel
from rich.table import Table

from victor.ui.slash.protocol import BaseSlashCommand, CommandContext, CommandMetadata
from victor.ui.slash.registry import register_command
from victor.agent.conversation.interop.formats import TranscriptFormat

logger = logging.getLogger(__name__)

_FORMAT_KEYS = {
    "auto": None,
    "claude": "claude",
    "codex": "codex",
    "chatgpt": "chatgpt",
    "victor": "victor",
    "anthropic": "anthropic",
    "json": "json",
}


def _resolve_format_flag(ctx: CommandContext, cmd: BaseSlashCommand) -> Optional[str]:
    raw = cmd._get_flag_value(ctx, "--format") or cmd._get_flag_value(ctx, "-f") or "auto"
    key = raw.strip().lower()
    if key not in _FORMAT_KEYS:
        return "$invalid"
    value = _FORMAT_KEYS[key]
    return value  # None means auto-detect


@register_command
class ImportCommand(BaseSlashCommand):
    """Import an external transcript as a victor session."""

    @property
    def metadata(self) -> CommandMetadata:
        return CommandMetadata(
            name="import",
            description="Import a claude/codex/chatgpt/victor transcript as a session",
            usage=(
                "/import <path|claude:<uuid>|codex:<uuid>> "
                "[--format auto|claude|codex|chatgpt|victor|json] [--dry-run] | /import --list"
            ),
            category="session",
            requires_agent=False,
        )

    def execute(self, ctx: CommandContext) -> None:
        positional = [a for a in ctx.args if not a.startswith("--")]
        if self._has_flag(ctx, "--list", "-l"):
            _print_store_listings(ctx)
            return
        if not positional:
            ctx.console.print(
                f"[red]Usage:[/] {self.metadata.usage}\n"
                "[dim]Examples: /import claude:<session-uuid> · "
                "/import ~/.claude/projects/-Users-me-myrepo/<uuid>.jsonl[/]"
            )
            return

        from victor.agent.conversation.interop import (
            UnsupportedFormatError,
            import_transcript,
        )
        from victor.agent.conversation.interop.stores import (
            SessionAddressError,
            resolve_import_target,
        )

        try:
            path, _source = resolve_import_target(positional[0])
        except SessionAddressError as exc:
            ctx.console.print(f"[red]Cannot import:[/] {exc}")
            return
        if not path.is_file():
            ctx.console.print(f"[red]File not found:[/] {path}")
            return

        fmt_key = _resolve_format_flag(ctx, self)
        if fmt_key == "$invalid":
            ctx.console.print(
                "[red]Unknown --format value[/] (auto|claude|codex|chatgpt|victor|json)"
            )
            return
        explicit = TranscriptFormat(fmt_key) if fmt_key else None

        try:
            imported = import_transcript(path, fmt=explicit)
        except UnsupportedFormatError as exc:
            ctx.console.print(f"[red]Cannot import:[/] {exc}")
            return
        except Exception as exc:  # noqa: BLE001 — surface any parse failure cleanly
            ctx.console.print(f"[red]Import failed:[/] {exc}")
            logger.exception("transcript import failed")
            return

        report = imported.report
        session = imported.session
        if not session.messages:
            ctx.console.print("[yellow]No convertible messages found — nothing imported.[/]")
            return

        body = report.render_summary() + f"\nSession: {session.session_id}"
        if report.warnings:
            body += "\n[dim]Warnings: " + "; ".join(report.warnings[:3]) + "[/]"
        if self._has_flag(ctx, "--dry-run", "-d"):
            ctx.console.print(Panel(body, title="Import Preview (dry run)", border_style="cyan"))
            return

        from victor.agent.conversation.interop import to_persist_messages
        from victor.agent.conversation.store import ConversationStore

        try:
            store = ConversationStore()
            messages = to_persist_messages(session)
            store.save_session(
                conversation={"messages": messages},
                model=session.model or "unknown",
                provider=session.provider or "unknown",
                session_id=session.session_id,
                title=session.title,
                tags=[f"import:{report.source_format}"],
                project_path=str(Path.cwd()),
            )
        except Exception as exc:  # noqa: BLE001
            ctx.console.print(f"[red]Failed to save imported session:[/] {exc}")
            logger.exception("import persistence failed")
            return

        total_tokens = sum(m.token_count for m in session.messages)
        hint = f"[dim]Use /resume {session.session_id} to continue it[/]"
        if session.max_tokens and total_tokens > session.max_tokens:
            hint += (
                f"\n[yellow]Note:[/] ~{total_tokens} tokens exceeds the "
                f"{session.max_tokens}-token window; run /compact after resuming."
            )
        ctx.console.print(
            Panel(body + "\n\n" + hint, title="Session Imported", border_style="green")
        )


@register_command
def _load_session_ir(ctx: CommandContext, session_id: Optional[str]):
    """Resolve a session id to the interop IR (errors printed to console).

    Shared by /export and /handoff: explicit id -> active session -> None.
    Returns a ConversationSession or None after printing the reason.
    """
    from datetime import datetime as _dt

    from victor.agent.conversation.interop import ConversationSession
    from victor.agent.conversation.store import ConversationStore
    from victor.agent.conversation.types import ConversationMessage

    if not session_id:
        ctx.console.print("[red]No session id given and no active session.[/]")
        return None
    loaded = ConversationStore().load_session(session_id)
    if loaded is None:
        ctx.console.print(f"[red]Session not found:[/] {session_id}")
        return None
    metadata = loaded.get("metadata", {})
    messages = [
        ConversationMessage(
            id=m.get("id"),
            role=m.get("role", "user"),
            content=m.get("content", ""),
            timestamp=m.get("timestamp"),
            token_count=m.get("token_count", 0),
            tool_name=m.get("tool_name"),
            tool_call_id=m.get("tool_call_id"),
            tool_calls=m.get("tool_calls"),
            metadata=m.get("metadata") or {},
        )
        for m in loaded.get("conversation", {}).get("messages", [])
    ]
    if not messages:
        ctx.console.print("[yellow]Session has no messages to export.[/]")
        return None
    # load_session returns ISO strings; ConversationMessage wants datetimes.
    for message in messages:
        if isinstance(message.timestamp, str):
            message.timestamp = _dt.fromisoformat(message.timestamp)
    return ConversationSession(
        session_id=metadata.get("session_id", session_id),
        title=metadata.get("title"),
        messages=messages,
        model=metadata.get("model"),
        provider=metadata.get("provider"),
    )


@register_command
class ExportCommand(BaseSlashCommand):
    """Export a victor session to another tool's transcript format."""

    @property
    def metadata(self) -> CommandMetadata:
        return CommandMetadata(
            name="export",
            description="Export a session to claude/codex/victor/json format",
            usage="/export [session_id] [--format claude|codex|victor|json] [--out <path>] [--install]",
            category="session",
            requires_agent=False,
        )

    def execute(self, ctx: CommandContext) -> None:
        from victor.agent.conversation.interop import export_session, install_session

        fmt_key = _resolve_format_flag(ctx, self)
        if fmt_key in (None, "$invalid"):
            ctx.console.print("[red]--format required[/] (claude|codex|victor|json)")
            return
        try:
            fmt = TranscriptFormat(fmt_key)
        except ValueError:
            ctx.console.print(f"[red]Format '{fmt_key}' is not exportable[/]")
            return
        if fmt not in (
            TranscriptFormat.CLAUDE_CODE,
            TranscriptFormat.CODEX_ROLLOUT,
            TranscriptFormat.VICTOR_JSON,
            TranscriptFormat.GENERIC_JSON,
        ):
            ctx.console.print(f"[red]Format '{fmt_key}' is import-only[/]")
            return

        install = self._has_flag(ctx, "--install", "-i")
        if install and fmt not in (TranscriptFormat.CLAUDE_CODE, TranscriptFormat.CODEX_ROLLOUT):
            ctx.console.print("[red]--install requires --format claude or codex[/]")
            return

        positional = [a for a in ctx.args if not a.startswith("--")]
        session_id = positional[0] if positional else getattr(ctx.agent, "active_session_id", None)
        session = _load_session_ir(ctx, session_id)
        if session is None:
            return

        try:
            if install:
                target = "claude" if fmt is TranscriptFormat.CLAUDE_CODE else "codex"
                out = install_session(session, target)
                hint = _target_hint(target, out)
                ctx.console.print(
                    Panel(
                        f"Installed into {target} store:\n{out}\n\n{hint}",
                        title="Session Exported",
                        border_style="green",
                    )
                )
                return
            out_raw = self._get_flag_value(ctx, "--out") or self._get_flag_value(ctx, "-o")
            default_name = f"victor-session-{session.session_id}.{_extension(fmt)}"
            out = Path(out_raw).expanduser() if out_raw else Path.cwd() / default_name
            export_session(session, fmt, out)
        except Exception as exc:  # noqa: BLE001
            ctx.console.print(f"[red]Export failed:[/] {exc}")
            logger.exception("session export failed")
            return

        hint = (
            _target_hint("claude" if fmt is TranscriptFormat.CLAUDE_CODE else "codex", out)
            if fmt in (TranscriptFormat.CLAUDE_CODE, TranscriptFormat.CODEX_ROLLOUT)
            else ""
        )
        ctx.console.print(
            Panel(
                f"{len(session.messages)} messages → {out}\n{hint}",
                title="Session Exported",
                border_style="green",
            )
        )
        _print_role_table(ctx, session.messages)


def _print_store_listings(ctx: CommandContext) -> None:
    """Show recent sessions available for address-based import."""
    from victor.agent.conversation.interop.stores import (
        SessionAddressError,
        list_store_sessions,
    )

    for source in ("claude", "codex"):
        table = Table(title=f"{source} sessions (import with {source}:<id>)")
        table.add_column("Session", style="cyan", no_wrap=True)
        table.add_column("First user message", overflow="fold")
        table.add_column("Modified", style="dim")
        try:
            entries = list_store_sessions(source, limit=10)
        except SessionAddressError as exc:
            ctx.console.print(f"[dim]{exc}[/]")
            continue
        for entry in entries:
            table.add_row(
                f"{source}:{entry.session_id}", entry.title, entry.modified.strftime("%m-%d %H:%M")
            )
        if entries:
            ctx.console.print(table)
        else:
            ctx.console.print(f"[dim]No {source} sessions found.[/]")


def _extension(fmt: TranscriptFormat) -> str:
    if fmt is TranscriptFormat.CLAUDE_CODE:
        return "claude.jsonl"
    if fmt is TranscriptFormat.CODEX_ROLLOUT:
        return "codex.jsonl"
    if fmt is TranscriptFormat.GENERIC_JSON:
        return "json"
    return "json"


def _target_hint(target: str, out: Path) -> str:
    if target == "claude":
        return (
            "[dim]Resume with: claude --resume <session-uuid> "
            "(file is already in the right project directory)[/]"
        )
    return "[dim]Resume with: codex resume <session-uuid>[/]"


def _print_role_table(ctx: CommandContext, messages) -> None:
    counts: dict = {}
    for m in messages:
        role = m.role if isinstance(m.role, str) else m.role.value
        counts[role] = counts.get(role, 0) + 1
    table = Table(title=None, box=None, show_header=False)
    table.add_column("role", style="cyan")
    table.add_column("count", justify="right")
    for role, count in sorted(counts.items()):
        table.add_row(role, str(count))
    ctx.console.print(table)


@register_command
class HandoffCommand(BaseSlashCommand):
    """Hand the current (or a specific) session to another coding agent."""

    @property
    def metadata(self) -> CommandMetadata:
        return CommandMetadata(
            name="handoff",
            description="Hand a session to claude or codex (export + install)",
            usage="/handoff <claude|codex> [session_id]",
            category="session",
            requires_agent=False,
        )

    def execute(self, ctx: CommandContext) -> None:
        from victor.agent.conversation.interop import install_session
        from victor.agent.conversation.interop.report import ConversionReport

        positional = [a for a in ctx.args if not a.startswith("--")]
        target = (positional[0] if positional else "").strip().lower()
        if target not in ("claude", "codex"):
            ctx.console.print(
                f"[red]Usage:[/] {self.metadata.usage}\n"
                "[dim]Example: /handoff codex  ·  /handoff claude s_abc123[/]"
            )
            return

        session_id = (
            positional[1] if len(positional) > 1 else getattr(ctx.agent, "active_session_id", None)
        )
        session = _load_session_ir(ctx, session_id)
        if session is None:
            return

        report = ConversionReport(
            source_format="victor", target_format=target, session_id=session.session_id
        )
        try:
            out = install_session(session, target, report)
        except Exception as exc:  # noqa: BLE001 — surface store problems cleanly
            ctx.console.print(f"[red]Handoff failed:[/] {exc}")
            logger.exception("session handoff failed")
            return

        summary = report.render_summary()
        resume_hint = (
            "claude --resume <uuid>  (this project's directory)"
            if target == "claude"
            else "codex resume <uuid>"
        )
        ctx.console.print(
            Panel(
                f"Handed off {len(session.messages)} messages to {target}.\n\n"
                f"{summary}\n\n[bold]Installed:[/] {out}\n\n"
                f"[green]Resume there with:[/] {resume_hint}",
                title=f"Handoff → {target}",
                border_style="green",
            )
        )
