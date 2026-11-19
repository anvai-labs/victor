# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
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

"""Long-term memory slash command: /memory (FEP-0048, FOU-A2d).

``/memory list`` (default) shows stored preferences (both scopes, with
id/scope/source/confidence), recent episodes, and the consolidation
proposals staged by the end-session hook — nothing is stored without the
explicit confirm path. ``/memory set`` captures a directive (user scope by
default, ``--project`` for project scope), ``/memory forget`` deletes by id
or content, and ``/memory confirm|discard`` drive the shared consolidator's
commit/discard path.

Storage access goes through ``victor.framework.memory`` (the client-layer
seam); imports are lazy to keep command-module import cheap (the lazy
slash-import contract in tests/unit/ui/test_lazy_slash_import.py).
"""

from __future__ import annotations

import logging

from rich.markup import escape
from rich.table import Table

from victor.ui.slash.protocol import BaseSlashCommand, CommandContext, CommandMetadata
from victor.ui.slash.registry import register_command

logger = logging.getLogger(__name__)

_USAGE = (
    "Usage: /memory [list | set <content> [--project] | " "forget <id|text> | confirm | discard]"
)

_PREFERENCE_CONTENT_WIDTH = 52
_EPISODE_SUMMARY_WIDTH = 60
_EPISODE_LIST_LIMIT = 10


def _truncate(text: str, width: int) -> str:
    return text if len(text) <= width else text[: width - 3] + "..."


@register_command
class MemoryCommand(BaseSlashCommand):
    """Inspect and manage long-term memory; confirm staged proposals."""

    @property
    def metadata(self) -> CommandMetadata:
        return CommandMetadata(
            name="memory",
            description="List, set, and forget long-term memory; confirm staged proposals",
            usage="/memory [list|set <content> [--project]|forget <id|text>|confirm|discard]",
            category="tools",
        )

    def execute(self, ctx: CommandContext) -> None:
        action = ctx.args[0].lower() if ctx.args else "list"

        if action in ("list", "show"):
            self._list(ctx)
        elif action == "set":
            self._set(ctx)
        elif action == "forget":
            self._forget(ctx)
        elif action == "confirm":
            self._confirm(ctx)
        elif action == "discard":
            self._discard(ctx)
        else:
            ctx.console.print(f"[red]Unknown /memory action:[/] {escape(action)}")
            ctx.console.print(f"[dim]{_USAGE}[/]")

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _list(self, ctx: CommandContext) -> None:
        from victor.framework.memory import (
            get_shared_memory_consolidator,
            get_shared_memory_store,
        )
        from victor.storage.memory.preference_episodic import MemoryScope

        store = get_shared_memory_store()
        preferences = store.get_preferences(MemoryScope.USER) + store.get_preferences(
            MemoryScope.PROJECT
        )
        episodes = store.recent_episodes(limit=_EPISODE_LIST_LIMIT)
        staged = get_shared_memory_consolidator().staged_proposals()

        if not preferences and not episodes and not staged:
            ctx.console.print("[yellow]No memories stored yet.[/]")
            ctx.console.print("[dim]Remember one with: /memory set <content>[/]")
            return

        if preferences:
            table = Table(title=f"Preferences ({len(preferences)})", show_header=True)
            table.add_column("ID", style="dim", no_wrap=True)
            table.add_column("Scope", style="cyan", width=7)
            table.add_column("Source", style="green", width=10)
            table.add_column("Conf", justify="right", width=5)
            table.add_column("Content")
            for pref in preferences:
                table.add_row(
                    pref.id,
                    pref.scope.value,
                    pref.source,
                    f"{pref.confidence:.1f}",
                    escape(_truncate(pref.content, _PREFERENCE_CONTENT_WIDTH * 4)),
                )
            ctx.console.print(table)

        if episodes:
            table = Table(title=f"Recent Episodes ({len(episodes)})", show_header=True)
            table.add_column("ID", style="dim", no_wrap=True)
            table.add_column("Summary")
            for episode in episodes:
                table.add_row(
                    episode.id,
                    escape(_truncate(episode.summary, _EPISODE_SUMMARY_WIDTH)),
                )
            ctx.console.print(table)

        if staged:
            table = Table(title=f"Staged Proposals ({len(staged)} pending)", show_header=True)
            table.add_column("Summary")
            for proposal in staged:
                table.add_row(escape(_truncate(proposal.summary, _EPISODE_SUMMARY_WIDTH)))
            ctx.console.print(table)
            ctx.console.print("[dim]Confirm with /memory confirm; discard with /memory discard[/]")

    def _set(self, ctx: CommandContext) -> None:
        from victor.framework.memory import get_shared_memory_store
        from victor.storage.memory.preference_episodic import (
            MemoryScope,
            MemoryWriteRefused,
        )

        args = ctx.args[1:]
        scope = MemoryScope.PROJECT if "--project" in args else MemoryScope.USER
        content = " ".join(a for a in args if not a.startswith("--")).strip()
        if not content:
            ctx.console.print(f"[yellow]{_USAGE}[/]")
            return

        try:
            preference = get_shared_memory_store().set_preference(content, scope=scope)
        except MemoryWriteRefused as exc:
            ctx.console.print(f"[red]Memory refused:[/] {escape(str(exc))}")
            return

        ctx.console.print(f"[green]Stored {preference.scope.value} preference[/] {preference.id}")
        ctx.console.print("[dim]Review with /memory list; forget with /memory forget[/]")

    def _forget(self, ctx: CommandContext) -> None:
        from victor.framework.memory import get_shared_memory_store
        from victor.storage.memory.preference_episodic import MemoryScope

        target = " ".join(ctx.args[1:]).strip()
        if not target:
            ctx.console.print(f"[yellow]{_USAGE}[/]")
            return

        store = get_shared_memory_store()
        if target.startswith("pref_"):
            deleted = store.forget(preference_id=target)
        else:
            # Content match sweeps both scopes (each query stays pinned to
            # its own database); report the combined deletions.
            deleted = sum(
                store.forget(content_match=target, scope=scope)
                for scope in (MemoryScope.PROJECT, MemoryScope.USER)
            )

        if deleted:
            ctx.console.print(f"[green]Forgot {deleted} memor{'y' if deleted == 1 else 'ies'}[/]")
        else:
            ctx.console.print("[yellow]No matching memories found[/]")

    def _confirm(self, ctx: CommandContext) -> None:
        from victor.framework.memory import get_shared_memory_consolidator

        stored = get_shared_memory_consolidator().commit_staged()
        if not stored:
            ctx.console.print("[yellow]No staged memory proposals[/]")
            return

        ctx.console.print(f"[green]Stored {len(stored)} episode(s) from staged proposals[/]")
        for episode in stored:
            ctx.console.print(f"  [dim]{episode.id}[/] {escape(episode.summary)}")

    def _discard(self, ctx: CommandContext) -> None:
        from victor.framework.memory import get_shared_memory_consolidator

        discarded = get_shared_memory_consolidator().discard_staged()
        if not discarded:
            ctx.console.print("[yellow]No staged memory proposals[/]")
            return

        ctx.console.print(f"[yellow]Discarded {discarded} staged proposal(s)[/]")
