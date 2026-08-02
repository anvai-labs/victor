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

"""Entity memory slash commands.

Provides commands for querying and managing entities extracted from
conversations and code.

Commands:
    /entities list [type]     - List entities (optionally by type)
    /entities search <query>  - Search for entities
    /entities show <name>     - Show entity details
    /entities related <name>  - Show related entities
    /entities stats           - Show entity memory statistics
    /entities clear           - Clear session entities
"""

from __future__ import annotations

import logging

from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree

from victor.ui.slash.protocol import BaseSlashCommand, CommandContext, CommandMetadata
from victor.ui.slash.registry import register_command

logger = logging.getLogger(__name__)


@register_command
class EntitiesCommand(BaseSlashCommand):
    """Entity memory management commands."""

    @property
    def metadata(self) -> CommandMetadata:
        return CommandMetadata(
            name="entities",
            description="Query and manage extracted entities",
            usage="/entities <list|search|show|related|stats|clear> [args]",
            category="entities",
            requires_agent=False,
        )

    def execute(self, ctx: CommandContext) -> None:
        """Execute entity command."""
        if not ctx.args:
            self._show_help(ctx)
            return

        subcommand = ctx.args[0].lower()
        subargs = ctx.args[1:]

        handlers = {
            "list": self._list_entities,
            "search": self._search_entities,
            "show": self._show_entity,
            "related": self._show_related,
            "stats": self._show_stats,
            "clear": self._clear_entities,
            "help": self._show_help,
        }

        handler = handlers.get(subcommand)
        if not handler:
            ctx.console.print(f"[yellow]Unknown subcommand:[/] {subcommand}")
            self._show_help(ctx)
            return

        handler(ctx, subargs)

    def _show_help(self, ctx: CommandContext, args: list = None) -> None:
        """Show entity command help."""
        help_text = """
[bold]Entity Memory Commands[/]

[cyan]/entities list[/] [type]
  List entities, optionally filtered by type
  Example: /entities list class

[cyan]/entities search[/] <query>
  Search entities by name
  Example: /entities search auth

[cyan]/entities show[/] <name>
  Show detailed entity info
  Example: /entities show UserAuth

[cyan]/entities related[/] <name>
  Show entities related to the named entity
  Example: /entities related UserAuth

[cyan]/entities stats[/]
  Show entity memory statistics

[cyan]/entities clear[/]
  Clear current session entities

[bold]Entity Types:[/]
  class, function, file, module, variable
  person, organization, team
  technology, concept, pattern
  project, repository, package
  service, endpoint, database
"""
        ctx.console.print(Panel(help_text, title="Entity Commands"))

    def _get_entity_memory(self, ctx: CommandContext):
        """Get entity memory from context or create one."""
        # Try to get from agent if available
        if ctx.agent:
            # Check if agent has entity memory
            if hasattr(ctx.agent, "entity_memory"):
                return ctx.agent.entity_memory

        return None

    def _list_entities(self, ctx: CommandContext, args: list) -> None:
        """List entities in memory."""
        try:
            from victor.storage.memory import EntityMemory, EntityType

            memory = self._get_entity_memory(ctx)
            if not memory:
                ctx.console.print("[yellow]No entity memory available.[/]")
                ctx.console.print(
                    "Entity memory is populated during conversations with code analysis."
                )
                return

            # Filter by type if specified
            _type_filter = None  # noqa: F841 - Will be used when async context is available
            if args:
                try:
                    _type_filter = EntityType(args[0].lower())
                except ValueError:
                    ctx.console.print(f"[red]Unknown entity type:[/] {args[0]}")
                    return

            # Get entities (this would need async support in a real implementation)
            ctx.console.print("[yellow]Entity listing requires async context.[/]")
            ctx.console.print("Use '/entities stats' to see memory overview.")

        except ImportError:
            ctx.console.print("[red]Entity memory module not available.[/]")
        except Exception as e:
            logger.error(f"Error listing entities: {e}")
            ctx.console.print(f"[red]Error:[/] {e}")

    def _search_entities(self, ctx: CommandContext, args: list) -> None:
        """Search for entities by name."""
        if not args:
            ctx.console.print("[yellow]Usage:[/] /entities search <query>")
            return

        query = " ".join(args)
        ctx.console.print(f"[cyan]Searching for:[/] {query}")

        memory = self._get_entity_memory(ctx)
        if not memory:
            ctx.console.print("[yellow]No entity memory available.[/]")
            return

        ctx.console.print("[yellow]Entity search requires async context.[/]")

    def _show_entity(self, ctx: CommandContext, args: list) -> None:
        """Show detailed entity information."""
        if not args:
            ctx.console.print("[yellow]Usage:[/] /entities show <name>")
            return

        name = " ".join(args)
        ctx.console.print(f"[cyan]Entity:[/] {name}")

        memory = self._get_entity_memory(ctx)
        if not memory:
            ctx.console.print("[yellow]No entity memory available.[/]")
            return

        ctx.console.print("[yellow]Entity details require async context.[/]")

    def _show_related(self, ctx: CommandContext, args: list) -> None:
        """Show entities related to the named entity."""
        if not args:
            ctx.console.print("[yellow]Usage:[/] /entities related <name>")
            return

        name = " ".join(args)
        ctx.console.print(f"[cyan]Related to:[/] {name}")

        memory = self._get_entity_memory(ctx)
        if not memory:
            ctx.console.print("[yellow]No entity memory available.[/]")
            return

        ctx.console.print("[yellow]Entity relations require async context.[/]")

    def _show_stats(self, ctx: CommandContext, args: list = None) -> None:
        """Show entity memory statistics."""
        memory = self._get_entity_memory(ctx)

        if not memory:
            table = Table(title="Entity Memory Statistics")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")

            table.add_row("Status", "Not initialized")
            table.add_row("Short-term entities", "0")
            table.add_row("Working memory", "0")
            table.add_row("Relations tracked", "0")

            ctx.console.print(table)
            ctx.console.print("\n[dim]Entity memory populates during code analysis.[/]")
            return

        ctx.console.print("[yellow]Stats require async context.[/]")

    def _clear_entities(self, ctx: CommandContext, args: list = None) -> None:
        """Clear session entities."""
        memory = self._get_entity_memory(ctx)

        if not memory:
            ctx.console.print("[yellow]No entity memory to clear.[/]")
            return

        ctx.console.print("[yellow]Clearing entities requires async context.[/]")
        ctx.console.print("Entity memory will be cleared on session end.")


# Export for registration
entities_command = EntitiesCommand()
