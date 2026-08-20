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

"""Plugin management slash command for interactive sessions."""

from __future__ import annotations

import logging

from rich.table import Table

from victor.ui.slash.protocol import BaseSlashCommand, CommandContext, CommandMetadata
from victor.ui.slash.registry import register_command

logger = logging.getLogger(__name__)


@register_command
class PluginCommand(BaseSlashCommand):
    """Manage external plugins in interactive sessions."""

    @property
    def metadata(self) -> CommandMetadata:
        return CommandMetadata(
            name="plugin",
            description="Manage external plugins",
            usage="/plugin [list|enable|disable|info] [plugin_id]",
            aliases=["plugins"],
            category="tools",
        )

    def execute(self, ctx: CommandContext) -> None:
        action = self._get_arg(ctx, 0, "list")
        plugin_id = self._get_arg(ctx, 1, "")

        if action == "list":
            self._list_plugins(ctx)
        elif action == "enable" and plugin_id:
            self._enable_plugin(ctx, plugin_id)
        elif action == "disable" and plugin_id:
            self._disable_plugin(ctx, plugin_id)
        elif action == "info" and plugin_id:
            self._show_info(ctx, plugin_id)
        elif action == "search":
            query = " ".join(ctx.args[1:]) if len(ctx.args) > 1 else ""
            self._search_plugins(ctx, query)
        else:
            ctx.console.print(
                "[dim]Usage: /plugin [list|enable|disable|info|search] [plugin_id|query][/]"
            )

    def _list_plugins(self, ctx: CommandContext) -> None:
        from victor.core.plugins.registry import PluginRegistry

        registry = PluginRegistry.get_instance()
        if not registry.is_discovered:
            registry.discover()

        entries = registry.list_all_with_type()

        if not entries:
            ctx.console.print("[dim]No plugins found.[/]")
            ctx.console.print("[dim]Install with: victor plugin install <path>[/]")
            return

        table = Table(title="Plugins")
        table.add_column("Name", style="cyan")
        table.add_column("Type")
        table.add_column("Version")
        table.add_column("Status")

        type_styles = {
            "vertical": "[blue]vertical[/]",
            "external": "[yellow]external[/]",
            "plugin": "[magenta]plugin[/]",
        }
        for entry in entries:
            status = "[green]enabled[/]" if entry["enabled"] else "[dim]disabled[/]"
            ctx.console.print() if False else None  # placeholder
            table.add_row(
                entry["name"],
                type_styles.get(entry["type"], entry["type"]),
                entry["version"],
                status,
            )

        ctx.console.print(table)

    def _enable_plugin(self, ctx: CommandContext, plugin_id: str) -> None:
        from victor.core.plugins.external import ExternalPluginManager

        manager = ExternalPluginManager()
        manager.discover_plugins()
        if manager.enable_plugin(plugin_id):
            ctx.console.print(f"[green]Enabled:[/] {plugin_id}")
        else:
            ctx.console.print(f"[red]Plugin not found:[/] {plugin_id}")

    def _disable_plugin(self, ctx: CommandContext, plugin_id: str) -> None:
        from victor.core.plugins.external import ExternalPluginManager

        manager = ExternalPluginManager()
        manager.discover_plugins()
        if manager.disable_plugin(plugin_id):
            ctx.console.print(f"[green]Disabled:[/] {plugin_id}")
        else:
            ctx.console.print(f"[red]Plugin not found:[/] {plugin_id}")

    def _show_info(self, ctx: CommandContext, plugin_id: str) -> None:
        from victor.core.plugins.external import ExternalPluginManager

        manager = ExternalPluginManager()
        manager.discover_plugins()

        if plugin_id not in manager.plugins:
            ctx.console.print(f"[red]Plugin not found:[/] {plugin_id}")
            return

        plugin = manager.plugins[plugin_id]
        m = plugin.manifest
        ctx.console.print(f"[bold]{m.name}[/] v{m.version}")
        ctx.console.print(f"  {m.description}")
        ctx.console.print(
            f"  Status: {'[green]enabled[/]' if plugin.enabled else '[dim]disabled[/]'}"
        )
        if m.tools:
            ctx.console.print(f"  Tools: {', '.join(t.name for t in m.tools)}")

    def _search_plugins(self, ctx: CommandContext, query: str) -> None:
        from victor.core.plugins.marketplace import search_marketplace

        if not query:
            ctx.console.print("[dim]Usage: /plugin search <query>[/]")
            return

        results = search_marketplace(query)
        if not results:
            ctx.console.print(f"[dim]No plugins match '{query}'[/]")
            return

        table = Table(title=f"Search: '{query}'")
        table.add_column("Name", style="cyan")
        table.add_column("Description")
        table.add_column("Install")

        for entry in results:
            table.add_row(
                entry["name"],
                entry["description"],
                entry.get("install_cmd", f"pip install {entry['name']}"),
            )

        ctx.console.print(table)
