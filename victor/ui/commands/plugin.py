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

"""Plugin management CLI commands.

Provides ``victor plugin`` subcommands for installing, managing, and
inspecting external subprocess-based plugins. Uses ExternalPluginManager
as the backend.

Usage:
    victor plugin list
    victor plugin install ./my-plugin
    victor plugin install https://github.com/user/my-plugin.git
    victor plugin enable my-plugin@external
    victor plugin disable my-plugin@external
    victor plugin uninstall my-plugin@external
    victor plugin info my-plugin@external
    victor plugin search keyword
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from victor.core.async_utils import run_sync

import typer
from rich.console import Console
from rich.table import Table

from victor.ui.json_utils import create_json_option, print_json_data

logger = logging.getLogger(__name__)

plugin_app = typer.Typer(name="plugin", help="Manage external plugins")
console = Console()


def _get_manager():
    """Create an ExternalPluginManager from current settings."""
    from victor.core.plugins.external import ExternalPluginManager

    return ExternalPluginManager()


@plugin_app.command("list")
def list_plugins(
    plugin_type: Optional[str] = typer.Option(
        None, "--type", "-t", help="Filter by type: vertical, external, plugin"
    ),
    enabled_only: bool = typer.Option(False, "--enabled", help="Show only enabled plugins"),
    json_output: bool = create_json_option(),
) -> None:
    """List all plugins (verticals, external plugins, and entry-point plugins)."""
    from victor.core.plugins.registry import PluginRegistry

    # Ensure discovery has run
    registry = PluginRegistry.get_instance()
    if not registry.is_discovered:
        registry.discover()

    entries = registry.list_all_with_type()

    if plugin_type:
        entries = [e for e in entries if e["type"] == plugin_type]
    if enabled_only:
        entries = [e for e in entries if e["enabled"]]

    if json_output:
        print_json_data({"plugins": entries, "count": len(entries)})
        return

    if not entries:
        console.print("[dim]No plugins found.[/]")
        console.print("[dim]Install with: victor plugin install <package-or-path>[/]")
        return

    table = Table(title="Plugins")
    table.add_column("Name", style="cyan")
    table.add_column("Type")
    table.add_column("Version")
    table.add_column("Status")

    for entry in entries:
        status = "[green]enabled[/]" if entry["enabled"] else "[dim]disabled[/]"
        type_style = {
            "vertical": "[blue]vertical[/]",
            "external": "[yellow]external[/]",
            "plugin": "[magenta]plugin[/]",
        }.get(entry["type"], entry["type"])
        table.add_row(entry["name"], type_style, entry["version"], status)

    console.print(table)
    console.print(f"\n[dim]{len(entries)} plugin(s) found[/]")


@plugin_app.command("install")
def install_plugin(
    source: str = typer.Argument(help="Local path or git URL to install from"),
    enable: bool = typer.Option(True, "--enable/--no-enable", help="Enable after install"),
) -> None:
    """Install a plugin from a local path or git URL."""
    from victor.core.plugins.external import ExternalPluginManager
    from victor.core.plugins.manifest import ManifestValidationError

    manager = _get_manager()

    source_type = "git_url" if source.endswith(".git") or "://" in source else "local_path"

    console.print(f"[cyan]Installing plugin from {source}...[/]")

    try:
        plugin = run_sync(manager.install_plugin(source, source_type))
    except ManifestValidationError as e:
        console.print("[red]Manifest validation failed:[/]")
        for error in e.errors:
            console.print(f"  - {error}")
        raise typer.Exit(1)
    except FileNotFoundError:
        console.print(f"[red]Source not found:[/] {source}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Install failed:[/] {e}")
        raise typer.Exit(1)

    if enable:
        manager.enable_plugin(plugin.plugin_id)
        console.print(f"[green]Installed and enabled:[/] {plugin.plugin_id}")
    else:
        console.print(f"[green]Installed (disabled):[/] {plugin.plugin_id}")

    console.print(f"  Version: {plugin.version}")
    console.print(f"  Tools: {len(plugin.manifest.tools)}")
    if plugin.manifest.tools:
        for tool in plugin.manifest.tools:
            console.print(f"    - {tool.name}: {tool.description}")


@plugin_app.command("uninstall")
def uninstall_plugin(
    plugin_id: str = typer.Argument(help="Plugin ID to uninstall (e.g., my-plugin@external)"),
    force: bool = typer.Option(False, "--force", help="Skip confirmation"),
) -> None:
    """Uninstall an external plugin."""
    manager = _get_manager()
    manager.discover_plugins()

    if plugin_id not in manager.plugins:
        console.print(f"[red]Plugin not found:[/] {plugin_id}")
        raise typer.Exit(1)

    plugin = manager.plugins[plugin_id]
    if not force:
        confirm = typer.confirm(
            f"Uninstall '{plugin.name}' ({plugin.version})? This cannot be undone."
        )
        if not confirm:
            console.print("[dim]Cancelled[/]")
            return

    if manager.uninstall_plugin(plugin_id):
        console.print(f"[green]Uninstalled:[/] {plugin_id}")
    else:
        console.print(f"[red]Failed to uninstall:[/] {plugin_id}")
        console.print("[dim]Bundled plugins can only be disabled, not uninstalled.[/]")
        raise typer.Exit(1)


@plugin_app.command("enable")
def enable_plugin(
    plugin_id: str = typer.Argument(help="Plugin ID to enable"),
) -> None:
    """Enable a disabled plugin."""
    manager = _get_manager()
    manager.discover_plugins()

    if manager.enable_plugin(plugin_id):
        console.print(f"[green]Enabled:[/] {plugin_id}")
    else:
        console.print(f"[red]Plugin not found:[/] {plugin_id}")
        raise typer.Exit(1)


@plugin_app.command("disable")
def disable_plugin(
    plugin_id: str = typer.Argument(help="Plugin ID to disable"),
) -> None:
    """Disable an enabled plugin."""
    manager = _get_manager()
    manager.discover_plugins()

    if manager.disable_plugin(plugin_id):
        console.print(f"[green]Disabled:[/] {plugin_id}")
    else:
        console.print(f"[red]Plugin not found:[/] {plugin_id}")
        raise typer.Exit(1)


@plugin_app.command("info")
def plugin_info(
    plugin_id: str = typer.Argument(help="Plugin ID to inspect"),
    json_output: bool = create_json_option(),
) -> None:
    """Show detailed information about a plugin."""
    manager = _get_manager()
    manager.discover_plugins()

    if plugin_id not in manager.plugins:
        console.print(f"[red]Plugin not found:[/] {plugin_id}")
        raise typer.Exit(1)

    plugin = manager.plugins[plugin_id]
    manifest = plugin.manifest

    if json_output:
        data = {
            "id": plugin.plugin_id,
            "name": manifest.name,
            "version": manifest.version,
            "description": manifest.description,
            "kind": plugin.kind.value,
            "enabled": plugin.enabled,
            "root_path": str(plugin.root_path),
            "permissions": manifest.permissions,
            "tools": [
                {
                    "name": t.name,
                    "description": t.description,
                    "permission": t.required_permission,
                }
                for t in manifest.tools
            ],
            "commands": [{"name": c.name, "description": c.description} for c in manifest.commands],
            "hooks": {
                "pre_tool_use": manifest.hooks.pre_tool_use,
                "post_tool_use": manifest.hooks.post_tool_use,
            },
        }
        print_json_data(data)
        return

    console.print(f"[bold]{manifest.name}[/] v{manifest.version}")
    console.print(f"  {manifest.description}")
    console.print()
    console.print(f"  ID:       {plugin.plugin_id}")
    console.print(f"  Kind:     {plugin.kind.value}")
    console.print(f"  Status:   {'[green]enabled[/]' if plugin.enabled else '[dim]disabled[/]'}")
    console.print(f"  Path:     {plugin.root_path}")

    if manifest.permissions:
        console.print(f"  Perms:    {', '.join(manifest.permissions)}")

    if manifest.tools:
        console.print(f"\n  [bold]Tools ({len(manifest.tools)}):[/]")
        for tool in manifest.tools:
            perm = f"[dim]({tool.required_permission})[/]"
            console.print(f"    - {tool.name}: {tool.description} {perm}")

    if manifest.commands:
        console.print(f"\n  [bold]Commands ({len(manifest.commands)}):[/]")
        for cmd in manifest.commands:
            console.print(f"    - /{cmd.name}: {cmd.description}")

    if manifest.hooks.pre_tool_use or manifest.hooks.post_tool_use:
        console.print("\n  [bold]Hooks:[/]")
        for h in manifest.hooks.pre_tool_use:
            console.print(f"    - PreToolUse: {h}")
        for h in manifest.hooks.post_tool_use:
            console.print(f"    - PostToolUse: {h}")


@plugin_app.command("search")
def search_plugins(
    query: str = typer.Argument(help="Search term"),
) -> None:
    """Search for plugins in the marketplace registry."""
    from victor.core.plugins.marketplace import search_marketplace

    results = search_marketplace(query)

    if not results:
        console.print(f"[dim]No plugins found matching '{query}'[/]")
        return

    table = Table(title=f"Plugin Search: '{query}'")
    table.add_column("Name", style="cyan")
    table.add_column("Description")
    table.add_column("Source")
    table.add_column("Install Command")

    for entry in results:
        table.add_row(
            entry["name"],
            entry["description"],
            entry.get("source_type", "git"),
            f"victor plugin install {entry['source']}",
        )

    console.print(table)


@plugin_app.command("init")
def init_plugin(
    name: str = typer.Argument(
        ...,
        help="Name of the new plugin (e.g., 'security', 'analytics')",
    ),
    description: str = typer.Option(
        None,
        "--description",
        "-d",
        help="Description of the plugin's purpose",
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
        help="Overwrite existing files if plugin already exists",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        "-n",
        help="Show what would be created without actually creating files",
    ),
) -> None:
    """Scaffold a new plugin from templates.

    Creates a plugin directory with a contract-first definition layer:
    - __init__.py - Package initialization and assistant export
    - assistant.py - Main plugin definition authored against victor_contracts
    - safety.py - Optional runtime-side safety notes placeholder
    - prompts.py - Optional serializable prompt metadata helper
    - mode_config.py - Optional runtime-side mode metadata placeholder
    - service_provider.py - DI container registration (optional)

    Examples:
        victor plugin init security --description "Security analysis assistant"
        victor plugin init analytics -d "Data analytics" --service-provider
        victor plugin init ml --dry-run
    """
    from victor.ui.commands.scaffold import scaffold_plugin

    scaffold_plugin(
        name=name,
        description=description,
        service_provider=service_provider,
        force=force,
        dry_run=dry_run,
        label="plugin",
    )
