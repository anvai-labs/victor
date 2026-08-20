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

"""Profile management commands.

Provides commands for:
- victor profiles list - List configured profiles from ~/.victor/profiles.yaml
- victor profiles templates - List built-in profile templates
- victor profiles show <name> - Show profile details
- victor profiles apply <name> - Apply a profile
- victor profiles current - Show current profile
"""

from pathlib import Path
from typing import Any, Dict, Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax
import yaml

from victor.config.profiles import (
    ProfileLevel,
    ProfileManager,
    PROFILES,
    generate_profile_yaml,
    get_profile,
    get_recommended_profile,
    install_profile,
    list_profiles,
)
from victor.config.settings import get_project_paths

profiles_app = typer.Typer(name="profiles", help="Manage configuration profiles.")
console = Console()


# =============================================================================
# Helper functions
# =============================================================================


def _resolve_config_dir(config_dir: Optional[str]) -> Path:
    """Resolve the target config directory through centralized Victor paths."""
    return Path(config_dir) if config_dir else get_project_paths().global_victor_dir


def _get_profile_manager(config_dir: Optional[str]) -> ProfileManager:
    """Get a ProfileManager for the given config directory."""
    config_path = _resolve_config_dir(config_dir)
    return ProfileManager.for_config_dir(config_path)


@profiles_app.command("list")
def profile_list(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed profile settings"),
    config_dir: Optional[str] = typer.Option(None, "--config-dir", "-d", help="Config directory"),
) -> None:
    """List configured profiles from ~/.victor/profiles.yaml."""
    mgr = _get_profile_manager(config_dir)
    data = mgr.load_profiles()
    profiles = data.get("profiles", {}) or {}
    config_path = _resolve_config_dir(config_dir)
    profiles_path = config_path / "profiles.yaml"
    default_profile = data.get("default_profile") or "default"

    console.print("\n[bold]Configured Profiles[/]")
    console.print(f"[dim]Source: {profiles_path}[/]")
    console.print("═" * 60)

    if not profiles:
        console.print("\n[yellow]No configured profiles found.[/]")
        console.print(
            "Create one with: [bold]victor profiles create <name> --provider <provider> --model <model>[/]"
        )
        console.print("Apply a template with: [bold]victor profiles apply basic[/]")
        console.print("View profile templates with: [bold]victor profiles templates[/]")
        return

    table = Table(show_header=True)
    table.add_column("Profile", style="cyan", no_wrap=True)
    table.add_column("Provider", style="green")
    table.add_column("Model", style="yellow")
    table.add_column("Temperature", justify="right")
    table.add_column("Max Tokens", justify="right")
    table.add_column("Account/Auth", style="magenta")

    for name, profile_data in sorted(profiles.items()):
        if not isinstance(profile_data, dict):
            profile_data = {}
        display_name = f"{name} ★" if name == default_profile else name
        provider = str(profile_data.get("provider", "—"))
        model = str(profile_data.get("model", "—"))
        temperature = _format_optional_value(profile_data.get("temperature"))
        max_tokens = _format_optional_value(
            profile_data.get("max_tokens", profile_data.get("max_completion_tokens"))
        )
        account_auth = _format_profile_auth(profile_data)

        table.add_row(
            display_name,
            provider,
            model,
            temperature,
            max_tokens,
            account_auth,
        )

    console.print(table)
    console.print(f"\n[dim]Total: {len(profiles)} configured profile(s). ★ = default[/]")
    console.print("[dim]Built-in templates: victor profiles templates[/]")

    if verbose:
        console.print("\n[bold]Raw Profile Settings:[/]")
        console.print("─" * 60)
        for name, profile_data in sorted(profiles.items()):
            console.print(f"\n[cyan bold]{name}[/]")
            syntax = Syntax(
                yaml.safe_dump(profile_data, default_flow_style=False, sort_keys=False).strip(),
                "yaml",
                theme="monokai",
            )
            console.print(syntax)


@profiles_app.command("templates")
def profile_templates(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed template settings"),
) -> None:
    """List built-in profile templates that can be applied to profiles.yaml."""
    _list_profile_templates_impl(verbose=verbose)


def _list_profile_templates_impl(*, verbose: bool) -> None:
    """List built-in profile templates."""
    all_profiles = list_profiles()

    console.print("\n[bold]Built-in Profile Templates[/]")
    console.print("═" * 60)

    table = Table(show_header=True, show_lines=True)
    table.add_column("Template", style="cyan")
    table.add_column("Level", style="magenta")
    table.add_column("Default Provider", style="green")
    table.add_column("Default Model", style="yellow")
    table.add_column("Description", style="white")

    for profile in sorted(all_profiles, key=lambda p: p.level.value):
        level_display = {
            ProfileLevel.BASIC: "[green]Basic[/]",
            ProfileLevel.ADVANCED: "[yellow]Advanced[/]",
            ProfileLevel.EXPERT: "[red]Expert[/]",
        }.get(profile.level, profile.level.value)

        table.add_row(
            f"{profile.display_name} ([dim]{profile.name}[/])",
            level_display,
            str(profile.settings.get("default_provider", "auto")),
            str(profile.settings.get("default_model", "auto")),
            profile.description,
        )

    console.print(table)

    if verbose:
        console.print("\n[bold]Detailed Settings:[/]")
        console.print("─" * 60)

        for profile in sorted(all_profiles, key=lambda p: p.level.value):
            console.print(f"\n[cyan bold]{profile.display_name} ([dim]{profile.name}[/])[/]")
            console.print(f"[dim]{profile.description}[/]")

            settings_table = Table(show_header=False, box=None)
            settings_table.add_column("Setting", style="yellow")
            settings_table.add_column("Value", style="white")

            for key, value in sorted(profile.settings.items()):
                if isinstance(value, dict):
                    value_str = f"<{len(value)} settings>"
                else:
                    value_str = str(value)
                settings_table.add_row(key, value_str)

            console.print(settings_table)

    # Show recommended
    recommended = get_recommended_profile()
    from victor.ui.emoji import get_icon

    console.print(
        f"\n{get_icon('info')} [yellow]Recommended for you:[/] {recommended.display_name}"
    )
    console.print(f"   Use: [bold]victor profiles apply {recommended.name}[/]")


def _format_optional_value(value: Any) -> str:
    if value is None:
        return "[dim]—[/]"
    return str(value)


def _format_profile_auth(profile_data: Dict[str, Any]) -> str:
    account = profile_data.get("account")
    if account:
        return f"account:{account}"

    auth = profile_data.get("auth")
    if isinstance(auth, dict):
        method = auth.get("method", "api_key")
        source = auth.get("source")
        return f"{method}/{source}" if source else str(method)
    if isinstance(auth, str):
        return auth

    auth_mode = profile_data.get("auth_mode")
    if auth_mode:
        return str(auth_mode)

    return "[dim]—[/]"


@profiles_app.command("show")
def profile_show(
    name: str = typer.Argument(
        ..., help="Profile name (basic, advanced, expert, coding, research)"
    ),
    export_yaml: bool = typer.Option(False, "--yaml", "-y", help="Export as YAML"),
) -> None:
    """Show details for a specific profile."""
    profile = get_profile(name)

    if not profile:
        from victor.ui.emoji import get_icon

        console.print(f"{get_icon('error')} Profile '{name}' not found")
        console.print("\nAvailable profiles: " + ", ".join(PROFILES.keys()))
        raise typer.Exit(1)

    console.print(f"\n[bold cyan]{profile.display_name} Profile[/]")
    console.print(f"[dim]ID: {profile.name} | Level: {profile.level.value.upper()}[/]")
    console.print(f"\n{profile.description}")

    if export_yaml:
        yaml_content = generate_profile_yaml(profile)
        console.print("\n[bold]Generated YAML:[/]")
        syntax = Syntax(yaml_content, "yaml", theme="monokai", line_numbers=True)
        console.print(syntax)
    else:
        # Show settings in a table
        console.print("\n[bold]Settings:[/]")
        table = Table(show_header=False, box=None)
        table.add_column("Setting", style="yellow", width=30)
        table.add_column("Value", style="white")

        for key, value in sorted(profile.settings.items()):
            if isinstance(value, dict):
                value_str = f"<{len(value)} nested settings>"
            else:
                value_str = str(value)
            table.add_row(key, value_str)

        console.print(table)

        # Show provider settings
        if profile.provider_settings:
            console.print("\n[bold]Provider Settings:[/]")
            for provider, settings in profile.provider_settings.items():
                console.print(f"  [cyan]{provider}[/]:")
                for key, value in settings.items():
                    console.print(f"    {key}: {value}")


@profiles_app.command("apply")
def profile_apply(
    name: str = typer.Argument(..., help="Profile name to apply"),
    provider: Optional[str] = typer.Option(None, "--provider", "-p", help="Override provider"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Override model"),
    config_dir: Optional[str] = typer.Option(None, "--config-dir", "-d", help="Config directory"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show changes without applying"),
) -> None:
    """Apply a profile to your configuration.

    This will create or update ~/.victor/profiles.yaml with the profile settings.
    """
    profile = get_profile(name)

    if not profile:
        from victor.ui.emoji import get_icon

        console.print(f"{get_icon('error')} Profile '{name}' not found")
        console.print("\nAvailable profiles: " + ", ".join(PROFILES.keys()))
        raise typer.Exit(1)

    # Resolve config directory
    config_path = _resolve_config_dir(config_dir)

    # Generate YAML content
    yaml_content = generate_profile_yaml(profile, provider_override=provider, model_override=model)

    if dry_run:
        console.print(f"\n[bold]Dry Run: {profile.display_name} Profile[/]")
        console.print(f"[dim]Config dir: {config_path}[/]")
        console.print("\n[bold]Generated profiles.yaml:[/]")
        syntax = Syntax(yaml_content, "yaml", theme="monokai", line_numbers=True)
        console.print(syntax)
        console.print("\n[yellow]Run without --dry-run to apply this profile.[/]")
        return

    # Apply the profile
    try:
        profiles_path = install_profile(
            profile,
            config_dir=config_path,
            provider_override=provider,
            model_override=model,
        )

        from victor.ui.emoji import get_icon

        console.print(
            f"\n{get_icon('success')} Applied [bold cyan]{profile.display_name}[/] profile"
        )
        console.print(f"[dim]Config written to: {profiles_path}[/]")

        # Show summary
        console.print("\n[bold]Settings Applied:[/]")
        table = Table(show_header=False, box=None)
        table.add_column("", style="yellow")
        table.add_column("", style="white")

        # Show key settings
        table.add_row("Profile Level", profile.level.value.upper())
        table.add_row("Provider", provider or profile.settings.get("default_provider", "ollama"))
        table.add_row("Model", model or profile.settings.get("default_model", "auto"))
        table.add_row("Max Tools", str(profile.settings.get("fallback_max_tools", 10)))

        optimizations = []
        if profile.settings.get("framework_preload_enabled"):
            optimizations.append("Preloading")
        if profile.settings.get("http_connection_pool_enabled"):
            optimizations.append("HTTP Pooling")
        if profile.settings.get("tool_selection_cache_enabled"):
            optimizations.append("Tool Cache")

        table.add_row("Optimizations", ", ".join(optimizations))

        console.print(table)

        console.print(f"\n{get_icon('info')} Next steps:[/]")
        console.print("  1. [bold]victor doctor[/] - Verify your configuration")
        console.print("  2. [bold]victor chat[/] - Start using Victor")
        console.print("\n[dim]To change profiles, run: victor profiles apply <name>[/]")

    except Exception as e:
        console.print(f"\n[red]✗[/] Failed to apply profile: {e}")
        raise typer.Exit(1)


@profiles_app.command("current")
def profile_current(
    config_dir: Optional[str] = typer.Option(None, "--config-dir", "-d", help="Config directory"),
) -> None:
    """Show the current active profile."""
    mgr = _get_profile_manager(config_dir)
    profile_name = mgr.get_current_profile_name()
    config_path = _resolve_config_dir(config_dir)

    if not profile_name:
        console.print("\n[yellow]⚠[/] No profile detected")
        console.print(f"[dim]Config directory: {config_path}[/]")
        console.print("\n[yellow]💡 To get started:[/]")
        console.print("  [bold]victor profiles apply basic[/] - Apply basic profile")
        console.print("  [bold]victor profiles list[/] - List all profiles")
        return

    profile = get_profile(profile_name)
    if profile:
        console.print("\n[bold]Current Profile:[/]")
        console.print(f"  Name: [cyan]{profile.display_name}[/] ([dim]{profile.name}[/])")
        console.print(f"  Level: [yellow]{profile.level.value.upper()}[/]")
        console.print(f"  Description: {profile.description}")
        console.print(f"\n[dim]Config directory: {config_path}[/]")
    else:
        console.print(f"\n[dim]Current profile: {profile_name}[/] (custom configuration)")


@profiles_app.command("create")
def profile_create(
    name: str = typer.Argument(..., help="Profile name to create"),
    provider: str = typer.Option("ollama", "--provider", "-p", help="LLM provider"),
    model: str = typer.Option("llama2", "--model", "-m", help="Model name"),
    account: Optional[str] = typer.Option(None, "--account", "-a", help="Provider account name"),
    auth_method: Optional[str] = typer.Option(None, "--auth-method", help="Auth method"),
    auth_source: Optional[str] = typer.Option(None, "--auth-source", help="Auth source"),
    temperature: Optional[float] = typer.Option(None, "--temperature", "-t", help="Temperature"),
    max_tokens: Optional[int] = typer.Option(None, "--max-tokens", help="Max tokens"),
    description: Optional[str] = typer.Option(None, "--description", "-d", help="Description"),
    set_default: bool = typer.Option(False, "--default", help="Set this profile as default"),
    config_dir: Optional[str] = typer.Option(None, "--config-dir", help="Config directory"),
) -> None:
    """Create a new custom profile."""
    mgr = _get_profile_manager(config_dir)
    data = mgr.load_profiles()
    profiles = data.get("profiles", {})

    if name in profiles:
        console.print(f"[red]✗[/] Profile '{name}' already exists")
        console.print("Use [bold]victor profiles edit[/] to modify it.")
        return

    profile_data: Dict[str, Any] = {"provider": provider, "model": model}
    if account is not None:
        profile_data["account"] = account
    if auth_method is not None or auth_source is not None:
        profile_data["auth"] = {
            "method": auth_method or "api_key",
            "source": auth_source or "keyring",
        }
    if temperature is not None:
        profile_data["temperature"] = temperature
    if max_tokens is not None:
        profile_data["max_tokens"] = max_tokens
    if description is not None:
        profile_data["description"] = description

    profiles[name] = profile_data
    data["profiles"] = profiles
    if set_default:
        data["default_profile"] = name
    try:
        mgr.save_profiles(data)
    except IOError as e:
        console.print(f"[red]Error writing profiles file: {e}[/]")
        raise typer.Exit(1)

    from victor.ui.emoji import get_icon

    console.print(f"\n{get_icon('success')} Created profile '{name}'")
    if set_default:
        console.print("[dim]Default profile: yes[/]")


@profiles_app.command("edit")
def profile_edit(
    name: str = typer.Argument(..., help="Profile name to edit"),
    provider: Optional[str] = typer.Option(None, "--provider", "-p", help="LLM provider"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Model name"),
    temperature: Optional[float] = typer.Option(None, "--temperature", "-t", help="Temperature"),
    max_tokens: Optional[int] = typer.Option(None, "--max-tokens", help="Max tokens"),
    description: Optional[str] = typer.Option(None, "--description", "-d", help="Description"),
    config_dir: Optional[str] = typer.Option(None, "--config-dir", help="Config directory"),
) -> None:
    """Edit an existing custom profile."""
    mgr = _get_profile_manager(config_dir)
    data = mgr.load_profiles()
    profiles = data.get("profiles", {})

    if name not in profiles:
        from victor.ui.emoji import get_icon

        console.print(f"{get_icon('error')} Profile '{name}' not found")
        return

    updates: Dict[str, Any] = {}
    if provider is not None:
        updates["provider"] = provider
    if model is not None:
        updates["model"] = model
    if temperature is not None:
        updates["temperature"] = temperature
    if max_tokens is not None:
        updates["max_tokens"] = max_tokens
    if description is not None:
        updates["description"] = description

    if not updates:
        console.print("[yellow]No changes specified[/]")
        return

    profiles[name].update(updates)
    data["profiles"] = profiles
    try:
        mgr.save_profiles(data)
    except IOError as e:
        console.print(f"[red]Error writing profiles file: {e}[/]")
        raise typer.Exit(1)

    console.print(f"[green]✓[/] Updated profile '{name}'")


@profiles_app.command("delete")
def profile_delete(
    name: str = typer.Argument(..., help="Profile name to delete"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
    config_dir: Optional[str] = typer.Option(None, "--config-dir", help="Config directory"),
) -> None:
    """Delete a custom profile."""
    mgr = _get_profile_manager(config_dir)
    data = mgr.load_profiles()
    profiles = data.get("profiles", {})

    if name not in profiles:
        from victor.ui.emoji import get_icon

        console.print(f"{get_icon('error')} Profile '{name}' not found")
        return

    if not force:
        confirm = typer.confirm(f"Delete profile '{name}'?")
        if not confirm:
            console.print("[yellow]Cancelled[/]")
            return

    del profiles[name]
    data["profiles"] = profiles
    try:
        mgr.save_profiles(data)
    except IOError as e:
        console.print(f"[red]Error writing profiles file: {e}[/]")
        raise typer.Exit(1)

    console.print(f"[green]✓[/] Deleted profile '{name}'")


@profiles_app.command("set-default")
def profile_set_default(
    name: str = typer.Argument(..., help="Profile name to set as default"),
    config_dir: Optional[str] = typer.Option(None, "--config-dir", help="Config directory"),
) -> None:
    """Set a profile as the default."""
    mgr = _get_profile_manager(config_dir)
    data = mgr.load_profiles()
    profiles = data.get("profiles", {})

    if name not in profiles:
        from victor.ui.emoji import get_icon

        console.print(f"{get_icon('error')} Profile '{name}' not found")
        return

    current_default = data.get("default_profile")
    if current_default == name:
        console.print(f"[yellow]'{name}' is already the default[/]")
        return

    data["default_profile"] = name
    try:
        mgr.save_profiles(data)
    except IOError as e:
        console.print(f"[red]Error writing profiles file: {e}[/]")
        raise typer.Exit(1)

    console.print(f"[green]✓[/] Set '{name}' as the default profile")


# Add profiles_app to the main CLI
# This is imported by cli.py
