import typer
from rich.console import Console
from rich.table import Table
from victor.ui.console import get_console
import yaml
from pathlib import Path
from typing import Any

from victor.core.async_utils import run_sync
from victor.config.settings import load_settings, get_project_paths
from victor.config.validation import validate_configuration, format_validation_result

config_app = typer.Typer(name="config", help="Validate configuration files and profiles.")
console = get_console()


def _get_config_dir() -> Path:
    """Resolve the global Victor config directory through centralized paths."""
    config_dir = get_project_paths().global_victor_dir
    return Path(config_dir)


def _validation_passed(result: Any) -> bool:
    """Return whether a validation result represents success.

    Supports both legacy property-style `is_valid` values and the newer
    callable `is_valid()` API exposed by some validators.
    """
    validator = getattr(result, "is_valid", None)
    if callable(validator):
        return bool(validator())
    return bool(validator)


@config_app.command("show")
def config_show(
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show detailed configuration including all sources",
    ),
) -> None:
    """Show effective configuration with source annotations.

    Displays the currently active configuration, showing:
    - Default provider and model
    - Active profile (if any)
    - Configuration sources (files, env vars)
    - Tool settings and limits
    """
    from rich.table import Table
    from rich.panel import Panel

    config_dir = _get_config_dir()
    settings_file = config_dir / "settings.yaml"
    profiles_file = config_dir / "profiles.yaml"
    api_keys_file = config_dir / "api_keys.yaml"

    try:
        settings = load_settings()
    except Exception as e:
        console.print(f"[red]✗ Failed to load configuration:[/] {e}")
        raise typer.Exit(1)

    console.print("\n[bold cyan]Victor Configuration[/]\n")

    # Provider and Model
    provider_table = Table(show_header=False, box=None, padding=(0, 2))
    provider_table.add_column("Setting", style="cyan")
    provider_table.add_column("Value", style="white")
    provider_table.add_column("Source", style="dim")

    provider_table.add_row(
        "Default Provider",
        settings.provider.default_provider,
        "settings.yaml" if settings_file.exists() else "default",
    )
    provider_table.add_row(
        "Default Model",
        settings.provider.default_model,
        "profiles.yaml" if profiles_file.exists() else "default",
    )
    provider_table.add_row("Temperature", str(settings.provider.default_temperature), "default")
    provider_table.add_row("Max Tokens", str(settings.provider.default_max_tokens), "default")

    console.print("[bold]Provider Configuration[/]")
    console.print(provider_table)
    console.print()

    # Tool Settings
    tools_table = Table(show_header=False, box=None, padding=(0, 2))
    tools_table.add_column("Setting", style="cyan")
    tools_table.add_column("Value", style="white")
    tools_table.add_column("Source", style="dim")

    tools_table.add_row(
        "Tool Budget",
        str(settings.tools.fallback_max_tools),
        "profiles.yaml" if profiles_file.exists() else "default",
    )
    tools_table.add_row(
        "Cache Enabled",
        "Yes" if settings.tools.tool_selection_cache_enabled else "No",
        "default",
    )
    tools_table.add_row(
        "Deduplication",
        "Yes" if settings.tools.enable_tool_deduplication else "No",
        "default",
    )

    console.print("[bold]Tool Settings[/]")
    console.print(tools_table)
    console.print()

    # Configuration Files
    files_table = Table(show_header=False, box=None, padding=(0, 2))
    files_table.add_column("File", style="cyan")
    files_table.add_column("Status", style="white")

    files_table.add_row(
        "profiles.yaml",
        "[green]Exists[/]" if profiles_file.exists() else "[dim]Not found[/]",
    )
    files_table.add_row(
        "settings.yaml",
        "[green]Exists[/]" if settings_file.exists() else "[dim]Not found[/]",
    )
    files_table.add_row(
        "api_keys.yaml",
        "[yellow]Deprecated[/]" if api_keys_file.exists() else "[dim]Not found[/]",
    )

    console.print("[bold]Configuration Files[/]")
    console.print(files_table)
    console.print(f"[dim]Config directory: {config_dir}[/]")
    console.print()

    # Environment Variables
    import os

    env_vars = {
        "VICTOR_LOG_LEVEL": os.getenv("VICTOR_LOG_LEVEL"),
        "ANTHROPIC_API_KEY": ("[red]***SET***[/]" if os.getenv("ANTHROPIC_API_KEY") else None),
        "OPENAI_API_KEY": "[red]***SET***[/]" if os.getenv("OPENAI_API_KEY") else None,
        "GOOGLE_API_KEY": "[red]***SET***[/]" if os.getenv("GOOGLE_API_KEY") else None,
    }

    # Filter to only set env vars
    set_env_vars = {k: v for k, v in env_vars.items() if v is not None}

    if set_env_vars or verbose:
        env_table = Table(show_header=False, box=None, padding=(0, 2))
        env_table.add_column("Variable", style="cyan")
        env_table.add_column("Value", style="white")

        for var, val in set_env_vars.items():
            env_table.add_row(var, val)

        console.print("[bold]Environment Variables[/]")
        if set_env_vars:
            console.print(env_table)
        else:
            console.print("[dim]No relevant environment variables set[/]")
        console.print()

    # Help text
    console.print("[dim]Configuration precedence:[/]")
    console.print("[dim]1. Command-line flags (highest priority)[/]")
    console.print("[dim]2. Environment variables[/]")
    console.print("[dim]3. profiles.yaml (active profile)[/]")
    console.print("[dim]4. settings.yaml (global settings)[/]")
    console.print("[dim]5. Default values (lowest priority)[/]")
    console.print()

    console.print("[dim]Edit configuration:[/]")
    console.print("[dim]  victor profiles list[/]  - List available profiles")
    console.print("[dim]  victor profiles set-default <name>[/]  - Change default profile")
    console.print("[dim]  Edit ~/.victor/profiles.yaml[/]  - Manual configuration")
    console.print()


@config_app.command("validate")
def config_validate(
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show detailed validation results",
    ),
    check_connectivity: bool = typer.Option(
        False,
        "--check-connectivity",
        "-c",
        help="Test provider connectivity (requires network/services)",
    ),
    fix: bool = typer.Option(
        False,
        "--fix",
        "-f",
        help="Automatically fix common configuration issues",
    ),
) -> None:
    """Validate configuration files and profiles.

    Performs comprehensive validation including:
    - Configuration file syntax and structure
    - Provider availability and API keys
    - Tool configuration
    - Performance optimization settings
    - Environment variables
    """
    console = get_console()

    # First, run the comprehensive validation
    try:
        settings = load_settings()
        result = validate_configuration(settings)

        # Display validation results
        print(format_validation_result(result))

        if not _validation_passed(result):
            raise typer.Exit(1)

    except Exception as e:
        console.print(f"[red]✗ Configuration validation failed:[/] {e}")
        raise typer.Exit(1)

    # If verbose or --fix, run additional detailed checks
    if verbose or fix:
        errors: list[str] = []
        warnings: list[str] = []
        checks_passed = 0
        checks_total = 0

        config_dir = settings.get_config_dir()
        profiles_file = config_dir / "profiles.yaml"

        # Check 1: Config directory exists
        checks_total += 1
        if config_dir.exists():
            checks_passed += 1
            if verbose:
                console.print(f"[green]✓[/] Config directory exists: {config_dir}")
        else:
            errors.append(f"Config directory not found: {config_dir}")
            console.print(f"[red]✗[/] Config directory not found: {config_dir}")
            console.print("\nRun [bold]victor init[/] to create configuration")
            raise typer.Exit(1)

        # Check 2: Profiles file exists
        checks_total += 1
        if profiles_file.exists():
            checks_passed += 1
            if verbose:
                console.print(f"[green]✓[/] Profiles file exists: {profiles_file}")
        else:
            errors.append(f"Profiles file not found: {profiles_file}")
            console.print(f"[red]✗[/] Profiles file not found: {profiles_file}")
            console.print("\nRun [bold]victor init[/] to create configuration")
            raise typer.Exit(1)

        # Check 3: Valid YAML syntax
        checks_total += 1
        try:
            with open(profiles_file, "r") as f:
                raw_config = yaml.safe_load(f)
            checks_passed += 1
            if verbose:
                console.print("[green]✓[/] YAML syntax is valid")
        except yaml.YAMLError as e:
            errors.append(f"Invalid YAML syntax: {e}")
            console.print(f"[red]✗[/] Invalid YAML syntax: {e}")
            raise typer.Exit(1)

        # Check 4: Profiles section exists
        checks_total += 1
        if raw_config and "profiles" in raw_config:
            checks_passed += 1
            if verbose:
                console.print("[green]✓[/] 'profiles' section found")
        else:
            errors.append("Missing 'profiles' section in configuration")
            console.print("[red]✗[/] Missing 'profiles' section in configuration")
            raise typer.Exit(1)

        # Load and validate profiles
        try:
            profiles = settings.load_profiles()
        except Exception as e:
            errors.append(f"Failed to load profiles: {e}")
            console.print(f"[red]✗[/] Failed to load profiles: {e}")
            raise typer.Exit(1)

        # Check 5: At least one profile exists
        checks_total += 1
        if profiles:
            checks_passed += 1
            if verbose:
                console.print(f"[green]✓[/] Found {len(profiles)} profile(s)")
        else:
            warnings.append("No profiles defined")
            console.print("[yellow]⚠[/] No profiles defined")

        # Validate each profile
        from victor.providers.registry import ProviderRegistry

        available_providers = ProviderRegistry.list_providers()

        if verbose:
            console.print("\n[bold]Profile Validation:[/]")

        for name, profile in profiles.items():
            # Check provider is valid
            checks_total += 1
            if profile.provider in available_providers:
                checks_passed += 1
                if verbose:
                    console.print(f"  [green]✓[/] [{name}] Provider '{profile.provider}' is valid")
            else:
                errors.append(f"[{name}] Unknown provider: {profile.provider}")
                console.print(f"  [red]✗[/] [{name}] Unknown provider: {profile.provider}")

            # Check model is specified
            checks_total += 1
            if profile.model:
                checks_passed += 1
                if verbose:
                    console.print(f"  [green]✓[/] [{name}] Model specified: {profile.model}")
            else:
                errors.append(f"[{name}] No model specified")
                console.print(f"  [red]✗[/] [{name}] No model specified")

            # Check temperature range
            checks_total += 1
            if 0.0 <= profile.temperature <= 2.0:
                checks_passed += 1
                if verbose:
                    console.print(
                        f"  [green]✓[/] [{name}] Temperature {profile.temperature} is valid"
                    )
            else:
                errors.append(f"[{name}] Temperature {profile.temperature} out of range [0.0, 2.0]")
                console.print(f"  [red]✗[/] [{name}] Temperature out of range")

            # Check max_tokens
            checks_total += 1
            if profile.max_tokens > 0:
                checks_passed += 1
                if verbose:
                    console.print(
                        f"  [green]✓[/] [{name}] Max tokens {profile.max_tokens} is valid"
                    )
            else:
                errors.append(f"[{name}] Invalid max_tokens: {profile.max_tokens}")
                console.print(f"  [red]✗[/] [{name}] Invalid max_tokens")

            # Check API keys for cloud providers
            if profile.provider in ["anthropic", "openai", "google", "xai", "grok"]:
                checks_total += 1
                provider_settings = settings.get_provider_settings(profile.provider)
                api_key = provider_settings.get("api_key")
                if api_key:
                    checks_passed += 1
                    if verbose:
                        console.print(
                            f"  [green]✓[/] [{name}] API key configured for {profile.provider}"
                        )
                else:
                    warnings.append(f"[{name}] No API key for {profile.provider}")
                    console.print(
                        f"  [yellow]⚠[/] [{name}] No API key configured for {profile.provider}"
                    )

        # Optional connectivity checks
        if check_connectivity:
            console.print("\n[bold]Connectivity Checks:[/]")
            run_sync(_check_connectivity(settings, profiles, verbose))

        # Summary
        console.print("\n" + "─" * 50)
        if errors:
            console.print(f"[red]Validation failed with {len(errors)} error(s)[/]")
            for err in errors:
                console.print(f"  [red]•[/] {err}")
            raise typer.Exit(1)
        elif warnings:
            console.print(f"[yellow]Validation passed with {len(warnings)} warning(s)[/]")
            for warn in warnings:
                console.print(f"  [yellow]•[/] {warn}")
            console.print(f"\n[green]✓[/] {checks_passed}/{checks_total} checks passed")
        else:
            console.print(
                f"[green]✓ Validation passed![/] {checks_passed}/{checks_total} checks passed"
            )


async def _check_connectivity(settings: Any, profiles: dict, verbose: bool) -> None:
    """Check provider connectivity for profiles."""
    checked_providers: set[str] = set()

    for _name, profile in profiles.items():
        provider = profile.provider
        if provider in checked_providers:
            continue
        checked_providers.add(provider)

        if provider == "ollama":
            from victor.providers.ollama_provider import OllamaProvider

            provider_settings = settings.get_provider_settings(provider)
            try:
                ollama = OllamaProvider(**provider_settings)
                models = await ollama.list_models()
                if models:
                    console.print(f"  [green]✓[/] Ollama: Connected ({len(models)} models)")
                else:
                    console.print("  [yellow]⚠[/] Ollama: Connected but no models installed")
                await ollama.close()
            except Exception as e:
                console.print(f"  [red]✗[/] Ollama: Cannot connect - {e}")
        else:
            if verbose:
                console.print(f"  [dim]→[/] {provider}: Connectivity check not implemented")
