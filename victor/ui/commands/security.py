import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Confirm
from victor.ui.console import get_console
from pathlib import Path
import os
from typing import Optional

from victor.config.secure_paths import (
    get_security_status,
    trust_plugin as do_trust_plugin,
    untrust_plugin as do_untrust_plugin,
    list_trusted_plugins,
    verify_cache_integrity,
    get_victor_dir,
    create_cache_manifest,
    get_sandbox_summary,
    validate_victor_dir_name,
    get_secure_home,
    get_secure_xdg_config_home,
    get_secure_xdg_data_home,
)
from victor.config.api_keys import is_keyring_available, get_configured_providers

security_app = typer.Typer(name="security", help="Security status and plugin trust management.")
console = get_console()


@security_app.callback(invoke_without_command=True)
def security(
    ctx: typer.Context,
    status: bool = typer.Option(True, "--status", "-s", help="Show security status"),
    trust_plugin: Optional[str] = typer.Option(
        None, "--trust-plugin", help="Trust a plugin by path"
    ),
    untrust_plugin: Optional[str] = typer.Option(
        None, "--untrust-plugin", help="Remove plugin from trust store"
    ),
    list_plugins: bool = typer.Option(False, "--list-plugins", "-l", help="List trusted plugins"),
    verify_cache: bool = typer.Option(
        False, "--verify-cache", help="Verify embedding cache integrity"
    ),
    verify_all: bool = typer.Option(
        False, "--verify-all", "-a", help="Run comprehensive security verification"
    ),
):
    """Security status and plugin trust management."""
    if ctx.invoked_subcommand is None:
        if trust_plugin:
            _trust_plugin(trust_plugin)
        elif untrust_plugin:
            _untrust_plugin(untrust_plugin)
        elif list_plugins:
            _list_plugins()
        elif verify_cache:
            _verify_cache()
        elif verify_all:
            _verify_all()
        else:
            _status()


def _trust_plugin(plugin_path: str):
    # Trust a plugin
    plugin_path = Path(plugin_path).expanduser().resolve()
    if not plugin_path.exists():
        console.print(f"[red]Plugin not found:[/] {plugin_path}")
        raise typer.Exit(1)

    if do_trust_plugin(plugin_path):
        console.print(f"[green]✓[/] Trusted plugin: [cyan]{plugin_path.name}[/]")
    else:
        console.print("[red]Failed to trust plugin[/]")
        raise typer.Exit(1)


def _untrust_plugin(plugin_name: str):
    # Untrust a plugin
    if do_untrust_plugin(plugin_name):
        console.print(f"[green]✓[/] Removed [cyan]{plugin_name}[/] from trust store")
    else:
        console.print(f"[yellow]Plugin {plugin_name} not found in trust store[/]")


def _list_plugins():
    # List trusted plugins
    plugins = list_trusted_plugins()
    if not plugins:
        console.print("[dim]No trusted plugins[/]")
        return

    table = Table(title="Trusted Plugins", show_header=True)
    table.add_column("Name", style="cyan")
    table.add_column("Hash (truncated)")
    table.add_column("Path")

    for plugin in plugins:
        table.add_row(plugin["name"], plugin.get("hash", "")[:16] + "...", plugin.get("path", ""))

    console.print(table)


def _verify_cache():
    # Verify cache integrity
    embeddings_dir = get_victor_dir() / "embeddings"
    if not embeddings_dir.exists():
        console.print("[dim]No embeddings cache found[/]")
        return

    console.print("[cyan]Verifying cache integrity...[/]")
    is_valid, tampered = verify_cache_integrity(embeddings_dir)

    if is_valid:
        console.print("[green]✓ Cache integrity verified[/]")
    else:
        console.print("[red]✗ Cache integrity check failed![/]")
        for file in tampered[:5]:
            console.print(f"  [red]Tampered:[/] {file}")
        if len(tampered) > 5:
            console.print(f"  ... and {len(tampered) - 5} more")

    # Offer to recreate manifest
    if not is_valid:
        if Confirm.ask("[yellow]Recreate manifest with current files?[/]"):
            if create_cache_manifest(embeddings_dir):
                console.print("[green]✓ Manifest recreated[/]")
            else:
                console.print("[red]Failed to recreate manifest[/]")


def _verify_all():
    # Comprehensive security verification
    console.print(Panel.fit("[bold cyan]Comprehensive Security Verification[/]"))
    console.print()

    all_passed = True
    issues = []

    # 1. HOME manipulation check
    console.print("[cyan]1. Checking HOME environment validation...[/]")
    secure_home = get_secure_home()
    env_home = os.environ.get("HOME", "")
    if str(secure_home) == env_home:
        console.print("   [green]✓ HOME environment matches passwd database[/]")
    else:
        console.print(f"   [yellow]⚠ HOME differs: env={env_home}, passwd={secure_home}[/]")
        issues.append("HOME environment may be manipulated")

    # 2. VICTOR_DIR_NAME validation
    console.print("[cyan]2. Checking VICTOR_DIR_NAME validation...[/]")
    dir_name = os.environ.get("VICTOR_DIR_NAME", ".victor")
    validated_name, is_safe = validate_victor_dir_name(dir_name)
    if is_safe:
        console.print(f"   [green]✓ VICTOR_DIR_NAME '{validated_name}' is valid[/]")
    else:
        console.print(f"   [red]✗ VICTOR_DIR_NAME blocked: {dir_name}[/]")
        all_passed = False
        issues.append(f"VICTOR_DIR_NAME contains path traversal: {dir_name}")

    # 3. XDG path validation
    console.print("[cyan]3. Checking XDG path validation...[/]")
    xdg_config = get_secure_xdg_config_home()
    xdg_data = get_secure_xdg_data_home()
    console.print(f"   [green]✓ XDG_CONFIG_HOME: {xdg_config}[/]")
    console.print(f"   [green]✓ XDG_DATA_HOME: {xdg_data}[/]")

    # 4. Keyring availability
    console.print("[cyan]4. Checking keyring availability...[/]")
    if is_keyring_available():
        console.print("   [green]✓ System keyring is available[/]")
    else:
        console.print("   [yellow]⚠ System keyring not available (install keyring package)[/]")
        issues.append("Keyring not installed - API keys stored in plaintext file")

    # 5. Cache integrity
    console.print("[cyan]5. Checking cache integrity...[/]")
    embeddings_dir = get_victor_dir() / "embeddings"
    if embeddings_dir.exists():
        is_valid, tampered = verify_cache_integrity(embeddings_dir)
        if is_valid:
            console.print("   [green]✓ Embeddings cache integrity verified[/]")
        else:
            console.print(f"   [red]✗ Cache integrity failed: {len(tampered)} tampered files[/]")
            all_passed = False
            issues.append(f"Cache tampering detected: {len(tampered)} files modified")
    else:
        console.print("   [dim]• No embeddings cache found[/]")

    # 6. Plugin sandbox status
    console.print("[cyan]6. Checking plugin sandbox configuration...[/]")
    sandbox = get_sandbox_summary()
    policy = sandbox["policy"]
    console.print(
        f"   • Trust required: {'[green]Yes[/]' if policy['require_trust'] else '[yellow]No[/]'}"
    )
    console.print(
        f"   • Network allowed: {'[yellow]Yes[/]' if policy['allow_network'] else '[green]Restricted[/]'}"
    )
    console.print(
        f"   • Subprocess allowed: {'[yellow]Yes[/]' if policy['allow_subprocess'] else '[green]Restricted[/]'}"
    )
    console.print(f"   • Trusted plugins: {sandbox['trusted_plugins']['count']}")

    # 7. API key source check
    console.print("[cyan]7. Checking API key sources...[/]")
    configured = get_configured_providers()
    for provider in configured[:5]:
        # Check source
        env_var = os.environ.get(f"{provider.upper()}_API_KEY")
        if env_var:
            console.print(f"   • {provider}: [green]environment (secure)[/]")
        else:
            console.print(f"   • {provider}: [yellow]file-based[/]")
    if len(configured) > 5:
        console.print(f"   ... and {len(configured) - 5} more")

    # Summary
    console.print()
    if all_passed and not issues:
        console.print("[bold green]✓ All security checks passed![/]")
    else:
        console.print("[bold yellow]Security Issues Found:[/]")
        for issue in issues:
            console.print(f"  [red]•[/] {issue}")


def _status():
    # Default: show security status
    sec_status = get_security_status()

    # Platform info
    console.print(
        Panel.fit(
            f"[cyan]Platform:[/] {sec_status['platform']['system']} {sec_status['platform']['release']}",
            title="Security Status",
        )
    )

    # Security checks table
    table = Table(show_header=True)
    table.add_column("Component", style="cyan")
    table.add_column("Status")
    table.add_column("Details")

    # Home security
    table.add_row(
        "Home Validation",
        "[green]✓ Secure[/]",
        sec_status["home_security"]["secure_home"],
    )

    # Keyring
    if sec_status["keyring"]["available"]:
        table.add_row("System Keyring", "[green]✓ Available[/]", sec_status["keyring"]["backend"])
    else:
        table.add_row("System Keyring", "[yellow]Not installed[/]", "pip install keyring")

    # API keys
    configured = get_configured_providers()
    table.add_row(
        "API Keys",
        f"[green]{len(configured)} configured[/]" if configured else "[dim]None[/]",
        ", ".join(configured[:3]) + ("..." if len(configured) > 3 else ""),
    )

    # Plugin trust
    trusted_plugins = sec_status["plugins"]["trusted_count"]
    table.add_row(
        "Trusted Plugins",
        f"[green]{trusted_plugins}[/]" if trusted_plugins else "[dim]None[/]",
        f"{len(sec_status['plugins']['plugin_dirs'])} plugin dirs",
    )

    # Cache integrity
    if sec_status["cache_integrity"]["embeddings_verified"]:
        table.add_row("Cache Integrity", "[green]✓ Verified[/]", "Embeddings cache valid")
    else:
        table.add_row("Cache Integrity", "[dim]Not verified[/]", "Run --verify-cache")

    console.print(table)

    # Recommendations
    recommendations = []
    if not sec_status["keyring"]["available"]:
        recommendations.append("Install keyring: [cyan]pip install keyring[/]")
    if not configured:
        recommendations.append("Configure API keys: [cyan]victor keys --setup[/]")
    if not sec_status["cache_integrity"]["embeddings_verified"]:
        recommendations.append("Verify cache: [cyan]victor security --verify-cache[/]")

    if recommendations:
        console.print()
        console.print("[yellow]Recommendations:[/]")
        for rec in recommendations:
            console.print(f"  • {rec}")
