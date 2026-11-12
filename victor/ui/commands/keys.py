import typer
from rich.console import Console
from rich.table import Table
from rich.prompt import Confirm
from victor.ui.console import get_console
import platform
import os
import getpass
from typing import Optional

from victor.config.api_keys import (
    APIKeyManager,
    get_configured_providers,
    get_configured_services,
    create_api_keys_template,
    DEFAULT_KEYS_FILE,
    PROVIDER_ENV_VARS,
    SERVICE_ENV_VARS,
    is_keyring_available,
    _get_key_from_keyring,
    _set_key_in_keyring,
    _delete_key_from_keyring,
    get_service_key,
    set_service_key,
    delete_service_key_from_keyring,
)
from victor.providers.registry import ProviderRegistry

keys_app = typer.Typer(name="keys", help="Manage API keys for cloud providers.")

console = get_console()


@keys_app.callback(invoke_without_command=True)
def keys(
    ctx: typer.Context,
    setup: bool = typer.Option(False, "--setup", "-s", help="Create API keys template file"),
    list_keys: bool = typer.Option(True, "--list", "-l", help="List configured providers"),
    provider: Optional[str] = typer.Option(None, "--set", help="Set API key for a provider"),
    service: Optional[str] = typer.Option(
        None, "--set-service", help="Set API key for a service (e.g., finnhub, fred)"
    ),
    keyring: bool = typer.Option(
        False, "--keyring", "-k", help="Store key in system keyring (secure)"
    ),
    migrate: bool = typer.Option(False, "--migrate", help="Migrate keys from file to keyring"),
    delete_keyring: Optional[str] = typer.Option(
        None, "--delete-keyring", help="Delete provider key from keyring"
    ),
    delete_service_keyring: Optional[str] = typer.Option(
        None, "--delete-service-keyring", help="Delete service key from keyring"
    ),
    list_services: bool = typer.Option(
        False, "--services", help="List configured services (finnhub, fred, etc.)"
    ),
    set_oauth_client_id: Optional[str] = typer.Option(
        None,
        "--set-oauth-client-id",
        help="Set OAuth client_id for a provider (openai, qwen)",
    ),
    delete_oauth_client_id: Optional[str] = typer.Option(
        None, "--delete-oauth-client-id", help="Delete OAuth client_id from keyring"
    ),
):
    """Manage API keys for cloud providers and external services.

    [yellow]Deprecated:[/] Use 'victor auth' instead for a more unified experience.
    This command continues to work for backward compatibility.
    """
    if ctx.invoked_subcommand is None:
        # Show deprecation notice
        console.print("[yellow]Note:[/] 'victor keys' is deprecated. Use 'victor auth' instead.")
        console.print("[dim]New command features:[/]")
        console.print("  • victor auth setup - Interactive setup wizard")
        console.print("  • victor auth add - Quick add provider account")
        console.print("  • victor auth list - List configured accounts")
        console.print("  • victor auth test - Test provider connection")
        console.print("[dim]Run: victor auth --help[/]")
        console.print()
        if setup:
            _setup()
        elif delete_keyring:
            _delete_keyring(delete_keyring)
        elif delete_service_keyring:
            _delete_service_keyring(delete_service_keyring)
        elif set_oauth_client_id:
            _set_oauth_client_id(set_oauth_client_id)
        elif delete_oauth_client_id:
            _delete_oauth_client_id(delete_oauth_client_id)
        elif migrate:
            _migrate()
        elif provider:
            _set_key(provider, keyring)
        elif service:
            _set_service_key(service, keyring)
        elif list_services:
            _list_services()
        else:
            _list_keys()


def _setup():
    # Create template file
    if DEFAULT_KEYS_FILE.exists():
        if not Confirm.ask(f"[yellow]{DEFAULT_KEYS_FILE} already exists. Overwrite?[:]"):
            console.print("[dim]Cancelled[/]")
            return

    DEFAULT_KEYS_FILE.parent.mkdir(parents=True, exist_ok=True)
    template = create_api_keys_template()
    DEFAULT_KEYS_FILE.write_text(template)

    # Set secure permissions
    os.chmod(DEFAULT_KEYS_FILE, 0o600)

    console.print(f"[green]✓[/] Created API keys template at [cyan]{DEFAULT_KEYS_FILE}[/]")
    console.print("\n[yellow]Next steps:[/]")
    console.print(f"  1. Edit [cyan]{DEFAULT_KEYS_FILE}[/]")
    console.print("  2. Replace placeholder values with your actual API keys")
    console.print("  3. File permissions already set to 0600 (owner-only)")
    console.print()
    console.print("[dim]Or use keyring for more secure storage:[/]")
    console.print("  victor keys --set anthropic --keyring")


def _delete_keyring(provider_name: str):
    # Delete key from keyring
    provider_name = provider_name.lower()
    if not is_keyring_available():
        console.print("[red]Keyring not available.[/] Install with: pip install keyring")
        raise typer.Exit(1)

    if _delete_key_from_keyring(provider_name):
        console.print(f"[green]✓[/] Deleted [cyan]{provider_name}[/] from keyring")
    else:
        console.print(f"[yellow]Key for {provider_name} not found in keyring")


def _migrate():
    # Migrate keys from file to keyring
    if not is_keyring_available():
        console.print("[red]Keyring not available.[/] Install with: pip install keyring")
        console.print("\n[dim]Platform requirements:[/]")
        console.print("  macOS: Built-in Keychain support")
        console.print("  Windows: Built-in Credential Manager")
        console.print("  Linux: Install gnome-keyring or kwallet")
        raise typer.Exit(1)

    backend_name, status = _get_keyring_info()
    console.print(f"[cyan]Keyring backend:[/] {backend_name} {status}")
    console.print()

    if not DEFAULT_KEYS_FILE.exists():
        console.print(f"[yellow]No keys file found at {DEFAULT_KEYS_FILE}[/]")
        raise typer.Exit(1)

    # Load keys from file
    import yaml

    with open(DEFAULT_KEYS_FILE) as f:
        data = yaml.safe_load(f) or {}

    api_keys = data.get("api_keys", data)
    migrated = 0
    failed = 0

    for prov, key in api_keys.items():
        if key and not key.startswith("your_"):
            if _set_key_in_keyring(prov, key):
                console.print(f"  [green]✓[/] Migrated [cyan]{prov}[/]")
                migrated += 1
            else:
                console.print(f"  [red]✗[/] Failed: [cyan]{prov}[/]")
                failed += 1

    console.print()
    console.print(f"[green]Migrated {migrated} keys to keyring")
    if failed:
        console.print(f"[red]Failed to migrate {failed} keys")

    if migrated > 0:
        console.print()
        console.print("[yellow]Security recommendation:[/]")
        console.print(f"  Delete the file: rm {DEFAULT_KEYS_FILE}")
        console.print("  Keys are now stored securely in system keyring")


def _set_key(provider: str, keyring: bool):
    # Set API key for a provider
    provider = provider.lower()
    if provider not in PROVIDER_ENV_VARS:
        console.print(f"[red]Unknown provider:[/] {provider}")
        console.print(f"Valid providers: {', '.join(sorted(PROVIDER_ENV_VARS.keys()))}")
        raise typer.Exit(1)

    # Check keyring availability if requested
    if keyring and not is_keyring_available():
        console.print("[red]Keyring not available.[/] Install with: pip install keyring")
        console.print("[dim]Falling back to file storage...[/]")
        keyring = False

    storage_type = "keyring" if keyring else "file"
    console.print(f"[cyan]Setting API key for {provider}[/] (storage: {storage_type})")
    console.print("[dim]Paste your API key (input hidden):[/]")

    key = getpass.getpass("")

    if not key.strip():
        console.print("[red]No key provided. Cancelled.[/]")
        raise typer.Exit(1)

    manager = APIKeyManager()
    if manager.set_key(provider, key.strip(), use_keyring=keyring):
        location = "system keyring" if keyring else str(DEFAULT_KEYS_FILE)
        console.print(f"[green]✓[/] API key for [cyan]{provider}[/] saved to {location}")
    else:
        console.print("[red]Failed to save API key")
        raise typer.Exit(1)


def _list_keys():
    """List configured providers with source info using APIKeyManager.get_status()."""
    manager = APIKeyManager()
    provider_status = manager.get_status()
    provider_aliases = ProviderRegistry.get_aliases()

    # Build primary to aliases mapping for display
    primary_to_aliases: dict[str, list[str]] = {}
    for alias, primary in provider_aliases.items():
        if alias != primary:  # Don't count self-mapping
            if primary not in primary_to_aliases:
                primary_to_aliases[primary] = []
            primary_to_aliases[primary].append(alias)

    table = Table(title="API Keys Status", show_header=True)
    table.add_column("Provider", style="cyan")
    table.add_column("Status")
    table.add_column("Source")
    table.add_column("Env Var")
    table.add_column("Aliases", style="dim")

    # Get primary providers only (exclude local providers from listing)
    primary_providers = {
        k: v for k, v in PROVIDER_ENV_VARS.items() if k not in {"ollama", "lmstudio", "vllm"}
    }

    configured_count = 0
    for prov, env_var in sorted(primary_providers.items()):
        # Get status from manager
        status_info = provider_status.get(prov, {})
        is_configured = status_info.get("configured", False)
        source = status_info.get("source")

        if is_configured:
            configured_count += 1

        # Format source for display
        source_display = "[dim]--[/]"
        if source == "env":
            source_display = "[green]env[/]"
        elif source == "keyring":
            source_display = "[blue]keyring[/]"
        elif source == "file":
            source_display = "[yellow]file[/]"

        status_display = "[green]✓ Configured[/]" if is_configured else "[dim]Not set[/]"
        aliases = ", ".join(sorted(primary_to_aliases.get(prov, [])))
        table.add_row(prov, status_display, source_display, env_var, aliases)

    console.print(table)

    # Count stats
    total_count = len(primary_providers)
    total_aliases = sum(len(aliases) for aliases in primary_to_aliases.values())

    console.print(
        f"\n[dim]Configured: {configured_count}/{total_count} providers ({total_aliases} aliases hidden)[/]"
    )

    # Show keyring status
    backend_name, status = _get_keyring_info()
    keyring_status = (
        "[green]✓ Available[/]" if is_keyring_available() else "[yellow]Not installed[/]"
    )
    console.print(f"[dim]Keyring:[/] {backend_name} {keyring_status}")
    console.print(f"[dim]Keys file:[/] {DEFAULT_KEYS_FILE}")

    if not configured_count:
        console.print()
        console.print("[yellow]No API keys configured.[/]")
        console.print("  [cyan]victor keys --setup[/]          Create template file")
        console.print("  [cyan]victor keys --set anthropic --keyring[/]  Store in keyring (secure)")


def _get_keyring_info() -> tuple[str, str]:
    """Get keyring backend and platform info."""
    system = platform.system()
    if system == "Darwin":
        return "macOS Keychain", "[green]Supported[/]"
    elif system == "Windows":
        return "Windows Credential Manager", "[green]Supported[/]"
    elif system == "Linux":
        return "Secret Service (GNOME Keyring/KWallet)", "[green]Supported[/]"
    else:
        return "Unknown", "[yellow]May not be supported[/]"


def _set_service_key(service_name: str, use_keyring: bool):
    """Set API key for an external service."""
    service_name = service_name.lower()
    if service_name not in SERVICE_ENV_VARS:
        console.print(f"[red]Unknown service:[/] {service_name}")
        console.print(f"Valid services: {', '.join(sorted(SERVICE_ENV_VARS.keys()))}")
        raise typer.Exit(1)

    # Check keyring availability if requested
    if use_keyring and not is_keyring_available():
        console.print("[red]Keyring not available.[/] Install with: pip install keyring")
        console.print("[dim]Falling back to file storage...[/]")
        use_keyring = False

    storage_type = "keyring" if use_keyring else "file"
    env_var = SERVICE_ENV_VARS[service_name]
    console.print(f"[cyan]Setting API key for service {service_name}[/] (storage: {storage_type})")
    console.print(f"[dim]Environment variable: {env_var}[/]")
    console.print("[dim]Paste your API key (input hidden):[/]")

    key = getpass.getpass("")

    if not key.strip():
        console.print("[red]No key provided. Cancelled.[/]")
        raise typer.Exit(1)

    if set_service_key(service_name, key.strip(), use_keyring=use_keyring):
        location = "system keyring" if use_keyring else str(DEFAULT_KEYS_FILE)
        console.print(
            f"[green]✓[/] API key for service [cyan]{service_name}[/] saved to {location}"
        )
    else:
        console.print("[red]Failed to save API key")
        raise typer.Exit(1)


def _delete_service_keyring(service_name: str):
    """Delete service key from keyring."""
    service_name = service_name.lower()
    if not is_keyring_available():
        console.print("[red]Keyring not available.[/] Install with: pip install keyring")
        raise typer.Exit(1)

    if delete_service_key_from_keyring(service_name):
        console.print(f"[green]✓[/] Deleted service [cyan]{service_name}[/] from keyring")
    else:
        console.print(f"[yellow]Key for service {service_name} not found in keyring")


def _list_services():
    """List configured external services using APIKeyManager.get_services_status()."""
    manager = APIKeyManager()
    services_status = manager.get_services_status()

    table = Table(title="External Service API Keys Status", show_header=True)
    table.add_column("Service", style="cyan")
    table.add_column("Status")
    table.add_column("Source")
    table.add_column("Env Var")
    table.add_column("Description", style="dim")

    # Service descriptions
    service_descriptions = {
        "finnhub": "Stock data, analyst estimates",
        "fred": "Federal Reserve Economic Data",
        "alphavantage": "Stock/forex/crypto data",
        "polygon": "Real-time market data",
        "tiingo": "Stock/crypto/forex data",
        "iex": "IEX Cloud market data",
        "quandl": "Nasdaq Data Link",
        "nasdaq": "Nasdaq Data Link",
        "newsapi": "News aggregation",
        "marketaux": "Financial news API",
        "sec": "SEC EDGAR API",
        "openweather": "Weather data",
        "geocoding": "Geocoding services",
    }

    configured_count = 0
    for svc, env_var in sorted(SERVICE_ENV_VARS.items()):
        # Get status from manager
        status_info = services_status.get(svc, {})
        is_configured = status_info.get("configured", False)
        source = status_info.get("source")

        if is_configured:
            configured_count += 1

        # Format source for display
        source_display = "[dim]--[/]"
        if source == "env":
            source_display = "[green]env[/]"
        elif source == "keyring":
            source_display = "[blue]keyring[/]"
        elif source == "file":
            source_display = "[yellow]file[/]"

        status_display = "[green]✓ Configured[/]" if is_configured else "[dim]Not set[/]"
        description = service_descriptions.get(svc, "")
        table.add_row(svc, status_display, source_display, env_var, description)

    console.print(table)

    # Count stats
    total_count = len(SERVICE_ENV_VARS)

    console.print(f"\n[dim]Configured: {configured_count}/{total_count} services[/]")

    # Show keyring status
    backend_name, status = _get_keyring_info()
    keyring_status = (
        "[green]✓ Available[/]" if is_keyring_available() else "[yellow]Not installed[/]"
    )
    console.print(f"[dim]Keyring:[/] {backend_name} {keyring_status}")

    if not configured_count:
        console.print()
        console.print("[yellow]No service API keys configured.[/]")
        console.print(
            "  [cyan]victor keys --set-service finnhub --keyring[/]  Store Finnhub key in keyring"
        )
        console.print(
            "  [cyan]victor keys --set-service fred --keyring[/]     Store FRED key in keyring"
        )


def _set_oauth_client_id(provider: str):
    """Set OAuth client_id for a provider."""
    provider = provider.lower()

    # Validate provider
    oauth_providers = {"openai", "qwen"}
    if provider not in oauth_providers:
        console.print(f"[red]Provider '{provider}' does not support OAuth.[/]")
        console.print(f"Supported OAuth providers: {', '.join(sorted(oauth_providers))}")
        raise typer.Exit(1)

    # Check keyring availability
    if not is_keyring_available():
        console.print("[red]Keyring not available.[/] Install with: pip install keyring")
        raise typer.Exit(1)

    console.print(f"[cyan]Setting OAuth client_id for {provider}[/]")
    console.print("[dim]Paste your OAuth client_id (input hidden):[/]")

    client_id = getpass.getpass("")

    if not client_id.strip():
        console.print("[red]No client_id provided. Cancelled.[/]")
        raise typer.Exit(1)

    # Import the OAuth manager function
    try:
        from victor.providers.oauth_manager import _set_oauth_client_id_in_keyring

        if _set_oauth_client_id_in_keyring(provider, client_id.strip()):
            console.print(f"[green]✓[/] OAuth client_id for [cyan]{provider}[/] saved to keyring")
            console.print()
            console.print("[yellow]Important:[/]")
            console.print(f"  This client_id must be registered with {provider.upper()}")
            console.print(
                f'  You can now use: victor chat --provider {provider} --auth-mode oauth "Your message"'
            )
        else:
            console.print("[red]Failed to save OAuth client_id")
            raise typer.Exit(1)
    except ImportError as e:
        console.print(f"[red]Failed to import OAuth manager: {e}[/]")
        raise typer.Exit(1)


def _delete_oauth_client_id(provider: str):
    """Delete OAuth client_id from keyring."""
    provider = provider.lower()

    # Validate provider
    oauth_providers = {"openai", "qwen"}
    if provider not in oauth_providers:
        console.print(f"[red]Provider '{provider}' does not support OAuth.[/]")
        console.print(f"Supported OAuth providers: {', '.join(sorted(oauth_providers))}")
        raise typer.Exit(1)

    # Check keyring availability
    if not is_keyring_available():
        console.print("[red]Keyring not available.[/] Install with: pip install keyring")
        raise typer.Exit(1)

    try:
        import keyring

        # Delete the OAuth client_id from keyring
        keyring.delete_password("victor", f"oauth_client_id_{provider}")
        console.print(f"[green]✓[/] Deleted OAuth client_id for [cyan]{provider}[/] from keyring")
    except Exception as e:
        console.print(f"[yellow]Failed to delete OAuth client_id: {e}[/]")
        console.print("[dim]The client_id may not be stored in the keyring.[/]")
