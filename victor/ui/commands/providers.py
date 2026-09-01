from typing import Dict, List, Optional, Tuple

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from victor.core.async_utils import run_sync
from victor.providers.registry import ProviderRegistry
from victor.providers.health import ProviderHealthChecker
from victor.ui.json_utils import create_json_option, print_json_data

providers_app = typer.Typer(name="provider", help="List and manage providers.")
console = Console()


@providers_app.callback(invoke_without_command=True)
def providers_callback(
    ctx: typer.Context,
    list_flag: bool = typer.Option(
        False,
        "--list",
        help="List providers. Compatibility alias for 'victor provider list'.",
    ),
) -> None:
    """Handle providers command with optional subcommand.

    If no subcommand is provided, lists all available providers.
    """
    if list_flag or ctx.invoked_subcommand is None:
        # No subcommand provided, default to listing providers
        _list_providers_impl()


@providers_app.command("list")
def list_providers() -> None:
    """List all available providers."""
    _list_providers_impl()


def _list_providers_impl() -> None:
    available_providers = ProviderRegistry.list_providers()
    raw_aliases = ProviderRegistry.get_aliases()
    aliases = raw_aliases if isinstance(raw_aliases, dict) else {}
    configured_by_provider, default_provider = _load_configured_provider_summary()

    # Build reverse mapping: primary provider -> list of aliases
    primary_to_aliases: Dict[str, List[str]] = {}
    primary_providers = set()
    for provider in available_providers:
        primary = aliases.get(provider, provider)
        if not isinstance(primary, str) or not primary:
            primary = provider
        primary_providers.add(primary)
        if provider != primary:
            primary_to_aliases.setdefault(primary, []).append(provider)

    if not primary_providers:
        primary_providers = set(available_providers)

    # Provider info: (status, features)
    provider_info = {
        # Tested and working providers
        "ollama": ("✅ Ready", "Local models, Free, Tool calling"),
        "anthropic": ("✅ Ready", "Claude 3.5/4, Tool calling, Streaming"),
        "openai": ("✅ Ready", "GPT-4.1/4o/o1/o3, Function calling, Vision"),
        "google": ("✅ Ready", "Gemini 2.5, 1M context, Multimodal"),
        "xai": ("✅ Ready", "Grok, Real-time info, Vision"),
        "lmstudio": ("✅ Ready", "Local models via LMStudio"),
        "vllm": ("✅ Ready", "High-throughput local inference"),
        "moonshot": ("✅ Ready", "Kimi K2, 256K context, Reasoning"),
        "deepseek": ("✅ Ready", "DeepSeek-V3, 128K, Cheap"),
        "groqcloud": ("✅ Ready", "Ultra-fast LPU, Free tier, Llama/GPT-OSS"),
        "cerebras": ("✅ Ready", "Ultra-fast inference, Free tier, Qwen/Llama"),
        # Local inference
        "llamacpp": ("✅ Ready", "Local GGUF models, CPU/GPU"),
        "inferflux": ("✅ Ready", "Self-hosted, Batched decode, KV-prefix reuse, Sandhi-metered"),
        # Tested providers
        "mistral": ("✅ Ready", "Mistral Large/Codestral, 500K tokens/min free"),
        # Known but untested providers
        "together": ("⚠️ Untested", "Together AI, $25 free credits"),
        "openrouter": ("✅ Ready", "Unified gateway, 350+ models, Free tier"),
        "fireworks": ("✅ Ready", "Fast inference, $1 free credits, Tool calling"),
        "zai": ("✅ Ready", "GLM-5/4.7, Coding Plan, Thinking mode, OpenAI-compat"),
        "qwen": ("✅ Ready", "Qwen3.5, OAuth + API-key, Coding Plan, OpenAI-compat"),
        "huggingface": ("⚠️ Untested", "HuggingFace Inference API"),
        "replicate": ("⚠️ Untested", "Replicate, Open models"),
        # Enterprise providers (require setup)
        "vertex": ("🏢 Enterprise", "Google Cloud Vertex AI"),
        "azure": ("🏢 Enterprise", "Azure OpenAI Service"),
        "bedrock": ("🏢 Enterprise", "AWS Bedrock (Claude, Llama, Titan)"),
    }

    table = Table(title="Available Providers", show_header=True)
    table.add_column("Provider", style="cyan", no_wrap=True)
    table.add_column("Status", style="green")
    table.add_column("Configured", style="yellow")
    table.add_column("Features")
    table.add_column("Aliases", style="dim")

    sorted_primary_providers = sorted(primary_providers)

    for provider in sorted_primary_providers:
        info = provider_info.get(provider)
        if info:
            status, features = info
            provider_aliases = primary_to_aliases.get(provider, [])
            aliases_str = ", ".join(sorted(provider_aliases)) if provider_aliases else ""
            table.add_row(
                provider,
                status,
                _format_configured_provider(provider, configured_by_provider, default_provider),
                features,
                aliases_str,
            )
        else:
            table.add_row(
                provider,
                "❓ Unknown",
                _format_configured_provider(provider, configured_by_provider, default_provider),
                "",
                "",
            )

    console.print(table)
    alias_count = sum(
        1 for provider in available_providers if aliases.get(provider, provider) != provider
    )
    console.print(
        f"\n[dim]Total: {len(sorted_primary_providers)} providers ({alias_count} aliases hidden)[/]"
    )
    console.print("[dim]Configured accounts: ~/.victor/config.yaml via 'victor auth list'[/]")
    console.print("[dim]Configured profiles: ~/.victor/profiles.yaml via 'victor profiles list'[/]")


def _load_configured_provider_summary() -> Tuple[Dict[str, List[str]], Optional[str]]:
    """Return configured account names grouped by provider plus default provider."""
    try:
        from victor.config.accounts import get_account_manager

        manager = get_account_manager()
        config = manager.load_config()
        default_account = config.defaults.account
        configured: Dict[str, List[str]] = {}
        default_provider: Optional[str] = None
        for account in config.list_accounts():
            configured.setdefault(account.provider, []).append(account.name)
            if account.name == default_account:
                default_provider = account.provider
        return configured, default_provider
    except Exception:
        return {}, None


def _format_configured_provider(
    provider: str,
    configured_by_provider: Dict[str, List[str]],
    default_provider: Optional[str],
) -> str:
    accounts = configured_by_provider.get(provider, [])
    if not accounts:
        return "[dim]—[/]"
    marker = " ★" if provider == default_provider else ""
    if len(accounts) == 1:
        return f"{accounts[0]}{marker}"
    return f"{len(accounts)} accounts{marker}"


@providers_app.command("check")
def check_provider(
    provider: str = typer.Argument(..., help="Provider name (e.g., deepseek, anthropic, ollama)"),
    model: str = typer.Option(
        None, help="Model to check (default: the provider's own policy default)"
    ),
    connectivity: bool = typer.Option(
        False, "--connectivity", "-c", help="Perform connectivity test (slower)"
    ),
    timeout: float = typer.Option(5.0, help="Timeout for connectivity check (seconds)"),
    json_output: bool = create_json_option(),
):
    """Check if a provider is properly configured.

    Example:
        victor providers check deepseek
        victor providers check anthropic --connectivity
        victor providers check ollama
    """
    if model is None:
        model = _default_check_model(provider)
    run_sync(
        _check_provider_async(
            provider=provider,
            model=model,
            connectivity=connectivity,
            timeout=timeout,
            json_output=json_output,
        )
    )


def _default_check_model(provider: str) -> str:
    """The provider's own policy default, not one hardcoded vendor's model.

    OpenAI-compat providers read their YAML policy tier's ``default_model``;
    everything else keeps the historical deepseek default only when the
    provider has no policy entry.
    """
    key = provider.strip().lower()
    try:
        from victor.providers.openai_compat_model_policy import (
            get_openai_compat_provider_spec,
        )

        spec = get_openai_compat_provider_spec(key)
        default = getattr(spec, "default_model", None)
        if default:
            return str(default)
    except Exception:
        pass
    return "deepseek-chat"


async def _check_provider_async(
    *,
    provider: str,
    model: str,
    connectivity: bool,
    timeout: float,
    json_output: bool,
) -> None:
    checker = ProviderHealthChecker()
    result = await checker.check_provider(
        provider=provider,
        model=model,
        check_connectivity=connectivity,
        timeout=timeout,
    )

    if json_output:
        print_json_data(result.to_dict())
        return

    # Display results in a nice format
    if result.healthy:
        status_panel = Panel(
            "[bold green]✓ HEALTHY[/bold green]",
            title=f"[bold]{provider.upper()}[/bold]",
            subtitle=f"Model: {model}",
        )
        console.print(status_panel)

        # Show info
        if result.info:
            info_table = Table(title="Configuration", show_header=False)
            info_table.add_column("Key", style="cyan")
            info_table.add_column("Value", style="green")

            for key, value in result.info.items():
                info_table.add_row(key, str(value))

            console.print(info_table)

        # Show warnings if any
        if result.warnings:
            console.print("\n[bold yellow]Warnings:[/bold yellow]")
            for warning in result.warnings:
                console.print(f"  ⚠️  {warning}")
    else:
        status_panel = Panel(
            "[bold red]✗ UNHEALTHY[/bold red]",
            title=f"[bold]{provider.upper()}[/bold]",
            subtitle=f"Model: {model}",
        )
        console.print(status_panel)

        # Show issues
        console.print("\n[bold red]Issues:[/bold red]")
        for i, issue in enumerate(result.issues, 1):
            console.print(f"  {i}. {issue}")

        # Show warnings if any
        if result.warnings:
            console.print("\n[bold yellow]Warnings:[/bold yellow]")
            for warning in result.warnings:
                console.print(f"  ⚠️  {warning}")


@providers_app.command("verify")
def verify_provider(
    provider: str = typer.Argument(..., help="Provider name"),
    model: str = typer.Option(..., help="Model to verify"),
    api_key: Optional[str] = typer.Option(None, help="API key (overrides other sources)"),
):
    """Verify provider configuration with detailed diagnostics.

    This is an enhanced health check that provides detailed information
    about where the API key is coming from and whether it's valid.

    Example:
        victor providers verify deepseek --model deepseek-chat
        victor providers verify anthropic --model claude-3-5-haiku --api-key sk-...
    """
    run_sync(
        _verify_provider_async(
            provider=provider,
            model=model,
            api_key=api_key,
        )
    )


async def _verify_provider_async(
    *,
    provider: str,
    model: str,
    api_key: Optional[str],
) -> None:
    from victor.providers.resolution import UnifiedApiKeyResolver, APIKeyNotFoundError

    console.print(f"[bold]Verifying Provider:[/bold] {provider}")
    console.print(f"[bold]Model:[/bold] {model}\n")

    # Check if provider is registered
    try:
        ProviderRegistry.get(provider)
        console.print("[green]✓[/green] Provider is registered")
    except Exception:
        console.print(f"[red]✗[/red] Provider '{provider}' is not registered")
        available = ProviderRegistry.list_providers()
        console.print(f"\nAvailable providers: {', '.join(available)}")
        raise typer.Exit(1)

    # Check API key resolution
    resolver = UnifiedApiKeyResolver()
    key_result = resolver.get_api_key(provider, explicit_key=api_key)

    console.print("\n[bold]API Key Resolution:[/bold]")
    for i, source in enumerate(key_result.sources_attempted, 1):
        status = "[green]✓[/green]" if source.found else "[red]✗[/red]"
        console.print(f"  {i}. {status} {source.description}")

        if source.found and source.value_preview:
            console.print(f"     Preview: {source.value_preview}")

    if key_result.key is None:
        error = APIKeyNotFoundError(
            provider=provider,
            sources_attempted=key_result.sources_attempted,
            non_interactive=key_result.non_interactive,
            model=model,
        )
        console.print(f"\n[red]{error}[/red]")
        raise typer.Exit(1)

    # Run health check
    console.print("\n[bold]Health Check:[/bold]")
    checker = ProviderHealthChecker()
    health_result = await checker.check_provider(
        provider=provider,
        model=model,
        check_connectivity=False,
    )

    if health_result.healthy:
        console.print("[green]✓ All checks passed![/green]")
    else:
        console.print("[red]✗ Health check failed:[/red]")
        for issue in health_result.issues:
            console.print(f"  • {issue}")


# ---------------------------------------------------------------------------
# victor providers auth login / logout / status
# ---------------------------------------------------------------------------

auth_app = typer.Typer(name="auth", help="Manage OAuth authentication for providers.")
providers_app.add_typer(auth_app)

OAUTH_SUPPORTED_PROVIDERS = ["openai", "qwen", "google", "github-copilot"]


@auth_app.command("login")
def auth_login(
    provider: str = typer.Argument(
        ..., help=f"Provider to authenticate ({', '.join(OAUTH_SUPPORTED_PROVIDERS)})"
    ),
    force: bool = typer.Option(
        False, "--force", "-f", help="Force re-authentication even if token is cached"
    ),
) -> None:
    """Log in to a provider via OAuth (opens browser).

    Example:
        victor providers auth login openai
        victor providers auth login qwen --force
    """
    provider = provider.lower()
    if provider not in OAUTH_SUPPORTED_PROVIDERS:
        console.print(
            f"[red]✗[/] OAuth not supported for '{provider}'. "
            f"Supported: {', '.join(OAUTH_SUPPORTED_PROVIDERS)}"
        )
        raise typer.Exit(1)

    run_sync(_auth_login_async(provider=provider, force=force))


async def _auth_login_async(*, provider: str, force: bool) -> None:
    from victor.providers.oauth_manager import OAuthTokenManager

    mgr = OAuthTokenManager(provider)

    if not force:
        cached = mgr._load_cached()
        if cached is not None and not cached.is_expired:
            console.print(f"[green]✓[/] Already authenticated with {provider}")
            console.print(f"  Token expires: {cached.expires_at.strftime('%Y-%m-%d %H:%M UTC')}")
            console.print("[dim]Use --force to re-authenticate[/]")
            return

    console.print(f"[cyan]Opening browser for {provider} OAuth login...[/]")
    try:
        token = await mgr.get_valid_token()
        if token:
            console.print(f"[green]✓[/] Successfully authenticated with {provider}")
            console.print("  Token saved to ~/.victor/oauth_tokens.yaml")
        else:
            console.print(f"[red]✗[/] Authentication failed for {provider}")
            raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]✗[/] OAuth login failed: {e}")
        raise typer.Exit(1)


@auth_app.command("logout")
def auth_logout(
    provider: str = typer.Argument(
        ..., help=f"Provider to log out from ({', '.join(OAUTH_SUPPORTED_PROVIDERS)})"
    ),
) -> None:
    """Remove cached OAuth tokens for a provider.

    Example:
        victor providers auth logout openai
        victor providers auth logout qwen
    """
    provider = provider.lower()
    if provider not in OAUTH_SUPPORTED_PROVIDERS:
        console.print(
            f"[red]✗[/] OAuth not supported for '{provider}'. "
            f"Supported: {', '.join(OAUTH_SUPPORTED_PROVIDERS)}"
        )
        raise typer.Exit(1)

    from victor.providers.oauth_manager import OAuthTokenManager

    mgr = OAuthTokenManager(provider)
    mgr._clear_cached()
    console.print(f"[green]✓[/] Logged out from {provider} (tokens cleared)")


@auth_app.command("status")
def auth_status(
    provider: Optional[str] = typer.Argument(
        None, help="Check specific provider (or all if omitted)"
    ),
) -> None:
    """Show OAuth authentication status for providers.

    Example:
        victor providers auth status
        victor providers auth status openai
    """
    from victor.providers.oauth_manager import OAuthTokenManager

    providers_to_check = [provider.lower()] if provider else OAUTH_SUPPORTED_PROVIDERS

    table = Table(title="OAuth Authentication Status", show_header=True)
    table.add_column("Provider", style="cyan")
    table.add_column("Status")
    table.add_column("Expires")
    table.add_column("Token Preview", style="dim")

    for prov in providers_to_check:
        if prov not in OAUTH_SUPPORTED_PROVIDERS:
            table.add_row(prov, "[red]Not supported[/]", "", "")
            continue

        mgr = OAuthTokenManager(prov)
        cached = mgr._load_cached()

        if cached is None:
            table.add_row(prov, "[dim]Not authenticated[/]", "", "")
        elif cached.is_expired:
            table.add_row(
                prov,
                "[yellow]Expired[/]",
                (cached.expires_at.strftime("%Y-%m-%d %H:%M UTC") if cached.expires_at else ""),
                "",
            )
        else:
            preview = cached.access_token[:8] + "..." if cached.access_token else ""
            table.add_row(
                prov,
                "[green]✓ Active[/]",
                (cached.expires_at.strftime("%Y-%m-%d %H:%M UTC") if cached.expires_at else ""),
                preview,
            )

    console.print(table)
    console.print("\n[dim]Login:  victor providers auth login <provider>[/]")
    console.print("[dim]Logout: victor providers auth logout <provider>[/]")
