# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
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

"""Interactive authentication and account management for Victor.

This module provides a comprehensive auth command with subcommands for:
- setup: Interactive wizard for first-time setup
- add: Quick add a provider account
- list: Show configured accounts
- remove: Remove an account
- migrate: Migrate from old configuration format
- test: Test a provider connection

The new unified configuration format replaces the old profiles.yaml + api_keys.yaml
with a single config.yaml file.
"""

from __future__ import annotations

import asyncio
import json
import os
import shlex
import subprocess
import sys
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt


class AuthStatus(str, Enum):
    """OAuth authentication status for a provider."""

    AUTHENTICATED = "authenticated"  # Valid token exists and is not expired
    EXPIRED = "expired"  # Token exists but has expired
    PENDING = "pending"  # No valid token (not authenticated or token missing)


from rich.table import Table
from rich.text import Text

from victor.config.accounts import (
    AccountManager,
    AuthConfig,
    ProviderAccount,
    get_account_manager,
)
from victor.config.api_keys import LOCAL_PROVIDERS, APIKeyManager
from victor.config.connection_validation import ConnectionValidator, ConnectionValidationResult
from victor.ui.json_utils import create_json_option, print_json_data
from victor.config.migration import (
    ConfigMigrator,
    check_migration_needed,
    run_migration,
)
from victor.config.profiles import ProfileManager
from victor.config.settings import get_project_paths


def _sync_profile_from_account(
    account: ProviderAccount,
    config_dir: Optional[Path] = None,
) -> bool:
    """Create or update the chat profile for a saved provider account."""
    if config_dir is None:
        config_dir = get_project_paths().global_victor_dir
    return ProfileManager.for_config_dir(config_dir).upsert_account_profile(account)


# =============================================================================
# Popular provider models (for quick selection)
# =============================================================================

POPULAR_MODELS: Dict[str, List[str]] = {
    "anthropic": [
        "claude-sonnet-4-5",
        "claude-sonnet-4-6",
        "claude-opus-4-6",
    ],
    "openai": [
        "gpt-5-nano",
        "gpt-5-mini",
        "gpt-5",
        "gpt-5.2",
        "o4-mini",
    ],
    "google": [
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "gemini-2.0-flash-exp",
    ],
    "zai": [
        "glm-4.6:coding",  # Coding Plan endpoint
        "glm-4.6:standard",  # Standard endpoint
        "glm-4.6:thinking",  # Thinking mode endpoint
        "glm-5.0",
    ],
    "xai": [
        "grok-2-1212",
        "grok-2-vision-1212",
    ],
    "qwen": [
        "qwen-coder-plus",
        "qwen-max",
        "qwen-plus",
    ],
    "moonshot": [
        "moonshot-v1-128k",
        "moonshot-v1-32k",
    ],
    "deepseek": [
        "deepseek-chat",
        "deepseek-coder",
        "deepseek-reasoner",
    ],
}

# =============================================================================
# OAuth-enabled providers
# =============================================================================

OAUTH_PROVIDERS = ["openai", "qwen"]

# =============================================================================
# Auth app
# =============================================================================

auth_app = typer.Typer(name="auth", help="Manage authentication and provider accounts.")
console = Console()


def _get_oauth_tokens_file() -> Path:
    """Resolve the OAuth token store through centralized Victor paths."""
    return get_project_paths().global_victor_dir / "oauth_tokens.yaml"


def _get_oauth_status(provider: str, source: str = "victor") -> AuthStatus:
    """Check OAuth token status for a provider without triggering login.

    Returns:
        AuthStatus enum: AUTHENTICATED if valid token exists, EXPIRED if token expired, PENDING otherwise.
    """
    if source in {"codex", "claude-code"}:
        try:
            from victor.providers.oauth_manager import OAuthTokenManager

            tokens = OAuthTokenManager(provider, token_source=source)._load_cached()
            if tokens is None:
                return AuthStatus.PENDING
            return AuthStatus.EXPIRED if tokens.is_expired else AuthStatus.AUTHENTICATED
        except Exception:
            return AuthStatus.PENDING

    token_file = _get_oauth_tokens_file()

    if not token_file.exists():
        return AuthStatus.PENDING

    try:
        import yaml

        with open(token_file) as f:
            all_tokens = yaml.safe_load(f) or {}
        data = all_tokens.get(provider)
        if data is None or not data.get("access_token"):
            return AuthStatus.PENDING

        # Check expiry
        expires_at = data.get("expires_at")
        if expires_at:
            from datetime import datetime, timezone

            exp = datetime.fromisoformat(expires_at)
            if exp <= datetime.now(timezone.utc):
                return AuthStatus.EXPIRED

        return AuthStatus.AUTHENTICATED
    except Exception:
        return AuthStatus.PENDING


# =============================================================================
# Setup Command
# =============================================================================


@auth_app.command("setup")
def auth_setup() -> None:
    """Interactive wizard for first-time authentication setup.

    This wizard guides you through:
    1. Detecting local providers (Ollama, LM Studio)
    2. Choosing a cloud provider
    3. Selecting authentication method (API key vs OAuth)
    4. Collecting credentials
    5. Testing the connection
    6. Saving the account

    Example:
        victor auth setup
    """
    wizard = AuthSetupWizard(console)
    exit_code = wizard.run()
    raise typer.Exit(exit_code)


# =============================================================================
# Add Command
# =============================================================================


@auth_app.command("add")
def auth_add(
    provider: str = typer.Option(
        ..., "--provider", "-p", help="Provider name (e.g., anthropic, openai)"
    ),
    model: str = typer.Option(..., "--model", "-m", help="Model name (e.g., claude-sonnet-4-5)"),
    name: str = typer.Option("default", "--name", "-n", help="Account name"),
    auth_method: str = typer.Option(
        "api_key", "--auth-method", help="Authentication method (api_key, oauth, none)"
    ),
    source: str = typer.Option(
        "keyring",
        "--source",
        help="Credential source (keyring, env, file, sentinelpass)",
    ),
    endpoint: Optional[str] = typer.Option(None, "--endpoint", "-e", help="Custom endpoint URL"),
    api_key: Optional[str] = typer.Option(
        None, "--api-key", help="API key (will prompt if not provided)"
    ),
    sentinelpass_domain: Optional[str] = typer.Option(
        None,
        "--sentinelpass-domain",
        help="SentinelPass lookup domain when --source sentinelpass is used",
    ),
    tags: Optional[str] = typer.Option(None, "--tags", help="Comma-separated tags"),
    temperature: Optional[float] = typer.Option(None, "--temperature", "-t", help="Temperature"),
    max_tokens: Optional[int] = typer.Option(None, "--max-tokens", help="Max output tokens"),
    set_default: bool = typer.Option(False, "--default", help="Set this account as default"),
) -> None:
    """Quick add a provider account.

    Example:
        victor auth add --provider anthropic --model claude-sonnet-4-5
        victor auth add --provider zai --model glm-4.6:coding --name glm-coding
        victor auth add --provider openai --model gpt-5-nano --auth-method oauth --source codex
        victor auth add --provider anthropic --model claude-haiku-4-5-20251001 --auth-method oauth --source claude-code
        victor auth add --provider anthropic --model claude-sonnet-4-5 --source sentinelpass
    """
    source = source.lower()
    valid_sources = {"keyring", "env", "file", "sentinelpass", "codex", "claude-code"}
    if source not in valid_sources:
        console.print(f"[red]✗[/] Unknown credential source: {source}")
        console.print(f"[dim]Valid sources: {', '.join(sorted(valid_sources))}[/]")
        raise typer.Exit(1)
    if source == "codex" and not (provider == "openai" and auth_method == "oauth"):
        console.print("[red]✗[/] --source codex is only valid with OpenAI OAuth accounts")
        console.print(
            "[dim]Use: victor auth add --provider openai --model gpt-5-nano "
            "--auth-method oauth --source codex[/]"
        )
        raise typer.Exit(1)
    if source == "claude-code" and not (provider == "anthropic" and auth_method == "oauth"):
        console.print("[red]✗[/] --source claude-code is only valid with Anthropic OAuth accounts")
        console.print(
            "[dim]Use: victor auth add --provider anthropic --model claude-haiku-4-5-20251001 "
            "--auth-method oauth --source claude-code[/]"
        )
        raise typer.Exit(1)

    # Parse tags
    tag_list = tags.split(",") if tags else []

    # Get API key if needed
    if auth_method == "api_key" and source != "sentinelpass" and not api_key:
        api_key = Prompt.ask(f"Enter API key for {provider}", password=True)

    # Create auth config
    auth = AuthConfig(method=auth_method, source=source)
    if source == "sentinelpass":
        auth.value = sentinelpass_domain or provider
    elif api_key:
        auth.value = api_key

    # Create account
    account = ProviderAccount(
        name=name,
        provider=provider,
        model=model,
        auth=auth,
        endpoint=endpoint,
        tags=tag_list,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    # Save account and matching chat profile
    manager = get_account_manager()
    manager.save_account(account)
    profile_synced = _sync_profile_from_account(account)
    if set_default:
        config = manager.load_config()
        config.defaults.account = name
        manager.save_config(config)

    # Save API key to keyring if provided
    if api_key and auth_method == "api_key" and source == "keyring":
        try:
            from victor.config.api_keys import _set_key_in_keyring

            _set_key_in_keyring(provider, api_key)
            console.print("[green]✓[/] API key saved to keyring")
        except Exception as e:
            console.print(f"[yellow]⚠[/] Could not save to keyring: {e}")
            console.print("[dim]API key stored in config file instead[/]")

    console.print(f"[green]✓[/] Account '{name}' added successfully")
    if profile_synced:
        console.print(f"[green]✓[/] Profile '{name}' added to profiles.yaml")
    console.print(f"[dim]Provider: {provider}[/]")
    console.print(f"[dim]Model: {model}[/]")
    if set_default:
        console.print("[dim]Default: yes[/]")
    if source == "sentinelpass":
        console.print(f"[dim]SentinelPass domain: {auth.value}[/]")
    if source == "codex":
        console.print("[dim]OAuth source: Codex ~/.codex/auth.json[/]")
    if source == "claude-code":
        console.print("[dim]OAuth source: Claude Code credentials[/]")


# =============================================================================
# List Command
# =============================================================================


def _provider_key_cell(provider: str, key_status: dict, *, plain: bool = False) -> str:
    """Render whether a provider has an API key, and from where.

    Local providers need no key. Cloud providers show the resolution source
    (env/keyring/file) or a clear 'missing' so a user can see at a glance why a
    profile/account works or doesn't — the gap behind 'why is my zai profile invisible'.
    """
    if provider in LOCAL_PROVIDERS:
        return "local" if plain else "[dim]— local[/]"
    st = key_status.get(provider) or {}
    if st.get("configured"):
        src = st.get("source") or "?"
        return f"key:{src}" if plain else f"[green]✓ {src}[/]"
    return "missing" if plain else "[red]✗ missing[/]"


def _oauth_status_value(account) -> str:
    """OAuth status as a clean lowercase string (no leaked AuthStatus enum repr)."""
    raw = _get_oauth_status(account.provider, account.auth.source)
    return str(getattr(raw, "value", raw)).lower()


@auth_app.command("list")
def auth_list(
    json_output: bool = create_json_option(),
    show_tags: bool = typer.Option(
        False, "--tags", help="Show the Tags column (incl. migration bookkeeping)"
    ),
) -> None:
    """List provider accounts AND profiles, with key status.

    Unifies the two registries Victor uses: accounts (config.yaml, written by `auth add`)
    and profiles (profiles.yaml, used by `--profile`). A profile you hand-edited (e.g.
    `zai-coding`) shows up here even without a matching account, and the Key column shows
    whether its provider's key is configured and from where.

    Example:
        victor auth list
        victor auth list --json
    """
    from victor.framework.runtime_discovery import list_runtime_profiles

    manager = get_account_manager()
    accounts = manager.list_accounts()
    try:
        profiles = list_runtime_profiles()
    except Exception:
        profiles = []
    try:
        key_status = APIKeyManager().get_status()
    except Exception:
        key_status = {}

    default_account = manager.load_config().defaults.account
    acct_by_name = {a.name: a for a in accounts}
    prof_by_name = {p.name: p for p in profiles}
    all_names = sorted(set(acct_by_name) | set(prof_by_name))

    if not all_names:
        if json_output:
            print_json_data({"entries": []})
            return
        console.print("[yellow]No accounts or profiles configured.[/]")
        console.print("[dim]Run 'victor auth setup' to add your first account.[/]")
        return

    entries: list[dict] = []
    for name in all_names:
        acct = acct_by_name.get(name)
        prof = prof_by_name.get(name)
        provider = acct.provider if acct else prof.provider
        model = acct.model if acct else prof.model
        if acct and prof:
            kind = "account+profile"
        elif acct:
            kind = "account"
        else:
            kind = "profile"
        is_default = bool(
            (acct and name == default_account) or (prof and getattr(prof, "is_default", False))
        )
        if acct and acct.auth.method == "oauth":
            status = _oauth_status_value(acct)
        elif acct and acct.auth.method == "none":
            status = "local"
        elif acct:
            status = "configured"
        else:
            status = "profile-only"
        st = key_status.get(provider) or {}
        entries.append(
            {
                "name": name,
                "kind": kind,
                "provider": provider,
                "model": model,
                "key_configured": provider in LOCAL_PROVIDERS or bool(st.get("configured")),
                "key_source": ("local" if provider in LOCAL_PROVIDERS else st.get("source")),
                "status": status,
                "default": is_default,
                "tags": list(acct.tags) if acct else [],
            }
        )

    if json_output:
        print_json_data({"entries": entries, "total": len(entries)})
        return

    status_render = {
        "authenticated": "[green]authenticated[/]",
        "pending": "[yellow]pending login[/]",
        "configured": "[green]configured[/]",
        "local": "[dim]local[/]",
        "profile-only": "[dim]profile-only[/]",
    }

    table = Table(title="Accounts & Profiles", show_header=True)
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("Kind", style="magenta")
    table.add_column("Provider", style="green")
    table.add_column("Model", style="yellow")
    table.add_column("Key", style="dim")
    table.add_column("Status", style="dim")
    if show_tags:
        table.add_column("Tags", style="dim")

    for e in entries:
        name_display = f"{e['name']} ★" if e["default"] else e["name"]
        row = [
            name_display,
            e["kind"],
            e["provider"],
            e["model"],
            _provider_key_cell(e["provider"], key_status),
            status_render.get(e["status"], f"[red]{e['status']}[/]"),
        ]
        if show_tags:
            row.append(", ".join(e["tags"]))
        table.add_row(*row)

    console.print(table)
    n_acct = sum(1 for e in entries if "account" in e["kind"])
    n_prof = sum(1 for e in entries if "profile" in e["kind"])
    console.print(
        f"\n[dim]Total: {len(entries)} ({n_acct} account(s), {n_prof} profile(s)) · "
        f"★ = default · use 'victor auth show <name>' for details[/]"
    )


# =============================================================================
# Show Command
# =============================================================================


@auth_app.command("show")
def auth_show(
    name: str = typer.Argument(..., help="Account name, profile name, or provider"),
    json_output: bool = create_json_option(),
) -> None:
    """Show what a name resolves to and whether its key is ready.

    Answers 'what will `--profile <name>` actually use?' — the resolved provider, model,
    endpoint, and whether the provider's API key is configured (and from where). Accepts an
    account name, a profile name, or a bare provider name.

    Example:
        victor auth show zai-coding
        victor auth show zai --json
    """
    from victor.framework.runtime_discovery import list_runtime_profiles

    manager = get_account_manager()
    acct = next((a for a in manager.list_accounts() if a.name == name), None)
    try:
        prof = next((p for p in list_runtime_profiles() if p.name == name), None)
    except Exception:
        prof = None
    try:
        key_status = APIKeyManager().get_status()
    except Exception:
        key_status = {}

    if acct or prof:
        provider = acct.provider if acct else prof.provider
        model = acct.model if acct else prof.model
        kind = "account+profile" if (acct and prof) else ("account" if acct else "profile")
    elif name in key_status or name in LOCAL_PROVIDERS:
        provider, model, kind = name, None, "provider"
    else:
        console.print(f"[red]✗[/] No account, profile, or provider named '{name}'.")
        console.print("[dim]Run 'victor auth list' to see available names.[/]")
        raise typer.Exit(1)

    st = key_status.get(provider) or {}
    key_configured = provider in LOCAL_PROVIDERS or bool(st.get("configured"))
    key_source = "local" if provider in LOCAL_PROVIDERS else st.get("source")
    endpoint = getattr(acct, "endpoint", None) if acct else None
    oauth = _oauth_status_value(acct) if (acct and acct.auth.method == "oauth") else None

    if json_output:
        print_json_data(
            {
                "name": name,
                "kind": kind,
                "provider": provider,
                "model": model,
                "endpoint": endpoint,
                "key_configured": key_configured,
                "key_source": key_source,
                "oauth_status": oauth,
            }
        )
        return

    lines = [
        f"[bold cyan]{name}[/]  [magenta]({kind})[/]",
        f"  provider : [green]{provider}[/]",
        f"  model    : [yellow]{model or '—'}[/]",
    ]
    if endpoint:
        lines.append(f"  endpoint : {endpoint}")
    lines.append(f"  key      : {_provider_key_cell(provider, key_status)}")
    if oauth:
        lines.append(f"  oauth    : {oauth}")
    if not key_configured and provider not in LOCAL_PROVIDERS:
        env_var = (st.get("env_var")) or f"{provider.upper()}_API_KEY"
        lines.append(
            f"  [dim]→ set a key with[/] victor auth add -p {provider} "
            f"[dim]or[/] export {env_var}=..."
        )
    console.print(Panel("\n".join(lines), title="auth show", border_style="cyan"))


# =============================================================================
# Env Command (load keys into the shell for headless / follow-on sessions)
# =============================================================================


def _stdout_is_tty() -> bool:
    """Whether stdout is an interactive terminal (extracted for testability)."""
    return sys.stdout.isatty()


@auth_app.command("env")
def auth_env(
    provider: Optional[str] = typer.Option(
        None, "--provider", "-p", help="Only emit this provider's key (recommended)"
    ),
) -> None:
    """Emit `export <VAR>=<key>` lines to load API keys into the environment.

    Use with eval so follow-on / headless sessions resolve keys from env (the keyring is
    skipped without a TTY, but env is checked first and always works):

        eval "$(victor auth env -p zai)"
        victor chat --profile zai-coding --headless -m "..."

    For safety this REFUSES to print keys to an interactive terminal — the export lines are
    only emitted when stdout is captured (eval, a pipe, or a file). Keys are never logged.
    """
    # Never display secrets on a real terminal; only emit when captured by eval/pipe/file.
    if _stdout_is_tty():
        suffix = f" -p {provider}" if provider else ""
        console.print(
            "[yellow]Refusing to print API keys to a terminal.[/] Run it through eval:\n"
            f'  [cyan]eval "$(victor auth env{suffix})"[/]'
        )
        raise typer.Exit(1)

    mgr = APIKeyManager()
    try:
        status = mgr.get_status()
    except Exception:
        status = {}

    targets = [provider] if provider else [p for p, s in status.items() if s.get("configured")]

    emitted = 0
    for p in targets:
        if p in LOCAL_PROVIDERS:
            continue
        # get_key (unlike the non-interactive get_api_key) reads env -> keyring -> file and
        # returns the value, so this works as the keyring -> env bridge it's meant to be.
        key = mgr.get_key(p)
        if not key:
            continue
        env_var = (status.get(p) or {}).get("env_var") or f"{p.upper()}_API_KEY"
        # Plain stdout for eval; shlex.quote guards shell metacharacters in the key value.
        print(f"export {env_var}={shlex.quote(key)}")
        emitted += 1

    if emitted == 0:
        # Diagnostics go to STDERR so a surrounding eval captures nothing.
        print(
            f"# victor auth env: no configured key for {provider or 'any cloud provider'}",
            file=sys.stderr,
        )
        raise typer.Exit(1)


# =============================================================================
# Remove Command
# =============================================================================


@auth_app.command("remove")
def auth_remove(
    name: str = typer.Argument(..., help="Account name to remove"),
) -> None:
    """Remove a provider account.

    Example:
        victor auth remove my-account
    """
    manager = get_account_manager()

    # Confirm
    account = manager.get_account(name)
    if not account:
        console.print(f"[red]✗[/] Account '{name}' not found")
        raise typer.Exit(1)

    console.print("[yellow]Removing account:[/]")
    console.print(f"  Name: {account.name}")
    console.print(f"  Provider: {account.provider}")
    console.print(f"  Model: {account.model}")

    if not Confirm.ask("Continue?", default=False):
        console.print("[yellow]Cancelled[/]")
        return

    # Remove account
    if manager.remove_account(name):
        console.print(f"[green]✓[/] Account '{name}' removed")
    else:
        console.print("[red]✗[/] Failed to remove account")
        raise typer.Exit(1)


# =============================================================================
# Migrate Command
# =============================================================================


@auth_app.command("migrate")
def auth_migrate(
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite existing config"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be done"),
) -> None:
    """Migrate from old configuration format.

    Migrates from:
      ~/.victor/profiles.yaml
      ~/.victor/api_keys.yaml

    To:
      ~/.victor/config.yaml

    Example:
        victor auth migrate
        victor auth migrate --dry-run
        victor auth migrate --force
    """
    if not check_migration_needed():
        console.print("[dim]No old configuration found.[/]")
        console.print("[green]✓[/] Already using new configuration format")
        return

    console.print("[cyan]Migrating to new configuration format...[/]")
    console.print()

    # Show what will be migrated
    migrator = ConfigMigrator(dry_run=True)

    if migrator.old_profiles_file.exists():
        console.print(f"[dim]Found: {migrator.old_profiles_file}[/]")
    if migrator.old_api_keys_file.exists():
        console.print(f"[dim]Found: {migrator.old_api_keys_file}[/]")

    console.print()
    console.print(f"[dim]New config: {migrator.new_config_file}[/]")

    if dry_run:
        console.print()
        console.print("[yellow]DRY RUN[/] - No changes will be made")
        return

    if not Confirm.ask("Continue?", default=True):
        console.print("[yellow]Cancelled[/]")
        return

    # Run migration
    result = run_migration(prompt=False, force=force)

    if result.success:
        console.print("[green]✓[/] Migration successful!")
        console.print(f"[dim]Accounts migrated: {result.migrated_accounts}[/]")
        console.print(f"[dim]API keys migrated: {result.migrated_keys}[/]")
        console.print(f"[dim]Backup: {result.backup_path}[/]")

        if result.warnings:
            console.print()
            console.print("[yellow]Warnings:[/]")
            for warning in result.warnings:
                console.print(f"  ⚠ {warning}")
    else:
        console.print("[red]✗[/] Migration failed")
        for error in result.errors:
            console.print(f"  ✗ {error}")
        raise typer.Exit(1)


# =============================================================================
# Test Command
# =============================================================================


@auth_app.command("test")
def auth_test(
    name: Optional[str] = typer.Option(None, "--name", "-n", help="Account name to test"),
    provider: Optional[str] = typer.Option(None, "--provider", "-p", help="Provider to test"),
) -> None:
    """Test a provider connection.

    Tests the connection to a provider account, validating:
    - Authentication configuration
    - API endpoint accessibility
    - Model availability

    Example:
        victor auth test
        victor auth test --name my-account
        victor auth test --provider anthropic
    """
    manager = get_account_manager()

    # Get account to test
    if name:
        account = manager.get_account(name)
        if not account:
            console.print(f"[red]✗[/] Account '{name}' not found")
            raise typer.Exit(1)
    elif provider:
        # First try as provider name
        accounts = [a for a in manager.list_accounts() if a.provider == provider]
        if accounts:
            account = accounts[0]
        else:
            # Fall back: treat as account name (user may have used -p instead of -n)
            account = manager.get_account(provider)
            if not account:
                console.print(f"[red]✗[/] No account or provider named '{provider}'")
                console.print("[dim]Hint: use --name/-n to test by account name[/]")
                raise typer.Exit(1)
            console.print(f"[dim]Matched account '{provider}' (hint: use -n for account names)[/]")
    else:
        # Use default account
        account = manager.get_account()
        if not account:
            console.print("[red]✗[/] No default account found")
            console.print("[dim]Run 'victor auth setup' to configure[/]")
            raise typer.Exit(1)

    console.print(f"[cyan]Testing connection to {account.provider}/{account.model}...[/]")
    console.print()

    # Run test
    validator = ConnectionValidator()
    result = validator.test_account_sync(account)

    # Display results
    if result.success:
        console.print("[green]✓[/] Connection successful!")
        console.print()

        # Show validation details
        for validation in result.validations:
            if validation.status == "success":
                console.print(f"  [green]✓[/] {validation.message}")
            elif validation.status == "warning":
                console.print(f"  [yellow]⚠[/] {validation.message}")
                if validation.details:
                    console.print(f"    [dim]{validation.details}[/]")

        if result.latency_ms:
            console.print()
            console.print(f"[dim]Latency: {result.latency_ms}ms[/]")
    else:
        console.print("[red]✗[/] Connection failed")
        console.print()

        if result.error:
            console.print(f"  [red]Error:[/] {result.error}")

        # Show validation details
        for validation in result.validations:
            if validation.status == "failed":
                console.print(f"  [red]✗[/] {validation.message}")
                if validation.details:
                    console.print(f"    [dim]{validation.details}[/]")
            elif validation.status == "warning":
                console.print(f"  [yellow]⚠[/] {validation.message}")

        raise typer.Exit(1)


# =============================================================================
# OAuth Commands (consolidated from victor providers auth)
# =============================================================================

OAUTH_LOGIN_PROVIDERS = ["openai", "qwen", "google", "github-copilot"]
OAUTH_SUPPORTED_PROVIDERS = [*OAUTH_LOGIN_PROVIDERS, "anthropic"]


@auth_app.command("login")
def auth_login(
    provider: str = typer.Argument(
        ..., help=f"Provider to authenticate ({', '.join(OAUTH_LOGIN_PROVIDERS)})"
    ),
    force: bool = typer.Option(
        False, "--force", "-f", help="Force re-authentication even if token is cached"
    ),
) -> None:
    """Log in to a provider via OAuth (opens browser).

    Example:
        victor auth login openai
        victor auth login qwen --force
    """
    provider = provider.lower()
    if provider not in OAUTH_LOGIN_PROVIDERS:
        console.print(
            f"[red]\u2717[/] OAuth not supported for '{provider}'. "
            f"Supported: {', '.join(OAUTH_LOGIN_PROVIDERS)}"
        )
        raise typer.Exit(1)

    from victor.core.async_utils import run_sync
    from victor.providers.oauth_manager import OAuthTokenManager

    async def _login():
        mgr = OAuthTokenManager(provider)
        if not force:
            cached = mgr._load_cached()
            if cached is not None and not cached.is_expired:
                console.print(f"[green]\u2713[/] Already authenticated with {provider}")
                console.print(
                    f"  Token expires: {cached.expires_at.strftime('%Y-%m-%d %H:%M UTC')}"
                )
                console.print("[dim]Use --force to re-authenticate[/]")
                return
        console.print(f"[cyan]Opening browser for {provider} OAuth login...[/]")
        try:
            token = await mgr.get_valid_token()
            if token:
                console.print(f"[green]\u2713[/] Successfully authenticated with {provider}")
            else:
                console.print(f"[red]\u2717[/] Authentication failed for {provider}")
                raise typer.Exit(1)
        except typer.Exit:
            raise
        except Exception as e:
            console.print(f"[red]\u2717[/] OAuth login failed: {e}")
            raise typer.Exit(1)

    run_sync(_login())


@auth_app.command("import-codex")
def auth_import_codex(
    provider: str = typer.Argument("openai", help="Provider to import for (currently: openai)"),
    codex_auth: Optional[Path] = typer.Option(
        None,
        "--codex-auth",
        help="Path to Codex auth.json (default: ~/.codex/auth.json)",
    ),
    storage_dir: Optional[Path] = typer.Option(
        None,
        "--storage-dir",
        help="Victor config directory (default: ~/.victor)",
    ),
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite existing Victor token"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Validate without writing tokens"),
) -> None:
    """Import OpenAI OAuth tokens from an existing Codex CLI login.

    This reuses local ChatGPT/Codex OAuth credentials without printing token values.

    Example:
        victor auth import-codex openai
        victor auth import-codex openai --force
    """
    provider = provider.lower()
    if provider != "openai":
        console.print("[red]✗[/] Codex OAuth import currently supports only OpenAI.")
        raise typer.Exit(1)

    codex_auth_path = codex_auth or Path.home() / ".codex" / "auth.json"
    victor_storage_dir = storage_dir or get_project_paths().global_victor_dir

    try:
        tokens = _load_codex_openai_tokens(codex_auth_path)
    except ValueError as e:
        console.print(f"[red]✗[/] {e}")
        raise typer.Exit(1) from e

    from victor.providers.oauth_manager import OAuthTokenManager

    manager = OAuthTokenManager(provider, storage_dir=victor_storage_dir)
    if manager._load_cached() is not None and not force:
        console.print("[yellow]Victor already has OpenAI OAuth tokens.[/]")
        console.print("[dim]Use --force to replace them, or --dry-run to validate Codex tokens.[/]")
        raise typer.Exit(1)

    console.print(f"[dim]Source: {codex_auth_path}[/]")
    console.print(f"[dim]Destination: {victor_storage_dir / 'oauth_tokens.yaml'}[/]")

    if dry_run:
        console.print("[green]✓[/] Codex OpenAI OAuth tokens are importable")
        return

    imported = manager.save_imported_tokens(tokens, overwrite=force)
    if not imported:
        console.print("[yellow]Victor already has OpenAI OAuth tokens.[/]")
        console.print("[dim]Use --force to replace them.[/]")
        raise typer.Exit(1)

    account = ProviderAccount(
        name="openai-oauth",
        provider="openai",
        model="gpt-5.4",
        auth=AuthConfig(method="oauth", source="keyring"),
    )
    account_manager = get_account_manager()
    if account_manager.get_account(name=account.name) is None:
        account_manager.save_account(account)
    profile_synced = _sync_profile_from_account(account, config_dir=victor_storage_dir)

    console.print("[green]✓[/] Imported OpenAI OAuth tokens from Codex")
    if profile_synced:
        console.print("[green]✓[/] Profile 'openai-oauth' added to profiles.yaml")
    console.print("[dim]Validate with: victor auth oauth-status openai[/]")
    console.print(
        "[dim]Add or select an OAuth account with: victor auth add --provider openai "
        "--model <model> --auth-method oauth --name <name>[/]"
    )


def _load_codex_openai_tokens(codex_auth_path: Path):
    """Load OpenAI OAuth tokens from Codex auth.json without exposing token values."""
    if not codex_auth_path.exists():
        raise ValueError(f"Codex auth file not found: {codex_auth_path}")

    try:
        data = json.loads(codex_auth_path.read_text())
    except json.JSONDecodeError as e:
        raise ValueError(f"Codex auth file is not valid JSON: {codex_auth_path}") from e

    tokens_data = data.get("tokens")
    if not isinstance(tokens_data, dict):
        raise ValueError("Codex auth file does not contain a tokens object")

    access_token = tokens_data.get("access_token")
    if not access_token:
        raise ValueError("Codex auth file does not contain an access_token")

    scopes = tokens_data.get("scopes") or tokens_data.get("scope") or []
    if isinstance(scopes, str):
        scopes = scopes.split()
    if not isinstance(scopes, list):
        scopes = []

    from victor.workflows.services.credentials import SSOTokens

    return SSOTokens(
        access_token=access_token,
        refresh_token=tokens_data.get("refresh_token"),
        id_token=tokens_data.get("id_token"),
        token_type=tokens_data.get("token_type", "Bearer"),
        expires_at=None,
        scopes=scopes,
    )


@auth_app.command("logout")
def auth_logout(
    provider: str = typer.Argument(
        ..., help=f"Provider to log out from ({', '.join(OAUTH_SUPPORTED_PROVIDERS)})"
    ),
) -> None:
    """Remove cached OAuth tokens for a provider.

    Example:
        victor auth logout openai
    """
    provider = provider.lower()
    if provider not in OAUTH_SUPPORTED_PROVIDERS:
        console.print(
            f"[red]\u2717[/] OAuth not supported for '{provider}'. "
            f"Supported: {', '.join(OAUTH_SUPPORTED_PROVIDERS)}"
        )
        raise typer.Exit(1)

    from victor.providers.oauth_manager import OAuthTokenManager

    mgr = OAuthTokenManager(provider)
    mgr.clear()
    console.print(f"[green]\u2713[/] Logged out from {provider} (tokens cleared)")


@auth_app.command("oauth-status")
def auth_oauth_status(
    provider: Optional[str] = typer.Argument(
        None, help="Check specific provider (or all if omitted)"
    ),
) -> None:
    """Show OAuth authentication status for providers.

    Example:
        victor auth oauth-status
        victor auth oauth-status openai
    """
    from victor.providers.oauth_manager import OAuthTokenManager

    providers_to_check = [provider.lower()] if provider else OAUTH_SUPPORTED_PROVIDERS

    table = Table(title="OAuth Authentication Status", show_header=True)
    table.add_column("Provider", style="cyan")
    table.add_column("Status")
    table.add_column("Expires")
    table.add_column("Token Store", style="dim")

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
                "stored",
            )
        else:
            table.add_row(
                prov,
                "[green]\u2713 Active[/]",
                (cached.expires_at.strftime("%Y-%m-%d %H:%M UTC") if cached.expires_at else ""),
                "stored",
            )

    console.print(table)
    console.print("\n[dim]Login: victor auth login <provider>[/]")


# =============================================================================
# Auth Setup Wizard
# =============================================================================


class AuthSetupWizard:
    """Interactive authentication setup wizard."""

    def __init__(self, console: Console, first_run: bool = False):
        """Initialize the wizard.

        Args:
            console: Rich console instance
            first_run: When True the wizard runs as the core of the first-run
                onboarding flow — the caller owns welcome/completion framing,
                and low-value prompts (account name, tags) are auto-answered
                to keep time-to-first-chat short.
        """
        self.console = console
        self.first_run = first_run
        self.account_manager = get_account_manager()
        self.state: Dict[str, Any] = {
            "local_providers": [],
            "selected_provider": None,
            "selected_model": None,
            "auth_method": None,
        }

    def run(self) -> int:
        """Run the setup wizard.

        Returns:
            Exit code (0 for success, 1 for error)
        """
        try:
            if not self.first_run:
                self._show_welcome()

            # Check for migration
            if self._check_migration():
                return 0

            # Step 1: Detect environment
            self._detect_environment()

            # Step 2: Choose provider type
            if not self._choose_provider_type():
                return 0

            # Step 3: Select provider and model
            if not self._select_provider_and_model():
                return 0

            # Steps 4-6: authentication → name → test, looping back to
            # credential entry when the connection test fails and the user
            # asks to try again.
            while True:
                if not self._configure_authentication():
                    return 0

                if not self._name_account():
                    return 0

                outcome = self._test_connection()
                if outcome == "ok":
                    break
                if outcome == "abort":
                    return 0
                # outcome == "retry": re-enter credentials

            # Step 7: Save account
            self._save_account()

            # Step 8: Complete
            if not self.first_run:
                self._show_completion()

            return 0

        except KeyboardInterrupt:
            self.console.print("\n\n[yellow]Setup cancelled.[/]")
            return 0
        except Exception as e:
            self.console.print(f"\n[red]✗[/] An error occurred: {e}")
            return 1

    def _show_welcome(self) -> None:
        """Display welcome screen."""
        welcome_text = Text()
        welcome_text.append("Victor Authentication Setup\n\n", style="bold cyan")
        welcome_text.append(
            "This wizard will help you configure a provider account.\n",
            style="white",
        )

        panel = Panel.fit(
            welcome_text,
            title="[bold cyan]Setup[/]",
            border_style="cyan",
            padding=(1, 2),
        )

        self.console.print(panel)
        self.console.print()

    def _check_migration(self) -> bool:
        """Check if migration is needed and offer to migrate.

        Returns:
            True if migration was performed, False otherwise
        """
        if not check_migration_needed():
            return False

        self.console.print("[yellow]⚠ Old configuration detected[/]")
        self.console.print("[dim]We've upgraded our configuration system.[/]")
        self.console.print()

        if Confirm.ask("Migrate to new format?", default=True):
            result = run_migration(prompt=False)
            if result.success:
                self.console.print(f"[green]✓[/] Migrated {result.migrated_accounts} accounts")
                self.console.print()
                return True
            else:
                self.console.print("[red]✗[/] Migration failed")
                for error in result.errors:
                    self.console.print(f"  {error}")
                return True

        return False

    def _detect_environment(self) -> None:
        """Detect local providers and environment."""
        self.console.print("[bold cyan]Step 1/6: Environment Detection[/]")
        self.console.print("─" * 50)
        self.console.print()

        # Check Ollama
        self.console.print("[yellow]🔍[/] Checking for Ollama...")
        if self._check_ollama():
            self.console.print("  [green]✓[/] Ollama is running")
            self.state["local_providers"].append("ollama")
        else:
            self.console.print("  [dim]Ollama not detected[/]")

        # Check LM Studio
        self.console.print("[yellow]🔍[/] Checking for LM Studio...")
        if self._check_lmstudio():
            self.console.print("  [green]✓[/] LM Studio is running")
            self.state["local_providers"].append("lmstudio")
        else:
            self.console.print("  [dim]LM Studio not detected[/]")

        self.console.print()

    def _choose_provider_type(self) -> bool:
        """Choose between local and cloud provider.

        Returns:
            False if user cancelled
        """
        self.console.print("[bold cyan]Step 2/6: Provider Type[/]")
        self.console.print("─" * 50)
        self.console.print()

        if self.state["local_providers"]:
            self.console.print("[green]Found local providers:[/]")
            for provider in self.state["local_providers"]:
                self.console.print(f"  • {provider.capitalize()}")
            self.console.print()

            if Confirm.ask("Use a local provider?", default=False):
                self.state["provider_type"] = "local"
                return True

        self.console.print("[cyan]Cloud providers[/]")
        self.console.print("  • Anthropic (Claude)")
        self.console.print("  • OpenAI (GPT)")
        self.console.print("  • Google (Gemini)")
        self.console.print("  • ZhipuAI (GLM)")
        self.console.print("  • And 18 more...")
        self.console.print()

        if not Confirm.ask("Use a cloud provider?", default=True):
            return False

        self.state["provider_type"] = "cloud"
        return True

    def _select_provider_and_model(self) -> bool:
        """Select provider and model.

        Returns:
            False if user cancelled
        """
        self.console.print("[bold cyan]Step 3/6: Provider & Model[/]")
        self.console.print("─" * 50)
        self.console.print()

        if self.state["provider_type"] == "local":
            # Use first available local provider
            provider = self.state["local_providers"][0]
            self.state["selected_provider"] = provider
            self.state["selected_model"] = "default"
            self.console.print(f"[green]✓[/] Using {provider.capitalize()}")
            return True

        # Cloud provider selection
        self.console.print("[cyan]Popular providers:[/]")
        popular = ["anthropic", "openai", "google", "zai"]
        for i, p in enumerate(popular, 1):
            self.console.print(f"  {i}. {p.capitalize()}")

        self.console.print("  0. Other")
        self.console.print()

        choice = Prompt.ask(
            "Select provider",
            choices=[str(i) for i in range(len(popular) + 1)],
            default="1",
        )

        if choice == "0":
            # Show all providers
            provider = Prompt.ask("Enter provider name")
        else:
            provider = popular[int(choice) - 1]

        self.state["selected_provider"] = provider.lower()

        # Model selection
        if provider.lower() in POPULAR_MODELS:
            models = POPULAR_MODELS[provider.lower()]
            self.console.print(f"\n[cyan]Popular {provider.capitalize()} models:[/]")
            for i, model in enumerate(models, 1):
                self.console.print(f"  {i}. {model}")
            self.console.print("  0. Other (enter model name)")
            self.console.print()

            model_choice = Prompt.ask(
                "Select model",
                choices=[str(i) for i in range(len(models) + 1)],
                default="1",
            )

            if model_choice != "0":
                self.state["selected_model"] = models[int(model_choice) - 1]
            else:
                self.state["selected_model"] = Prompt.ask("Enter model name")
        else:
            self.state["selected_model"] = Prompt.ask("Enter model name")

        self.console.print()
        self.console.print(f"[green]✓[/] Selected: {provider}/{self.state['selected_model']}")
        return True

    def _configure_authentication(self) -> bool:
        """Configure authentication.

        Returns:
            False if user cancelled
        """
        self.console.print()
        self.console.print("[bold cyan]Step 4/6: Authentication[/]")
        self.console.print("─" * 50)
        self.console.print()

        provider = self.state["selected_provider"]

        # Local providers don't need auth
        if provider in AccountManager.LOCAL_PROVIDERS:
            self.state["auth_method"] = "none"
            self.console.print("[green]✓[/] No authentication required")
            return True

        # Check if OAuth is available
        if provider in OAUTH_PROVIDERS:
            self.console.print(f"[cyan]OAuth is available for {provider.capitalize()}[/]")
            use_oauth = Confirm.ask(
                "Use OAuth (recommended)?",
                default=True,
            )

            if use_oauth:
                self.state["auth_method"] = "oauth"
                self.console.print("[green]✓[/] Will use OAuth")
                return True

        # Default to API key
        self.state["auth_method"] = "api_key"
        self.console.print("[cyan]Enter your API key[/]")
        self.console.print("[dim]Your key will be stored securely in the system keyring[/]")

        api_key = Prompt.ask("API key", password=True)
        self.state["api_key"] = api_key

        self.console.print("[green]✓[/] API key received")
        return True

    def _name_account(self) -> bool:
        """Name the account.

        Returns:
            False if user cancelled
        """
        # Suggest a name
        provider = self.state["selected_provider"]
        model = self.state["selected_model"]

        if ":" in model:
            base_model, variant = model.rsplit(":", 1)
            suggestion = f"{provider}-{variant}"
        else:
            suggestion = provider

        if self.first_run:
            # Keep first run short: accept the suggested name, no tags.
            self.state["account_name"] = suggestion
            self.state["tags"] = []
            return True

        self.console.print()
        self.console.print("[bold cyan]Step 5/6: Account Name[/]")
        self.console.print("─" * 50)
        self.console.print()

        name = Prompt.ask("Account name", default=suggestion)
        self.state["account_name"] = name

        # Ask for tags
        tags_input = Prompt.ask(
            "Tags (comma-separated, optional)",
            default="",
        )
        self.state["tags"] = [t.strip() for t in tags_input.split(",") if t.strip()]

        self.console.print(f"[green]✓[/] Account name: {name}")
        return True

    def _test_connection(self) -> str:
        """Test the connection.

        Returns:
            "ok" on success, "retry" to loop back to credential entry,
            "abort" to exit the wizard.
        """
        self.console.print()
        self.console.print("[bold cyan]Step 6/6: Test Connection[/]")
        self.console.print("─" * 50)
        self.console.print()

        provider = self.state["selected_provider"]
        model = self.state["selected_model"]
        auth_method = self.state["auth_method"]

        self.console.print(f"[cyan]Testing {provider}/{model}...[/]")
        self.console.print()

        # Create test account
        auth = AuthConfig(method=auth_method, source="keyring")
        if auth_method == "api_key" and "api_key" in self.state:
            auth.value = self.state["api_key"]

        account = ProviderAccount(
            name="test",
            provider=provider,
            model=model,
            auth=auth,
            tags=self.state.get("tags", []),
        )

        # Run test
        validator = ConnectionValidator()
        result = validator.test_account_sync(account)

        # Show results
        if result.success:
            self.console.print("[green]✓ Connection successful![/]")
            return "ok"

        self.console.print("[red]✗ Connection failed[/]")
        if result.error:
            self.console.print(f"  Error: {result.error}")
        self.console.print("[dim]Run 'victor doctor' for a full environment diagnosis.[/]")

        self.console.print()
        if Confirm.ask("Try again with different credentials?", default=False):
            return "retry"

        self.console.print("[yellow]Setup not completed.[/] Resume anytime with: victor auth setup")
        return "abort"

    def _save_account(self) -> None:
        """Save the account."""
        self.console.print()
        self.console.print("[bold cyan]Saving Account[/]")
        self.console.print("─" * 50)
        self.console.print()

        # Create account
        auth = AuthConfig(method=self.state["auth_method"], source="keyring")
        if self.state["auth_method"] == "api_key" and "api_key" in self.state:
            auth.value = self.state["api_key"]

        account = ProviderAccount(
            name=self.state["account_name"],
            provider=self.state["selected_provider"],
            model=self.state["selected_model"],
            auth=auth,
            tags=self.state.get("tags", []),
        )

        # Save to config and matching chat profile
        self.account_manager.save_account(account)
        profile_synced = _sync_profile_from_account(account)
        # Expose the persisted account so wrapping flows (first-run
        # onboarding) can re-sync the chat profile after rewriting
        # profiles.yaml.
        self.state["saved_account"] = account

        # Save API key to keyring
        if self.state["auth_method"] == "api_key" and "api_key" in self.state:
            try:
                from victor.config.api_keys import _set_key_in_keyring

                _set_key_in_keyring(
                    self.state["selected_provider"],
                    self.state["api_key"],
                )
                self.console.print("[green]✓[/] API key saved to keyring")
            except Exception as e:
                self.console.print(f"[yellow]⚠[/] Could not save to keyring: {e}")

        # Set as default if it's the first account
        accounts = self.account_manager.list_accounts()
        if len(accounts) == 1:
            config = self.account_manager.load_config()
            config.defaults.account = account.name
            self.account_manager.save_config(config)
            self.console.print("[green]✓[/] Set as default account")

        self.console.print(f"[green]✓[/] Account '{account.name}' saved")
        if profile_synced:
            self.console.print(f"[green]✓[/] Profile '{account.name}' added to profiles.yaml")

    def _show_completion(self) -> None:
        """Show completion message."""
        self.console.print()
        self.console.print("[bold green]✓ Setup Complete![/]")
        self.console.print()
        self.console.print("[cyan]Next steps:[/]")
        self.console.print("  • Start chatting: victor chat")
        self.console.print("  • List accounts: victor auth list")
        self.console.print("  • Test connection: victor auth test")
        self.console.print()

    # ========================================================================
    # Helper methods
    # ========================================================================

    def _check_ollama(self) -> bool:
        """Check if Ollama is available."""
        try:
            result = subprocess.run(
                ["ollama", "--version"],
                capture_output=True,
                timeout=2,
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def _check_lmstudio(self) -> bool:
        """Check if LM Studio is available."""
        import socket

        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex(("127.0.0.1", 1234))
            sock.close()
            return result == 0
        except Exception:
            return False
