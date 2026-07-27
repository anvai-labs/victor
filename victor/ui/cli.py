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

"""Command-line interface for Victor - Open-source agentic AI framework."""

import click
import typer
from rich.console import Console
from typing import Optional

# NOTE: ``run_sync`` is imported function-locally in ``run_command`` (see below);
# keeping it out of module scope avoids pulling ``victor.core`` eagerly at CLI
# import time.

try:
    from victor import __version__
except (ImportError, AttributeError):
    # Handle shadowing/namespace issues
    try:
        from importlib.metadata import version as _v

        __version__ = _v("victor-ai")
    except Exception:
        try:
            from .. import __version__
        except (ImportError, ValueError):
            __version__ = "0.0.0"

from victor.core.utils.capability_loader import load_coding_analyze_app
from victor.ui.commands.benchmark import benchmark_app
from victor.ui.commands.bayesian import bayesian_app
from victor.ui.commands.capabilities import capabilities_app
from victor.ui.commands.chat import (
    chat_app,
    _run_default_interactive,
)
from victor.ui.commands.config import config_app
from victor.ui.commands.doctor import run_doctor
from victor.ui.commands.dashboard import dashboard_app
from victor.ui.commands.gateway import gateway_app
from victor.ui.commands.docs import docs_app
from victor.ui.commands.observability import app as observability_app
from victor.ui.commands.embeddings import embeddings_app
from victor.ui.commands.examples import examples_app
from victor.ui.commands.experiments import experiment_app
from victor.ui.commands.fep import fep_app
from victor.ui.commands.graph import graph_app
from victor.ui.commands.index import index_app
from victor.ui.commands.init import init_app
from victor.ui.commands.ml import ml_app


# Lazy imports for deprecated/hidden commands (startup performance)
def _get_keys_app():
    """Lazy import for deprecated keys command."""
    from victor.ui.commands.keys import keys_app

    return keys_app


def _get_test_provider_app():
    """Lazy import for hidden test_provider command."""
    from victor.ui.commands.test_provider import test_provider_app

    return test_provider_app


# Regular imports (alphabetically ordered)
from victor.ui.commands.mcp import mcp_app
from victor.ui.commands.models import models_app
from victor.ui.commands.profiles import profiles_app
from victor.ui.commands.providers import providers_app
from victor.ui.commands.security import security_app
from victor.ui.commands.serve import serve_app
from victor.ui.commands.ui import ui_app
from victor.ui.commands.tools import tools_app
from victor.ui.commands.scaffold import scaffold_app
from victor.ui.commands.scheduler import scheduler_app
from victor.ui.commands.sessions import sessions_app
from victor.ui.commands.db import db_app
from victor.ui.commands.vertical import vertical_app
from victor.ui.commands.plugin import plugin_app
from victor.ui.commands.skills import skills_app
from victor.ui.commands.workflow import workflow_app
from victor.ui.commands.ab_testing import ab_app

# Import new unified auth command
from victor.ui.commands.auth import auth_app

# Import verification command
from victor.ui.commands.verify import app as verify_app
from victor.ui.cli_group import SuggestingGroup, suggest_command
from victor.ui.theme import victor_theme

app = typer.Typer(
    name="victor",
    help="Victor - Open-source agentic AI framework with multi-provider support.",
    cls=SuggestingGroup,
    add_completion=False,
)

console = Console(theme=victor_theme)


@app.command(
    "help",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def help_command(ctx: typer.Context) -> None:
    """Show help for Victor, or for one of its commands.

    A ``help`` command is what most CLIs expose, but Typer does not register one
    by default. This mirrors ``--help`` and additionally descends into a
    subcommand so ``victor help db vacuum`` works.

    \b
    Examples:
      victor help             # top-level help
      victor help db          # help for the 'db' group
      victor help db vacuum   # help for the 'db vacuum' command
    """
    root_ctx = ctx.parent
    root_group = root_ctx.command
    tokens = [t for t in ctx.args if not t.startswith("-")]

    if not tokens:
        typer.echo(root_ctx.get_help())
        raise typer.Exit()

    current_cmd = root_group
    current_ctx = root_ctx
    for name in tokens:
        if not isinstance(current_cmd, typer.core.TyperGroup):
            raise typer.BadParameter(
                f"'{name}' is not valid: '{current_cmd.name}' has no subcommands."
            )
        nxt = current_cmd.get_command(current_ctx, name)
        if nxt is None:
            hint = suggest_command(current_ctx, name, current_cmd)
            raise typer.BadParameter(hint or f"No such command '{name}'.")
        current_cmd = nxt
        current_ctx = click.Context(nxt, info_name=name, parent=current_ctx)

    typer.echo(current_cmd.get_help(current_ctx))
    raise typer.Exit()


# Define doctor_command BEFORE registering it
@app.command("doctor")
def doctor_command(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed diagnostic output"),
    fix: bool = typer.Option(False, "--fix", "-f", help="Automatically fix common issues"),
    config: bool = typer.Option(
        False,
        "--config",
        "-c",
        help="Also validate configuration files (profiles.yaml, settings)",
    ),
    providers: bool = typer.Option(
        False,
        "--providers",
        "-p",
        help="Also check provider connectivity",
    ),
) -> None:
    """Run system diagnostics and troubleshooting.

    Performs comprehensive system checks:
    - Python version and environment
    - Dependencies and packages
    - Provider connectivity and configuration
    - Tool availability
    - File permissions
    - Performance optimizations

    Use --config to include configuration validation.
    Use --providers to include provider health checks.
    """
    exit_code = run_doctor(verbose=verbose, fix=fix)

    if config:
        console.print("\n[bold]Configuration Validation[/]")
        console.print("-" * 50)
        try:
            from victor.config.settings import load_settings
            from victor.config.validation import (
                validate_configuration,
                format_validation_result,
            )

            settings = load_settings()
            result = validate_configuration(settings)
            print(format_validation_result(result))
            if not result.is_valid():
                exit_code = 1
        except Exception as e:
            console.print(f"[red]\u2717 Config validation failed:[/] {e}")
            exit_code = 1

    if providers:
        console.print("\n[bold]Provider Health Checks[/]")
        console.print("-" * 50)
        try:
            from victor.config.settings import load_settings

            settings = load_settings()
            profiles = settings.load_profiles()

            # Collect unique providers from profiles
            seen_providers = set()
            for _name, profile_config in profiles.items():
                prov = profile_config.provider.lower()
                if prov not in seen_providers:
                    seen_providers.add(prov)

            for prov in sorted(seen_providers):
                try:
                    from victor.providers.registry import ProviderRegistry

                    if prov in ProviderRegistry.list_providers():
                        console.print(f"  [green]\u2713[/] {prov}: registered")
                    else:
                        console.print(f"  [yellow]?[/] {prov}: not in registry")
                except Exception:
                    console.print(f"  [red]\u2717[/] {prov}: check failed")
        except Exception as e:
            console.print(f"[red]\u2717 Provider check failed:[/] {e}")
            exit_code = 1

    raise typer.Exit(exit_code)


@app.command("run")
def run_command(
    message: str = typer.Argument(..., help="Message to send"),
    profile: str = typer.Option("default", "-p", "--profile", help="Profile to use"),
    provider: Optional[str] = typer.Option(None, "--provider", help="Override provider"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Override model"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
    code_only: bool = typer.Option(False, "--code-only", help="Extract only code blocks"),
    stdin: bool = typer.Option(False, "--stdin", help="Read input from stdin"),
    input_file: Optional[str] = typer.Option(
        None, "-f", "--input-file", help="Read input from file"
    ),
) -> None:
    """One-shot query with plain output (non-interactive, pipe-friendly).

    Examples:
        victor run "explain this function"
        victor run -p deepseek "write a hello world"
        echo "explain X" | victor run --stdin
        victor run --code-only "write hello.py" > hello.py
    """
    import sys

    from victor.config.settings import load_settings
    from victor.core.async_utils import run_sync
    from victor.framework.session_config import SessionConfig
    from victor.ui.commands.chat import run_oneshot
    from victor.ui.commands.utils import setup_logging

    # Read from stdin if requested
    if stdin:
        message = sys.stdin.read().strip()
    elif input_file:
        with open(input_file) as f:
            message = f.read().strip()

    if not message:
        console.print("[bold red]Error:[/] No message provided.")
        raise typer.Exit(1)

    setup_logging(command="run", cli_log_level="ERROR")
    settings = load_settings()
    try:
        session_config = SessionConfig.from_cli_flags(
            agent_profile=profile,
            provider=provider,
            model=model,
        )
    except ValueError as exc:
        console.print(f"[bold red]Error:[/] {exc}")
        raise typer.Exit(1)

    run_sync(
        run_oneshot(
            message=message,
            settings=settings,
            profile=profile,
            stream=False,
            thinking=False,
            formatter=None,
            preindex=False,
            renderer_choice="text",
            show_reasoning=False,
            session_config=session_config,
        )
    )


@app.command("onboarding")
def onboarding_command(
    force: bool = typer.Option(
        False, "--force", "-f", help="Re-run onboarding even if completed before"
    ),
) -> None:
    """Run the interactive onboarding wizard for first-time setup.

    The onboarding wizard guides you through:
    - Environment detection (Ollama, LM Studio, API keys)
    - Provider and model selection
    - Credential entry (stored in the system keyring)
    - A live connection test
    - Default profile installation and example prompts

    Use --force to re-run onboarding even if you've completed it before.
    """
    from victor.ui.commands.onboarding import run_onboarding
    from victor.config.settings import is_first_time_user

    # Check if onboarding was already completed
    if not force and not is_first_time_user():
        console.print("\n[yellow]⚠[/] Onboarding already completed.")
        console.print("")
        console.print("To re-run onboarding, use:")
        console.print("  [cyan]victor onboarding --force[/]")
        console.print("")
        console.print("Or reconfigure your profile with:")
        console.print("  [cyan]victor profiles list[/]")
        console.print("  [cyan]victor profiles set-default <profile>[/]")
        return

    # Run the onboarding wizard (it owns welcome/completion framing)
    exit_code = run_onboarding()

    if exit_code != 0:
        console.print("\n[yellow]Onboarding interrupted.[/]")
        console.print("Run [cyan]victor onboarding[/] again to complete setup.")

    raise typer.Exit(exit_code)


# =============================================================================
# Backward Compatibility Aliases
# =============================================================================


@app.command("profile")
def profile_alias(
    verb: str = typer.Argument(
        ...,
        help="Subcommand (list, show, apply, current, create, edit, delete, set-default)",
    ),
    args: Optional[list[str]] = typer.Argument(
        None, help="Additional arguments for the subcommand"
    ),
) -> None:
    """Backward-compatible alias for 'victor profiles' commands.

    This command exists for backward compatibility and redirects to
    the canonical 'victor profiles' command group.

    Deprecated: Use 'victor profiles <verb>' directly instead.
    """
    import sys
    import warnings

    # Issue deprecation warning
    warnings.warn(
        "'victor profile' is deprecated, use 'victor profiles' instead. "
        "Example: 'victor profiles list' instead of 'victor profile list'",
        DeprecationWarning,
        stacklevel=2,
    )

    # Reinvoke as "victor profiles <verb> <args>"
    # Reconstruct sys.argv for the profiles command
    sys.argv = ["victor", "profiles", verb] + (args or [])

    # Import and run the profiles app directly
    from victor.ui.commands.profiles import profiles_app

    try:
        profiles_app()
    except SystemExit as e:
        # Propagate the exit code from profiles app
        sys.exit(e.code if e.code is not None else 0)


# =============================================================================
# Register all subcommands with rich_help_panel grouping
# =============================================================================

# --- Core Commands (default panel, shown first) ---
app.add_typer(chat_app)
app.add_typer(auth_app, name="auth", help="Manage authentication and provider accounts.")
app.add_typer(init_app)
app.add_typer(models_app)
app.add_typer(ui_app, name="ui", help="Launch the Victor web chat UI (Chainlit).")

# --- Verb groups (UX P2): satellites nest under their owning verb; the old
# flat names stay invocable below as hidden aliases with a migration hint. ---
config_app.add_typer(profiles_app, name="profiles")
config_app.add_typer(providers_app)

index_app.add_typer(graph_app)
index_app.add_typer(embeddings_app)

tools_app.add_typer(capabilities_app)
tools_app.add_typer(examples_app)

observe_app = typer.Typer(name="observe", help="Observability, benchmarking, and experiments.")
observe_app.add_typer(dashboard_app)
observe_app.add_typer(bayesian_app)
observe_app.add_typer(observability_app, name="observability")
observe_app.add_typer(gateway_app, name="gateway")
observe_app.add_typer(benchmark_app)
observe_app.add_typer(experiment_app)
observe_app.add_typer(ab_app)
observe_app.add_typer(ml_app)

data_app = typer.Typer(name="data", help="Sessions, databases, and stored artifacts.")
data_app.add_typer(sessions_app)
data_app.add_typer(db_app)

# --- Configuration ---
app.add_typer(config_app, rich_help_panel="Configuration")

# --- Development ---
app.add_typer(tools_app, rich_help_panel="Development")
app.add_typer(skills_app, rich_help_panel="Development")
app.add_typer(workflow_app, rich_help_panel="Development")
app.add_typer(index_app, rich_help_panel="Development")
app.add_typer(plugin_app, rich_help_panel="Development")
app.add_typer(serve_app, rich_help_panel="Development")
app.add_typer(mcp_app, rich_help_panel="Development")
app.add_typer(verify_app, name="verify", rich_help_panel="Development")

# --- Observability / Data verbs ---
app.add_typer(observe_app, rich_help_panel="Observability")
app.add_typer(data_app, rich_help_panel="Data & Sessions")

# --- Documentation ---
app.add_typer(docs_app, rich_help_panel="Documentation")
vertical_app.add_typer(scaffold_app, name="scaffold")
app.add_typer(vertical_app, rich_help_panel="Documentation")

# --- Advanced ---
app.add_typer(security_app, rich_help_panel="Advanced")
app.add_typer(fep_app, rich_help_panel="Advanced")
coding_app = typer.Typer(name="coding", help="Coding vertical specialized commands.")
app.add_typer(coding_app, rich_help_panel="Advanced")

# --- Legacy flat aliases (UX P2): hidden but fully functional. Resolution
# prints a one-line pointer to the new verb path (SuggestingGroup). Remove
# after one minor release per the deprecation policy. ---
_LEGACY_VERB_PATHS = {
    # keys are the apps' intrinsic top-level names (some are singular)
    "profiles": "config profiles",
    "provider": "config provider",
    "graph": "index graph",
    "embedding": "index embedding",
    "capability": "tool capability",
    "examples": "tool examples",
    "dashboard": "observe dashboard",
    "bayesian": "observe bayesian",
    "observability": "observe observability",
    "gateway": "observe gateway",
    "benchmark": "observe benchmark",
    "experiment": "observe experiment",
    "ab": "observe ab",
    "ml": "observe ml",
    "session": "data session",
    "db": "data db",
}
SuggestingGroup.legacy_paths = dict(_LEGACY_VERB_PATHS)

app.add_typer(profiles_app, name="profiles", hidden=True)
app.add_typer(providers_app, hidden=True)
app.add_typer(graph_app, hidden=True)
app.add_typer(embeddings_app, hidden=True)
app.add_typer(capabilities_app, hidden=True)
app.add_typer(examples_app, hidden=True)
app.add_typer(dashboard_app, hidden=True)
app.add_typer(bayesian_app, hidden=True)
app.add_typer(observability_app, name="observability", hidden=True)
app.add_typer(gateway_app, name="gateway", hidden=True)
app.add_typer(benchmark_app, hidden=True)
app.add_typer(experiment_app, hidden=True)
app.add_typer(ab_app, hidden=True)
app.add_typer(ml_app, hidden=True)
app.add_typer(sessions_app, hidden=True)
app.add_typer(db_app, hidden=True)


# --- Deprecated / Hidden (lazy-loaded for startup performance) ---
# Lazy loading wrapper for deprecated/hidden commands
class LazyTyper:
    """Lazy wrapper for Typer apps to defer import until first use."""

    def __init__(self, import_func):
        self._import_func = import_func
        self._app = None

    @property
    def app(self):
        if self._app is None:
            self._app = self._import_func()
        return self._app

    def __getattr__(self, name):
        return getattr(self.app, name)


# Register lazy-loaded deprecated/hidden commands
app.add_typer(LazyTyper(_get_keys_app), deprecated=True)
app.add_typer(LazyTyper(_get_test_provider_app), hidden=True)


def _register_plugin_commands():
    """Discover and register CLI commands from plugins."""
    try:
        from victor.core.plugins.registry import PluginRegistry
        from victor.core.bootstrap import ensure_bootstrapped

        # Ensure the framework is bootstrapped, which triggers plugin registration
        ensure_bootstrapped()

        registry = PluginRegistry.get_instance()

        # 1. Register commands from PluginContext (New method)
        analyze_registered = False
        if registry.context:
            for name, sub_app in registry.context.commands.items():
                try:
                    if name == "analyze":
                        analyze_registered = True
                        # Register under both coding subcommand AND top level
                        coding_app.add_typer(sub_app, name=name)
                        app.add_typer(sub_app, name=name)
                    else:
                        # Register command under its specified name at top level
                        app.add_typer(sub_app, name=name)
                except Exception as e:
                    console.print(f"[yellow]Warning:[/] Failed to load CLI command '{name}': {e}")

        # 2. Register apps from get_cli_app (Legacy method)
        for plugin in registry.discover():
            if plugin.name == "coding":
                # Skip coding as it's already registered via the new method
                continue

            try:
                sub_app = plugin.get_cli_app()
                if sub_app:
                    # Legacy hook often doesn't specify name, uses app name or plugin name
                    app.add_typer(sub_app)
            except Exception as e:
                console.print(
                    f"[yellow]Warning:[/] Failed to load legacy CLI app from plugin '{plugin.name}': {e}"
                )

        # 3. Fallback: register analyze from contrib if plugin context didn't provide it
        if not analyze_registered:
            try:
                _analyze_fallback = load_coding_analyze_app()

                coding_app.add_typer(_analyze_fallback, name="analyze")
                app.add_typer(_analyze_fallback, name="analyze")
            except (ImportError, Exception):
                pass  # Neither plugin nor contrib available
    except Exception as e:
        # Surface plugin loading failures as warnings (Phase 4 UX improvement)
        console.print(f"[yellow]Warning:[/] Failed to load plugin commands: {e}\n")
        pass


# Register plugin commands after builtin ones
_register_plugin_commands()


def version_callback(value: bool) -> None:
    """Show version and exit."""
    if value:
        console.print(f"Victor v{__version__}")
        raise typer.Exit()


@app.callback(invoke_without_command=True)
def callback(
    ctx: typer.Context,
    version: Optional[bool] = typer.Option(
        None,
        "--version",
        "-v",
        callback=version_callback,
        is_eager=True,
        help="Show version and exit",
    ),
    skip_onboarding: bool = typer.Option(
        False,
        "--skip-onboarding",
        "--no-setup",
        help="Skip first-time setup wizard",
    ),
) -> None:
    """Victor - Open-source agentic AI framework with multi-provider support."""
    # Install in-process hang/crash diagnostics as early as possible so a wedged
    # session can be introspected with `kill -USR1 <pid>` (no root / py-spy needed,
    # which matters on macOS hardened-runtime Python where attach is blocked).
    from victor.ui.diagnostics import install_fault_diagnostics

    install_fault_diagnostics()

    if ctx.invoked_subcommand is None:
        # Check if this is a first-time user
        if not skip_onboarding:
            from victor.config.settings import is_first_time_user

            if is_first_time_user():
                from victor.ui.commands.onboarding import run_onboarding

                # offer_chat=False: this path always drops into chat below.
                exit_code = run_onboarding(offer_chat=False)
                if exit_code != 0:
                    console.print("\n[yellow]Onboarding interrupted. Starting chat anyway...[/]")
                else:
                    console.print("\n[green]✓[/] Setup complete! Starting chat...\n")

        _run_default_interactive()


if __name__ == "__main__":
    app()
