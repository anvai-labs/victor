from __future__ import annotations

import asyncio
import logging
import typer
from typer.models import ArgumentInfo, OptionInfo
import importlib
import os
import shutil
import sys
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Optional, Any
from pathlib import Path

from rich import box
from rich.console import Console, Group
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table
from rich.text import Text

# ✅ PROPER: Import SessionConfig and framework-owned session runner
from victor.framework.session_config import SessionConfig
from victor.framework.session_runner import FrameworkSessionRunner
from victor.config.settings import get_project_paths, load_settings
from victor.core.async_utils import run_sync

# Framework-first boundary: UI composes VictorClient/SessionConfig only.
from victor.core.verticals import get_vertical, list_verticals
from victor.core.errors import (
    ConfigurationError,
    ProviderError,
    ProviderConnectionError,
    ProviderAuthError,
    ProviderNotFoundError,
    VictorError,
)
from victor.ui.output_formatter import InputReader, create_formatter
from victor.ui.history_utils import (
    count_prompt_toolkit_history_entries,
    load_input_history_from_db,
    sanitize_prompt_toolkit_history_file,
)
from victor.ui.delegate_follow_up import (
    DelegateFollowUpContractError,
    load_delegate_follow_up_contract_file,
)
from victor.ui.rendering.utils import render_status_message
from victor.ui.commands.utils import (
    preload_semantic_index,
    check_codebase_index,
    get_rl_profile_suggestion,
    setup_safety_confirmation,
    setup_logging,
    graceful_shutdown,
)
from victor.workflows import (
    load_workflow_from_file,
    YAMLWorkflowError,
    StateGraphExecutor,
    ExecutorConfig,
)
from victor.workflows.visualization import (
    WorkflowVisualizer,
    OutputFormat as VizFormat,
    RenderBackend,
    get_available_backends,
)

# Contextual error formatting
try:
    from victor.framework.contextual_errors import format_exception_for_user
except ImportError:
    # Fallback if framework module is not available
    def format_exception_for_user(e):
        return str(e)


chat_app = typer.Typer(
    name="chat",
    help="""Start interactive chat or send a one-shot message.

    **Basic Usage:**
        victor chat                    # Start interactive chat
        victor chat "Hello, Victor!"    # Send one-shot message
        /shortcuts                     # Show interactive keyboard shortcuts

    **Advanced Options:**
        Use --help-full to see all 37 options organized by category.
        Workflow options: Use 'victor workflow' command instead.
        Session options: Use 'victor sessions' command instead.
    """,
)
console = Console()

DEFAULT_CODING_AGENT_MODES = ("build", "plan", "review", "delegate")
ADVANCED_CODING_AGENT_MODES = ("explore",)
ALL_CODING_AGENT_MODES = DEFAULT_CODING_AGENT_MODES + ADVANCED_CODING_AGENT_MODES


def normalize_chat_mode(mode: Optional[str]) -> Optional[str]:
    """Normalize and validate the narrow coding-agent mode surface."""
    if mode is None:
        return None

    normalized = mode.strip().lower()
    if not normalized:
        return None

    if normalized not in ALL_CODING_AGENT_MODES:
        default_modes = ", ".join(DEFAULT_CODING_AGENT_MODES)
        advanced_modes = ", ".join(ADVANCED_CODING_AGENT_MODES)
        raise ValueError(
            "Invalid mode. Choose one of default modes: "
            f"{default_modes}. Advanced opt-in modes: {advanced_modes}."
        )

    return normalized


def _resolve_typer_default(value: Any) -> Any:
    """Unwrap Typer metadata objects when callback functions are invoked directly."""
    if isinstance(value, (OptionInfo, ArgumentInfo)):
        return value.default
    return value


def __getattr__(name: str) -> Any:
    """Provide lazy compatibility shims for legacy test and patch targets."""
    if name == "AgentOrchestrator":
        module = importlib.import_module("victor.agent.orchestrator")
        return module.AgentOrchestrator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _collect_runtime_override_kwargs(
    *,
    compaction_threshold: Optional[float],
    adaptive_threshold: Optional[bool],
    compaction_min_threshold: Optional[float],
    compaction_max_threshold: Optional[float],
    enable_smart_routing: bool,
    routing_profile: str,
    fallback_chain: Optional[str],
    tool_preview: bool,
    enable_pruning: bool,
    # Bayesian orchestration parameters
    enable_bayesian: bool,
    force_bayesian: bool,
    simple_threshold: float,
    complex_threshold: float,
    enable_voi: bool,
    enable_correlation: bool,
    min_agents_for_bayesian: int,
) -> dict[str, Any]:
    """Return only non-default runtime override kwargs for chat execution helpers."""
    kwargs: dict[str, Any] = {}

    if compaction_threshold is not None:
        kwargs["compaction_threshold"] = compaction_threshold
    if adaptive_threshold is not None:
        kwargs["adaptive_threshold"] = adaptive_threshold
    if compaction_min_threshold is not None:
        kwargs["compaction_min_threshold"] = compaction_min_threshold
    if compaction_max_threshold is not None:
        kwargs["compaction_max_threshold"] = compaction_max_threshold
    if enable_smart_routing:
        kwargs["enable_smart_routing"] = enable_smart_routing
    if routing_profile != "balanced":
        kwargs["routing_profile"] = routing_profile
    if fallback_chain is not None:
        kwargs["fallback_chain"] = fallback_chain
    if not tool_preview:
        kwargs["tool_preview"] = tool_preview
    if enable_pruning:
        kwargs["enable_pruning"] = enable_pruning

    # Bayesian orchestration defaults (only add if non-default)
    if not enable_bayesian:
        kwargs["enable_bayesian"] = enable_bayesian
    if force_bayesian:
        kwargs["force_bayesian"] = force_bayesian
    if simple_threshold != 0.3:
        kwargs["simple_threshold"] = simple_threshold
    if complex_threshold != 0.7:
        kwargs["complex_threshold"] = complex_threshold
    if not enable_voi:
        kwargs["enable_voi"] = enable_voi
    if not enable_correlation:
        kwargs["enable_correlation"] = enable_correlation
    if min_agents_for_bayesian != 2:
        kwargs["min_agents_for_bayesian"] = min_agents_for_bayesian

    return kwargs


def _build_session_config(
    *,
    agent_profile: Optional[str],
    tool_budget: Optional[int],
    max_iterations: Optional[int],
    compaction_threshold: Optional[float],
    adaptive_threshold: Optional[bool],
    compaction_min_threshold: Optional[float],
    compaction_max_threshold: Optional[float],
    enable_smart_routing: bool,
    routing_profile: str,
    fallback_chain: Optional[str],
    tool_preview: bool,
    enable_pruning: bool,
    planning_enabled: Optional[bool],
    planning_model: Optional[str],
    mode: Optional[str],
    show_reasoning: bool,
    observability_logging: Optional[bool] = None,
    auto_skill_enabled: Optional[bool] = None,
    one_shot_mode: Optional[bool] = None,
    verify_mode: str = "none",
    lsp_perception: bool = False,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    endpoint: Optional[str] = None,
    auth_mode: Optional[str] = None,
    coding_plan: bool = False,
    # Bayesian orchestration parameters
    enable_bayesian: bool = True,
    force_bayesian: bool = False,
    simple_threshold: float = 0.3,
    complex_threshold: float = 0.7,
    enable_voi: bool = True,
    enable_correlation: bool = True,
    min_agents_for_bayesian: int = 2,
) -> SessionConfig:
    """Create a normalized SessionConfig from chat CLI flags."""
    return SessionConfig.from_cli_flags(
        agent_profile=agent_profile,
        tool_budget=tool_budget,
        max_iterations=max_iterations,
        compaction_threshold=compaction_threshold,
        adaptive_threshold=adaptive_threshold,
        compaction_min_threshold=compaction_min_threshold,
        compaction_max_threshold=compaction_max_threshold,
        enable_smart_routing=enable_smart_routing,
        routing_profile=routing_profile,
        fallback_chain=fallback_chain,
        tool_preview=tool_preview,
        enable_pruning=enable_pruning,
        planning_enabled=planning_enabled,
        planning_model=planning_model,
        mode=mode,
        show_reasoning=show_reasoning,
        observability_logging=observability_logging,
        auto_skill_enabled=auto_skill_enabled,
        one_shot_mode=one_shot_mode,
        verify_mode=verify_mode,
        lsp_perception=lsp_perception,
        provider=provider,
        model=model,
        endpoint=endpoint,
        auth_mode=auth_mode,
        coding_plan=coding_plan,
        enable_bayesian=enable_bayesian,
        force_bayesian=force_bayesian,
        simple_threshold=simple_threshold,
        complex_threshold=complex_threshold,
        enable_voi=enable_voi,
        enable_correlation=enable_correlation,
        min_agents_for_bayesian=min_agents_for_bayesian,
    )


def _display_skill_preview(con: Console, agent: Any, message: str) -> None:
    """Show skill auto-selection feedback before the LLM response.

    Does a preview match (deterministic, same as coordinator will use)
    to display which skill(s) will be activated.
    """
    matcher = getattr(agent, "_skill_matcher", None)
    if matcher is None or not getattr(matcher, "_initialized", False):
        return
    if getattr(agent, "_skill_auto_disabled", False):
        return
    try:
        matches = matcher.match_multiple_sync(message)
        if not matches:
            return
        if len(matches) == 1:
            skill, score = matches[0]
            con.print(f"[dim]\U0001f3af Skill: {skill.name} ({score:.2f})[/]")
        else:
            names = " \u2192 ".join(s.name for s, _ in matches)
            con.print(f"[dim]\U0001f3af Skills: {names}[/]")
    except Exception:
        pass


def _print_tool_output_mode_banner(con: Console, tool_settings: Any) -> None:
    """Render a compact banner describing tool-output preview behavior."""
    con.print(_describe_tool_output_mode(tool_settings))


def _describe_tool_output_mode(tool_settings: Any) -> str:
    """Return the user-facing tool-output mode summary."""
    summary, preview_state = _tool_output_mode_parts(tool_settings)
    icon = "[green]✓[/]" if summary == "safe read-heavy pruning" else "[yellow]![/]"
    return f"{icon} [bold]Tool output[/] [dim]{summary} • {preview_state}[/]"


def _summarize_tool_output_mode(tool_settings: Any) -> str:
    """Return a plain-text startup message describing tool-output behavior."""
    summary, preview_state = _tool_output_mode_parts(tool_settings)
    return f"Tool output: {summary}, {preview_state}"


def _build_file_watcher_exec_context(settings: Any) -> dict[str, Any]:
    """Build the execution context used to initialize chat file watchers."""
    return {
        "cwd": os.getcwd(),
        "settings": settings,
    }


async def _initialize_file_watchers_background(exec_ctx: dict[str, Any]) -> None:
    """Initialize chat file watchers without making watcher failures fatal."""
    from victor.core.indexing.watcher_initializer import initialize_from_context

    try:
        await initialize_from_context(exec_ctx)
    except asyncio.CancelledError:
        raise
    except Exception as e:
        logging.getLogger(__name__).warning(
            "Failed to initialize file watchers; continuing without automatic file watching: %s",
            e,
        )


def _start_file_watcher_initialization(settings: Any) -> asyncio.Task[None]:
    """Start non-blocking file watcher initialization for an interactive session."""
    return asyncio.create_task(
        _initialize_file_watchers_background(_build_file_watcher_exec_context(settings))
    )


def _should_start_file_watchers_on_startup(settings: Any) -> bool:
    """Return True when chat should eagerly start repository file watchers."""
    env_value = os.getenv("VICTOR_CHAT_FILE_WATCHERS")
    if env_value is not None:
        return env_value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(getattr(settings, "chat_file_watchers_on_startup", False))


async def _cancel_file_watcher_initialization(
    task: Optional[asyncio.Task[None]],
) -> None:
    """Cancel a pending watcher initialization task during chat shutdown."""
    if task is None or task.done():
        return
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


def _tool_output_mode_parts(tool_settings: Any) -> tuple[str, str]:
    """Return summary parts for tool-output startup messaging."""
    preview_enabled = bool(getattr(tool_settings, "tool_output_preview_enabled", True))
    safe_only = bool(getattr(tool_settings, "tool_output_pruning_safe_only", True))

    if safe_only:
        summary = "safe read-heavy pruning"
    else:
        summary = "broader pruning"

    preview_state = "preview on" if preview_enabled else "preview off"
    return summary, preview_state


def _should_render_cli_chrome(formatter: Any) -> bool:
    """Return True when stdout can safely include Rich-only CLI chrome."""
    config = getattr(formatter, "config", None)
    if config is None:
        return True

    quiet = getattr(config, "quiet", False)
    if isinstance(quiet, bool) and quiet:
        return False

    mode = getattr(config, "mode", None)
    if mode is None:
        return True

    mode_value = getattr(mode, "value", mode)
    return str(mode_value).lower() == "rich"


def _should_render_interactive_tool_banner() -> bool:
    """Return True when interactive startup should print CLI tool-output chrome."""
    return True


def _should_prompt_for_model_warning(
    *,
    actual_message: Optional[str],
    automation_mode: bool,
    input_file: Optional[Path],
) -> bool:
    """Return whether CLI startup should block on a non-fatal model warning.

    Interactive chat can prompt for confirmation. One-shot, stdin/file-driven,
    or automation-oriented invocations should stay non-blocking and continue
    after surfacing the warning.
    """
    return not (
        bool(actual_message)
        or automation_mode
        or input_file is not None
        or not sys.stdin.isatty()
        or not sys.stdout.isatty()
    )


def _profile_not_found_message(profile: str, profiles: dict[str, Any]) -> str:
    """Return a concise diagnostic for a missing non-default profile."""
    message = (
        f"Profile '{profile}' not found. "
        "Use --profile for configured profiles; use --provider/-P and --model "
        "to select a provider/model directly."
    )
    if profiles:
        available = ", ".join(sorted(profiles.keys()))
        message += f" Available profiles: {available}."
    else:
        message += " No profiles are configured."
    return message


def _require_existing_non_default_profile(
    settings: Any,
    profile: str,
) -> tuple[dict[str, Any], Any]:
    """Load profiles and reject missing explicit profile names."""
    profiles = settings.load_profiles()
    profile_config = profiles.get(profile)
    if profile_config is None and profile != "default":
        raise ValueError(_profile_not_found_message(profile, profiles))
    return profiles, profile_config


def _resolve_profile_display(
    *,
    config: SessionConfig,
    profile_config: Any,
    settings: Any,
) -> SimpleNamespace:
    """Resolve the provider/model displayed in chat chrome."""
    provider_override = config.provider_override
    if provider_override.is_active:
        return SimpleNamespace(
            provider=provider_override.provider or settings.provider.default_provider,
            model=provider_override.model or settings.provider.default_model,
        )
    return profile_config or SimpleNamespace(
        provider=settings.provider.default_provider,
        model=settings.provider.default_model,
    )


def _print_interactive_startup_messages(con: Console, messages: list[str]) -> None:
    """Print queued startup notices for interactive CLI surfaces."""
    for message in messages:
        render_status_message(con, message)


@dataclass(frozen=True)
class ChatGraphWatchHandle:
    """Lifecycle handle for a graph-watch daemon ensured by interactive chat."""

    messages: list[str]
    project_root: Optional[Path] = None
    started_by_chat: bool = False


def _ensure_graph_watch_handle_for_chat(*, enabled: bool) -> ChatGraphWatchHandle:
    """Ensure the project-scoped graph watch daemon exists for interactive chat."""
    if not enabled:
        return ChatGraphWatchHandle(messages=[])

    paths = get_project_paths(Path.cwd())
    paths.ensure_project_dirs()

    try:
        from victor.ui.commands.graph import (
            _read_graph_watch_manifest,
            ensure_graph_watch_daemon,
            summarize_graph_watch_startup,
        )

        state = ensure_graph_watch_daemon(
            paths.project_root,
            enable_ccg=True,
            # Interactive chat must not kick off a potentially large graph rebuild
            # while the user is waiting on provider output. The daemon will keep
            # watching future changes; explicit graph commands can still request
            # an immediate build with --build-now.
            build_now=False,
            owner="chat",
        )
    except Exception as exc:
        return ChatGraphWatchHandle(
            messages=[f"Warning: failed to ensure graph watch daemon: {exc}"]
        )

    manifest = _read_graph_watch_manifest(paths.project_root)
    return ChatGraphWatchHandle(
        messages=summarize_graph_watch_startup(paths.project_root, state, manifest=manifest),
        project_root=paths.project_root,
        started_by_chat=bool(state.started),
    )


def _ensure_graph_watch_for_chat(*, enabled: bool) -> list[str]:
    """Ensure graph watch for chat and return user-facing startup messages."""
    return _ensure_graph_watch_handle_for_chat(enabled=enabled).messages


def _cleanup_graph_watch_for_chat(handle: ChatGraphWatchHandle) -> None:
    """Stop chat-started graph watch daemons without touching explicit persistent daemons."""
    if not handle.started_by_chat or handle.project_root is None:
        return

    try:
        from victor.ui.commands.graph import stop_graph_watch_daemon

        stop_graph_watch_daemon(handle.project_root)
    except Exception as exc:
        logging.getLogger(__name__).debug(
            "Failed to stop chat-started graph watch daemon: %s",
            exc,
        )


def _summarize_smart_routing(
    settings: Any, enable_smart_routing: bool, routing_profile: str
) -> list[str]:
    """Return startup messages describing smart-routing state."""
    if not enable_smart_routing:
        return []
    if getattr(settings, "smart_routing_enabled", False):
        return [f"Smart routing enabled (profile={routing_profile})"]
    return ["Smart routing is currently disabled via feature flag."]


def _summarize_compaction_overrides(
    *,
    compaction_threshold: Optional[float] = None,
    adaptive_threshold: Optional[bool] = None,
    compaction_min_threshold: Optional[float] = None,
    compaction_max_threshold: Optional[float] = None,
) -> list[str]:
    """Return startup messages describing CLI compaction overrides."""
    messages: list[str] = []
    if compaction_threshold is not None:
        messages.append(f"Compaction threshold set to {compaction_threshold:.0%}")
    if adaptive_threshold is True:
        min_thresh = compaction_min_threshold or 0.35
        max_thresh = compaction_max_threshold or 0.70
        messages.append(f"Adaptive threshold enabled ({min_thresh:.0%}-{max_thresh:.0%})")
    elif adaptive_threshold is False:
        messages.append("Adaptive threshold disabled")
    return messages


class _NoopStatus:
    """No-op replacement for Rich status in automation output modes."""

    def __enter__(self) -> "_NoopStatus":
        return self

    def __exit__(self, *_args: Any) -> bool:
        return False

    def update(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    def stop(self) -> None:
        pass


def _build_cli_panel(
    profile_config: Any,
    vertical_name: Optional[str] = None,
    rl_suggestion: Optional[tuple[str, str, float]] = None,
    settings: Any = None,
) -> Panel:
    """Build the interactive CLI header panel."""
    brand = Text()
    brand.append("Victor", style="bold white")
    brand.append("  ", style="dim")
    brand.append("Open-source agentic AI framework", style="dim")

    meta = Text()
    meta.append("Provider ", style="dim")
    # Handle None profile_config (default settings)
    provider_name = (
        profile_config.provider if profile_config else settings.provider.default_provider
    )
    model_name = profile_config.model if profile_config else settings.provider.default_model
    meta.append(str(provider_name), style="bold cyan")
    meta.append("  •  ", style="dim")
    meta.append("Model ", style="dim")
    meta.append(str(model_name), style="bold green")
    if vertical_name:
        meta.append("  •  ", style="dim")
        meta.append("Vertical ", style="dim")
        meta.append(vertical_name, style="bold magenta")

    controls = Text()
    controls.append("/help", style="bold")
    controls.append(" commands  •  ", style="dim")
    controls.append("/exit", style="bold")
    controls.append(" or Ctrl+D  •  history with Up/Down", style="dim")

    lines: list[Any] = [brand, meta, controls]

    if rl_suggestion:
        profile_name, provider_name, q_value = rl_suggestion
        hint = Text()
        hint.append("Routing hint ", style="dim")
        hint.append(provider_name, style="bold cyan")
        hint.append(f"  Q={q_value:.2f}", style="yellow")
        hint.append("  •  victor chat --profile ", style="dim")
        hint.append(profile_name, style="bold cyan")
        lines.append(hint)

    return Panel(
        Group(*lines),
        title="Victor CLI",
        border_style="bright_blue",
        box=box.ROUNDED,
        padding=(1, 2),
    )


def _configure_agent_compaction(
    agent: Any,
    *,
    compaction_threshold: Optional[float] = None,
    adaptive_threshold: Optional[bool] = None,
    compaction_min_threshold: Optional[float] = None,
    compaction_max_threshold: Optional[float] = None,
    con: Console = console,
    show_status: bool = True,
) -> None:
    """Apply CLI compaction overrides to an agent/orchestrator if supported."""
    if not any(
        (
            compaction_threshold is not None,
            adaptive_threshold is not None,
            compaction_min_threshold is not None,
            compaction_max_threshold is not None,
        )
    ):
        return

    orchestrator_getter = getattr(agent, "get_orchestrator", None)
    orchestrator = orchestrator_getter() if callable(orchestrator_getter) else agent
    compactor = getattr(orchestrator, "context_compactor", None)
    if compactor is None:
        return

    if compaction_threshold is not None:
        if not (0.1 <= compaction_threshold <= 0.95):
            if show_status:
                con.print(f"\n[red]✗[/] Invalid compaction threshold: {compaction_threshold}")
                con.print("[yellow]Threshold must be between 0.1 (10%) and 0.95 (95%)[/]")
            raise typer.Exit(code=1)
        compactor.config.proactive_threshold = compaction_threshold
        if show_status:
            con.print(f"[dim]Compaction threshold set to {compaction_threshold:.0%}[/]")

    if adaptive_threshold is not None:
        if adaptive_threshold:
            from victor.agent.adaptive_compaction import AdaptiveCompactionThreshold

            min_thresh = compaction_min_threshold or 0.35
            max_thresh = compaction_max_threshold or 0.70
            if min_thresh >= max_thresh:
                if show_status:
                    con.print(
                        f"\n[red]✗[/] Min threshold ({min_thresh}) must be less than max "
                        f"({max_thresh})"
                    )
                raise typer.Exit(code=1)

            adaptive = AdaptiveCompactionThreshold(
                min_threshold=min_thresh,
                max_threshold=max_thresh,
            )
            compactor.set_adaptive_threshold(adaptive)
            if show_status:
                con.print(f"[dim]Adaptive threshold enabled ({min_thresh:.0%}-{max_thresh:.0%})[/]")
        else:
            compactor.disable_adaptive_threshold()
            if show_status:
                con.print("[dim]Adaptive threshold disabled[/]")


@chat_app.callback(invoke_without_command=True)
def chat(
    ctx: typer.Context,
    help_full: bool = typer.Option(
        False,
        "--help-full",
        help="Show full help with all 37 options (advanced usage).",
        is_eager=True,
    ),
    message: Optional[str] = typer.Argument(
        None,
        help="Message to send (put BEFORE options, or use -m instead). Interactive mode if omitted.",
    ),
    message_opt: Optional[str] = typer.Option(
        None,
        "--message",
        "-m",
        help="Message to send (alternative to positional arg, works anywhere in command).",
    ),
    # Core options (always visible - beginner friendly)
    profile: str = typer.Option(
        "default",
        "--profile",
        "-p",
        help="Profile to use from profiles.yaml",
    ),
    provider: Optional[str] = typer.Option(
        None,
        "--provider",
        "-P",
        help="Override provider (e.g., ollama, anthropic, openai). Uses provider's default model if --model not specified.",
        case_sensitive=False,
    ),
    model: Optional[str] = typer.Option(
        None,
        "--model",
        help="Override model identifier (optional if --provider specified).",
    ),
    stream: bool = typer.Option(
        True,
        "--stream/--no-stream",
        help="Stream responses in real-time",
    ),
    thinking: bool = typer.Option(
        False,
        "--thinking/--no-thinking",
        help="Enable extended thinking/reasoning mode (Claude models). Shows model's reasoning process.",
    ),
    tui: Optional[bool] = typer.Option(
        None,
        "--tui/--repl",
        help="Force the interactive multi-pane TUI (--tui) or the line-oriented REPL "
        "(--repl). Default: the TUI on capable terminals, the REPL on dumb terminals / "
        "pipes / CI.",
        rich_help_panel="Advanced Agent Behavior",
    ),
    headless: bool = typer.Option(
        False,
        "--headless",
        help="Run without prompts, auto-approve safe actions.",
        rich_help_panel="Advanced Agent Behavior",
    ),
    # Output options
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output response as JSON object (for automation/scripting).",
        rich_help_panel="Output Format",
    ),
    plain: bool = typer.Option(
        False,
        "--plain",
        help="Output plain text without Rich formatting.",
        rich_help_panel="Output Format",
    ),
    code_only: bool = typer.Option(
        False,
        "--code-only",
        help="Extract and output only code blocks from response.",
        rich_help_panel="Output Format",
    ),
    quiet: bool = typer.Option(
        False,
        "--quiet",
        "-q",
        help="Suppress status messages (only output response).",
        rich_help_panel="Output Format",
    ),
    # Input options
    stdin: bool = typer.Option(
        False,
        "--stdin",
        help="Read input from stdin (supports multi-line).",
        rich_help_panel="Input",
    ),
    input_file: Optional[str] = typer.Option(
        None,
        "--input-file",
        "-f",
        help="Read input from file instead of argument.",
        rich_help_panel="Input",
    ),
    # Logging options
    log_level: str = typer.Option(
        None,
        "--log-level",
        "-l",
        help="Set logging level (DEBUG, INFO, WARN, ERROR). Defaults to WARNING or VICTOR_LOG_LEVEL env var.",
        case_sensitive=False,
        rich_help_panel="Logging",
    ),
    # Advanced options (grouped for progressive disclosure)
    debug_modules: str = typer.Option(
        None,
        "--debug-modules",
        help="Comma-separated modules to set to DEBUG level (e.g., code_search,agent_adapter).",
        rich_help_panel="Advanced Logging",
    ),
    enable_observability: bool = typer.Option(
        True,
        "--observability/--no-observability",
        help="Enable observability integration for event tracking.",
        rich_help_panel="Advanced Logging",
    ),
    log_events: bool = typer.Option(
        False,
        "--log-events",
        help="Enable JSONL event logging to ~/.victor/logs/victor.log for dashboard visualization.",
        rich_help_panel="Advanced Logging",
    ),
    show_reasoning: bool = typer.Option(
        False,
        "--show-reasoning",
        help="Show LLM reasoning/thinking content in output.",
        rich_help_panel="Advanced Output",
    ),
    renderer: str = typer.Option(
        "auto",
        "--renderer",
        help="Renderer to use for streaming output: auto, rich, rich-text, or text.",
        case_sensitive=False,
        rich_help_panel="Advanced Output",
    ),
    mode: Optional[str] = typer.Option(
        None,
        "--mode",
        help=(
            "Initial coding-agent mode: build, plan, review, or delegate. "
            "Advanced opt-in: explore."
        ),
        case_sensitive=False,
        rich_help_panel="Advanced Agent Behavior",
    ),
    tool_budget: Optional[int] = typer.Option(
        None,
        "--tool-budget",
        help="Override tool call budget for this session.",
        rich_help_panel="Advanced Agent Behavior",
    ),
    max_iterations: Optional[int] = typer.Option(
        None,
        "--max-iterations",
        help="Override maximum total iterations for this session.",
        rich_help_panel="Advanced Agent Behavior",
    ),
    preindex: bool = typer.Option(
        False,
        "--preindex",
        help="Preload semantic code index at startup (avoids 20-30s delay on first search).",
        rich_help_panel="Advanced Agent Behavior",
    ),
    graph_watch: bool = typer.Option(
        True,
        "--graph-watch/--no-graph-watch",
        help="Ensure a project-scoped graph watch daemon is running for interactive chat sessions.",
        rich_help_panel="Advanced Agent Behavior",
    ),
    vertical: Optional[str] = typer.Option(
        None,
        "--vertical",
        "-V",
        help=f"Vertical template to use ({', '.join(list_verticals()) or 'coding, research, devops'}).",
        rich_help_panel="Advanced Agent Behavior",
    ),
    verify_mode: str = typer.Option(
        "none",
        "--verify",
        help="Verify the agent's work after it claims done and retry on failure (FEP-0018). "
        "'pytest' runs python -m pytest in the workspace; 'lint' runs ruff check; "
        "'lsp' checks LSP diagnostics on edited files (FEP-0019, multi-language, "
        "needs victor-coding); 'none' disables verification (default).",
        rich_help_panel="Advanced Agent Behavior",
    ),
    lsp_perception: bool = typer.Option(
        False,
        "--lsp-perception/--no-lsp-perception",
        help="Inject live LSP symbols + errors each turn so generated code matches the "
        "codebase on the first try (FEP-0019 Phase 3, experimental; needs victor-coding).",
        rich_help_panel="Advanced Agent Behavior",
    ),
    auto_skill: Optional[bool] = typer.Option(
        None,
        "--auto-skill/--no-auto-skill",
        help="Enable/disable automatic skill selection based on message content.",
        rich_help_panel="Advanced Agent Behavior",
    ),
    enable_planning: Optional[bool] = typer.Option(
        None,
        "--planning/--no-planning",
        help="Enable structured planning for complex multi-step tasks.",
        rich_help_panel="Advanced Agent Behavior",
    ),
    planning_model: Optional[str] = typer.Option(
        None,
        "--planning-model",
        help="Override model for planning tasks (e.g., 'deepseek-chat').",
        rich_help_panel="Advanced Agent Behavior",
    ),
    # Compaction options
    compaction_threshold: Optional[float] = typer.Option(
        None,
        "--compaction-threshold",
        help="Compaction threshold (0.1-0.95). Default: 0.50. Lower = earlier compaction.",
        rich_help_panel="Compaction",
    ),
    adaptive_threshold: Optional[bool] = typer.Option(
        None,
        "--adaptive-threshold/--no-adaptive-threshold",
        help="Enable adaptive threshold based on conversation patterns (35-70% range).",
        rich_help_panel="Compaction",
    ),
    compaction_min_threshold: Optional[float] = typer.Option(
        None,
        "--compaction-min-threshold",
        help="Minimum adaptive threshold (0.1-0.8). Default: 0.35. Used with --adaptive-threshold.",
        rich_help_panel="Compaction",
    ),
    compaction_max_threshold: Optional[float] = typer.Option(
        None,
        "--compaction-max-threshold",
        help="Maximum adaptive threshold (0.2-0.95). Default: 0.70. Used with --adaptive-threshold.",
        rich_help_panel="Compaction",
    ),
    # Workflow options (grouped separately)
    workflow: Optional[str] = typer.Option(
        None,
        "--workflow",
        "-w",
        help="Path to YAML workflow file to execute. Runs workflow instead of chat mode.",
        rich_help_panel="Workflow",
    ),
    validate_workflow: bool = typer.Option(
        False,
        "--validate",
        help="Validate YAML workflow file without executing. Use with --workflow.",
        rich_help_panel="Workflow",
    ),
    render_format: Optional[str] = typer.Option(
        None,
        "--render",
        "-r",
        help="Render workflow DAG (ascii, mermaid, d2, dot, plantuml, svg, png).",
        rich_help_panel="Workflow",
    ),
    render_output: Optional[str] = typer.Option(
        None,
        "--render-output",
        "-o",
        help="Output file for rendered diagram. Required for svg/png formats.",
        rich_help_panel="Workflow",
    ),
    delegate_follow_up_contract: Optional[str] = typer.Option(
        None,
        "--delegate-follow-up-contract",
        help=(
            "Path to a JSON delegate follow-up contract to inject into workflow state "
            "for TeamStep resume/review/merge execution. Use with --workflow."
        ),
        rich_help_panel="Workflow",
    ),
    delegate_next_step_id: Optional[str] = typer.Option(
        None,
        "--delegate-next-step-id",
        help="Explicit follow-up step_id to execute from the delegate follow-up contract.",
        rich_help_panel="Workflow",
    ),
    # Session options (grouped separately)
    list_sessions: bool = typer.Option(
        False,
        "--sessions",
        help="List saved sessions and exit (top 20).",
        rich_help_panel="Session",
    ),
    session_id: Optional[str] = typer.Option(
        None,
        "--sessionid",
        help="Resume specific session by ID.",
        rich_help_panel="Session",
    ),
    # Auth & Compatibility (expert options)
    endpoint: Optional[str] = typer.Option(
        None,
        "--endpoint",
        help="Override base URL for local providers (ollama, lmstudio, vllm).",
        rich_help_panel="Expert Auth & Compatibility",
    ),
    auth_mode: Optional[str] = typer.Option(
        None,
        "--auth-mode",
        help="Authentication mode: 'api_key' (default) or 'oauth' (for OpenAI Codex, Qwen).",
        case_sensitive=False,
        rich_help_panel="Expert Auth & Compatibility",
    ),
    coding_plan: bool = typer.Option(
        False,
        "--coding-plan",
        help="Use coding plan endpoint (Z.AI). Routes to api.z.ai/api/coding/paas/v4/.",
        rich_help_panel="Expert Auth & Compatibility",
    ),
    # Smart Routing options (Phase 11 - Intelligent Provider Selection)
    enable_smart_routing: bool = typer.Option(
        False,
        "--enable-smart-routing",
        help="Enable automatic provider routing based on health, resources, cost, latency, and performance.",
        rich_help_panel="Smart Routing",
    ),
    routing_profile: str = typer.Option(
        "balanced",
        "--routing-profile",
        help="Routing profile to use (balanced, cost-optimized, performance, local-first).",
        rich_help_panel="Smart Routing",
    ),
    fallback_chain: Optional[str] = typer.Option(
        None,
        "--fallback-chain",
        help="Custom fallback chain (comma-separated provider names, e.g., 'ollama,anthropic,openai').",
        rich_help_panel="Smart Routing",
    ),
    # Tool Output Preview options (safe-default pruning)
    tool_preview: bool = typer.Option(
        True,
        "--tool-preview/--no-tool-preview",
        help="Show tool output preview (default: yes). Disable to show only status line.",
        rich_help_panel="Tool Output",
    ),
    enable_pruning: bool = typer.Option(
        False,
        "--enable-pruning",
        help="Broaden tool output pruning beyond the safe default read-heavy scope.",
        rich_help_panel="Tool Output",
    ),
    # Bayesian Orchestration options (complexity-based routing)
    enable_bayesian: bool = typer.Option(
        True,
        "--enable-bayesian/--no-enable-bayesian",
        help="Enable Bayesian orchestration for complex queries (default: yes).",
        rich_help_panel="Bayesian Orchestration",
    ),
    force_bayesian: bool = typer.Option(
        False,
        "--force-bayesian",
        help="Force ALL queries through Bayesian orchestration (for testing).",
        rich_help_panel="Bayesian Orchestration",
    ),
    simple_threshold: float = typer.Option(
        0.3,
        "--simple-threshold",
        help="Complexity score below which queries use simple fast path (0.0-1.0, default: 0.3).",
        rich_help_panel="Bayesian Orchestration",
    ),
    complex_threshold: float = typer.Option(
        0.7,
        "--complex-threshold",
        help="Complexity score above which queries use Bayesian orchestration (0.0-1.0, default: 0.7).",
        rich_help_panel="Bayesian Orchestration",
    ),
    enable_voi: bool = typer.Option(
        True,
        "--enable-voi/--no-enable-voi",
        help="Enable Value of Information-based agent selection (default: yes).",
        rich_help_panel="Bayesian Orchestration",
    ),
    enable_correlation: bool = typer.Option(
        True,
        "--enable-correlation/--no-enable-correlation",
        help="Enable correlation-aware consensus (default: yes).",
        rich_help_panel="Bayesian Orchestration",
    ),
    min_agents_for_bayesian: int = typer.Option(
        2,
        "--min-agents-for-bayesian",
        help="Minimum number of agents required to trigger Bayesian orchestration (default: 2).",
        rich_help_panel="Bayesian Orchestration",
    ),
):
    """Start interactive chat or send a one-shot message.

    \f
    **Quick Start:**
        victor chat                    # Interactive with defaults
        victor chat "Your message"     # One-shot message

    **For advanced options and examples:**
        victor chat --help-full
    """
    help_full = _resolve_typer_default(help_full)
    debug_modules = _resolve_typer_default(debug_modules)
    auto_skill = _resolve_typer_default(auto_skill)
    compaction_threshold = _resolve_typer_default(compaction_threshold)
    adaptive_threshold = _resolve_typer_default(adaptive_threshold)
    compaction_min_threshold = _resolve_typer_default(compaction_min_threshold)
    compaction_max_threshold = _resolve_typer_default(compaction_max_threshold)
    enable_smart_routing = _resolve_typer_default(enable_smart_routing)
    routing_profile = _resolve_typer_default(routing_profile)
    fallback_chain = _resolve_typer_default(fallback_chain)
    tool_preview = _resolve_typer_default(tool_preview)
    enable_pruning = _resolve_typer_default(enable_pruning)
    enable_bayesian = _resolve_typer_default(enable_bayesian)
    force_bayesian = _resolve_typer_default(force_bayesian)
    simple_threshold = _resolve_typer_default(simple_threshold)
    complex_threshold = _resolve_typer_default(complex_threshold)
    enable_voi = _resolve_typer_default(enable_voi)
    enable_correlation = _resolve_typer_default(enable_correlation)
    min_agents_for_bayesian = _resolve_typer_default(min_agents_for_bayesian)
    graph_watch = _resolve_typer_default(graph_watch)
    delegate_follow_up_contract = _resolve_typer_default(delegate_follow_up_contract)
    delegate_next_step_id = _resolve_typer_default(delegate_next_step_id)

    # Handle --help-full flag for comprehensive help
    if help_full:
        from rich.markdown import Markdown

        full_help = """
# Victor Chat - Full Help

**Victor** has 39 options organized into 11 categories for progressive disclosure.

## Quick Reference
- Basic chat: `victor chat`
- One-shot: `victor chat "your message"`
- Switch provider: `victor chat -p anthropic`
- Use profile: `victor chat --profile coding`
- In-chat shortcuts: `/shortcuts` or F1

## Core Options (Beginner-Friendly)
These are the most commonly used options:

| Option | Description | Example |
|--------|-------------|---------|
| `--profile`, `-p` | Profile from profiles.yaml | `-p coding` |
| `--provider` | Override provider | `--provider ollama` |
| `--model` | Override model | `--model llama3.2` |
| `--stream` | Enable streaming (default: ON) | `--no-stream` |
| `--thinking` | Enable extended reasoning | `--thinking` |

## Category Reference

### Output Format (4 options)
`--json`, `--plain`, `--code-only`, `--quiet`, `-q`

### Input (2 options)
`--stdin`, `--input-file`, `-f`

### Logging (2 options)
`--log-level`, `-l`, `--debug-modules`

### Advanced Agent Behavior (8 options)
`--mode`, `--tool-budget`, `--max-iterations`, `--preindex`,
`--vertical`, `-V`, `--auto-skill`, `--planning`, `--planning-model`

### Workflow (6 options)
Use `victor workflow` command instead for full workflow features.
`--workflow`, `-w`, `--validate`, `--render`, `-r`, `--render-output`, `-o`,
`--delegate-follow-up-contract`, `--delegate-next-step-id`

### Session (2 options)
Use `victor sessions` command instead for session management.
`--sessions`, `--sessionid`

### Expert Options (7 options)
Advanced debugging and compatibility options:
- **Logging**: `--observability`, `--log-events`
- **Output**: `--show-reasoning`, `--renderer`
- **Auth**: `--endpoint`, `--auth-mode`, `--coding-plan`

## Examples

**Interactive Chat:**
```bash
victor chat                              # Interactive with defaults
victor chat -p anthropic --thinking      # Use Claude with reasoning
victor chat -p ollama --model llama3.2  # Use local model
```

**One-Shot Messages:**
```bash
victor chat "Explain recursion"         # Basic message
victor chat "Analyze this file" -f code.py --preindex
victor chat "Plan a REST API" --mode plan
```

**Automation:**
```bash
victor chat "Generate tests" --json --quiet > tests.json
victor chat "Refactor" --code-only > refactored.py
```

**Workflows:**
```bash
victor workflow validate ./workflow.yaml
victor workflow render ./workflow.yaml --format svg -o diagram.svg
```

**Sessions:**
```bash
victor chat --sessions                    # List recent sessions
victor chat --sessionid abc123            # Resume session
```

## See Also
- `victor workflow --help` - Workflow commands
- `victor sessions --help` - Session management
- `victor profiles --help` - Profile management
- `victor init --wizard` - First-time setup
"""
        console.print(Markdown(full_help))
        raise typer.Exit(0)

    if ctx.invoked_subcommand is None:
        renderer = renderer.lower()
        if renderer not in {"auto", "rich", "rich-text", "text"}:
            console.print(
                "[bold red]Error:[/] Invalid renderer. Choose from auto, rich, rich-text, text."
            )
            raise typer.Exit(1)

        try:
            mode = normalize_chat_mode(mode)
        except ValueError as exc:
            console.print(f"[bold red]Error:[/] {exc}")
            raise typer.Exit(1)

        # Handle workflow mode (--workflow and --validate/--render)
        workflow_flag_requested = (
            validate_workflow
            or render_format
            or delegate_follow_up_contract
            or delegate_next_step_id
        )
        if workflow or workflow_flag_requested:
            if workflow_flag_requested and not workflow:
                console.print(
                    "[bold red]Error:[/] Workflow options require --workflow to specify a file."
                )
                raise typer.Exit(1)

            run_sync(
                run_workflow_mode(
                    workflow_path=workflow,  # type: ignore
                    validate_only=validate_workflow,
                    render_format=render_format,
                    render_output=render_output,
                    profile=profile,
                    vertical=vertical,
                    log_level=log_level,
                    delegate_follow_up_contract=delegate_follow_up_contract,
                    delegate_next_step_id=delegate_next_step_id,
                )
            )
            return

        automation_mode = json_output or plain or code_only

        # Use ERROR level for automation modes (cleaner output)
        if log_level is None and automation_mode:
            log_level = "ERROR"

        # Validate log level if explicitly provided
        if log_level is not None:
            log_level = log_level.upper()
            valid_levels = ["DEBUG", "INFO", "WARN", "WARNING", "ERROR", "CRITICAL"]

            if log_level not in valid_levels:
                console.print(
                    f"[bold red]Error:[/ ] Invalid log level '{log_level}'. Valid options: {', '.join(valid_levels)}"
                )
                raise typer.Exit(1)

            if log_level == "WARN":
                log_level = "WARNING"

            # For troubleshooting, prefer plain text output when verbose logging is enabled
            if log_level in {"DEBUG", "INFO"}:
                renderer = "text"

        # Use centralized logging config with session ID if resuming
        setup_logging(
            command="chat",
            cli_log_level=log_level,
            cli_debug_modules=debug_modules,
            stream=sys.stderr,
            session_id=session_id,  # Use --sessionid flag value for logging
        )

        # Handle --sessions flag (list sessions and exit)
        if list_sessions:
            from victor.agent.conversation.store import ConversationStore

            store = ConversationStore()
            sessions = store.list_sessions(limit=20)

            if not sessions:
                console.print("[dim]No sessions found[/]")
                raise typer.Exit(0)

            table = Table(title="Saved Sessions (top 20)")
            table.add_column("Session ID", style="cyan", no_wrap=True)
            table.add_column("Title", style="white")
            table.add_column("Model", style="yellow")
            table.add_column("Provider", style="blue")
            table.add_column("Messages", justify="right")
            table.add_column("Created", style="dim")

            for session in sessions:
                date_str = session.created_at.strftime("%Y-%m-%d %H:%M")
                title = "Untitled"  # ConversationSession doesn't have title
                table.add_row(
                    session.session_id,
                    title,
                    session.model or "unknown",
                    session.provider or "unknown",
                    str(len(session.messages)),
                    date_str,
                )

            console.print(table)
            console.print(f"\n[dim]Total: {len(sessions)} session(s)[/]")
            console.print("[dim]Use 'victor chat --sessionid <id>' to resume a session[/]")
            raise typer.Exit(0)

        from victor.agent.debug_logger import configure_logging_levels

        # Apply debug logger levels if explicitly specified
        if log_level:
            configure_logging_levels(log_level)

        formatter = create_formatter(
            json_mode=json_output,
            plain=plain,
            code_only=code_only,
            quiet=quiet,
            stream=stream and not json_output,
        )

        # --message / -m option takes precedence over positional argument
        effective_message = message_opt or message

        actual_message = InputReader.read_message(
            argument=effective_message,
            from_stdin=stdin,
            input_file=input_file,
        )

        settings = load_settings()
        try:
            session_config = _build_session_config(
                agent_profile=profile,
                tool_budget=tool_budget,
                max_iterations=max_iterations,
                compaction_threshold=compaction_threshold,
                adaptive_threshold=adaptive_threshold,
                compaction_min_threshold=compaction_min_threshold,
                compaction_max_threshold=compaction_max_threshold,
                enable_smart_routing=enable_smart_routing,
                routing_profile=routing_profile,
                fallback_chain=fallback_chain,
                tool_preview=tool_preview,
                enable_pruning=enable_pruning,
                planning_enabled=enable_planning,
                planning_model=planning_model,
                mode=mode,
                show_reasoning=show_reasoning,
                observability_logging=True if log_events else None,
                auto_skill_enabled=auto_skill,
                one_shot_mode=bool(actual_message),
                provider=provider,
                model=model,
                endpoint=endpoint,
                auth_mode=auth_mode,
                coding_plan=coding_plan,
                enable_bayesian=enable_bayesian,
                force_bayesian=force_bayesian,
                simple_threshold=simple_threshold,
                complex_threshold=complex_threshold,
                enable_voi=enable_voi,
                enable_correlation=enable_correlation,
                min_agents_for_bayesian=min_agents_for_bayesian,
                verify_mode=verify_mode,
                lsp_perception=lsp_perception,
            )
        except ValueError as exc:
            console.print(f"[bold red]Error:[/] {exc}")
            raise typer.Exit(1)

        if provider and not model and session_config.provider_override.model:
            console.print(
                f"[dim]Using default model for {session_config.provider_override.provider}: "
                f"{session_config.provider_override.model}[/]"
            )

        # Periodic cleanup: check if history file needs rotation (once per 7 days)
        try:
            history_file = get_project_paths().project_victor_dir / "chat_history"
            max_entries = settings.ui.cli_history_max_entries  # Use configured max entries
            _maybe_rotate_history_file(history_file, max_entries=max_entries)
        except Exception:
            pass  # Periodic cleanup is best-effort

        setup_safety_confirmation()

        if actual_message:
            runtime_override_kwargs = _collect_runtime_override_kwargs(
                compaction_threshold=compaction_threshold,
                adaptive_threshold=adaptive_threshold,
                compaction_min_threshold=compaction_min_threshold,
                compaction_max_threshold=compaction_max_threshold,
                enable_smart_routing=enable_smart_routing,
                routing_profile=routing_profile,
                fallback_chain=fallback_chain,
                tool_preview=tool_preview,
                enable_pruning=enable_pruning,
                enable_bayesian=enable_bayesian,
                force_bayesian=force_bayesian,
                simple_threshold=simple_threshold,
                complex_threshold=complex_threshold,
                enable_voi=enable_voi,
                enable_correlation=enable_correlation,
                min_agents_for_bayesian=min_agents_for_bayesian,
            )
            run_sync(
                run_oneshot(
                    actual_message,
                    settings,
                    profile,
                    stream and not json_output,
                    thinking,
                    formatter=formatter,
                    preindex=preindex,
                    renderer_choice=renderer,
                    mode=mode,
                    tool_budget=tool_budget,
                    max_iterations=max_iterations,
                    vertical=vertical,
                    enable_observability=enable_observability,
                    enable_planning=enable_planning,
                    planning_model=planning_model,
                    show_reasoning=show_reasoning,
                    session_config=session_config,
                    **runtime_override_kwargs,
                )
            )
        elif stdin or input_file:
            formatter.error("No input received from stdin or file")
            raise typer.Exit(1)
        else:
            runtime_override_kwargs = _collect_runtime_override_kwargs(
                compaction_threshold=compaction_threshold,
                adaptive_threshold=adaptive_threshold,
                compaction_min_threshold=compaction_min_threshold,
                compaction_max_threshold=compaction_max_threshold,
                enable_smart_routing=enable_smart_routing,
                routing_profile=routing_profile,
                fallback_chain=fallback_chain,
                tool_preview=tool_preview,
                enable_pruning=enable_pruning,
                enable_bayesian=enable_bayesian,
                force_bayesian=force_bayesian,
                simple_threshold=simple_threshold,
                complex_threshold=complex_threshold,
                enable_voi=enable_voi,
                enable_correlation=enable_correlation,
                min_agents_for_bayesian=min_agents_for_bayesian,
            )
            run_sync(
                run_interactive(
                    settings,
                    profile,
                    stream,
                    thinking,
                    preindex=preindex,
                    renderer_choice=renderer,
                    mode=mode,
                    tool_budget=tool_budget,
                    max_iterations=max_iterations,
                    vertical=vertical,
                    enable_observability=enable_observability,
                    enable_planning=enable_planning,
                    planning_model=planning_model,
                    resume_session_id=session_id,
                    show_reasoning=show_reasoning,
                    graph_watch=graph_watch,
                    surface=_resolve_surface(tui),
                    session_config=session_config,
                    **runtime_override_kwargs,
                )
            )
            return


def _configure_smart_routing(
    settings: Any,
    console: Console,
    enable_smart_routing: bool,
    routing_profile: str,
    fallback_chain: Optional[str],
    show_status: bool = True,
) -> None:
    """Display smart-routing state after SessionConfig normalization."""
    if not enable_smart_routing:
        return
    from victor.core.feature_flags import get_feature_flag_manager, FeatureFlag

    feature_manager = get_feature_flag_manager()
    if feature_manager.is_enabled(FeatureFlag.USE_SMART_ROUTING):
        if show_status:
            console.print(f"[green]✓[/] Smart routing enabled (profile={routing_profile})")
    else:
        if show_status:
            console.print("[yellow]Smart routing is currently disabled via feature flag.[/]")
            console.print(
                "Enable with: [cyan]export VICTOR_USE_SMART_ROUTING=true[/] or "
                "[cyan]--enable-smart-routing[/] flag in beta period"
            )


def _run_default_interactive() -> None:
    """Run the default interactive CLI mode with default options."""
    # Use centralized logging config (respects ~/.victor/config.yaml and env vars)
    setup_logging(command="chat")

    settings = load_settings()
    setup_safety_confirmation()
    run_sync(run_interactive(settings, "default", True, False))


def _resolve_surface(tui: Optional[bool]) -> str:
    """Map the tri-state ``--tui/--repl`` flag to an interactive surface (ADR-020).

    ``None`` (neither flag) → ``"auto"`` (TUI on capable terminals, REPL otherwise);
    ``True`` (``--tui``) → forced ``"tui"``; ``False`` (``--repl``) → forced ``"repl"``.
    """
    if tui is None:
        return "auto"
    return "tui" if tui else "repl"


def _tui_capable() -> bool:
    """True when the current terminal can host the interactive Textual UI."""
    try:
        from victor.ui.rendering.terminal_capabilities import (
            TerminalCapabilities,
            TerminalCapability,
        )

        caps = TerminalCapabilities()
        return (
            caps.is_interactive()
            and not caps.is_ci_environment()
            and caps.get_capability_level() == TerminalCapability.FULL
        )
    except Exception:
        return False


async def _run_tui_app(
    client: Any,
    agent: Any,
    settings: Any,
    *,
    mode: Optional[str] = None,
    tool_budget: Optional[int] = None,
    theme: str = "dark",
) -> None:
    """Launch the interactive Textual TUI over an already-initialized session."""
    from victor.ui.tui.app import VictorTUIApp

    app = VictorTUIApp(
        client=client,
        agent=agent,
        settings=settings,
        mode=mode,
        tool_budget=tool_budget,
        theme=theme,
    )
    await app.run_async()


def run_tui_entry(
    profile: str = "default",
    mode: Optional[str] = None,
    resume_session_id: Optional[str] = None,
    theme: str = "dark",
) -> None:
    """Entry point for the ``victor tui`` command — the opt-in interactive TUI."""
    setup_logging(command="tui")
    settings = load_settings()
    setup_safety_confirmation()
    run_sync(
        run_interactive(
            settings,
            profile,
            True,
            False,
            mode=mode,
            resume_session_id=resume_session_id,
            surface="tui",
            tui_theme=theme,
        )
    )


async def run_oneshot(
    message: str,
    settings: Any,
    profile: str,
    stream: bool,
    thinking: bool = False,
    formatter: Optional[Any] = None,
    preindex: bool = False,
    renderer_choice: str = "auto",
    mode: Optional[str] = None,
    tool_budget: Optional[int] = None,
    max_iterations: Optional[int] = None,
    vertical: Optional[str] = None,
    enable_observability: bool = True,
    enable_planning: Optional[bool] = None,
    planning_model: Optional[str] = None,
    show_reasoning: bool = False,
    compaction_threshold: Optional[float] = None,
    adaptive_threshold: Optional[bool] = None,
    compaction_min_threshold: Optional[float] = None,
    compaction_max_threshold: Optional[float] = None,
    # Smart routing parameters
    enable_smart_routing: bool = False,
    routing_profile: str = "balanced",
    fallback_chain: Optional[str] = None,
    # Tool output preview parameters
    tool_preview: bool = True,
    enable_pruning: bool = False,
    # Bayesian orchestration parameters
    enable_bayesian: bool = True,
    force_bayesian: bool = False,
    simple_threshold: float = 0.3,
    complex_threshold: float = 0.7,
    enable_voi: bool = True,
    enable_correlation: bool = True,
    min_agents_for_bayesian: int = 2,
    session_config: Optional[SessionConfig] = None,
) -> None:
    """Run a single message and exit.

    Uses VictorClient to bridge CLI with framework features like
    observability and vertical configuration.
    """
    from victor.framework.agent_factory import InitializationError
    from victor.ui.output_formatter import create_formatter

    if formatter is None:
        formatter = create_formatter()

    start_time = time.time()
    show_cli_chrome = _should_render_cli_chrome(formatter)

    config = session_config or _build_session_config(
        agent_profile=profile,
        tool_budget=tool_budget,
        max_iterations=max_iterations,
        compaction_threshold=compaction_threshold,
        adaptive_threshold=adaptive_threshold,
        compaction_min_threshold=compaction_min_threshold,
        compaction_max_threshold=compaction_max_threshold,
        enable_smart_routing=enable_smart_routing,
        routing_profile=routing_profile,
        fallback_chain=fallback_chain,
        tool_preview=tool_preview,
        enable_pruning=enable_pruning,
        planning_enabled=enable_planning,
        planning_model=planning_model,
        mode=mode,
        show_reasoning=show_reasoning,
        enable_bayesian=enable_bayesian,
        force_bayesian=force_bayesian,
        simple_threshold=simple_threshold,
        complex_threshold=complex_threshold,
        enable_voi=enable_voi,
        enable_correlation=enable_correlation,
        min_agents_for_bayesian=min_agents_for_bayesian,
    )
    session_runner = FrameworkSessionRunner(settings, config)
    prepared_state = session_runner.prepare_state(
        one_shot_mode=True,
        stream=stream,
        show_reasoning=show_reasoning,
    )
    config = prepared_state.config
    show_reasoning = prepared_state.show_reasoning

    try:
        _require_existing_non_default_profile(settings, profile)
    except ValueError as exc:
        if show_cli_chrome:
            console.print(f"[bold red]Error:[/] {exc}")
        else:
            formatter.error(str(exc))
        raise typer.Exit(1) from exc

    _configure_smart_routing(
        settings,
        console,
        enable_smart_routing,
        routing_profile,
        fallback_chain,
        show_status=show_cli_chrome,
    )

    # Configure tool output preview (safe-default pruning)
    from victor.config.tool_settings import ToolSettings

    tool_settings = settings.tool_settings if hasattr(settings, "tool_settings") else ToolSettings()
    if show_cli_chrome:
        _print_tool_output_mode_banner(console, tool_settings)

    client = None
    agent = None
    try:
        from victor.framework.task import TaskComplexityService as ComplexityClassifier

        classifier = ComplexityClassifier(use_semantic=False)
        classification = classifier.classify(message)
        _task_type = classification.complexity.value  # Currently unused, logged for debugging
    except Exception:
        pass

    # If planning is enabled, disable thinking mode to avoid context bloat
    # Planning mode generates structured plans which don't need extended reasoning
    if enable_planning and thinking:
        if show_cli_chrome:
            console.print(
                "[yellow]Planning mode enabled: disabling thinking mode to avoid context overflow.[/]"
            )
        thinking = False

    agent = None
    # ✅ PROPER: No FrameworkShim needed (VictorClient handles framework features)
    # shim: Optional[FrameworkShim] = None  # REMOVED
    provider_for_suggestions = getattr(settings.provider, "default_provider", "unknown")
    client = None  # ✅ NEW: VictorClient (replaces orchestrator/shim)

    try:
        # ✅ PROPER: Create VictorClient with SessionConfig (replaces AgentFactory)
        # VictorClient handles Agent.create() internally with proper service layering

        # Show initialization progress for first-time setup (can take 20-30s)
        status_context = (
            console.status("[bold green]Initializing Victor...[/]", spinner="dots")
            if show_cli_chrome
            else _NoopStatus()
        )
        with status_context as status:
            # Step 1: Load and validate configuration
            status.update("Loading configuration...")

            # Configuration validation (P1-6: Display configuration validation status)
            try:
                from victor.config.validation import (
                    format_validation_result,
                )

                status.update("Validating configuration...")
                validation_result = session_runner.validate_configuration()

                if not validation_result.is_valid():
                    # Configuration has errors - display and exit
                    status.stop()  # Stop the spinner to show errors clearly
                    formatted_validation = format_validation_result(validation_result)
                    if show_cli_chrome:
                        console.print("[bold red]Configuration Validation Failed:[/]")
                        console.print(formatted_validation)
                        console.print("")
                        console.print(
                            "[yellow]Run 'victor config validate' for detailed diagnostics[/]"
                        )
                    else:
                        formatter.error("Configuration validation failed", formatted_validation)
                    raise typer.Exit(1)
                elif validation_result.has_warnings():
                    # Configuration has warnings but is valid - show summary
                    warning_count = len(validation_result.warnings)
                    if show_cli_chrome and warning_count > 0:
                        console.print(
                            f"[dim]✓ Configuration valid with {warning_count} warning(s)[/]"
                        )
                else:
                    # Configuration is completely valid
                    if show_cli_chrome:
                        console.print("[dim]✓ Configuration valid[/]")
            except typer.Exit:
                raise
            except Exception as e:
                # If validation itself fails, log it but don't block startup
                import logging

                logger = logging.getLogger(__name__)
                logger.warning(f"Configuration validation skipped due to error: {e}")
                if show_cli_chrome:
                    console.print("[dim yellow]⚠ Configuration validation skipped[/]")

            # Validate default model existence (Phase 1 UX improvement)
            model_valid, model_warning = session_runner.validate_default_model()
            if not model_valid and model_warning:
                prompt_for_model_warning = _should_prompt_for_model_warning(
                    actual_message=message,
                    automation_mode=not show_cli_chrome,
                    input_file=None,
                )
                if not show_cli_chrome:
                    if prompt_for_model_warning:
                        formatter.error("Configuration warning", model_warning)
                        raise typer.Exit(code=1)
                    formatter.info("Configuration warning", model_warning)

                # Model validation failed - show warning and offer to continue
                if show_cli_chrome:
                    console.print("\n[yellow]⚠ Configuration Warning:[/]")
                    console.print(model_warning)
                    console.print()

                    if prompt_for_model_warning:
                        from rich.prompt import Confirm

                        if not Confirm.ask("Continue anyway?", default=False, show_default=True):
                            console.print("\n[yellow]Setup cancelled.[/]")
                            console.print("Fix the issue above and run 'victor chat' again.\n")
                            raise typer.Exit(code=1)
                    else:
                        console.print("[dim]Continuing in non-interactive mode.[/]")

            # Step 2: Initialize VictorClient with SessionConfig
            status.update("Initializing VictorClient...")
            try:
                client = session_runner.create_client(config)
            except ConfigurationError as e:
                console.print(f"\n[red]✗[/] Configuration error: {e}")
                console.print("\n[yellow]Suggestions:[/]")
                console.print("  • Run 'victor doctor' to diagnose configuration issues")
                console.print("  • Check your profile configuration: victor profiles list")
                console.print("  • Validate config: victor config validate\n")
                raise typer.Exit(code=1)
            except (ProviderConnectionError, ProviderNotFoundError) as e:
                console.print(f"\n[red]✗[/] Provider error: {e}")
                console.print("\n[yellow]Suggestions:[/]")
                console.print("  • Check if provider is available: victor doctor --providers")
                console.print("  • Verify provider configuration in profiles.yaml")
                console.print("  • Try a different profile: victor profiles list\n")
                raise typer.Exit(code=1)
            except ProviderAuthError as e:
                console.print(f"\n[red]✗[/] Authentication error: {e}")
                console.print("\n[yellow]Suggestions:[/]")
                console.print("  • Verify API key is set: victor doctor --credentials")
                console.print("  • Check API key has required permissions")
                console.print("  • Try re-exporting your API key\n")
                raise typer.Exit(code=1)
            except Exception as e:
                # Unexpected error - show full traceback in debug mode
                console.print(f"\n[red]✗[/] Failed to initialize agent factory: {e}")
                console.print("\n[yellow]Suggestions:[/]")
                console.print("  • Run 'victor doctor' to diagnose configuration issues")
                console.print("  • Check your profile configuration: victor profiles list")
                console.print("  • Try default profile: victor chat --profile default\n")
                if os.getenv("VICTOR_DEBUG"):
                    import traceback

                    console.print(traceback.format_exc())
                raise typer.Exit(code=1)

            # Step 3: Ensure VictorClient is initialized
            status.update("Creating agent...")
            try:
                agent = await session_runner.initialize_client(
                    client,
                    planning_model=planning_model,
                )
            except ConfigurationError as e:
                console.print(f"\n[red]✗[/] Configuration error during agent creation: {e}")
                console.print("\n[yellow]Suggestions:[/]")
                console.print("  • Check vertical configuration: victor vertical list")
                console.print("  • Validate config: victor config validate")
                console.print("  • Try default vertical: victor chat --vertical default\n")
                raise typer.Exit(code=1)
            except ProviderError as e:
                console.print(f"\n[red]✗[/] Provider error during agent creation: {e}")
                console.print("\n[yellow]Suggestions:[/]")
                console.print("  • Check provider status: victor doctor --providers")
                console.print("  • Verify provider is running (for local models)")
                console.print("  • Try a different provider: victor chat --provider ollama\n")
                raise typer.Exit(code=1)
            except Exception as e:
                # Unexpected error - show full traceback in debug mode
                console.print(f"\n[red]✗[/] Failed to create agent: {e}")
                console.print("\n[yellow]Suggestions:[/]")
                console.print("  • Check if provider is configured: victor doctor --providers")
                console.print("  • Verify API keys or local model availability")
                console.print("  • Try a different profile: victor profiles list\n")
                if os.getenv("VICTOR_DEBUG"):
                    import traceback

                    console.print(traceback.format_exc())
                raise typer.Exit(code=1)

            # Step 4: Configure agent settings
            status.update("Configuring agent settings...")
            # SessionConfig already carries runtime overrides into Agent/VictorClient creation.
            # Avoid mutating the runtime again here so the UI stays on framework surfaces.

            if mode:
                try:
                    session_runner.apply_agent_mode(mode)
                except Exception as e:
                    if show_cli_chrome:
                        console.print(f"\n[yellow]Warning:[/] Failed to set mode '{mode}': {e}")
                        console.print("  Continuing with default mode.\n")

            # Step 5: Preload semantic index if requested
            if preindex:
                status.update("Preloading semantic index...")
                try:
                    await preload_semantic_index(os.getcwd(), settings, console)
                except Exception as e:
                    if show_cli_chrome:
                        console.print(
                            f"\n[yellow]Warning:[/] Failed to preload semantic index: {e}"
                        )
                        console.print(
                            "  Continuing without preindex (first search will be slower).\n"
                        )

            # Step 6: Start embedding preload
            status.update("Starting embedding preload...")
            try:
                await session_runner.start_embedding_preload(client)
            except Exception as e:
                if show_cli_chrome:
                    console.print(f"\n[yellow]Warning:[/] Failed to start embedding preload: {e}")
                    console.print(
                        "  Continuing without embeddings (some features may be slower).\n"
                    )

            # Planning mode requires non-streaming (plan generation → step execution → summary)
            use_streaming = prepared_state.use_streaming

            # Display skill auto-selection feedback before response
            if show_cli_chrome:
                _display_skill_preview(console, agent, message)

            if use_streaming:
                from victor.ui.rendering import (
                    FormatterRenderer,
                    LiveDisplayRenderer,
                    stream_response,
                )

                use_live = show_cli_chrome and renderer_choice in {"rich", "auto"}
                if renderer_choice in {"rich-text", "text"}:
                    use_live = False
                renderer = (
                    LiveDisplayRenderer(
                        console,
                        tool_budget_source=lambda: _coerce_cli_int(
                            getattr(agent, "tool_budget", None)
                        ),
                    )
                    if use_live
                    else FormatterRenderer(formatter, console)
                )
                await stream_response(
                    client, message, renderer, suppress_thinking=not show_reasoning
                )
            else:
                # Use streaming pipeline with BufferedRenderer to capture
                # tool calls and reasoning that agent.chat() would swallow
                from victor.ui.rendering import (
                    BufferedRenderer,
                    stream_response,
                )

                buffered = BufferedRenderer(
                    show_reasoning=show_reasoning,
                    plain=formatter._plain if hasattr(formatter, "_plain") else False,
                )
                await stream_response(
                    client, message, buffered, suppress_thinking=not show_reasoning
                )
                if show_cli_chrome:
                    buffered.flush(console)
                else:
                    formatter.response(content=buffered.finalize())
                    # Surface buffered statuses (provider errors land here via
                    # EventDispatcher ERROR routing) so a failed turn is never
                    # silent in pipe-friendly output.
                    for status_message in buffered.statuses():
                        formatter.info(status_message)

            await client.get_session_metrics()

    except InitializationError as e:
        formatter.error(f"{e.stage}: {e.message}")
        for s in e.suggestions:
            formatter.info(f"  → {s}")
        if e.run_command:
            formatter.info(f"Run: {e.run_command}")
        raise typer.Exit(1)
    except typer.Exit:
        raise
    except Exception as e:
        # Use contextual error formatting for better UX
        error_message = format_exception_for_user(e)
        formatter.error(error_message)

        # Suggest provider switching for provider-specific errors
        error_str = str(e).lower()
        is_provider_error = any(
            term in error_str
            for term in [
                "api key",
                "unauthorized",
                "rate limit",
                "timeout",
                "connection",
                "network",
            ]
        )

        if is_provider_error:
            if provider_for_suggestions != "ollama":
                formatter.info(
                    "💡 Try switching to a different provider:\n"
                    "  victor chat -p ollama  # Local models (free)\n"
                    "  victor chat -p openai  # GPT models"
                )
            else:
                formatter.info(
                    "💡 Try switching to a cloud provider:\n"
                    "  victor chat -p anthropic  # Claude models\n"
                    "  victor chat -p openai  # GPT models"
                )

        formatter.info("Run 'victor doctor' for diagnostics")

        # Show traceback in debug mode only
        if os.getenv("VICTOR_DEBUG"):
            import traceback

            formatter.error(traceback.format_exc())

        raise typer.Exit(1)
    finally:
        _duration = time.time() - start_time
        if client is not None:
            await client.close()


async def run_interactive(
    settings: Any,
    profile: str,
    stream: bool,
    thinking: bool = False,
    preindex: bool = False,
    renderer_choice: str = "auto",
    mode: Optional[str] = None,
    tool_budget: Optional[int] = None,
    max_iterations: Optional[int] = None,
    vertical: Optional[str] = None,
    enable_observability: bool = True,
    enable_planning: Optional[bool] = None,
    planning_model: Optional[str] = None,
    resume_session_id: Optional[str] = None,
    show_reasoning: bool = False,
    graph_watch: bool = True,
    compaction_threshold: Optional[float] = None,
    adaptive_threshold: Optional[bool] = None,
    compaction_min_threshold: Optional[float] = None,
    compaction_max_threshold: Optional[float] = None,
    # Smart routing parameters
    enable_smart_routing: bool = False,
    routing_profile: str = "balanced",
    fallback_chain: Optional[str] = None,
    # Tool output preview parameters
    tool_preview: bool = True,
    enable_pruning: bool = False,
    # Bayesian orchestration parameters
    enable_bayesian: bool = True,
    force_bayesian: bool = False,
    simple_threshold: float = 0.3,
    complex_threshold: float = 0.7,
    enable_voi: bool = True,
    enable_correlation: bool = True,
    min_agents_for_bayesian: int = 2,
    session_config: Optional[SessionConfig] = None,
    surface: str = "auto",
    tui_theme: str = "dark",
) -> None:
    """Run interactive CLI mode.

    Uses VictorClient to bridge CLI with framework features like
    observability and vertical configuration.

    Args:
        show_reasoning: If True, show LLM reasoning/thinking content in output.
        surface: ``"repl"`` (default) for the prompt-toolkit REPL, or ``"tui"`` to
            launch the interactive Textual UI when the terminal can host it.
    """
    config = session_config or _build_session_config(
        agent_profile=profile,
        tool_budget=tool_budget,
        max_iterations=max_iterations,
        compaction_threshold=compaction_threshold,
        adaptive_threshold=adaptive_threshold,
        compaction_min_threshold=compaction_min_threshold,
        compaction_max_threshold=compaction_max_threshold,
        enable_smart_routing=enable_smart_routing,
        routing_profile=routing_profile,
        fallback_chain=fallback_chain,
        tool_preview=tool_preview,
        enable_pruning=enable_pruning,
        planning_enabled=enable_planning,
        planning_model=planning_model,
        mode=mode,
        show_reasoning=show_reasoning,
        enable_bayesian=enable_bayesian,
        force_bayesian=force_bayesian,
        simple_threshold=simple_threshold,
        complex_threshold=complex_threshold,
        enable_voi=enable_voi,
        enable_correlation=enable_correlation,
        min_agents_for_bayesian=min_agents_for_bayesian,
    )
    session_runner = FrameworkSessionRunner(settings, config)
    prepared_state = session_runner.prepare_state(
        one_shot_mode=False,
        stream=stream,
        show_reasoning=show_reasoning,
    )
    config = prepared_state.config
    show_reasoning = prepared_state.show_reasoning
    stream = prepared_state.use_streaming

    agent = None
    client = None  # ✅ NEW: VictorClient (replaces orchestrator/shim)
    watcher_init_task: Optional[asyncio.Task[None]] = None
    show_startup_cli_chrome = _should_render_interactive_tool_banner()
    smart_routing_status_shown = False
    compaction_status_shown = False
    graph_watch_handle = ChatGraphWatchHandle(messages=[])

    _configure_smart_routing(
        settings,
        console,
        enable_smart_routing,
        routing_profile,
        fallback_chain,
        show_status=show_startup_cli_chrome,
    )
    smart_routing_status_shown = show_startup_cli_chrome and enable_smart_routing
    # Configure tool output preview (safe-default pruning)
    from victor.config.tool_settings import ToolSettings

    tool_settings = settings.tool_settings if hasattr(settings, "tool_settings") else ToolSettings()
    tool_banner_shown = False
    if show_startup_cli_chrome:
        _print_tool_output_mode_banner(console, tool_settings)
        tool_banner_shown = True
    try:
        profiles, profile_config = _require_existing_non_default_profile(settings, profile)
    except ValueError as exc:
        console.print(f"[bold red]Error:[/] {exc}")
        raise typer.Exit(1) from exc

    graph_watch_handle = _ensure_graph_watch_handle_for_chat(enabled=graph_watch)
    _print_interactive_startup_messages(console, graph_watch_handle.messages)

    try:
        profile_display = _resolve_profile_display(
            config=config,
            profile_config=profile_config,
            settings=settings,
        )

        # Unified initialization via AgentFactory
        from victor.framework.agent_factory import InitializationError

        # ✅ PROPER: Create VictorClient (replaces AgentFactory)
        try:
            client = session_runner.create_client(config)
            agent = await session_runner.initialize_client(
                client,
                planning_model=planning_model,
            )

            # Resume session if requested
            if resume_session_id:
                from victor.agent.sqlite_session_persistence import (
                    get_sqlite_session_persistence,
                )
                from victor.agent.message_history import MessageHistory
                from victor.agent.conversation_state import ConversationStateMachine

                persistence = get_sqlite_session_persistence()
                session_data = persistence.load_session(resume_session_id)

                if not session_data:
                    console.print(f"[bold red]Error:[/ ] Session not found: {resume_session_id}")
                    raise typer.Exit(1)

                # Restore conversation
                metadata = session_data.get("metadata", {})
                conversation_dict = session_data.get("conversation", {})

                agent.conversation = MessageHistory.from_dict(conversation_dict)
                agent.active_session_id = resume_session_id

                # Restore conversation state if available
                conversation_state_dict = session_data.get("conversation_state")
                if conversation_state_dict:
                    try:
                        agent.conversation_state = ConversationStateMachine.from_dict(
                            conversation_state_dict
                        )
                    except Exception as e:
                        console.print(
                            f"[yellow]Warning:[/] Failed to restore conversation state: {e}"
                        )

                resumed_message = (
                    f"Resumed session: {metadata.get('title', 'Untitled')} "
                    f"({metadata.get('message_count', 0)} messages)"
                )
                console.print(f"[green]✓[/] {resumed_message}\n")

            # FEP-0012 Phase 0: ensure a non-empty session_id so every decision
            # logged this session can join to its outcome. Fresh chat starts with
            # active_session_id=None (→ get_session_id() returns ""), which would
            # make record_session_outcome a no-op. Mirror the benchmark adapter
            # (agent_adapter.py). Resumed sessions keep their id from above.
            try:
                import uuid as _uuid

                from victor.core.context import set_session_id

                _chat_sid = getattr(agent, "active_session_id", None) or _uuid.uuid4().hex
                agent.active_session_id = _chat_sid
                _orch = agent.get_orchestrator() if hasattr(agent, "get_orchestrator") else None
                if _orch is not None:
                    _orch.active_session_id = _chat_sid
                set_session_id(_chat_sid)
            except Exception:
                pass

            # Note: Observability (shim) is handled by AgentFactory internally
            # The factory creates the agent with framework features already wired

            if _should_start_file_watchers_on_startup(settings):
                # Optional eager watcher mode. Default is demand-driven startup
                # through graph/code_search so large repos do not cold-scan here.
                watcher_init_task = _start_file_watcher_initialization(settings)
        except InitializationError as e:
            console.print(f"[red]Error ({e.stage}):[/] {e.message}")
            for s in e.suggestions:
                console.print(f"  → {s}")
            if e.run_command:
                console.print(f"\nRun: [bold]{e.run_command}[/]")
            raise typer.Exit(1)

        if tool_budget is not None:
            agent.set_tool_budget(tool_budget, user_override=True)
        if max_iterations is not None:
            agent.set_max_iterations(max_iterations, user_override=True)
        _configure_agent_compaction(
            agent,
            compaction_threshold=compaction_threshold,
            adaptive_threshold=adaptive_threshold,
            compaction_min_threshold=compaction_min_threshold,
            compaction_max_threshold=compaction_max_threshold,
            con=console,
            show_status=show_startup_cli_chrome,
        )
        compaction_status_shown = show_startup_cli_chrome and any(
            (
                compaction_threshold is not None,
                adaptive_threshold is not None,
                compaction_min_threshold is not None,
                compaction_max_threshold is not None,
            )
        )
        if mode:
            try:
                session_runner.apply_agent_mode(mode)
            except Exception:
                pass

        if enable_smart_routing and not smart_routing_status_shown:
            _configure_smart_routing(
                settings,
                console,
                enable_smart_routing,
                routing_profile,
                fallback_chain,
                show_status=True,
            )
            smart_routing_status_shown = True
        if not tool_banner_shown:
            _print_tool_output_mode_banner(console, tool_settings)
            tool_banner_shown = True
        if not compaction_status_shown and any(
            (
                compaction_threshold is not None,
                adaptive_threshold is not None,
                compaction_min_threshold is not None,
                compaction_max_threshold is not None,
            )
        ):
            _configure_agent_compaction(
                agent,
                compaction_threshold=compaction_threshold,
                adaptive_threshold=adaptive_threshold,
                compaction_min_threshold=compaction_min_threshold,
                compaction_max_threshold=compaction_max_threshold,
                con=console,
                show_status=True,
            )
            compaction_status_shown = True
        from victor.ui.commands import SlashCommandHandler

        cmd_handler = SlashCommandHandler(console, settings, agent)

        # Interactive TUI (ADR-020/021) is the default on capable terminals; the REPL is
        # the fallback for dumb terminals / pipes / CI. ``surface="auto"`` (the default)
        # selects silently by capability; an explicit ``--tui`` on a terminal that can't
        # host it says so before falling back; ``--repl`` forces the REPL.
        if surface in ("tui", "auto"):
            if _tui_capable():
                await _run_tui_app(
                    client,
                    agent,
                    settings,
                    mode=mode,
                    tool_budget=tool_budget,
                    theme=tui_theme,
                )
                return
            if surface == "tui":
                console.print(
                    "[yellow]This terminal can't host the interactive TUI; "
                    "using the REPL instead.[/]"
                )

        rl_suggestion = get_rl_profile_suggestion(profile_display.provider, profiles)
        await _run_cli_repl(
            agent,
            settings,
            cmd_handler,
            profile_display,
            stream,
            rl_suggestion,
            renderer_choice=renderer_choice,
            vertical_name=vertical,
            enable_planning=enable_planning,
            show_reasoning=show_reasoning,
            profile_name=profile,
            startup_messages=None,
        )

        if hasattr(agent, "get_session_metrics"):
            metrics = agent.get_session_metrics()
            _ = metrics.get("tool_calls", 0) if metrics else 0

    except Exception as e:
        # Use contextual error formatting for better UX
        error_message = format_exception_for_user(e)
        # format_exception_for_user already includes markup, so print directly
        console.print(error_message)

        # Show traceback in debug mode only
        import os

        if os.getenv("VICTOR_DEBUG"):
            import traceback

            console.print(traceback.format_exc())

        # Suggest provider switching for provider-specific errors
        error_str = str(e).lower()
        is_provider_error = any(
            term in error_str
            for term in [
                "api key",
                "unauthorized",
                "rate limit",
                "timeout",
                "connection",
                "network",
            ]
        )

        if is_provider_error:
            console.print(
                "\n[yellow]💡 Try switching to a different provider:[/]"
                "\n  [dim]victor chat -p ollama[/]  # Local models (free)"
                "\n  [dim]victor chat -p anthropic[/]  # Claude models"
            )

        console.print("\n[yellow]💡 Run 'victor doctor' for diagnostics[/]")
        raise typer.Exit(1)
    finally:
        _cleanup_graph_watch_for_chat(graph_watch_handle)
        await _cancel_file_watcher_initialization(watcher_init_task)

        # Emit session end event
        # ✅ PROPER: No shim needed - session cleanup handled by agent/services
        # Note: shim.emit_session_end() removed (FrameworkShim no longer used)
        if agent:
            await graceful_shutdown(agent)

        # Cleanup file watchers
        try:
            from victor.core.indexing.watcher_initializer import cleanup_session

            await cleanup_session()
        except Exception as e:
            # Non-fatal: log warning but don't fail shutdown
            console.print(f"[dim]Note: Failed to cleanup file watchers: {e}[/]")


def _prune_history_file(history_file, max_entries: int = 250) -> None:
    """Sanitize history file and keep only the most recent complete entries.

    This prevents the history file from growing unbounded, which would
    slow down prompt_toolkit's history search on every keystroke.

    Args:
        history_file: Path to history file
        max_entries: Maximum number of entries to keep (default 200)
    """
    try:
        sanitize_prompt_toolkit_history_file(history_file, max_entries=max_entries)
    except Exception as e:
        import logging

        logger = logging.getLogger(__name__)
        logger.debug(f"Failed to prune history file {history_file}: {e}")


def _maybe_rotate_history_file(history_file, max_entries: int = 250) -> None:
    """Periodically rotate history file if it grows too large.

    This is called periodically to prevent the history file from
    growing unbounded over time.

    Args:
        history_file: Path to history file
        max_entries: Maximum entries before rotation
    """
    if not history_file.exists():
        return

    try:
        import time

        # Only check once per session
        rotation_marker = history_file.with_suffix(".last_rotation")

        if rotation_marker.exists():
            last_rotation = rotation_marker.stat().st_mtime
            # Only rotate if more than 7 days since last rotation
            if time.time() - last_rotation < 7 * 24 * 3600:
                return

        entry_count = count_prompt_toolkit_history_entries(history_file)

        # If too large, prune
        if entry_count > max_entries * 1.5:  # 50% buffer
            _prune_history_file(history_file, max_entries)

            # Update rotation marker
            rotation_marker.touch()

    except Exception as e:
        import logging

        logger = logging.getLogger(__name__)
        logger.debug(f"Failed to rotate history file: {e}")


def _check_history_health(history_file, max_entries: int = 250) -> None:
    """Check history file health and warn if too large.

    Args:
        history_file: Path to history file
        max_entries: Configured maximum entries (for warning threshold)
    """
    if not history_file.exists():
        return

    try:
        entry_count = count_prompt_toolkit_history_entries(history_file)
        size_kb = history_file.stat().st_size / 1024

        # Warning threshold: 2x the configured max_entries or 500, whichever is higher
        warning_threshold = max(max_entries * 2, 500)
        size_warning_threshold = 50  # KB

        if entry_count > warning_threshold or size_kb > size_warning_threshold:
            console.print(
                f"[yellow]Warning:[/] CLI history file is large "
                f"({entry_count:,} entries, {size_kb:.1f} KB). "
                f"This may slow down typing. "
                f"Run 'victor chat cleanup-history' to optimize."
            )
    except Exception:
        pass


def _resolve_cli_display_values(
    *,
    settings: Any = None,
    profile_config: Any = None,
    profile_name: str = "default",
    vertical_name: Optional[str] = None,
) -> dict[str, str]:
    """Return compact prompt status values for the prompt-toolkit chrome."""
    provider = getattr(profile_config, "provider", None)
    model = getattr(profile_config, "model", None)
    if settings is not None and hasattr(settings, "provider"):
        provider = provider or getattr(settings.provider, "default_provider", None)
        model = model or getattr(settings.provider, "default_model", None)
    return {
        "profile": profile_name or "default",
        "provider": str(provider or "provider"),
        "model": str(model or "model"),
        "vertical": str(vertical_name or "core"),
    }


def _truncate_cli_value(value: str, max_len: int) -> str:
    """Keep toolbar values readable in narrow terminals."""
    text = str(value or "")
    if max_len <= 1 or len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"


def _chat_mode_label(enable_planning: Optional[bool]) -> str:
    if enable_planning is True:
        return "plan"
    if enable_planning is False:
        return "chat"
    return "auto"


def _coerce_cli_int(value: Any) -> Optional[int]:
    """Return value as int when it is a finite, non-negative count."""
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number >= 0 else None


def _format_cli_count(value: int) -> str:
    """Format large counts compactly for a single-line toolbar."""
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}m".replace(".0m", "m")
    if value >= 1_000:
        return f"{value / 1_000:.1f}k".replace(".0k", "k")
    return str(value)


def _extract_cli_message_content(message: Any) -> str:
    """Best-effort message content extraction for lightweight context estimates."""
    if isinstance(message, dict):
        content = message.get("content", "")
    else:
        content = getattr(message, "content", "")

    if isinstance(content, str):
        return content
    if content is None:
        return ""
    return str(content)


def _collect_cli_messages(agent: Any) -> Optional[list[Any]]:
    """Collect current conversation messages without assembling or scoring context."""
    if agent is None:
        return None

    get_messages = getattr(agent, "get_messages", None)
    if callable(get_messages):
        try:
            messages = get_messages()
            if messages is not None:
                return list(messages)
        except Exception:
            pass

    conversation = getattr(agent, "conversation", None)
    messages = getattr(conversation, "messages", None)
    if messages is not None:
        try:
            return list(messages)
        except TypeError:
            return None

    messages = getattr(agent, "messages", None)
    if messages is not None:
        try:
            return list(messages)
        except TypeError:
            return None

    return None


def _resolve_cli_context_window(provider: str, model: str) -> Optional[int]:
    """Resolve configured model context window from cached local provider limits."""
    if not provider or provider == "provider":
        return None
    try:
        from victor.config.config_loaders import get_provider_limits

        return _coerce_cli_int(get_provider_limits(provider, model).context_window)
    except Exception:
        return None


def _build_cli_runtime_segment(
    *,
    agent: Any = None,
    settings: Any = None,
    provider: str = "",
    model: str = "",
    compact: bool = False,
) -> Optional[str]:
    """Build live runtime budget/status text when reliable state is available."""
    if agent is None:
        return None

    parts: list[str] = []
    tool_calls_used = _coerce_cli_int(getattr(agent, "tool_calls_used", None))
    # No settings fallback: settings.tools.tool_call_budget is the whole-session cap
    # (BUDGET_LIMITS.max_session_budget = 2000), not the per-turn budget this counter
    # is measured against. Pairing them showed "Tools 16/2000" while 20 was enforced.
    tool_budget = _coerce_cli_int(getattr(agent, "tool_budget", None))
    if tool_calls_used is not None and tool_budget is not None:
        label = "t" if compact else "Tools"
        parts.append(f"{label} {tool_calls_used}/{tool_budget}")

    iterations = _coerce_cli_int(getattr(agent, "iteration_count", None))
    max_iterations = _coerce_cli_int(getattr(agent, "max_iterations", None))
    if iterations is not None and max_iterations is not None:
        label = "i" if compact else "Iter"
        parts.append(f"{label} {iterations}/{max_iterations}")

    messages = _collect_cli_messages(agent)
    conversation = getattr(agent, "conversation", None)
    message_count = None
    if conversation is not None and callable(getattr(conversation, "message_count", None)):
        try:
            message_count = _coerce_cli_int(conversation.message_count())
        except Exception:
            message_count = None
    if message_count is None and messages is not None:
        message_count = len(messages)
    if message_count is not None:
        label = "msg" if compact else "Msg"
        parts.append(f"{label} {message_count}")

    if messages:
        estimated_tokens = (
            sum(len(_extract_cli_message_content(message)) for message in messages) // 4
        )
        context_window = _resolve_cli_context_window(provider, model)
        if estimated_tokens > 0 and context_window:
            label = "ctx" if compact else "Ctx"
            parts.append(
                f"{label} ~{_format_cli_count(estimated_tokens)}/"
                f"{_format_cli_count(context_window)}"
            )

    return " ".join(parts) if parts else None


def _cli_work_status_message(enable_planning: Optional[bool]) -> str:
    """Return the transient status shown while non-streaming chat is running."""
    if enable_planning is True:
        return "Planning..."
    return "Thinking..."


def _build_cli_prompt_fragments(profile_name: str = "default") -> list[tuple[str, str]]:
    """Build a styled prompt that stays compact in narrow terminals."""
    label = profile_name or "default"
    return [
        ("class:prompt.brand", "victor"),
        ("class:prompt.profile", f"[{label}]"),
        ("class:prompt.arrow", " > "),
    ]


def _build_cli_bottom_toolbar(
    *,
    agent: Any = None,
    settings: Any = None,
    profile_config: Any = None,
    profile_name: str = "default",
    vertical_name: Optional[str] = None,
    enable_planning: Optional[bool] = None,
    stream: Optional[bool] = None,
    renderer_choice: str = "auto",
    show_reasoning: bool = False,
    width: Optional[int] = None,
) -> list[tuple[str, str]]:
    """Build a professional, glanceable toolbar for prompt-toolkit CLI mode."""
    values = _resolve_cli_display_values(
        settings=settings,
        profile_config=profile_config,
        profile_name=profile_name,
        vertical_name=vertical_name,
    )
    columns = width or shutil.get_terminal_size((120, 24)).columns
    compact = columns < 100
    provider = _truncate_cli_value(values["provider"], 14 if compact else 20)
    model = _truncate_cli_value(values["model"], 18 if compact else 32)
    profile = _truncate_cli_value(values["profile"], 14 if compact else 20)
    vertical = _truncate_cli_value(values["vertical"], 12 if compact else 18)
    mode = _chat_mode_label(enable_planning)
    stream_label = "stream" if stream else "sync"
    renderer = _truncate_cli_value(renderer_choice or "auto", 8)
    reasoning = "reason" if show_reasoning else "hide-thoughts"
    runtime_segment = _build_cli_runtime_segment(
        agent=agent,
        settings=settings,
        provider=values["provider"],
        model=values["model"],
        compact=compact,
    )

    if compact:
        fragments = [
            ("class:toolbar.label", " "),
            ("class:toolbar.value", profile),
            ("class:toolbar.separator", " | "),
            ("class:toolbar.value", f"{provider}/{model}"),
            ("class:toolbar.separator", " | "),
            ("class:toolbar.value", f"{mode}/{stream_label}"),
        ]
        if runtime_segment:
            fragments.extend(
                [
                    ("class:toolbar.separator", " | "),
                    ("class:toolbar.value", runtime_segment),
                ]
            )
        fragments.extend(
            [
                ("class:toolbar.separator", " | "),
                (
                    "class:toolbar.hint",
                    "Enter send  Alt+Enter newline  Tab cmds  Ctrl+O expand  F1 help",
                ),
            ]
        )
        return fragments

    fragments = [
        ("class:toolbar.label", " Profile "),
        ("class:toolbar.value", profile),
        ("class:toolbar.separator", "  |  "),
        ("class:toolbar.label", "Provider "),
        ("class:toolbar.value", provider),
        ("class:toolbar.separator", " / "),
        ("class:toolbar.value", model),
        ("class:toolbar.separator", "  |  "),
        ("class:toolbar.label", "Context "),
        ("class:toolbar.value", vertical),
        ("class:toolbar.separator", "  |  "),
        ("class:toolbar.label", "Mode "),
        ("class:toolbar.value", mode),
        ("class:toolbar.separator", " / "),
        ("class:toolbar.value", stream_label),
        ("class:toolbar.separator", " / "),
        ("class:toolbar.value", renderer),
        ("class:toolbar.separator", " / "),
        ("class:toolbar.value", reasoning),
    ]
    if runtime_segment:
        fragments.extend(
            [
                ("class:toolbar.separator", "  |  "),
                ("class:toolbar.value", runtime_segment),
            ]
        )
    fragments.extend(
        [
            ("class:toolbar.separator", "  |  "),
            (
                "class:toolbar.hint",
                "Enter send  Alt+Enter newline  Up/Down history  Tab commands  "
                "Esc clear  Ctrl+O expand  F1 help",
            ),
        ]
    )
    return fragments


def _build_cli_shortcuts_panel() -> Panel:
    """Build prompt-toolkit shortcut help for the interactive CLI."""
    table = Table.grid(padding=(0, 2))
    table.add_column("Key", style="cyan", no_wrap=True)
    table.add_column("Action", style="white")
    table.add_row("Enter", "send the current message")
    table.add_row("Alt+Enter", "insert a newline")
    table.add_row("Tab", "complete commands and known command arguments")
    table.add_row("Up/Down", "browse prompt history")
    table.add_row("Esc", "clear the current input")
    table.add_row("Ctrl+O", "expand the last tool output")
    table.add_row("Ctrl+D", "exit chat")
    table.add_row("F1 or /shortcuts", "show this shortcut reference")

    return Panel(
        table,
        title="CLI Shortcuts",
        border_style="cyan",
    )


def _build_cli_right_prompt() -> list[tuple[str, str]]:
    """Build compact right-side prompt hints for prompt-toolkit CLI mode."""
    return [
        ("class:rprompt.hint", "Ctrl+D exit"),
    ]


CLI_COMMAND_COMPLETIONS = {
    "/help": "show slash-command help",
    "/shortcuts": "show prompt keyboard shortcuts",
    "/keys": "show prompt keyboard shortcuts",
    "/model": "inspect or switch the active model",
    "/mode": "switch build, plan, review, delegate, or explore mode",
    "/profiles": "list and inspect configured profiles",
    "/provider": "inspect or switch provider",
    "/tools": "list available tools",
    "/sessions": "list or resume saved sessions",
    "/skills": "list available skills",
    "/plugins": "inspect plugin state",
    "/mcp": "inspect MCP servers",
    "/status": "show runtime status",
    "/history": "show conversation history",
    "/resume": "resume a saved conversation",
    "/clear": "clear the current conversation",
    "/exit": "leave interactive chat",
    "/quit": "leave interactive chat",
    "/expand": "expand the last tool output",
    "/e": "expand the last tool output",
    "clear": "clear the current conversation",
    "exit": "leave interactive chat",
    "quit": "leave interactive chat",
}

CLI_COMMAND_ALIASES = (
    ("/?", "/help", "alias for /help"),
    (":help", "/shortcuts", "alias for /shortcuts"),
    (":q", "/quit", "alias for /quit"),
    (":clear", "/clear", "alias for /clear"),
)

CLI_COMMAND_ARGUMENT_COMPLETIONS = {
    "/mode": (
        ("build", "implementation mode"),
        ("plan", "planning and research mode"),
        ("review", "findings-first review mode"),
        ("delegate", "parallel-work delegation mode"),
        ("explore", "code navigation mode"),
    ),
    "/plan": (
        ("save", "save the current plan"),
        ("load", "load a saved plan"),
        ("list", "list saved plans"),
        ("show", "show current plan details"),
    ),
    "/model": (
        ("list", "list available models"),
        ("--resume", "resume after switching model"),
    ),
    "/provider": (
        ("openai", "OpenAI provider"),
        ("anthropic", "Anthropic provider"),
        ("ollama", "local Ollama provider"),
        ("lmstudio", "local LM Studio provider"),
        ("vllm", "vLLM-compatible provider"),
        ("zai", "Z.ai provider"),
        ("groq", "Groq provider"),
    ),
}


def _collect_cli_command_metadata() -> dict[str, str]:
    """Return slash command completion metadata from the registry plus CLI-only commands."""
    commands = dict(CLI_COMMAND_COMPLETIONS)
    commands.update({alias: description for alias, _target, description in CLI_COMMAND_ALIASES})

    try:
        from victor.ui.slash import get_command_registry

        registry = get_command_registry()
        if not any(registry.iter_commands()):
            registry.discover_commands()

        for name, meta in registry.list_commands():
            command_name = f"/{name}"
            commands.setdefault(command_name, meta.description)
            for alias in meta.aliases:
                commands.setdefault(f"/{alias}", f"alias for {command_name}")
    except Exception:
        pass

    return commands


def _build_cli_command_completer():
    """Create a metadata-rich command completer for the prompt-toolkit CLI."""
    from prompt_toolkit.completion import Completer, Completion

    commands = _collect_cli_command_metadata()
    alias_targets = {alias: target for alias, target, _description in CLI_COMMAND_ALIASES}

    class VictorCliCommandCompleter(Completer):
        """Complete slash and command-mode inputs without scanning arbitrary prompt text."""

        def get_completions(self, document, complete_event):
            text = document.text_before_cursor
            stripped = text.lstrip()
            if not stripped or text[: len(text) - len(stripped)].strip():
                return

            token = stripped.split(maxsplit=1)[0]
            if not (token.startswith("/") or token.startswith(":") or token.isalpha()):
                return

            if " " in stripped:
                parts = stripped.split(maxsplit=1)
                if len(parts) < 2:
                    command = parts[0]
                    arg_prefix = ""
                else:
                    command, arg_prefix = parts
                command = alias_targets.get(command, command)
                options = CLI_COMMAND_ARGUMENT_COMPLETIONS.get(command)
                if not options:
                    return

                active_arg = arg_prefix.split()[-1] if arg_prefix.split() else ""
                active_arg_lower = active_arg.lower()
                for value, description in options:
                    if value.lower().startswith(active_arg_lower):
                        yield Completion(
                            value,
                            start_position=-len(active_arg),
                            display=value,
                            display_meta=description,
                        )
                return

            token_lower = token.lower()
            for command, description in commands.items():
                if command.lower().startswith(token_lower):
                    yield Completion(
                        command,
                        start_position=-len(token),
                        display=command,
                        display_meta=description,
                    )

    return VictorCliCommandCompleter()


def _normalize_cli_input_alias(user_input: str) -> str:
    """Normalize prompt-toolkit command aliases before REPL dispatch."""
    stripped = user_input.strip()
    alias_map = {alias: target for alias, target, _description in CLI_COMMAND_ALIASES}
    return alias_map.get(stripped, user_input)


def _cli_mouse_support_enabled() -> bool:
    """Return whether interactive chat should let prompt_toolkit capture mouse events."""
    return os.getenv("VICTOR_CHAT_MOUSE_SUPPORT", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


@chat_app.command("cleanup-history")
def cleanup_history_command(
    max_entries: int = typer.Option(
        250,
        "--max-entries",
        "-m",
        help="Maximum entries to keep (default: 250, range: 10-1000).",
        min=10,
        max=1000,
    ),
):
    """Clean up CLI chat history to improve typing performance.

    Reduces the history file to the most recent entries, which improves
    prompt_toolkit's search performance and makes typing feel snappier.

    **Examples:**
        victor chat cleanup-history          # Keep default 250 entries
        victor chat cleanup-history -m 500   # Keep 500 entries
    """
    from victor.config.settings import get_project_paths

    history_file = get_project_paths().project_victor_dir / "chat_history"

    if not history_file.exists():
        console.print("[green]✓[/] No history file to clean up")
        return

    try:
        original_entries = count_prompt_toolkit_history_entries(history_file)
    except Exception as e:
        console.print(f"[bold red]Error:[/] Failed to read history file: {e}")
        raise typer.Exit(1)

    # Prune the history file
    _prune_history_file(history_file, max_entries=max_entries)

    # Count new entries
    try:
        new_entries = count_prompt_toolkit_history_entries(history_file)
    except Exception as e:
        console.print(f"[bold red]Error:[/] Failed to read history file after pruning: {e}")
        raise typer.Exit(1)

    removed = original_entries - new_entries
    size_kb = history_file.stat().st_size / 1024

    console.print("[green]✓[/] Cleaned up history file:")
    console.print(f"  Entries: {original_entries:,} → {new_entries:,} (removed {removed:,})")
    console.print(f"  File size: {size_kb:.1f} KB")
    console.print()
    console.print("[dim]Typing should feel snappier now![/]")


def _create_cli_prompt_session(
    agent: Any = None,
    settings=None,
    profile_config: Any = None,
    profile_name: str = "default",
    vertical_name: Optional[str] = None,
    enable_planning: Optional[bool] = None,
    stream: Optional[bool] = None,
    renderer_choice: str = "auto",
    show_reasoning: bool = False,
):
    """Create a prompt_toolkit PromptSession with persistent history.

    Loads previous user messages from the conversation database and persists
    new input to .victor/chat_history for project-specific history.

    Args:
        settings: Optional settings object (to avoid redundant load_settings call)

    Returns:
        Tuple of (PromptSession, renderer_holder_dict) where renderer_holder_dict
        is a mutable container for storing the current renderer reference.
    """
    from prompt_toolkit import PromptSession
    from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
    from prompt_toolkit.history import FileHistory, InMemoryHistory
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.keys import Keys
    from prompt_toolkit.shortcuts import CompleteStyle
    from prompt_toolkit.styles import Style

    # Get max_entries from settings or use default
    if settings and hasattr(settings, "ui"):
        max_entries = settings.ui.cli_history_max_entries
    else:
        max_entries = 250  # Default limit for CLI history

    # Mutable container for renderer reference (shared with key bindings)
    renderer_holder = {"ref": None}

    # Use a persistent history file in project-specific .victor/
    try:
        from victor.config.settings import get_project_paths

        history_file = get_project_paths().project_victor_dir / "chat_history"
        history_file.parent.mkdir(parents=True, exist_ok=True)

        # Prune history file if too large
        _prune_history_file(history_file, max_entries=max_entries)

        history = FileHistory(str(history_file))

        # Seed from conversation DB if history file is empty/new
        if not history_file.exists() or history_file.stat().st_size == 0:
            try:
                db_path = get_project_paths().project_db
                for msg in load_input_history_from_db(db_path, limit=min(max_entries, 100)):
                    history.store_string(msg)

                # Prune again after seeding to ensure we stay at max_entries
                _prune_history_file(history_file, max_entries=max_entries)
            except Exception:
                pass  # DB seeding is best-effort

    except Exception:
        history = InMemoryHistory()

    # Create key bindings for the session
    key_bindings = KeyBindings()

    @key_bindings.add("escape")
    def _clear_input(event):
        """Clear the active prompt input without leaving the interactive session."""
        event.app.current_buffer.reset()

    @key_bindings.add("enter")
    def _send_input(event):
        """Send the active prompt input, even when multiline compose is enabled."""
        event.app.current_buffer.validate_and_handle()

    @key_bindings.add("escape", "enter")
    def _insert_newline(event):
        """Insert a newline without taking over terminal scrollback or selection."""
        event.app.current_buffer.insert_text("\n")

    # Ctrl+O hotkey to expand last tool output.
    # renderer_holder is updated each turn so the binding always refers to
    # the most recent renderer instance.
    @key_bindings.add(Keys.ControlO)
    def _expand_output(event):
        """Expand last tool output to show full content."""
        renderer = renderer_holder["ref"]
        if renderer and hasattr(renderer, "expand_last_output"):
            # run_in_terminal temporarily suspends the prompt, prints, then restores it —
            # without this, console.print() inside a key binding gets overwritten immediately
            # by prompt_toolkit's own prompt re-render.
            event.app.run_in_terminal(renderer.expand_last_output)

    # F1 to show interactive shortcut help without disturbing the prompt.
    @key_bindings.add(Keys.F1)
    def _show_shortcuts(event):
        """Show interactive shortcut help without disturbing the prompt."""
        event.app.run_in_terminal(lambda: console.print(_build_cli_shortcuts_panel()))

    prompt_style = Style.from_dict(
        {
            "prompt.brand": "bold #7dd3fc",
            "prompt.profile": "#a7f3d0",
            "prompt.arrow": "bold #fbbf24",
            "toolbar.label": "#9ca3af",
            "toolbar.value": "bold #e5e7eb",
            "toolbar.separator": "#4b5563",
            "toolbar.hint": "#93c5fd",
            "rprompt.hint": "#6b7280",
            "rprompt.separator": "#374151",
        }
    )

    prompt_session = PromptSession(
        history=history,
        key_bindings=key_bindings,
        completer=_build_cli_command_completer(),
        auto_suggest=AutoSuggestFromHistory(),
        bottom_toolbar=lambda: _build_cli_bottom_toolbar(
            agent=agent,
            settings=settings,
            profile_config=profile_config,
            profile_name=profile_name,
            vertical_name=vertical_name,
            enable_planning=enable_planning,
            stream=stream,
            renderer_choice=renderer_choice,
            show_reasoning=show_reasoning,
        ),
        rprompt=_build_cli_right_prompt,
        style=prompt_style,
        complete_style=CompleteStyle.MULTI_COLUMN,
        reserve_space_for_menu=8,
        mouse_support=_cli_mouse_support_enabled(),
        multiline=True,
        wrap_lines=True,
        complete_while_typing=False,  # Keep completions explicit and cheap for large histories
        enable_history_search=False,  # Disable history search on typing for better performance
    )

    return prompt_session, renderer_holder


def _maybe_prompt_outcome(agent: Any, settings: Any, console: "Console") -> None:
    """FEP-0012 (passive): log a ``task_completion`` decision on fresh completions.

    On a fresh completion (the orchestrator's task-completion detector flips
    False→True across this turn, tracked per-session on the orchestrator) we
    log a ``task_completion`` decision so the session's decisions can join to
    an outcome. Reward recording is automatic (the serve→attribute→reward
    loop); explicit feedback lives in ``/rate`` — a one-time hint is printed
    instead of the old blocking "Did this resolve your task?" prompt, which
    only fires anymore if the deprecated ``enable_rl_feedback_prompt`` is
    explicitly set True.

    Non-invasive: uses a transition signal (not a detector reset), so it never
    alters completion-detection behavior. Interactive REPL only; best-effort —
    never raises into the REPL.
    """
    try:
        orch = agent.get_orchestrator() if hasattr(agent, "get_orchestrator") else None
        if orch is None:
            return
        detector = getattr(orch, "_task_completion_detector", None)
        if detector is None or not hasattr(detector, "should_stop"):
            return

        now_complete = bool(detector.should_stop())
        prev_complete = getattr(orch, "_rl_prev_completed", False)
        orch._rl_prev_completed = now_complete
        if not (now_complete and not prev_complete):
            return  # not a fresh completion this turn

        sid = getattr(orch, "active_session_id", "") or ""
        if not sid:
            return

        from victor.agent.decisions.chain import log_decision

        state = getattr(detector, "_state", None)
        log_decision(
            "task_completion",
            {
                "source": "task_completion_detector",
                "detector_verdict": "fulfilled",
                "completion_percentage": getattr(state, "completion_percentage", None),
            },
            result="fulfilled",
            source="task_completion_detector",
            session_id_override=sid,
        )

        if getattr(settings, "enable_rl_feedback_prompt", False):
            # Deprecated opt-in: legacy blocking reward prompt.
            from rich.prompt import Prompt

            answer = Prompt.ask(
                "[dim]Did this resolve your task?[/]",
                choices=["y", "n", "skip"],
                default="skip",
            )
            if answer == "skip":
                return
            success = answer == "y"
            from victor.agent.decisions.outcome import record_session_outcome

            count = record_session_outcome(
                sid, success=success, quality_score=1.0 if success else 0.0
            )
            console.print(
                f"[dim]✓ Recorded {'success' if success else 'failure'} "
                f"for {count} decision(s).[/]"
            )
        elif not getattr(orch, "_rate_hint_shown", False):
            orch._rate_hint_shown = True
            console.print("[dim]Tip: rate this session anytime with /rate.[/]")
    except Exception:
        pass  # never break the REPL


async def _run_cli_repl(
    agent: Any,
    settings: Any,
    cmd_handler: Any,
    profile_config: Any,
    stream: bool,
    rl_suggestion: Optional[tuple[str, str, float]] = None,
    renderer_choice: str = "auto",
    vertical_name: Optional[str] = None,
    enable_planning: Optional[bool] = None,
    show_reasoning: bool = False,
    profile_name: str = "default",
    startup_messages: Optional[list[str]] = None,
) -> None:
    """Run the CLI-based REPL (fallback for unsupported terminals)."""
    console.print(_build_cli_panel(profile_config, vertical_name, rl_suggestion, settings))
    if startup_messages:
        _print_interactive_startup_messages(console, startup_messages)

    # Set up prompt_toolkit with persistent history for Up/Down arrow navigation
    # Returns both the session and a mutable holder for the renderer reference
    prompt_session, renderer_holder = _create_cli_prompt_session(
        agent=agent,
        settings=settings,
        profile_config=profile_config,
        profile_name=profile_name,
        vertical_name=vertical_name,
        enable_planning=enable_planning,
        stream=stream,
        renderer_choice=renderer_choice,
        show_reasoning=show_reasoning,
    )

    def _autosave_recovery_session(
        conversation: Any,
        *,
        provider: str,
        model: str,
        profile_name: str,
    ) -> str:
        """Persist recovery sessions via the canonical ConversationStore."""
        from victor.agent.conversation.store import ConversationStore
        from victor.agent.conversation.types import MessageRole

        role_by_value = {role.value: role for role in MessageRole}
        conversation_data = (
            conversation.to_dict() if hasattr(conversation, "to_dict") else conversation
        )
        if not isinstance(conversation_data, dict):
            conversation_data = {"messages": []}

        store = ConversationStore()
        session = store.create_session(
            provider=provider,
            model=model,
            profile=profile_name,
        )

        for message in conversation_data.get("messages", []):
            if not isinstance(message, dict):
                continue

            role_name = str(message.get("role", "assistant"))
            role = role_by_value.get(role_name, MessageRole.ASSISTANT)
            metadata = {
                key: value
                for key, value in message.items()
                if key not in {"role", "content", "name", "tool_call_id", "tool_calls"}
            }
            tool_calls = message.get("tool_calls")
            store.add_message(
                session_id=session.session_id,
                role=role,
                content=str(message.get("content", "")),
                tool_name=(str(message["name"]) if message.get("name") is not None else None),
                tool_call_id=(
                    str(message["tool_call_id"])
                    if message.get("tool_call_id") is not None
                    else None
                ),
                metadata=metadata or None,
                tool_calls=tool_calls if isinstance(tool_calls, list) else None,
            )

        for preview in conversation_data.get("preview_messages", []):
            if not isinstance(preview, dict):
                continue
            preview_metadata = dict(preview.get("metadata", {}) or {})
            preview_metadata["interactive_preview"] = True
            preview_metadata["after_message_index"] = preview.get("after_message_index", 0)
            preview_role_name = str(preview.get("role", "system"))
            preview_role = role_by_value.get(preview_role_name, MessageRole.SYSTEM)
            store.add_message(
                session_id=session.session_id,
                role=preview_role,
                content=str(preview.get("content", "")),
                metadata=preview_metadata,
            )

        return session.session_id

    while True:
        renderer = None
        try:
            # Use prompt_async() — prompt_toolkit's native async input.
            # The sync .prompt() internally calls asyncio.run() which crashes
            # when we're already inside an event loop (run_sync → asyncio.run).
            user_input = await prompt_session.prompt_async(
                _build_cli_prompt_fragments(profile_name)
            )
            user_input = _normalize_cli_input_alias(user_input)

            if not user_input.strip():
                continue

            if user_input.strip().lower() in ("exit", "quit", "/exit", "/quit"):
                console.print("[dim]Goodbye![/]")
                break

            if user_input.strip().lower() in ("/expand", "/e"):
                current_renderer = renderer_holder["ref"]
                if current_renderer is not None and hasattr(current_renderer, "expand_last_output"):
                    current_renderer.expand_last_output()
                else:
                    console.print("[dim]No tool output to expand[/]")
                continue

            if user_input.strip().lower() in ("/shortcuts", "/keys"):
                console.print(_build_cli_shortcuts_panel())
                continue

            if cmd_handler.is_command(user_input):
                await cmd_handler.execute(user_input)
                continue

            if user_input.strip().lower() in ("clear", "/clear"):
                agent.reset_conversation()
                console.print("[green]Conversation cleared[/]")
                continue

            console.print("[bold cyan]Assistant[/]")

            # Planning mode requires non-streaming execution
            use_streaming = stream and not enable_planning

            if use_streaming:
                from victor.agent.response_sanitizer import sanitize_response
                from victor.ui.rendering import (
                    LiveDisplayRenderer,
                    FormatterRenderer,
                    stream_response,
                )

                use_live = renderer_choice in {"rich", "auto"}
                if use_live:
                    renderer = LiveDisplayRenderer(
                        console,
                        tool_budget_source=lambda: _coerce_cli_int(
                            getattr(agent, "tool_budget", None)
                        ),
                    )
                else:
                    formatter = create_formatter()
                    renderer = FormatterRenderer(formatter, console)

                # Release previous renderer before swapping to prevent Live display leaks.
                prev_renderer = renderer_holder.get("ref")
                if prev_renderer is not None and hasattr(prev_renderer, "cleanup"):
                    prev_renderer.cleanup()
                # Update binding target so Ctrl+O always refers to the current renderer
                renderer_holder["ref"] = renderer

                content_buffer = await stream_response(
                    agent,
                    user_input,
                    renderer,
                    suppress_thinking=not show_reasoning,
                )
                content_buffer = sanitize_response(content_buffer)

                # Renderers own streamed output. Re-printing the returned buffer here
                # duplicates the assistant response for FormatterRenderer.
            else:
                with console.status(
                    f"[dim]{_cli_work_status_message(enable_planning)}[/]",
                    spinner="dots",
                ):
                    response = await agent.chat(user_input, use_planning=enable_planning)
                console.print(Markdown(response.content))

            # FEP-0012: capture the RL reward label after the agent's turn. Both
            # the streaming and non-streaming paths converge here.
            _maybe_prompt_outcome(agent, settings, console)

        except KeyboardInterrupt:
            console.print("\n[dim]Use /exit or Ctrl+D to quit[/]")
            continue
        except EOFError:
            console.print("\n[dim]Goodbye![/]")
            break
        except Exception as e:
            # Use contextual error formatting for better UX
            error_message = format_exception_for_user(e)
            console.print(f"[bold red]Error:[/]\n{error_message}")

            current_provider = getattr(profile_config, "provider", None)
            if not isinstance(current_provider, str) or not current_provider.strip():
                current_provider = getattr(
                    getattr(settings, "provider", None), "default_provider", None
                )
            current_provider = str(current_provider or "unknown")

            current_model = getattr(profile_config, "model", None)
            if not isinstance(current_model, str) or not current_model.strip():
                current_model = getattr(getattr(settings, "provider", None), "default_model", None)
            current_model = str(current_model or "unknown")

            # Show traceback in debug mode only
            import os  # Import os for getenv

            if os.getenv("VICTOR_DEBUG"):
                import traceback

                console.print(traceback.format_exc())

            # Save conversation state on error for recovery
            try:
                if hasattr(agent, "get_conversation_history"):
                    conversation = agent.get_conversation_history()

                    session_id = _autosave_recovery_session(
                        conversation,
                        model=current_model,
                        provider=current_provider,
                        profile_name=profile_name,
                    )
                    console.print(
                        f"\n[dim]💾 Conversation auto-saved. Resume with: victor chat --resume {session_id}[/]"
                    )
            except Exception as save_error:
                import logging

                logging.getLogger(__name__).warning(
                    "Interactive chat error recovery auto-save failed: %s",
                    save_error,
                )

            # Suggest provider switching for provider-specific errors
            error_str = str(e).lower()
            is_provider_error = any(
                term in error_str
                for term in [
                    "api key",
                    "unauthorized",
                    "rate limit",
                    "timeout",
                    "connection",
                    "network",
                ]
            )

            if is_provider_error and current_provider.lower() != "ollama":
                console.print(
                    "\n[yellow]💡 Try switching to a different provider:[/]"
                    "\n  [dim]victor chat -p ollama[/]  # Local models (free)[/]"
                    "\n  [dim]victor chat -p openai[/]  # GPT models[/]"
                )
            elif is_provider_error:
                console.print(
                    "\n[yellow]💡 Try switching to a cloud provider:[/]"
                    "\n  [dim]victor chat -p anthropic[/]  # Claude models[/]"
                    "\n  [dim]victor chat -p openai[/]  # GPT models[/]"
                )

            console.print("\n[yellow]💡 Run 'victor doctor' for diagnostics[/]")

        finally:
            # Ensure renderer cleanup happens in ALL exit paths
            if renderer is not None and hasattr(renderer, "cleanup"):
                renderer.cleanup()
                if renderer_holder.get("ref") == renderer:
                    renderer_holder["ref"] = None


def _create_compile_only_compiler() -> Any:
    """Create the canonical compile-only workflow compiler for CLI validation."""
    from victor.workflows.unified_compiler import UnifiedWorkflowCompiler

    return UnifiedWorkflowCompiler(enable_caching=False)


async def run_workflow_mode(
    workflow_path: str,
    validate_only: bool = False,
    render_format: Optional[str] = None,
    render_output: Optional[str] = None,
    profile: str = "default",
    vertical: Optional[str] = None,
    log_level: Optional[str] = None,
    delegate_follow_up_contract: Optional[str] = None,
    delegate_next_step_id: Optional[str] = None,
) -> None:
    """Run, validate, or render a YAML workflow file.

    Args:
        workflow_path: Path to the YAML workflow file
        validate_only: If True, only validate without executing
        render_format: Output format for rendering (ascii, mermaid, d2, dot, svg, png)
        render_output: Output file for rendered diagram
        profile: Profile to use for agent nodes
        vertical: Optional vertical for context
        log_level: Logging level
        delegate_follow_up_contract: Optional contract file to inject into graph state
        delegate_next_step_id: Optional follow-up step ID to execute from the contract
    """
    import json
    from pathlib import Path
    from rich.table import Table

    # Setup logging
    if log_level:
        setup_logging(command="workflow", cli_log_level=log_level)
    else:
        setup_logging(command="workflow")

    workflow_file = Path(workflow_path)
    if not workflow_file.exists():
        console.print(f"[bold red]Error:[/] Workflow file not found: {workflow_path}")
        raise typer.Exit(1)

    if workflow_file.suffix not in {".yaml", ".yml"}:
        console.print(f"[bold red]Error:[/] File must be .yaml or .yml: {workflow_path}")
        raise typer.Exit(1)

    console.print(f"\n[bold blue]Workflow:[/] {workflow_file.name}")
    console.print("[dim]" + "─" * 50 + "[/]")

    try:
        # Load and parse workflow(s)
        loaded = load_workflow_from_file(str(workflow_file))

        # Handle dict of workflows or single workflow
        from victor.workflows.definition import WorkflowDefinition

        workflows: dict[str, WorkflowDefinition] = {}
        if isinstance(loaded, dict):
            workflows = loaded
        else:
            workflows = {loaded.name: loaded}

        if not workflows:
            console.print("[bold red]Error:[/] No workflows found in file")
            raise typer.Exit(1)

        compiler = _create_compile_only_compiler()
        all_validated = True

        for wf_name, workflow in workflows.items():
            console.print(f"\n[bold cyan]Workflow:[/] {wf_name}")

            # Display workflow info
            table = Table(show_header=False, box=None)
            table.add_column("Property", style="cyan")
            table.add_column("Value")

            table.add_row("Description", workflow.description or "(none)")
            table.add_row("Nodes", str(len(workflow.nodes)))
            table.add_row("Start Node", workflow.start_node)

            # Count node types
            node_types: dict[str, int] = {}
            for node in workflow.nodes.values():
                node_type = type(node).__name__
                node_types[node_type] = node_types.get(node_type, 0) + 1

            types_str = ", ".join(f"{t}: {c}" for t, c in sorted(node_types.items()))
            table.add_row("Node Types", types_str)

            console.print(table)

            try:
                # Validate using compiler
                _compiled = compiler.compile_definition(workflow)
                console.print("[bold green]✓[/] Validation passed")
            except Exception as e:
                console.print(f"[bold red]✗[/] Validation failed: {e}")
                all_validated = False

        if not all_validated:
            raise typer.Exit(1)

        if validate_only and not render_format:
            console.print("\n[bold green]Validation complete.[/]")
            return

        # Handle rendering
        if render_format:
            # For rendering, use first workflow if multiple exist
            workflow = next(iter(workflows.values()))
            viz = WorkflowVisualizer(workflow)

            # Map format string to enum
            format_map = {
                "ascii": VizFormat.ASCII,
                "mermaid": VizFormat.MERMAID,
                "plantuml": VizFormat.PLANTUML,
                "dot": VizFormat.DOT,
                "d2": VizFormat.D2,
                "svg": VizFormat.SVG,
                "png": VizFormat.PNG,
            }

            fmt = format_map.get(render_format.lower())
            if not fmt:
                console.print(
                    f"[bold red]Error:[/] Unknown format '{render_format}'. "
                    f"Valid: {', '.join(format_map.keys())}"
                )
                raise typer.Exit(1)

            # SVG/PNG require output path
            if fmt in {VizFormat.SVG, VizFormat.PNG} and not render_output:
                # Generate default output path
                render_output = str(workflow_file.with_suffix(f".{render_format.lower()}"))
                console.print(f"[dim]Output: {render_output}[/]")

            console.print(f"\n[bold]Rendering as {render_format.upper()}...[/]")

            try:
                result = viz.render(fmt, render_output)

                if render_output:
                    console.print(f"[bold green]✓[/] Saved to {render_output}")
                else:
                    console.print("")
                    # Use markup=False to avoid interpreting [] as Rich markup
                    console.print(result, markup=False)

            except Exception as e:
                console.print(f"[bold red]✗[/] Rendering failed: {e}")

                # Show available backends
                backends = get_available_backends()
                console.print("\n[dim]Available backends:[/]")
                for name, avail in backends.items():
                    status = "[green]✓[/]" if avail else "[red]✗[/]"
                    console.print(f"  {status} {name}")

                raise typer.Exit(1)

            return

        initial_context: dict[str, Any] = {}
        if delegate_follow_up_contract:
            contract_path = Path(delegate_follow_up_contract)
            try:
                follow_up_contract = load_delegate_follow_up_contract_file(contract_path)
            except FileNotFoundError:
                console.print(
                    "[bold red]Error:[/] Delegate follow-up contract not found: " f"{contract_path}"
                )
                raise typer.Exit(1)
            except DelegateFollowUpContractError as e:
                console.print(f"[bold red]Error:[/] {e}")
                raise typer.Exit(1)

            initial_context["delegate_follow_up_contract"] = follow_up_contract
            if delegate_next_step_id:
                initial_context["delegate_next_step_id"] = delegate_next_step_id
        elif delegate_next_step_id:
            console.print(
                "[bold red]Error:[/] --delegate-next-step-id requires "
                "--delegate-follow-up-contract"
            )
            raise typer.Exit(1)

        # For execution, use first workflow if multiple exist
        workflow = next(iter(workflows.values()))

        # Execute workflow
        console.print(f"\n[bold]Executing workflow '{workflow.name}'...[/]\n")

        # Create orchestrator if agent nodes exist
        orchestrator = None
        from victor.workflows.definition import AgentNode

        has_agent_nodes = any(isinstance(node, AgentNode) for node in workflow.nodes.values())

        if has_agent_nodes:
            # ✅ PROPER: Use FrameworkSessionRunner/VictorClient instead of FrameworkShim
            settings = load_settings()
            config = SessionConfig.from_cli_flags(agent_profile=profile)
            workflow_session_runner = FrameworkSessionRunner(settings, config)
            client = workflow_session_runner.create_client(config)
            orchestrator = await workflow_session_runner.initialize_client(client)

        # Execute with StateGraphExecutor
        executor = StateGraphExecutor(
            orchestrator=orchestrator,
            config=ExecutorConfig(
                enable_checkpointing=False,  # Simple execution mode
                max_iterations=50,
            ),
        )

        result = await executor.execute(workflow, initial_context)

        # Display result
        console.print("[dim]" + "─" * 50 + "[/]")

        if result.success:
            console.print("[bold green]✓[/] Workflow completed successfully")
            console.print(f"  [dim]Duration: {result.duration_seconds:.2f}s[/]")
            console.print(f"  [dim]Nodes executed: {', '.join(result.nodes_executed)}[/]")
            console.print(f"  [dim]Iterations: {result.iterations}[/]")

            if result.state:
                console.print("\n[bold]Final State:[/]")
                # Convert Pydantic model to dict if needed
                state_dict = (
                    result.state.to_dict() if hasattr(result.state, "to_dict") else result.state
                )
                # Filter internal keys
                display_state = {
                    k: v
                    for k, v in state_dict.items()
                    if not k.startswith("_") and k != "node_results"
                }
                if display_state:
                    console.print(json.dumps(display_state, indent=2, default=str))
        else:
            console.print("[bold red]✗[/] Workflow failed")
            if result.error:
                console.print(f"  [red]{result.error}[/]")

        # Cleanup orchestrator
        if orchestrator:
            await graceful_shutdown(orchestrator)

    except YAMLWorkflowError as e:
        console.print("[bold red]✗[/] Workflow validation failed")
        console.print(f"  [red]{e}[/]")
        console.print("\n[yellow]💡 Run 'victor doctor' for diagnostics[/]")
        raise typer.Exit(1)
    except ConfigurationError as e:
        console.print(f"[bold red]✗[/] Workflow configuration error: {e}")
        console.print("\n[yellow]Suggestions:[/]")
        console.print("  • Check workflow YAML syntax")
        console.print("  • Validate workflow: victor workflow validate <file>")
        console.print("  • Run 'victor doctor' for diagnostics\n")
        raise typer.Exit(1)
    except VictorError as e:
        # Known Victor errors - use contextual formatting
        error_message = format_exception_for_user(e)
        console.print(f"[bold red]Error:[/]\n{error_message}")
        console.print("\n[yellow]💡 Run 'victor doctor' for diagnostics[/]")
        if os.getenv("VICTOR_DEBUG"):
            import traceback

            console.print(traceback.format_exc())
        raise typer.Exit(1)
    except Exception as e:
        # Unexpected error - always show in debug mode
        console.print(f"[bold red]Unexpected error:[/]\n{e}")
        console.print("\n[yellow]Suggestions:[/]")
        console.print("  • This may be a bug - please report it")
        console.print("  • Run 'victor doctor' for diagnostics")
        console.print("  • Enable debug mode: VICTOR_DEBUG=1 victor chat\n")
        import traceback

        if os.getenv("VICTOR_DEBUG"):
            console.print(traceback.format_exc())
        raise typer.Exit(1)
