import asyncio
import logging
import logging.handlers
import os
import signal
import sys
import uuid
from typing import Any, Dict, Optional, TYPE_CHECKING

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm
from pathlib import Path
import time

# ✅ PROPER: No AgentOrchestrator import (use Any type instead)
from victor.agent.safety import (
    ConfirmationRequest,
    OperationalRiskLevel,
    set_confirmation_callback,
)
from victor.config.settings import get_project_paths

from victor.core.container import get_container

try:
    from victor.framework.vertical_protocols import CodebaseIndexFactoryProtocol
except ImportError:
    CodebaseIndexFactoryProtocol = None  # type: ignore[misc,assignment]
# Shared symbols were promoted into core so that victor no longer hard-depends
# on the external victor-coding package. Only the vertical-specific index
# builder (_get_or_build_index) is loaded dynamically below.
from victor.config.settings import _resolve_graph_writer_mode
from victor.core.search.indexer import _INDEX_CACHE


def _load_code_search_index_builder():
    """Load ``_get_or_build_index`` from the victor-coding vertical.

    Discovery routes through ``capability_loader`` so core never holds a
    vertical string-literal import (core/vertical boundary guard).
    """
    from victor.core.utils.capability_loader import load_code_search_module

    return load_code_search_module()._get_or_build_index


if TYPE_CHECKING:
    from victor.config.config_loaders import LoggingConfig

logger = logging.getLogger(__name__)
console = Console()

# Global reference for signal handler cleanup (use Any instead of AgentOrchestrator)
_current_agent: Optional[Any] = None


def _get_default_log_dir() -> Path:
    """Resolve the default CLI log directory through centralized Victor paths."""
    return get_project_paths().global_victor_dir / "logs"


def _get_default_log_file() -> Path:
    """Resolve the default CLI log file through centralized Victor paths."""
    return _get_default_log_dir() / "victor.log"


def configure_logging_from_config(
    config: "LoggingConfig",
    stream: Optional[Any] = None,
    session_id: Optional[str] = None,
    repo_path: Optional[str] = None,
) -> None:
    """Configure logging from a centralized LoggingConfig object.

    This is the preferred method for configuring logging. It uses the
    centralized config system with proper priority chain:
    1. CLI arguments (passed when creating config)
    2. Environment variables
    3. User config (~/.victor/config.yaml)
    4. Command-specific overrides
    5. Package defaults

    Args:
        config: LoggingConfig with all settings
        stream: Stream for console output (default stderr)
        session_id: Optional session identifier for log context
        repo_path: Optional repo path for log context

    Example:
        from victor.config.config_loaders import get_logging_config
        config = get_logging_config(command="benchmark", cli_console_level="DEBUG")
        configure_logging_from_config(config)
    """
    # Get the root logger
    root_logger = logging.getLogger()

    # Clear existing handlers
    root_logger.handlers.clear()

    # Set root logger to lowest level (handlers will filter)
    root_logger.setLevel(logging.DEBUG)

    # Auto-detect repo from cwd if not provided
    if repo_path is None:
        repo_path = os.path.basename(os.getcwd())
    if session_id is None:
        session_id = str(uuid.uuid4())[:8]

    # Console handler — Live-aware so records print cleanly above an active
    # Rich Live display instead of tearing it (plain stream when no display).
    from victor.ui.rendering.log_handler import LiveAwareLogHandler

    console_formatter = logging.Formatter(config.console_format)
    console_handler: logging.Handler = LiveAwareLogHandler(
        stream, show_tracebacks=config.get_console_level_int() <= logging.DEBUG
    )
    console_handler.setLevel(config.get_console_level_int())
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # File handler (with rotation)
    if config.file_enabled:
        try:
            log_path = config.expanded_file_path
            log_path.parent.mkdir(parents=True, exist_ok=True)

            # Build file format with session context
            file_format = config.file_format.replace("%(session)s", f"{repo_path}-{session_id}")
            file_formatter = logging.Formatter(file_format)

            file_handler = logging.handlers.RotatingFileHandler(
                log_path,
                maxBytes=config.file_max_bytes,
                backupCount=config.file_backup_count,
                encoding="utf-8",
            )
            file_handler.setLevel(config.get_file_level_int())
            file_handler.setFormatter(file_formatter)
            root_logger.addHandler(file_handler)
            logger.debug(
                f"File logging enabled: {log_path} (session={session_id}, repo={repo_path})"
            )
        except Exception as e:
            logger.warning(f"Could not enable file logging: {e}")

    # Apply module-specific level overrides
    for module_name, level in config.module_levels.items():
        module_logger = logging.getLogger(module_name)
        module_logger.setLevel(getattr(logging, level.upper(), logging.WARNING))

    # EventBus → Logging integration
    if config.event_logging:
        try:
            from victor.core.events import get_observability_bus
            from victor.observability.exporters import LoggingExporter

            get_observability_bus()
            # Note: The canonical event system doesn't have exporters
            # This is a no-op for now, but kept for compatibility
            logger.debug("EventBus → Logging integration enabled (no-op for canonical system)")
        except Exception as e:
            logger.debug(f"Could not enable event logging: {e}")


def setup_logging(
    command: Optional[str] = None,
    cli_log_level: Optional[str] = None,
    stream: Optional[Any] = None,
    session_id: Optional[str] = None,
    repo_path: Optional[str] = None,
    cli_debug_modules: Optional[str] = None,
    cli_file_level: Optional[str] = None,
) -> "LoggingConfig":
    """Convenience function to load config and configure logging in one call.

    This is the recommended entry point for subcommands to set up logging.
    It handles the full priority chain and applies the configuration.

    Args:
        command: Command name for command-specific overrides (e.g., "chat", "benchmark")
        cli_log_level: CLI-provided log level (highest priority, applies to console only;
            the persistent file level stays at its configured value so one verbose
            session does not flood ~/.victor/logs/victor.log)
        stream: Stream for console output (default stderr)
        session_id: Optional session identifier for log context
        repo_path: Optional repo path for log context
        cli_file_level: Explicit file log level override. Callers that genuinely
            want verbose file logs must pass this (or set VICTOR_LOG_FILE_LEVEL).

    Returns:
        The LoggingConfig that was applied (for inspection if needed)

    Example:
        # In a subcommand
        config = setup_logging(command="benchmark", cli_log_level=log_level)
        # Logging is now configured, proceed with command
    """
    from victor.config.config_loaders import get_logging_config

    config = get_logging_config(
        command=command,
        cli_console_level=cli_log_level,
        cli_file_level=cli_file_level,
        cli_debug_modules=cli_debug_modules,
    )
    configure_logging_from_config(
        config,
        stream=stream,
        session_id=session_id,
        repo_path=repo_path,
    )
    return config


def configure_logging(
    log_level: str,
    stream: Optional[Any] = None,
    file_logging: bool = True,
    file_level: str = "INFO",
    console_level: str = "WARNING",
    log_file: Optional[Path] = None,
    event_logging: bool = True,
    session_id: Optional[str] = None,
    repo_path: Optional[str] = None,
) -> None:
    """Configure logging with dual output: file (INFO) and console (WARNING).

    DEPRECATED: Use setup_logging() or configure_logging_from_config() instead.
    This function is kept for backward compatibility.

    Args:
        log_level: Overall log level (for backward compatibility)
        stream: Stream for console output (default stderr)
        file_logging: Enable file logging (default True)
        file_level: Log level for file output (default INFO)
        console_level: Log level for console output (default WARNING)
        log_file: Path to log file (default ~/.victor/logs/victor.log)
        event_logging: Enable EventBus → logging integration (default True)
        session_id: Optional session identifier for log context
        repo_path: Optional repo path for log context
    """
    # Get the root logger
    root_logger = logging.getLogger()

    # Clear existing handlers
    root_logger.handlers.clear()

    # Set root logger to lowest level (handlers will filter)
    root_logger.setLevel(logging.DEBUG)

    # Console log format (concise)
    console_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    console_formatter = logging.Formatter(console_format)

    # File log format (includes session/repo context for multi-session debugging)
    # Auto-detect repo from cwd if not provided
    if repo_path is None:
        repo_path = os.path.basename(os.getcwd())
    if session_id is None:
        session_id = str(uuid.uuid4())[:8]  # Short session ID

    file_format = f"%(asctime)s - {repo_path}-{session_id} - %(name)s - %(levelname)s - %(message)s"
    file_formatter = logging.Formatter(file_format)

    # Console handler (WARNING by default, can be overridden by log_level)
    from victor.ui.rendering.log_handler import LiveAwareLogHandler

    console_handler: logging.Handler = LiveAwareLogHandler(stream)
    effective_console_level = getattr(logging, log_level.upper(), None)
    if effective_console_level is None:
        effective_console_level = getattr(logging, console_level.upper(), logging.WARNING)
    console_handler.setLevel(effective_console_level)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # File handler (INFO by default) with rotation
    if file_logging:
        try:
            log_path = log_file or _get_default_log_file()
            log_path.parent.mkdir(parents=True, exist_ok=True)

            # Use RotatingFileHandler to prevent unbounded growth
            # Max 10MB per file, keep 5 backup files
            file_handler = logging.handlers.RotatingFileHandler(
                log_path,
                maxBytes=10 * 1024 * 1024,  # 10MB
                backupCount=5,
                encoding="utf-8",
            )
            file_handler.setLevel(getattr(logging, file_level.upper(), logging.INFO))
            file_handler.setFormatter(file_formatter)
            root_logger.addHandler(file_handler)
            logger.debug(
                f"File logging enabled: {log_path} (session={session_id}, repo={repo_path})"
            )
        except Exception as e:
            # Don't fail if file logging can't be set up
            logger.warning(f"Could not enable file logging: {e}")

    # EventBus → Logging integration
    # This routes observability events to the logging system
    if event_logging:
        try:
            from victor.core.events import get_observability_bus

            get_observability_bus()
            # Note: The canonical event system doesn't have exporters
            # This is a no-op for now, but kept for compatibility
            logger.debug("EventBus → Logging integration enabled (no-op for canonical system)")
        except Exception as e:
            # Don't fail if event logging can't be set up
            logger.debug(f"Could not enable event logging: {e}")


def configure_logging_simple(log_level: str, stream: Optional[Any] = None) -> None:
    """Simple logging configuration (backward compatible, console only).

    Use this for quick scripts or when file logging is not desired.
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.WARNING),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=stream or sys.stderr,
        force=True,
    )


def flush_logging() -> None:
    """Flush all logging handlers before shutdown messages."""
    for handler in logging.root.handlers:
        handler.flush()


async def graceful_shutdown(agent: Optional[Any]) -> None:
    """Perform graceful shutdown of the agent.

    Args:
        agent: Agent instance (any type - duck typing used for runtime access)
    """
    if agent is None:
        return
    try:
        from victor.framework.rl.base import RLOutcome
        from victor.framework.rl.coordinator import get_rl_coordinator

        coordinator = get_rl_coordinator()
        learner = coordinator.get_learner("model_selector")
        if learner and agent.provider and hasattr(agent, "message_count"):
            msg_count = agent.message_count
            if msg_count > 0:
                import uuid

                metrics = {}
                if hasattr(agent, "get_session_metrics"):
                    metrics = agent.get_session_metrics() or {}

                # Compute quality score based on session metrics
                latency = metrics.get("total_latency", 0)
                tool_calls = metrics.get("tool_calls", 0)
                quality_score = 1.0
                if latency > 30:
                    quality_score -= min(0.1 * (latency - 30) / 30, 0.5)
                if tool_calls > 0:
                    quality_score += min(0.05 * tool_calls, 0.2)

                outcome = RLOutcome(
                    provider=agent.provider.name,
                    model=getattr(agent.provider, "model", "unknown"),
                    task_type="chat",
                    success=True,
                    quality_score=max(0.0, min(1.0, quality_score)),
                    metadata={
                        "session_id": str(uuid.uuid4())[:8],
                        "latency_seconds": latency,
                        "token_count": metrics.get("total_tokens", 0),
                        "tool_calls_made": tool_calls,
                        "message_count": msg_count,
                    },
                    vertical="coding",
                )
                coordinator.record_outcome("model_selector", outcome, "coding")

                # Get updated Q-value for logging
                rankings = learner.get_provider_rankings()
                provider_ranking = next(
                    (r for r in rankings if r["provider"] == agent.provider.name), None
                )
                new_q = provider_ranking["q_value"] if provider_ranking else 0.0

                logger.info(
                    f"RL session feedback: {agent.provider.name} "
                    f"({msg_count} messages, {tool_calls} tools) → Q={new_q:.3f}"
                )
    except Exception as e:
        logger.debug(f"RL feedback recording skipped: {e}")

    # Record session outcome for prompt optimizer (lightweight — no LLM call).
    # This updates posteriors for the active GEPA candidate so it learns
    # from every chat session, not just benchmarks.
    try:
        from victor.framework.rl.base import RLOutcome

        coordinator = get_rl_coordinator()
        prompt_learner = coordinator.get_learner("prompt_optimizer")
        if prompt_learner and agent.provider:
            session_metrics = {}
            if hasattr(agent, "get_session_metrics"):
                session_metrics = agent.get_session_metrics() or {}
            tool_calls = session_metrics.get("tool_calls", 0)
            if tool_calls > 0:  # Only record if tools were used
                outcome = RLOutcome(
                    provider=agent.provider.name,
                    model=getattr(agent.provider, "model", "unknown"),
                    task_type="chat",
                    success=True,
                    quality_score=min(1.0, 0.5 + tool_calls * 0.05),
                    metadata={"prompt_section": "ASI_TOOL_EFFECTIVENESS_GUIDANCE"},
                )
                prompt_learner.record_outcome(outcome)
                logger.debug("Prompt optimizer: recorded chat session outcome")
    except Exception as e:
        logger.debug(f"Prompt optimizer feedback skipped: {e}")

    try:
        shutdown_results = await agent.graceful_shutdown()
        logger.debug(f"Graceful shutdown results: {shutdown_results}")
        await agent.shutdown()
    except Exception as e:
        logger.warning(f"Error during graceful shutdown: {e}")
        try:
            if agent.provider:
                await agent.provider.close()
        except Exception as close_error:
            logger.debug(f"Error closing provider: {close_error}")
    finally:
        flush_logging()


def setup_signal_handlers(loop: asyncio.AbstractEventLoop) -> None:
    """Set up signal handlers for graceful shutdown."""

    def signal_handler(sig: int, frame: Any) -> None:
        global _current_agent
        sig_name = signal.Signals(sig).name
        logger.info(f"Received {sig_name}, initiating graceful shutdown...")
        if _current_agent is not None:
            try:
                asyncio.create_task(graceful_shutdown(_current_agent))
            except RuntimeError:
                logger.debug("Event loop not running, attempting direct cleanup")
                try:
                    if _current_agent.provider:
                        pass
                except Exception:
                    pass
        if sig == signal.SIGINT:
            raise KeyboardInterrupt()

    if sys.platform != "win32":
        signal.signal(signal.SIGTERM, signal_handler)


async def check_codebase_index(cwd: str, console_obj: Console, silent: bool = False) -> None:
    """Check codebase index status at startup and reindex if needed."""
    if CodebaseIndexFactoryProtocol is None:
        if not silent:
            logger.debug("Codebase indexing not available - victor-coding package not installed")
        return

    from victor.config.settings import load_settings

    _container = get_container()
    _factory = _container.get_optional(CodebaseIndexFactoryProtocol)
    if _factory is None:
        if not silent:
            logger.debug("Codebase indexing not available - victor-coding package not installed")
        return
    try:
        settings = load_settings()
        graph_writer_mode = _resolve_graph_writer_mode(settings)
        graph_store_name = getattr(settings, "codebase_graph_store", "sqlite")
        graph_path = getattr(settings, "codebase_graph_path", None)

        index = _factory.create(
            root_path=cwd,
            use_embeddings=False,
            enable_watcher=False,
            graph_writer_mode=str(graph_writer_mode),
            graph_store_name=graph_store_name,
            graph_path=Path(graph_path) if graph_path else None,
        )
        is_stale, modified, deleted = index.check_staleness_by_mtime()
        if not is_stale:
            if not silent:
                logger.debug("Codebase index is up to date")
            return
        total_changes = len(modified) + len(deleted)
        if not silent:
            console_obj.print(
                f"[dim]Index stale: {len(modified)} modified, {len(deleted)} deleted files[/]"
            )
        if total_changes <= 10:
            await index.incremental_reindex()
            if not silent:
                console_obj.print(f"[green]Incrementally reindexed {total_changes} files[/]")
        else:
            await index.reindex()
            if not silent:
                console_obj.print(f"[green]Full reindex completed ({len(index.files)} files)[/]")
    except ImportError:
        logger.debug("Codebase indexer not available")
    except Exception as e:
        logger.debug(f"Codebase index check failed: {e}")


async def preload_semantic_index(
    cwd: str, settings: Any, console_obj: Console, force: bool = False
) -> bool:
    """Preload semantic codebase index with embeddings and graph upfront."""
    try:
        root_path = Path(cwd).resolve()
        cache_entry = _INDEX_CACHE.get(str(root_path))
        if cache_entry and not force:
            console_obj.print("[dim]✓ Semantic index already loaded[/]")
            return True

        # Show progress during indexing (can take 20-30s)
        from rich.console import Console as RichConsole

        # Index builder lives in the victor-coding vertical; load it dynamically
        # so core does not hard-depend on the external package.
        _get_or_build_index = _load_code_search_index_builder()

        if isinstance(console_obj, RichConsole):
            with console_obj.status(
                "[bold green]Building semantic code index (one-time)...[/]",
                spinner="dots",
            ) as status:
                status.update("Indexing files...")
                start_time = time.time()
                index, rebuilt = await _get_or_build_index(root_path, settings, force_reindex=force)
                elapsed = time.time() - start_time
        else:
            console_obj.print("[dim]⏳ Building semantic code index (one-time)...[/]")
            start_time = time.time()
            index, rebuilt = await _get_or_build_index(root_path, settings, force_reindex=force)
            elapsed = time.time() - start_time

        # Get graph stats if available
        graph_stats = ""
        empty_graph_warning = False
        if index.graph_store:
            try:
                stats = await index.graph_store.stats()
                node_count = stats.get("nodes", 0)
                edge_count = stats.get("edges", 0)
                if node_count or edge_count:
                    graph_stats = f" | Graph: {node_count} symbols, {edge_count} edges"
                if node_count == 0:
                    empty_graph_warning = True
            except Exception:
                pass

        if rebuilt:
            console_obj.print(f"[green]✓ Semantic index built in {elapsed:.1f}s{graph_stats}[/]")
        else:
            console_obj.print(f"[green]✓ Semantic index loaded in {elapsed:.1f}s{graph_stats}[/]")
        if empty_graph_warning:
            console_obj.print(
                "[yellow]⚠ No symbols were added to the project graph. "
                "Verify --path points at your repo root and rerun `victor index --reset --debug` "
                "if you expected code to be indexed.[/yellow]"
            )
        return True
    except ImportError as e:
        logger.warning(f"Semantic indexing dependencies not available: {e}")
        return False
    except Exception as e:
        logger.warning(f"Failed to preload semantic index: {e}")
        console_obj.print(f"[yellow]⚠ Semantic index preload failed: {e}[/]")
        return False


async def cli_confirmation_callback(request: ConfirmationRequest) -> bool:
    """Prompt user for confirmation of dangerous operations."""
    risk_colors = {
        OperationalRiskLevel.SAFE: "green",
        OperationalRiskLevel.LOW: "blue",
        OperationalRiskLevel.MEDIUM: "yellow",
        OperationalRiskLevel.HIGH: "red",
        OperationalRiskLevel.CRITICAL: "bold red",
    }
    color = risk_colors.get(request.risk_level, "white")
    console.print()
    console.print(
        Panel(
            f"[{color}]{request.risk_level.value.upper()} RISK[/{color}]\n\n"
            f"Tool: [bold]{request.tool_name}[/bold]\n"
            f"Action: {request.description}\n\n"
            + (
                "Details:\n" + "\n".join(f"  • {d}" for d in request.details)
                if request.details
                else ""
            ),
            title="⚠️  Confirmation Required",
            border_style=color,
        )
    )
    try:
        return Confirm.ask("Proceed with this operation?", default=False)
    except (KeyboardInterrupt, EOFError):
        console.print("\n[dim]Operation cancelled[/dim]")
        return False


def setup_safety_confirmation() -> None:
    """Set up the CLI confirmation callback for dangerous operations."""
    set_confirmation_callback(cli_confirmation_callback)


def get_rl_profile_suggestion(
    current_provider: str, profiles: dict
) -> Optional[tuple[str, str, float]]:
    """Get RL-based profile suggestion if different from current."""
    try:
        from victor.framework.rl.coordinator import get_rl_coordinator
        import json

        coordinator = get_rl_coordinator()
        learner = coordinator.get_learner("model_selector")
        if not learner:
            return None

        # Get available providers from profiles
        available = list(set(cfg.provider for cfg in profiles.values()))

        # Get recommendation
        rec = coordinator.get_recommendation(
            "model_selector",
            json.dumps(available),
            "",
            "chat",
        )
        if not rec or rec.confidence < 0.3:
            return None
        if rec.value.lower() == current_provider.lower():
            return None

        # Find matching profile for recommended provider
        matching_profiles = [
            name for name, cfg in profiles.items() if cfg.provider.lower() == rec.value.lower()
        ]
        if matching_profiles:
            return (matching_profiles[0], rec.value, rec.confidence)
        return None
    except Exception:
        return None
