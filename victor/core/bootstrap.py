# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Bootstrap module for Victor dependency injection.

This module configures the DI container with default implementations
of all services. It's called once at application startup.

Design Principles:
- Explicit registration (no auto-wiring)
- Easy to substitute for testing
- Lazy initialization where appropriate
- Environment-aware configuration

Usage:
    from victor.core.bootstrap import bootstrap_container, get_container

    # Bootstrap with default settings
    container = bootstrap_container()

    # Or with custom settings
    container = bootstrap_container(settings=my_settings)

    # Resolve services
    metrics = container.get(MetricsServiceProtocol)
    logger = container.get(LoggerServiceProtocol)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Set, Type, TypeVar

from victor.config.settings import Settings, load_settings, get_project_paths
from victor.core.container import (
    ServiceContainer,
    ServiceLifetime,
    get_container,
    set_container,
    MetricsServiceProtocol,
    LoggerServiceProtocol,
    CacheServiceProtocol,
    EmbeddingServiceProtocol,
)

logger = logging.getLogger(__name__)

# Vertical package hints loaded dynamically from entry points.
# No hardcoded vertical names in core — verticals advertise
# themselves via victor.vertical_hints entry points.
_DEFAULT_VERTICAL_PACKAGE_HINTS: Dict[str, str] = {}
_REPORTED_MISSING_VERTICALS: Set[str] = set()


def _load_vertical_package_hints() -> Dict[str, str]:
    """Load vertical package hints from entry points with hardcoded fallbacks.

    Scans ``victor.vertical_hints`` entry points so that external vertical
    packages can advertise their pip-installable name without modifying
    this module.  Falls back to ``_DEFAULT_VERTICAL_PACKAGE_HINTS`` for
    any vertical not covered by an entry point.

    Each entry point should have ``name=<vertical>`` and its loaded value
    should be the pip package name (a plain string).
    """
    hints = dict(_DEFAULT_VERTICAL_PACKAGE_HINTS)
    try:
        from importlib.metadata import entry_points

        eps = entry_points()
        # Python 3.12+ returns a SelectableGroups, older returns dict
        group_eps = (
            eps.select(group="victor.vertical_hints")
            if hasattr(eps, "select")
            else eps.get("victor.vertical_hints", [])
        )  # noqa: E501
        for ep in group_eps:
            try:
                value = ep.load()
                if isinstance(value, str):
                    hints[ep.name] = value
            except Exception:
                pass
    except Exception:
        pass
    return hints


T = TypeVar("T")


# =============================================================================
# Default Service Implementations
# =============================================================================


class NullMetricsService:
    """No-op metrics service for when metrics are disabled."""

    def record_metric(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        pass

    def increment_counter(self, name: str, tags: Optional[Dict[str, str]] = None) -> None:
        pass


class ConsoleLoggerService:
    """Simple console-based logger service."""

    def __init__(self, enabled: bool = True):
        self._enabled = enabled
        self._logger = logging.getLogger("victor.analytics")

    def log(self, message: str, level: str = "info", **kwargs: Any) -> None:
        if not self._enabled:
            return

        log_level = getattr(logging, level.upper(), logging.INFO)
        extra = f" | {kwargs}" if kwargs else ""
        self._logger.log(log_level, f"{message}{extra}")

    @property
    def enabled(self) -> bool:
        return self._enabled


class InMemoryCacheService:
    """Simple in-memory cache service."""

    def __init__(self, max_size: int = 1000):
        self._cache: Dict[str, Any] = {}
        self._max_size = max_size

    def get(self, key: str) -> Optional[Any]:
        return self._cache.get(key)

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        # Simple LRU-style eviction if over limit
        if len(self._cache) >= self._max_size:
            # Remove oldest entry (first key)
            if self._cache:
                first_key = next(iter(self._cache))
                del self._cache[first_key]
        self._cache[key] = value

    def invalidate(self, key: str) -> None:
        self._cache.pop(key, None)

    def clear(self) -> None:
        self._cache.clear()


class LazyEmbeddingService:
    """Lazy-loading embedding service that initializes on first use."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self._model_name = model_name
        self._model = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name)
            logger.info(f"Loaded embedding model: {self._model_name}")
        except Exception as e:
            logger.warning(f"Failed to load embedding model: {e}")
            self._model = None

    def embed(self, texts: list[str]) -> list[list[float]]:
        self._ensure_loaded()
        if self._model is None:
            return [[0.0] * 384 for _ in texts]  # Return zero vectors
        return self._model.encode(texts, convert_to_numpy=False).tolist()

    def embed_single(self, text: str) -> list[float]:
        result = self.embed([text])
        return result[0] if result else [0.0] * 384


# =============================================================================
# Service Protocol Type Hints
# =============================================================================


class SignatureStoreProtocol:
    """Protocol for failed signature storage."""

    def is_known_failure(self, tool_name: str, args: Dict[str, Any]) -> bool: ...

    def record_failure(self, tool_name: str, args: Dict[str, Any], error_message: str) -> None: ...


class UsageLoggerProtocol:
    """Protocol for usage logging."""

    def log_event(self, event_type: str, data: Dict[str, Any]) -> None: ...

    def is_enabled(self) -> bool: ...


# =============================================================================
# Bootstrap Function
# =============================================================================


def bootstrap_container(
    settings: Optional[Settings] = None,
    vertical: Optional[str] = None,
    override_services: Optional[Dict[Type, Any]] = None,
) -> ServiceContainer:
    """Bootstrap the DI container with default service implementations.

    This function configures the container with all required services.
    Call once at application startup.

    Execution follows an explicit dependency DAG defined in
    :data:`_BOOTSTRAP_PHASES`. Each phase declares its dependencies,
    and phases are executed in topological order.

    Args:
        settings: Optional Settings instance (loads from config if None)
        vertical: Optional vertical name to activate (e.g., "coding", "research")
                 If None, uses settings.default_vertical or None (base mode)
        override_services: Optional dict of service type -> instance for testing

    Returns:
        Configured ServiceContainer
    """
    from victor.core.bootstrap_phases import BootstrapPhase, execute_phases

    if settings is None:
        settings = load_settings()

    active_vertical = _resolve_vertical_name(settings, vertical)

    container = ServiceContainer()

    # Shared mutable context for inter-phase data
    context: Dict[str, Any] = {
        "active_vertical": active_vertical,
        "override_services": override_services,
    }

    execute_phases(_BOOTSTRAP_PHASES, container, settings, context)

    logger.info("Bootstrapped service container")
    return container


# -------------------------------------------------------------------------
# Phase registration functions (thin wrappers for the DAG)
# Each accepts (container, settings, context) for uniform signatures.
# -------------------------------------------------------------------------


def _phase_settings(container, settings, context):
    """Phase: Register Settings singleton and domain config slices."""
    container.register_instance(Settings, settings)

    # Register each nested config group as an independent service.
    # Components can depend on e.g. ToolSettings instead of full Settings.
    from victor.config.settings import _NESTED_GROUPS

    for group_name, group_cls in _NESTED_GROUPS.items():
        group_obj = getattr(settings, group_name, None)
        if group_obj is not None:
            container.register_instance(group_cls, group_obj)


def _phase_core(container, settings, context):
    """Phase: Core services (cache, capability config, framework)."""
    _register_core_services(container, settings)


def _phase_events(container, settings, context):
    """Phase: Event backend, observability bus, message bus."""
    _register_event_services(container, settings)


def _phase_analytics(container, settings, context):
    """Phase: Metrics, logger, usage analytics."""
    _register_analytics_services(container, settings)


def _phase_embedding(container, settings, context):
    """Phase: Lazy embedding service."""
    _register_embedding_services(container, settings)


def _phase_plugins(container, settings, context):
    """Phase: Plugin discovery and registration."""
    from victor.core.plugins.registry import PluginRegistry

    plugin_registry = PluginRegistry.get_instance()
    plugin_registry.register_all(container)


def _phase_capabilities(container, settings, context):
    """Phase: Capability stubs + entry point discovery."""
    bootstrap_capabilities()
    _report_capability_health(context.get("active_vertical"), container)


def _phase_vertical_services(container, settings, context):
    """Phase: Vertical-specific bootstrap services via entry points.

    Scans ``victor.bootstrap_services`` entry points so external
    verticals can register bootstrap hooks without modifying this
    module.  No hardcoded vertical calls — all verticals participate
    equally through entry points.
    """
    # Discover bootstrap service hooks from entry points
    try:
        from importlib.metadata import entry_points as _ep

        eps = _ep()
        group_eps = (
            eps.select(group="victor.bootstrap_services")
            if hasattr(eps, "select")
            else eps.get("victor.bootstrap_services", [])
        )
        for ep in group_eps:
            try:
                hook = ep.load()
                hook(container, settings, context)
                logger.debug("Loaded bootstrap service entry point: %s", ep.name)
            except Exception as exc:
                logger.debug("Skipped bootstrap service %s: %s", ep.name, exc)
    except Exception as exc:
        logger.debug("Bootstrap service entry point discovery failed: %s", exc)


def _phase_signature(container, settings, context):
    """Phase: Signature store."""
    _register_signature_store(container, settings)


def _phase_orchestrator(container, settings, context):
    """Phase: 46+ orchestrator services from OrchestratorServiceProvider."""
    _register_orchestrator_services(container, settings)


def _phase_agent_services(container, settings, context):
    """Phase: canonical agent-service bootstrap."""
    _register_agent_services(container, settings)


def _phase_workflow(container, settings, context):
    """Phase: Workflow services."""
    _register_workflow_services(container, settings)


def _phase_compiler_plugins(container, settings, context):
    """Phase: Workflow compiler plugins."""
    _register_workflow_compiler_plugins(container, settings)


def _phase_extensions(container, settings, context):
    """Phase: Extension loader runtime configuration."""
    _configure_extension_loader_runtime(settings)


def _phase_vertical(container, settings, context):
    """Phase: Vertical-specific services."""
    active_vertical = context.get("active_vertical")
    _register_vertical_services(container, settings, active_vertical)


def _phase_overrides(container, settings, context):
    """Phase: Apply testing overrides."""
    override_services = context.get("override_services")
    if override_services:
        for service_type, instance in override_services.items():
            container.register_or_replace(
                service_type,
                lambda c, inst=instance: inst,
                ServiceLifetime.SINGLETON,
            )


def _phase_finalize(container, settings, context):
    """Phase: Set global container."""
    set_container(container)


# -------------------------------------------------------------------------
# Phase DAG definition
# -------------------------------------------------------------------------

from victor.core.bootstrap_phases import BootstrapPhase  # noqa: E402

_BOOTSTRAP_PHASES = [
    BootstrapPhase("settings", _phase_settings),
    BootstrapPhase("core", _phase_core, depends_on=("settings",)),
    BootstrapPhase("events", _phase_events, depends_on=("settings",)),
    BootstrapPhase("analytics", _phase_analytics, depends_on=("settings",)),
    BootstrapPhase("embedding", _phase_embedding, depends_on=("settings",)),
    BootstrapPhase(
        "plugins",
        _phase_plugins,
        depends_on=("core", "events", "analytics", "embedding"),
    ),
    BootstrapPhase("capabilities", _phase_capabilities, depends_on=("plugins",)),
    BootstrapPhase(
        "vertical_services",
        _phase_vertical_services,
        depends_on=("capabilities",),
        optional=True,
    ),
    BootstrapPhase("signature", _phase_signature, depends_on=("settings",)),
    BootstrapPhase("orchestrator", _phase_orchestrator, depends_on=("capabilities",)),
    BootstrapPhase(
        "agent_services",
        _phase_agent_services,
        depends_on=("orchestrator",),
        optional=True,
    ),
    BootstrapPhase("workflow", _phase_workflow, depends_on=("settings",)),
    BootstrapPhase("compiler_plugins", _phase_compiler_plugins, depends_on=("settings",)),
    BootstrapPhase("extensions", _phase_extensions, depends_on=("events",)),
    BootstrapPhase(
        "vertical",
        _phase_vertical,
        depends_on=("capabilities", "orchestrator"),
    ),
    BootstrapPhase("overrides", _phase_overrides, depends_on=("vertical",)),
    BootstrapPhase("finalize", _phase_finalize, depends_on=("overrides",)),
]


def _register_core_services(container: ServiceContainer, settings: Settings) -> None:
    """Register core infrastructure services."""
    from victor.framework.capability_config_service import CapabilityConfigService
    from victor.framework.framework_integration_registry_service import (
        FrameworkIntegrationRegistryService,
    )

    # Cache service
    container.register(
        CacheServiceProtocol,
        lambda c: InMemoryCacheService(max_size=1000),
        ServiceLifetime.SINGLETON,
    )

    # Capability config service (DI-global store with per-session scope buckets)
    container.register(
        CapabilityConfigService,
        lambda c: CapabilityConfigService(),
        ServiceLifetime.SINGLETON,
    )

    # Framework integration registry facade service
    container.register(
        FrameworkIntegrationRegistryService,
        lambda c: FrameworkIntegrationRegistryService(),
        ServiceLifetime.SINGLETON,
    )


def _register_event_services(container: ServiceContainer, settings: Settings) -> None:
    """Register event backend and observability services.

    This registers the canonical event system (core/events) as the
    primary event backend, replacing the legacy observability/event_bus.py.
    """
    from victor.core.events import (
        IEventBackend,
        ObservabilityBus,
        AgentMessageBus,
        create_event_backend,
    )
    from victor.core.events.backends import build_backend_config_from_settings

    backend_config = build_backend_config_from_settings(settings)
    lazy_init = bool(getattr(settings, "event_backend_lazy_init", True))

    # Register IEventBackend as singleton
    container.register(
        IEventBackend,
        lambda c, cfg=backend_config, lazy=lazy_init: create_event_backend(
            config=cfg, lazy_init=lazy
        ),
        ServiceLifetime.SINGLETON,
    )

    # Register ObservabilityBus as singleton
    container.register(
        ObservabilityBus,
        lambda c: ObservabilityBus(backend=c.get(IEventBackend)),
        ServiceLifetime.SINGLETON,
    )

    # Register AgentMessageBus as singleton
    container.register(
        AgentMessageBus,
        lambda c: AgentMessageBus(backend=c.get(IEventBackend)),
        ServiceLifetime.SINGLETON,
    )

    logger.info(
        "Registered event services with backend: %s (overflow_policy=%s, queue_maxsize=%s)",
        backend_config.backend_type.value,
        backend_config.extra.get("queue_overflow_policy"),
        backend_config.extra.get("queue_maxsize"),
    )


def _configure_extension_loader_runtime(settings: Settings) -> None:
    """Apply settings-level extension-loader pressure/reporter configuration."""
    try:
        from victor.core.verticals.extension_loader import (
            VerticalExtensionLoader,
            start_extension_loader_metrics_reporter,
            stop_extension_loader_metrics_reporter,
        )
    except Exception as e:
        logger.debug("Failed to import extension loader runtime config hooks: %s", e)
        return

    def _setting_int(name: str, default: int) -> int:
        value = getattr(settings, name, default)
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, (int, float)):
            return int(value)
        if isinstance(value, str):
            try:
                return int(value.strip())
            except ValueError:
                return default
        return default

    def _setting_float(name: str, default: float) -> float:
        value = getattr(settings, name, default)
        if isinstance(value, bool):
            return float(value)
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value.strip())
            except ValueError:
                return default
        return default

    def _setting_bool(name: str, default: bool) -> bool:
        value = getattr(settings, name, default)
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"1", "true", "yes", "on"}:
                return True
            if normalized in {"0", "false", "no", "off"}:
                return False
        return default

    def _setting_str(name: str, default: str) -> str:
        value = getattr(settings, name, default)
        return value if isinstance(value, str) and value.strip() else default

    VerticalExtensionLoader.configure_extension_loader_pressure(
        warn_queue_threshold=_setting_int("extension_loader_warn_queue_threshold", 24),
        error_queue_threshold=_setting_int("extension_loader_error_queue_threshold", 32),
        warn_in_flight_threshold=_setting_int("extension_loader_warn_in_flight_threshold", 6),
        error_in_flight_threshold=_setting_int("extension_loader_error_in_flight_threshold", 8),
        cooldown_seconds=_setting_float("extension_loader_pressure_cooldown_seconds", 5.0),
        emit_events=_setting_bool("extension_loader_emit_pressure_events", False),
    )

    if _setting_bool("extension_loader_metrics_reporter_enabled", False):
        start_extension_loader_metrics_reporter(
            interval_seconds=_setting_float(
                "extension_loader_metrics_reporter_interval_seconds", 60.0
            ),
            topic=_setting_str(
                "extension_loader_metrics_reporter_topic",
                "vertical.extensions.loader.metrics",
            ),
            source="BootstrapExtensionLoaderMetricsReporter",
            reset_after_emit=_setting_bool(
                "extension_loader_metrics_reporter_reset_after_emit", False
            ),
        )
    else:
        stop_extension_loader_metrics_reporter(timeout=2.0)


def _register_analytics_services(container: ServiceContainer, settings: Settings) -> None:
    """Register analytics and logging services."""

    # Metrics service
    container.register(
        MetricsServiceProtocol,
        lambda c: NullMetricsService(),  # No-op by default
        ServiceLifetime.SINGLETON,
    )

    # Logger service
    container.register(
        LoggerServiceProtocol,
        lambda c: ConsoleLoggerService(enabled=True),
        ServiceLifetime.SINGLETON,
    )

    # Enhanced usage logger
    paths = get_project_paths()

    # Check for test mode and redirect telemetry to test-specific location
    # This prevents MagicMock events from test suites from polluting global usage.jsonl
    import os

    if (
        os.getenv("PYTEST_XDIST_WORKER")
        or os.getenv("TEST_MODE")
        or os.getenv("PYTEST_CURRENT_TEST")
    ):
        # Running under pytest - use test-specific log file
        import tempfile

        test_log_dir = Path(tempfile.gettempdir()) / "victor_test_telemetry"
        test_log_dir.mkdir(exist_ok=True)
        usage_log_file = test_log_dir / "test_usage.jsonl"
    else:
        # Normal operation - use global logs directory
        usage_log_file = paths.global_logs_dir / "usage.jsonl"

    container.register(
        UsageLoggerProtocol,
        lambda c: _create_usage_logger(usage_log_file, settings),
        ServiceLifetime.SINGLETON,
    )


def _register_embedding_services(container: ServiceContainer, settings: Settings) -> None:
    """Register embedding/ML services.

    Registers both the EmbeddingServiceProtocol (for protocol-based injection)
    and the concrete EmbeddingService class (for direct injection), sharing
    the same singleton instance via get_instance().
    """
    from victor.storage.embeddings.service import EmbeddingService

    model_name = settings.embedding.unified_embedding_model

    container.register(
        EmbeddingServiceProtocol,
        lambda c: EmbeddingService.get_instance(model_name=model_name),
        ServiceLifetime.SINGLETON,
    )

    # Also register the concrete class so components can inject it directly
    container.register(
        EmbeddingService,
        lambda c: EmbeddingService.get_instance(model_name=model_name),
        ServiceLifetime.SINGLETON,
    )


def bootstrap_capabilities() -> None:
    """Register capability stubs, then discover enhanced providers from entry points.

    This function:
    1. Registers all contrib stub implementations as STUB
    2. Discovers enhanced implementations from 'victor.capabilities' entry points
    3. Enhanced providers override stubs automatically

    Safe to call multiple times — the registry tracks bootstrapped state.
    """
    from victor.core.capability_registry import CapabilityRegistry, CapabilityStatus
    from victor.framework.vertical_protocols import (
        CodebaseIndexFactoryProtocol,
        EditorProtocol,
        IgnorePatternsProtocol,
        LanguageRegistryProtocol,
        SymbolStoreFactoryProtocol,
        TaskClassifierPhraseProtocol,
        TaskTypeHintProtocol,
        TreeSitterAnalysisProtocol,
        TreeSitterExtractorProtocol,
        TreeSitterParserProtocol,
    )

    registry = CapabilityRegistry.get_instance()
    # Mark as bootstrapped early to prevent re-entry from get()/is_enhanced() calls
    registry._bootstrapped = True

    # 1. Register all stubs
    from victor.contrib.parsing.parser import NullTreeSitterParser
    from victor.contrib.parsing.extractor import NullTreeSitterExtractor
    from victor.contrib.parsing.analysis import NullTreeSitterAnalysis
    from victor.contrib.codebase.indexer import NullCodebaseIndexFactory
    from victor.contrib.codebase.symbol_store import NullSymbolStore
    from victor.contrib.codebase.ignore_patterns import BasicIgnorePatterns
    from victor.contrib.languages.registry import NullLanguageRegistry
    from victor.contrib.prompts.task_hints import NullTaskTypeHinter

    registry.register(TreeSitterParserProtocol, NullTreeSitterParser(), CapabilityStatus.STUB)
    registry.register(TreeSitterExtractorProtocol, NullTreeSitterExtractor(), CapabilityStatus.STUB)
    registry.register(TreeSitterAnalysisProtocol, NullTreeSitterAnalysis(), CapabilityStatus.STUB)
    registry.register(
        CodebaseIndexFactoryProtocol, NullCodebaseIndexFactory(), CapabilityStatus.STUB
    )
    registry.register(SymbolStoreFactoryProtocol, NullSymbolStore(), CapabilityStatus.STUB)
    registry.register(IgnorePatternsProtocol, BasicIgnorePatterns(), CapabilityStatus.STUB)
    registry.register(LanguageRegistryProtocol, NullLanguageRegistry(), CapabilityStatus.STUB)
    registry.register(TaskTypeHintProtocol, NullTaskTypeHinter(), CapabilityStatus.STUB)

    class _NullClassifierPhraseContributor:
        """Returns no additional phrases when no vertical enhances this."""

        def get_classifier_phrases(self) -> dict:
            return {}

    registry.register(
        TaskClassifierPhraseProtocol,
        _NullClassifierPhraseContributor(),
        CapabilityStatus.STUB,
    )

    from victor.contrib.editing.diff_editor import DiffEditor

    registry.register(EditorProtocol, DiffEditor(), CapabilityStatus.STUB)
    # LSPManagerProtocol and EmbeddingModelFactoryProtocol — no stubs registered;
    # capabilities.get() returns None when not available, which callers already handle.

    # 2. Discover enhanced providers from entry points.
    #    Scan the preferred semantic extension capability group plus legacy
    #    framework/SDK groups to bridge older packages during migration.
    #    Uses UnifiedEntryPointRegistry for single-pass lazy scanning.
    try:
        from victor.framework.entry_point_registry import (
            CAPABILITY_ENTRY_POINT_GROUPS,
            get_entry_point_registry,
        )

        registry_instance = get_entry_point_registry()

        for group in CAPABILITY_ENTRY_POINT_GROUPS:
            group_obj = registry_instance.get_group(group)
            if not group_obj:
                continue

            for ep_name, (ep, loaded) in group_obj.entry_points.items():
                try:
                    # Load entry point if not already loaded
                    if not loaded:
                        register_func = ep.load()
                    else:
                        register_func = loaded

                    register_func(registry)
                    logger.debug(f"Loaded capability entry point: {group}:{ep_name}")
                except Exception as e:
                    logger.debug(f"Skipped capability {group}:{ep_name}: {e}")
    except Exception as e:
        logger.debug(f"Entry point discovery failed: {e}")

    # 3. Discover enhanced capabilities from plugin verticals.
    #    Plugins (victor.plugins entry points) register verticals that declare
    #    capabilities via get_capability_registrations(). This path ensures
    #    tools like graph/search that run outside AgentFactory still get
    #    the enhanced CodebaseIndex factory from victor-coding.
    _discover_plugin_capabilities(registry)

    # 4. Auto-detect installed vertical packages that provide enhanced capabilities
    #    but haven't registered via entry points (e.g., victor-coding's CodebaseIndex).
    _auto_detect_enhanced_capabilities(registry)


# CONSOLIDATION: plugin-vertical unification — see memory plugin_vertical_consolidation.md
# Capability registration ultimately runs inside plugin.register(HostPluginContext),
# so this entry-point is the *single* side-effecting registration path. Do not
# add a parallel entry-point scan here — PluginRegistry owns discovery.
# Emit the "CodebaseIndex not enhanced" warning at most once per process. Plugin
# discovery runs several times (bootstrap + lazy ensure_bootstrapped), and the
# repeated warning is pure noise.
_codebase_index_unenhanced_warned = False


def _discover_plugin_capabilities(registry: Any) -> None:
    """Discover enhanced capabilities from installed plugin verticals.

    Triggers full plugin registration if not already done, which causes:
    plugin.register(context) → context.register_vertical(VerticalClass)
    → _auto_register_vertical_capabilities() → CapabilityRegistry.register(ENHANCED)

    This replaces the hardcoded _PLUGIN_VERTICAL_MAP with dynamic discovery.

    The check uses CapabilityRegistry state (not PluginRegistry.context) to
    decide whether to re-run discovery.  This handles the case where the
    CapabilityRegistry singleton was reset (e.g., by test fixtures) while
    the ServiceContainer singleton still has Settings registered, causing
    ensure_bootstrapped() to take the early-return path.
    """
    try:
        from victor.core.capability_registry import CapabilityRegistry as CapReg
        from victor.framework.vertical_protocols import CodebaseIndexFactoryProtocol

        cap_registry = CapReg.get_instance()

        # If enhanced capabilities already registered, nothing to do.
        if cap_registry.is_enhanced(CodebaseIndexFactoryProtocol):
            return

        from victor.core.plugins.registry import PluginRegistry

        plugin_registry = PluginRegistry.get_instance()
        # Force full registration — capabilities are missing even if
        # the plugin context was set by a previous (now stale) bootstrap.
        plugin_registry.register_all()
        # Verify registration succeeded (warn once per process — see flag above)
        if not cap_registry.is_enhanced(CodebaseIndexFactoryProtocol):
            global _codebase_index_unenhanced_warned
            if not _codebase_index_unenhanced_warned:
                _codebase_index_unenhanced_warned = True
                logger.warning(
                    "Plugin capability discovery completed but CodebaseIndex "
                    "still not enhanced — check victor-coding plugin registration"
                )
    except Exception as e:
        logger.warning("Plugin capability discovery failed: %s", e, exc_info=True)


# ---------------------------------------------------------------------------
# Data-driven auto-detection for enhanced capabilities
# ---------------------------------------------------------------------------
#
# Each entry maps a protocol to either:
#   - "import_path": direct class import (registered as-is)
#   - "factory": module:function that returns an instance (or None)
#
# To add a new auto-detected capability, append an entry here — no new
# function needed.
# DEPRECATED: Capabilities are now registered via the plugin system.
# Plugins call context.register_capability() or declare them via
# get_capability_registrations() on their VerticalBase subclass.
# See HostPluginContext.register_vertical() for the bridge logic.
_AUTO_DETECT_SPECS: list[dict[str, Any]] = []


def _auto_detect_enhanced_capabilities(registry: Any) -> None:
    """Auto-detect enhanced capabilities from installed vertical packages.

    Uses a declarative spec list (_AUTO_DETECT_SPECS) so adding a new
    auto-detected protocol requires only a dict entry, not a new function.
    """
    import importlib

    from victor.core.capability_registry import CapabilityStatus
    from victor.framework import vertical_protocols

    for spec in _AUTO_DETECT_SPECS:
        protocol = getattr(vertical_protocols, spec["protocol_attr"], None)
        if protocol is None:
            continue

        if registry.is_enhanced(protocol):
            continue

        label = spec.get("label", spec["protocol_attr"])

        # Strategy 1: direct class import
        if "import_path" in spec:
            module_path, class_name = spec["import_path"].rsplit(":", 1)
            try:
                module = importlib.import_module(module_path)
                provider = getattr(module, class_name)
                registry.register(protocol, provider, CapabilityStatus.ENHANCED)
                logger.info(f"Auto-detected enhanced {label}")
            except ImportError:
                logger.debug(f"{label} auto-detection skipped: package not installed")
            except Exception as e:
                logger.debug(f"{label} auto-detection failed: {e}")
            continue

        # Strategy 2: factory function
        if "factory" in spec:
            module_path, func_name = spec["factory"].rsplit(":", 1)
            try:
                module = importlib.import_module(module_path)
                factory_fn = getattr(module, func_name)
                provider = factory_fn()
                if provider is not None:
                    registry.register(protocol, provider, CapabilityStatus.ENHANCED)
                    logger.info(f"Auto-detected enhanced {label}")
            except Exception as e:
                logger.debug(f"{label} auto-detection skipped: {e}")


def _register_language_services(container: ServiceContainer, settings: Settings) -> None:
    """Register language plugin services (discovery, indexing).

    This ensures language plugins are discovered at startup, not mid-conversation.
    Uses the capability registry — no direct vertical imports.

    Note: Currently unused by bootstrap phases. Available for vertical-triggered
    language plugin discovery via ``victor.bootstrap_services`` entry points.
    """
    from victor.core.capability_registry import CapabilityRegistry
    from victor.framework.vertical_protocols import LanguageRegistryProtocol

    registry = CapabilityRegistry.get_instance()
    lang_registry = registry.get(LanguageRegistryProtocol)
    if lang_registry is not None and registry.is_enhanced(LanguageRegistryProtocol):
        try:
            count = lang_registry.discover_plugins()
            logger.info(f"Discovered {count} language plugins at startup")
        except Exception as e:
            logger.warning(f"Failed to discover language plugins: {e}")
    else:
        logger.debug("Language plugins not available - victor-coding package not installed")


def _register_signature_store(container: ServiceContainer, settings: Settings) -> None:
    """Register failed signature store service."""

    container.register(
        SignatureStoreProtocol,
        lambda c: _create_signature_store(),
        ServiceLifetime.SINGLETON,
    )


def _create_usage_logger(log_file: Path, settings: Settings) -> Any:
    """Create usage logger with enhanced features if available."""
    sampling_filter = _create_sampling_filter(settings)

    try:
        from victor.observability.analytics.enhanced_logger import EnhancedUsageLogger

        return EnhancedUsageLogger(
            log_file=log_file,
            enabled=True,
            scrub_pii=True,
            encrypt=False,  # Enable if needed
            max_log_size=10 * 1024 * 1024,  # 10MB
            backup_count=5,
            compress_rotated=True,
            sampling_filter=sampling_filter,
            # Batches writes instead of a rotation-check + disk write per
            # event (co-design review item 14). LifecycleManager.
            # graceful_shutdown() flushes on the way out; the crash window
            # is <=100 events / 5s, acceptable since consumers (query
            # service, benchmark mining) are offline readers.
            buffered=True,
        )
    except Exception as e:
        logger.warning(f"Failed to create enhanced logger: {e}")
        # Fall back to basic logger
        from victor.observability.analytics.logger import UsageLogger

        return UsageLogger(log_file, enabled=True, sampling_filter=sampling_filter)


def _create_sampling_filter(settings: Settings) -> Any:
    """Create semantic sampling filter if enabled in settings."""
    if not getattr(settings, "usage_sampling_enabled", True):
        return None
    try:
        from victor.observability.analytics.sampling_filter import (
            SamplingPolicy,
            SemanticSamplingFilter,
        )

        policy = SamplingPolicy(
            sample_rate=getattr(settings, "usage_content_sample_rate", 10),
            dedup_window_seconds=getattr(settings, "usage_dedup_window_seconds", 5.0),
        )
        return SemanticSamplingFilter(policy)
    except Exception as e:
        logger.debug("Sampling filter unavailable: %s", e)
        return None


def _create_signature_store() -> Any:
    """Create signature store for failed tool calls."""
    try:
        from victor.agent.signature_store import SignatureStore

        return SignatureStore()
    except Exception as e:
        logger.warning(f"Failed to create signature store: {e}")
        # Return a no-op store
        return NullSignatureStore()


class NullSignatureStore:
    """No-op signature store for when the real one fails to load."""

    def is_known_failure(self, tool_name: str, args: Dict[str, Any]) -> bool:
        return False

    def record_failure(self, tool_name: str, args: Dict[str, Any], error_message: str) -> None:
        pass

    def clear_signature(self, tool_name: str, args: Dict[str, Any]) -> bool:
        return False


def _register_orchestrator_services(container: ServiceContainer, settings: Settings) -> None:
    """Register orchestrator-related services.

    Part of Phase 10 DI Migration - registers services used by AgentOrchestrator.
    This enables:
    - Type-safe dependency injection for orchestrator components
    - Easy testing via mock substitution
    - Proper lifecycle management (singleton vs scoped)

    Args:
        container: DI container to register services in
        settings: Application settings
    """
    try:
        from victor.agent.service_provider import configure_orchestrator_services

        configure_orchestrator_services(container, settings)
        logger.debug("Registered orchestrator services")
    except Exception as e:
        # Don't fail bootstrap if orchestrator services can't be registered
        # The orchestrator will fall back to direct instantiation
        logger.warning(f"Failed to register orchestrator services: {e}")


def _register_agent_services(container: ServiceContainer, settings: Settings) -> None:
    """Register the canonical agent-service layer.

    The service layer is part of the default runtime shape. Bootstrap occurs
    whenever the required conversation and streaming collaborators are
    available; only optional decision-service registration remains feature-flag
    controlled inside :mod:`victor.core.bootstrap_services`.

    Args:
        container: DI container to register services in
        settings: Application settings
    """
    try:
        # Get conversation and streaming coordinators (required for ChatService)
        from victor.agent.protocols import (
            ConversationControllerProtocol,
            StreamingCoordinatorProtocol,
        )

        conversation_controller = container.get_optional(ConversationControllerProtocol)
        streaming_coordinator = container.get_optional(StreamingCoordinatorProtocol)

        if conversation_controller is not None and streaming_coordinator is not None:
            from victor.core.bootstrap_services import bootstrap_new_services

            bootstrap_new_services(
                container,
                conversation_controller=conversation_controller,
                streaming_coordinator=streaming_coordinator,
            )
            logger.debug("Bootstrapped canonical agent services")
        else:
            logger.debug("Skipping canonical agent-service bootstrap (missing dependencies)")
    except Exception as e:
        # Don't fail bootstrap if canonical services can't be registered.
        logger.debug("Failed to bootstrap canonical agent services: %s", e)


def _register_workflow_services(container: ServiceContainer, settings: Settings) -> None:
    """Register workflow-related services.

    Part of SOLID Refactoring - registers services used by the workflow system.
    This enables:
    - Type-safe dependency injection for workflow components
    - Protocol-based architecture (ISP, DIP compliance)
    - Proper lifecycle management (singleton vs scoped vs transient)

    Services registered:
    - Singleton: YAMLWorkflowLoader, WorkflowValidator, NodeExecutorFactory
    - Scoped: ExecutionContext, OrchestratorPool
    - Transient: WorkflowCompiler, WorkflowExecutor

    Args:
        container: DI container to register services in
        settings: Application settings
    """
    try:
        from victor.workflows.services.workflow_service_provider import (
            configure_workflow_services,
        )

        configure_workflow_services(container, settings)
        logger.debug("Registered workflow services")
    except Exception as e:
        # Don't fail bootstrap if workflow services can't be registered
        # The workflow system will fall back to direct instantiation
        logger.warning(f"Failed to register workflow services: {e}")


def _register_workflow_compiler_plugins(
    container: ServiceContainer,
    settings: Settings,
) -> None:
    """Register workflow compiler plugins.

    Part of Plugin Architecture - registers built-in compiler plugins.
    This enables:
    - Third-party plugin registration via WorkflowCompilerRegistry
    - Entry-point-based plugin discovery

    Note: UnifiedWorkflowCompiler is the canonical compiler API for framework
    use (caching, multi-source, execution). The plugin registry extends it
    with third-party compiler backends (e.g., S3, security-specific).

    Plugins registered:
    - YamlCompilerPlugin: YAML workflow compilation

    Args:
        container: DI container to register services in
        settings: Application settings
    """
    try:
        from victor.workflows.plugins import register_builtin_plugins

        register_builtin_plugins()
        logger.debug("Registered workflow compiler plugins")
    except Exception as e:
        # Don't fail bootstrap if plugins can't be registered
        # The plugin system will fall back to direct instantiation
        logger.warning(f"Failed to register workflow compiler plugins: {e}")


def _resolve_vertical_name(settings: Settings, requested_vertical: Optional[str]) -> Optional[str]:
    """Resolve which vertical name should be activated for this bootstrap.

    Returns:
        Vertical name string or None if no vertical should be activated.
    """
    if isinstance(requested_vertical, str) and requested_vertical.strip():
        return requested_vertical.strip()

    default_vertical = getattr(settings, "default_vertical", None)
    if isinstance(default_vertical, str) and default_vertical.strip():
        return default_vertical.strip()

    return None


def _report_capability_health(
    vertical_name: Optional[str],
    container: Optional[ServiceContainer] = None,
) -> None:
    """Log a single actionable warning when a requested vertical is unavailable."""
    if vertical_name is None:
        return

    normalized = vertical_name.strip()
    if not normalized or normalized in _REPORTED_MISSING_VERTICALS:
        return

    try:
        from victor.core.verticals.base import VerticalRegistry
        from victor.core.verticals.vertical_loader import get_vertical_loader
        from victor.core.events import get_observability_bus
    except Exception as exc:
        logger.debug(f"Capability health check skipped: {exc}")
        return

    available = set(VerticalRegistry.list_names())
    try:
        loader = get_vertical_loader()
        # Use discover_vertical_names (reads entry point metadata only)
        # instead of discover_verticals (imports + validates each class).
        available.update(loader.discover_vertical_names())
    except Exception as exc:
        logger.debug(f"Vertical discovery failed during health check: {exc}")

    if normalized in available:
        return

    _REPORTED_MISSING_VERTICALS.add(normalized)
    package_hint = _load_vertical_package_hints().get(normalized)
    if package_hint:
        remedy = (
            f"Install the '{package_hint}' package (e.g., `pip install {package_hint}`) "
            "or choose a different --vertical."
        )
    else:
        remedy = "Install the optional package that provides this capability or select another --vertical."

    logger.warning(
        "Vertical '%s' is unavailable via entry points; capability-dependent features "
        "will run in fallback mode. %s",
        normalized,
        remedy,
    )

    # Emit usage analytics so operators have structured telemetry.
    usage_logger = None
    try:
        target_container = container or get_container()
        usage_logger = target_container.get_optional(UsageLoggerProtocol)  # type: ignore[attr-defined]
        if usage_logger:
            usage_logger.log_event(
                "missing_vertical",
                {
                    "vertical": normalized,
                    "package_hint": package_hint,
                    "remedy": remedy,
                },
            )
    except Exception as exc:
        logger.debug(f"Usage analytics missing_vertical event failed: {exc}")

    # Emit observability event for fleet-wide alerting.
    try:
        bus = get_observability_bus()
        if bus:
            bus.emit_sync(
                topic="capabilities.vertical.missing",
                data={
                    "vertical": normalized,
                    "package_hint": package_hint,
                    "remedy": remedy,
                },
                source="CapabilityHealthMonitor",
            )
    except Exception as exc:
        logger.debug(f"Observability missing_vertical event failed: {exc}")


def _register_vertical_services(
    container: ServiceContainer,
    settings: Settings,
    vertical_name: Optional[str] = None,
) -> None:
    """Register vertical-specific services.

    Loads the specified vertical and registers its services with the container.
    This enables verticals to provide:
    - Custom middleware for tool execution
    - Safety patterns for dangerous operations
    - Prompt contributions for task hints
    - Mode configurations

    Args:
        container: DI container to register services in
        settings: Application settings
        vertical_name: Optional vertical name. If None, uses settings.default_vertical.
                       If settings.default_vertical is also empty, no vertical is
                       activated (base/framework-only mode).
    """
    target_vertical = _resolve_vertical_name(settings, vertical_name)
    if target_vertical is None:
        logger.debug("No vertical requested; skipping vertical service registration")
        return

    try:
        from victor.core.verticals.vertical_loader import activate_vertical_services
        from victor.core.verticals.protocols import VerticalExtensions

        activation = activate_vertical_services(container, settings, target_vertical)

        # Register the extensions as a service for framework access
        from victor.core.verticals.vertical_loader import get_vertical_loader

        loader = get_vertical_loader()
        extensions = loader.get_extensions()
        if extensions:
            if container.is_registered(VerticalExtensions):
                container.register_or_replace(
                    VerticalExtensions,
                    lambda c, ext=extensions: ext,
                    ServiceLifetime.SINGLETON,
                )
            else:
                container.register_instance(VerticalExtensions, extensions)

        logger.info(
            "Registered vertical services: %s (activated=%s, services_registered=%s)",
            target_vertical,
            activation.activated,
            activation.services_registered,
        )
    except ImportError as e:
        logger.debug(f"Vertical loading not available: {e}")
    except ValueError as e:
        logger.warning(f"Failed to load vertical '{target_vertical}': {e}")
    except Exception as e:
        # Don't fail bootstrap if vertical services can't be registered
        logger.warning(f"Failed to register vertical services: {e}")


# =============================================================================
# Convenience Functions
# =============================================================================


def get_service(service_type: Type[T]) -> T:
    """Get a service from the global container.

    Args:
        service_type: Type of service to retrieve

    Returns:
        Service instance
    """
    return get_container().get(service_type)


def get_service_optional(service_type: Type[T]) -> Optional[T]:
    """Get a service from the global container, or None if not found.

    Args:
        service_type: Type of service to retrieve

    Returns:
        Service instance or None
    """
    return get_container().get_optional(service_type)


def ensure_bootstrapped(
    settings: Optional[Settings] = None,
    vertical: Optional[str] = None,
) -> ServiceContainer:
    """Ensure the container is bootstrapped with correct vertical.

    Semi-idempotent - only bootstraps if not already done, but will
    re-activate vertical if a different one is requested.

    Args:
        settings: Optional Settings instance
        vertical: Optional vertical name (e.g., "coding", "data_analysis")

    Returns:
        Service container
    """
    container = get_container()

    # Check if already bootstrapped (Settings registered)
    if container.is_registered(Settings):
        # Container already bootstrapped — still run plugin discovery so capabilities
        # from installed verticals (e.g. CodebaseIndex for graph tool) are registered
        # even when no --vertical flag is passed.
        _discover_plugin_capabilities(None)
        # Check if we need to switch verticals
        if vertical is not None:
            _ensure_vertical_activated(container, settings or container.get(Settings), vertical)
        return container

    return bootstrap_container(settings, vertical=vertical)


def _ensure_vertical_activated(
    container: ServiceContainer,
    settings: Settings,
    vertical_name: str,
) -> None:
    """Ensure the specified vertical is activated, switching if needed.

    This handles the case where the container was bootstrapped with one
    vertical but we need to use a different one (e.g., --vertical CLI flag).

    Args:
        container: DI container
        settings: Application settings
        vertical_name: Vertical name to ensure is active
    """
    _report_capability_health(vertical_name, container)
    try:
        from victor.core.verticals.vertical_loader import (
            activate_vertical_services,
            get_vertical_loader,
        )
        from victor.core.verticals.protocols import VerticalExtensions

        loader = get_vertical_loader()
        current_vertical = loader.active_vertical_name

        # If no vertical active or different vertical requested, (re)activate
        if current_vertical is None or current_vertical != vertical_name:
            logger.info(f"Switching vertical: {current_vertical or 'none'} -> {vertical_name}")
            activation = activate_vertical_services(container, settings, vertical_name)

            # Update the extensions in container
            extensions = loader.get_extensions()
            if extensions:
                container.register_or_replace(
                    VerticalExtensions,
                    lambda c, ext=extensions: ext,
                    ServiceLifetime.SINGLETON,
                )

            logger.info(
                "Vertical switched to: %s (activated=%s, services_registered=%s)",
                vertical_name,
                activation.activated,
                activation.services_registered,
            )
    except ImportError as e:
        logger.debug(f"Vertical loading not available: {e}")
    except ValueError as e:
        logger.warning(f"Failed to switch to vertical '{vertical_name}': {e}")
    except Exception as e:
        logger.warning(f"Failed to ensure vertical activation: {e}")
