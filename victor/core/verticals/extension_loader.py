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

"""Vertical Extension Loading Capabilities.

This module provides a comprehensive extension loading system for Victor verticals,
enabling modular integration with framework components through lazy loading and
caching infrastructure.

Key Features:
- Fine-grained extension caching with composite keys to prevent collisions
- Lazy loading with auto-generated class names for reduced boilerplate
- Strict error handling with configurable extension requirements
- Support for 10+ extension types (middleware, safety, prompt, workflow, etc.)

Extension Types:
    - Middleware: Tool execution pipeline processing
    - Safety Extension: Dangerous operation pattern detection
    - Prompt Contributor: Task hints and prompt sections
    - Mode Config Provider: Operational mode configurations
    - Tool Dependency Provider: Tool execution patterns
    - Workflow Provider: Vertical-specific workflow definitions
    - Service Provider: DI container integration
    - RL Config Provider: Reinforcement learning configurations
    - Team Spec Provider: Multi-agent team formations
    - Enrichment Strategy: DSPy-like prompt optimization
    - Tiered Tool Config: Context-aware tool selection

Design Principles:
    - SRP Compliance: Extracted from VerticalBase for focused responsibility
    - Lazy Loading: Extensions loaded only when first accessed
    - Caching: Fine-grained cache with composite keys (ClassName:extension_key)
    - Error Tolerance: Graceful degradation with strict mode optional
    - Auto-Discovery: Auto-generated class names reduce boilerplate

Usage:
    from victor.core.verticals.extension_loader import VerticalExtensionLoader

    class MyVertical(VerticalExtensionLoader):
        @classmethod
        def get_safety_extension(cls):
            # Use generic factory (auto-generates "MyVerticalSafetyExtension")
            return cls._get_extension_factory(
                "safety_extension",
                "myvertical.safety",
            )

        @classmethod
        def get_prompt_contributor(cls):
            # Custom class name
            return cls._get_extension_factory(
                "prompt_contributor",
                "myvertical.prompts",
                "MyCustomPromptContributor",
            )

Error Handling:
    # Strict mode (any failure raises ExtensionLoadError)
    class StrictVertical(VerticalExtensionLoader):
        strict_extension_loading = True

    # Required extensions (must succeed even in non-strict mode)
    class CriticalVertical(VerticalExtensionLoader):
        required_extensions = {"safety", "middleware"}

Related Modules:
    - victor.core.verticals.base: VerticalBase (uses this loader)
    - victor.core.verticals.protocols: Extension protocol definitions
    - victor.core.verticals.metadata: Metadata provider
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import importlib
import importlib.util
import logging
import threading
import time
from abc import ABC, abstractmethod
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Dict,
    List,
    Optional,
    Set,
    Type,
)

from victor.core.verticals.extension_cache_manager import ExtensionCacheManager
from victor.core.verticals.extension_module_resolver import ExtensionModuleResolver
from victor.core.verticals.import_resolver import vertical_runtime_module_candidates
from victor.core.verticals.vertical_metadata import VerticalMetadata

if TYPE_CHECKING:
    from victor.core.verticals.protocols import VerticalExtensions
    from victor.core.vertical_types import TieredToolConfig

logger = logging.getLogger(__name__)

_EXTENSION_LOADER_METRICS_TOPIC = "vertical.extensions.loader.metrics"
_EXTENSION_LOADER_PRESSURE_TOPIC = "vertical.extensions.loader.pressure"

_METRICS_REPORTER_LOCK = threading.Lock()
_METRICS_REPORTER_SINGLETON: Optional["ExtensionLoaderMetricsReporter"] = None


class PressureLevel(str, Enum):
    """Pressure level for extension loader queue and in-flight operations."""

    ERROR = "error"  # Critical saturation (at or above error thresholds)
    WARN = "warn"  # Elevated load (at or above warning thresholds)
    OK = "ok"  # Normal operation (below all thresholds)


def _build_team_spec_provider_from_definition(definition: Any) -> Optional[Any]:
    """Build a team-spec provider from an SDK VerticalDefinition's team metadata."""
    from dataclasses import dataclass

    if not definition.team_metadata.teams:
        return None

    from victor.framework.team_schema import TeamSpec
    from victor.framework.teams import TeamMemberSpec
    from victor.teams.types import MemoryConfig, TeamAgentCategory, TeamFormation

    @dataclass(frozen=True)
    class _TeamSpecProvider:
        team_specs: Dict[str, Any]
        default_team: Optional[str] = None

        def get_team_specs(self) -> Dict[str, Any]:
            return dict(self.team_specs)

        def get_default_team(self) -> Optional[str]:
            return self.default_team

    team_specs: Dict[str, TeamSpec] = {}
    for team in definition.team_metadata.teams:
        members = []
        for member in team.members:
            memory_config = MemoryConfig(**member.memory_config) if member.memory_config else None
            members.append(
                TeamMemberSpec(
                    role=member.role,
                    goal=member.goal,
                    name=member.name,
                    tool_budget=member.tool_budget,
                    allowed_tools=member.allowed_tools or None,
                    agent_category=TeamAgentCategory(member.agent_category),
                    is_manager=member.is_manager,
                    priority=member.priority,
                    backstory=member.backstory,
                    expertise=member.expertise.copy(),
                    personality=member.personality,
                    max_delegation_depth=member.max_delegation_depth,
                    memory=member.memory,
                    memory_config=memory_config,
                    cache=member.cache,
                    verbose=member.verbose,
                    max_iterations=member.max_iterations,
                )
            )

        team_specs[team.team_id] = TeamSpec(
            name=team.name,
            description=team.description,
            vertical=definition.name,
            formation=TeamFormation(team.formation),
            members=members,
            total_tool_budget=team.total_tool_budget,
            max_iterations=team.max_iterations,
            tags=team.tags.copy(),
            task_types=team.task_types.copy(),
            metadata=dict(team.metadata),
        )

    return _TeamSpecProvider(
        team_specs=team_specs,
        default_team=definition.team_metadata.default_team,
    )


class ExtensionLoaderPressureMonitor:
    """Internal helper that owns all pressure monitoring and metrics state.

    Encapsulates the lightweight loader metrics counters, queue-pressure
    thresholds, cooldown logic, and event emission so that
    ``VerticalExtensionLoader`` can delegate to a single instance without
    spreading metrics bookkeeping across many class variables.

    Not part of the public API -- instantiated once as a class variable on
    ``VerticalExtensionLoader._pressure_monitor``.
    """

    def __init__(self) -> None:
        self._metrics: Dict[str, int] = {
            "submitted": 0,
            "completed": 0,
            "failed": 0,
            "in_flight": 0,
            "max_in_flight": 0,
            "queued": 0,
            "max_queued": 0,
            "queue_waits": 0,
            "pressure_warnings": 0,
            "pressure_errors": 0,
        }
        self._metrics_lock = threading.Lock()

        # Queue pressure policy (P3 reliability).
        self.warn_queue_threshold: int = 24
        self.error_queue_threshold: int = 32
        self.warn_in_flight_threshold: int = 14
        self.error_in_flight_threshold: int = 18
        self.pressure_cooldown_seconds: float = 5.0
        self.emit_pressure_events: bool = False
        self.last_pressure_level: PressureLevel = PressureLevel.OK
        self.last_pressure_emit_ts: float = 0.0

        # Track optional modules already reported missing (avoid log spam).
        self._missing_extension_modules: Set[str] = set()
        self._missing_extension_modules_lock = threading.RLock()

    # ------------------------------------------------------------------
    # Metric helpers
    # ------------------------------------------------------------------

    def increment_metric(self, metric: str, delta: int = 1) -> None:
        """Increment extension loader metric counter."""
        with self._metrics_lock:
            self._metrics[metric] = self._metrics.get(metric, 0) + delta

    def update_peak_metric(self, metric: str, value: int) -> None:
        """Update extension loader metric peak value."""
        with self._metrics_lock:
            current = self._metrics.get(metric, 0)
            if value > current:
                self._metrics[metric] = value

    def get_metric(self, metric: str) -> int:
        """Return current value of a single metric under lock."""
        with self._metrics_lock:
            return self._metrics.get(metric, 0)

    def get_metrics_snapshot(
        self,
        *,
        max_workers: int = 0,
        queue_limit: int = 0,
    ) -> Dict[str, Any]:
        """Return snapshot of all loader metrics plus threshold config.

        Args:
            max_workers: Executor max_workers value to include in snapshot.
            queue_limit: Executor queue_limit value to include in snapshot.
        """
        with self._metrics_lock:
            snapshot: Dict[str, Any] = dict(self._metrics)
        snapshot["max_workers"] = max_workers
        snapshot["queue_limit"] = queue_limit
        snapshot["warn_queue_threshold"] = self.warn_queue_threshold
        snapshot["error_queue_threshold"] = self.error_queue_threshold
        snapshot["warn_in_flight_threshold"] = self.warn_in_flight_threshold
        snapshot["error_in_flight_threshold"] = self.error_in_flight_threshold
        snapshot["pressure_level"] = self.last_pressure_level
        return snapshot

    def reset_metrics(self) -> None:
        """Reset all metric counters and pressure state."""
        with self._metrics_lock:
            for key in (
                "submitted",
                "completed",
                "failed",
                "in_flight",
                "max_in_flight",
                "queued",
                "max_queued",
                "queue_waits",
                "pressure_warnings",
                "pressure_errors",
            ):
                self._metrics[key] = 0
            self.last_pressure_level = "ok"
            self.last_pressure_emit_ts = 0.0

    # ------------------------------------------------------------------
    # Pressure level evaluation
    # ------------------------------------------------------------------

    def pressure_level(self, *, queued: int, in_flight: int) -> PressureLevel:
        """Return pressure level for current loader queue and in-flight counts.

        Args:
            queued: Number of extensions waiting in queue
            in_flight: Number of extensions currently being loaded

        Returns:
            PressureLevel enum: ERROR if at error thresholds, WARN if at warning thresholds, OK otherwise
        """
        if queued >= self.error_queue_threshold or in_flight >= self.error_in_flight_threshold:
            return PressureLevel.ERROR
        if queued >= self.warn_queue_threshold or in_flight >= self.warn_in_flight_threshold:
            return PressureLevel.WARN
        return PressureLevel.OK

    # ------------------------------------------------------------------
    # Pressure event emission
    # ------------------------------------------------------------------

    def emit_pressure_event(
        self, level: PressureLevel, snapshot: Dict[str, Any], reason: str
    ) -> None:
        """Emit queue-pressure signal for extension loader saturation."""
        try:
            from victor.core.events import get_observability_bus
            from victor.core.events.emit_helper import emit_event_sync

            bus = get_observability_bus()
            if bus is None:
                return
            emit_event_sync(
                bus,
                _EXTENSION_LOADER_PRESSURE_TOPIC,
                {
                    "level": level,
                    "reason": reason,
                    "metrics": snapshot,
                },
                source="VerticalExtensionLoader",
                use_background_loop=True,
                track_metrics=False,
            )
        except Exception as e:
            logger.debug("Failed emitting extension loader pressure event: %s", e)

    # ------------------------------------------------------------------
    # Combined pressure check (warn / error / emit)
    # ------------------------------------------------------------------

    def check_pressure(
        self,
        *,
        reason: str,
        max_workers: int = 0,
        queue_limit: int = 0,
    ) -> None:
        """Check queue pressure and optionally emit warning/error diagnostics."""
        snapshot = self.get_metrics_snapshot(
            max_workers=max_workers,
            queue_limit=queue_limit,
        )
        queued = int(snapshot.get("queued", 0))
        in_flight = int(snapshot.get("in_flight", 0))
        level = self.pressure_level(queued=queued, in_flight=in_flight)

        now = time.monotonic()
        should_emit = False
        with self._metrics_lock:
            previous_level = self.last_pressure_level
            cooldown = self.pressure_cooldown_seconds
            elapsed = now - self.last_pressure_emit_ts

            if level == PressureLevel.OK:
                self.last_pressure_level = PressureLevel.OK
                return

            if level != previous_level or elapsed >= cooldown:
                self.last_pressure_level = level
                self.last_pressure_emit_ts = now
                should_emit = True
                if level == PressureLevel.ERROR:
                    self._metrics["pressure_errors"] += 1
                else:
                    self._metrics["pressure_warnings"] += 1

        if not should_emit:
            return

        message = (
            "Extension loader pressure %s: queued=%s in_flight=%s "
            "(reason=%s, thresholds q:%s/%s in_f:%s/%s)"
        )
        args = (
            level.value.upper(),
            queued,
            in_flight,
            reason,
            self.warn_queue_threshold,
            self.error_queue_threshold,
            self.warn_in_flight_threshold,
            self.error_in_flight_threshold,
        )

        if level == "error":
            logger.error(message, *args)
        else:
            logger.warning(message, *args)

        if self.emit_pressure_events:
            self.emit_pressure_event(level, snapshot, reason)

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def configure(
        self,
        *,
        warn_queue_threshold: Optional[int] = None,
        error_queue_threshold: Optional[int] = None,
        warn_in_flight_threshold: Optional[int] = None,
        error_in_flight_threshold: Optional[int] = None,
        cooldown_seconds: Optional[float] = None,
        emit_events: Optional[bool] = None,
    ) -> None:
        """Configure queue-pressure warning/error thresholds."""
        with self._metrics_lock:
            if warn_queue_threshold is not None:
                self.warn_queue_threshold = max(0, int(warn_queue_threshold))
            if error_queue_threshold is not None:
                self.error_queue_threshold = max(0, int(error_queue_threshold))
            if warn_in_flight_threshold is not None:
                self.warn_in_flight_threshold = max(0, int(warn_in_flight_threshold))
            if error_in_flight_threshold is not None:
                self.error_in_flight_threshold = max(0, int(error_in_flight_threshold))
            if cooldown_seconds is not None:
                self.pressure_cooldown_seconds = max(0.0, float(cooldown_seconds))
            if emit_events is not None:
                self.emit_pressure_events = bool(emit_events)

    # ------------------------------------------------------------------
    # Metrics event emission
    # ------------------------------------------------------------------

    def emit_metrics_event(
        self,
        *,
        max_workers: int = 0,
        queue_limit: int = 0,
        event_bus: Optional[Any] = None,
        topic: str = _EXTENSION_LOADER_METRICS_TOPIC,
        source: str = "VerticalExtensionLoader",
        reset_after_emit: bool = False,
    ) -> Dict[str, Any]:
        """Emit current extension-loader metrics as an observability event."""
        metrics = self.get_metrics_snapshot(
            max_workers=max_workers,
            queue_limit=queue_limit,
        )

        bus = event_bus
        if bus is None:
            try:
                from victor.core.events import get_observability_bus

                bus = get_observability_bus()
            except Exception as e:
                logger.debug("Failed resolving observability bus for loader metrics: %s", e)
                bus = None

        if bus is not None:
            try:
                from victor.core.events.emit_helper import emit_event_sync

                emit_event_sync(
                    bus,
                    topic,
                    {"metrics": metrics},
                    source=source,
                    use_background_loop=True,
                    track_metrics=False,
                )
            except Exception as e:
                logger.debug("Failed emitting extension-loader metrics event: %s", e)

        if reset_after_emit:
            self.reset_metrics()
        return metrics

    # ------------------------------------------------------------------
    # Missing module tracking
    # ------------------------------------------------------------------

    def record_missing_module(self, cache_key: str) -> bool:
        """Record a missing module key. Return True if this is the first time."""
        with self._missing_extension_modules_lock:
            if cache_key not in self._missing_extension_modules:
                self._missing_extension_modules.add(cache_key)
                return True
        return False


class VerticalExtensionLoader(ABC):
    """Loader of vertical extensions.

    Handles loading and caching of vertical extensions including middleware,
    safety, prompt contributors, mode configs, and more.

    This is a mix-in class that provides extension loading capabilities
    to verticals while maintaining SRP compliance.

    Extension Loading Configuration:
        strict_extension_loading: When True, any extension loading failure
            raises ExtensionLoadError.
        required_extensions: Extensions that must load successfully even
            when strict_extension_loading=False.
    """

    # Extension loading configuration
    # When True, any extension loading failure raises ExtensionLoadError
    strict_extension_loading: ClassVar[bool] = False

    # Extensions that must load successfully even when strict_extension_loading=False
    # Valid values: "middleware", "safety", "prompt", "mode_config", "tool_deps",
    #               "workflow", "service", "rl_config", "team_spec", "enrichment",
    #               "tiered_tools"
    required_extensions: ClassVar[Set[str]] = set()

    # LSP capability for language intelligence
    _lsp_capability: ClassVar[Optional[Any]] = None

    # Extension cache (shared across all verticals)
    _cache_manager: ClassVar[ExtensionCacheManager] = ExtensionCacheManager()

    # Shared async extension loading infrastructure (P3)
    _extension_executor_max_workers: ClassVar[int] = 14
    _extension_executor_queue_limit: ClassVar[int] = 32
    _extension_load_timeout: ClassVar[float] = 10.0
    _extension_executor: ClassVar[Optional[concurrent.futures.ThreadPoolExecutor]] = None
    _extension_executor_lock: ClassVar[threading.RLock] = threading.RLock()
    _extension_executor_semaphores: ClassVar[Dict[int, asyncio.Semaphore]] = {}

    # Pressure monitoring (metrics, thresholds, cooldown, missing-module tracking).
    _pressure_monitor: ClassVar[ExtensionLoaderPressureMonitor] = ExtensionLoaderPressureMonitor()

    # Module resolver (delegates candidate resolution, availability checks,
    # attribute loading, and class-name generation).
    _module_resolver: ClassVar[ExtensionModuleResolver] = ExtensionModuleResolver(_pressure_monitor)

    @classmethod
    def _cache_namespace(cls) -> str:
        """Return namespaced cache prefix for this vertical class."""
        return f"{cls.__name__}:{cls.__module__}:{cls.__qualname__}"

    @classmethod
    def get_lsp(cls) -> Optional[Any]:
        """Get the LSP capability for this vertical.

        Returns:
            LSPCapability instance or None
        """
        return cls._lsp_capability

    @classmethod
    def set_lsp(cls, lsp_capability: Any) -> None:
        """Set the LSP capability for this vertical.

        Enables language intelligence features for extension loading
        and code-related operations.

        Args:
            lsp_capability: LSPCapability instance
        """
        cls._lsp_capability = lsp_capability

    @classmethod
    def _get_shared_extension_executor(cls) -> concurrent.futures.ThreadPoolExecutor:
        """Get shared bounded executor for async extension loading."""
        base = VerticalExtensionLoader
        with base._extension_executor_lock:
            if base._extension_executor is None:
                base._extension_executor = concurrent.futures.ThreadPoolExecutor(
                    max_workers=base._extension_executor_max_workers,
                    thread_name_prefix="victor-ext-loader",
                )
            return base._extension_executor

    @classmethod
    def _get_extension_load_semaphore(cls) -> asyncio.Semaphore:
        """Get per-event-loop semaphore for bounded async submission."""
        base = VerticalExtensionLoader
        loop = asyncio.get_running_loop()
        loop_id = id(loop)
        with base._extension_executor_lock:
            semaphore = base._extension_executor_semaphores.get(loop_id)
            if semaphore is None:
                semaphore = asyncio.Semaphore(base._extension_executor_queue_limit)
                base._extension_executor_semaphores[loop_id] = semaphore
            return semaphore

    @classmethod
    def _increment_loader_metric(cls, metric: str, delta: int = 1) -> None:
        """Increment extension loader metric counter."""
        VerticalExtensionLoader._pressure_monitor.increment_metric(metric, delta)

    @classmethod
    def _update_loader_peak_metric(cls, metric: str, value: int) -> None:
        """Update extension loader metric peak value."""
        VerticalExtensionLoader._pressure_monitor.update_peak_metric(metric, value)

    @classmethod
    def get_extension_loader_metrics(cls) -> Dict[str, Any]:
        """Return snapshot of shared async extension loader metrics."""
        base = VerticalExtensionLoader
        return base._pressure_monitor.get_metrics_snapshot(
            max_workers=base._extension_executor_max_workers,
            queue_limit=base._extension_executor_queue_limit,
        )

    @classmethod
    def reset_extension_loader_metrics(cls) -> None:
        """Reset async extension loader metrics counters."""
        VerticalExtensionLoader._pressure_monitor.reset_metrics()

    @classmethod
    def _extension_module_candidates(cls, module_suffix: str) -> List[str]:
        """Return possible module paths for an optional vertical extension."""
        vertical_name = getattr(cls, "name", None) or ""
        return cls._module_resolver.resolve_candidates(vertical_name, module_suffix)

    @classmethod
    def _extension_module_available(cls, module_path: str) -> bool:
        """Return True when the extension module can be imported."""
        return cls._module_resolver.is_available(
            module_path,
            vertical_display_name=getattr(cls, "name", cls.__name__),
            caller_class_name=cls.__name__,
        )

    @classmethod
    def _find_available_candidates(cls, suffix: str) -> List[str]:
        """Resolve and filter extension module candidates to available ones."""
        return [
            path
            for path in cls._extension_module_candidates(suffix)
            if cls._extension_module_available(path)
        ]

    @classmethod
    def _resolve_factory_extension(
        cls,
        extension_key: str,
        suffix: str,
        class_name: Optional[str] = None,
    ) -> Optional[Any]:
        """Common pattern: find candidates → try _get_extension_factory on each.

        Used by get_safety_extension, get_prompt_contributor, get_team_spec_provider,
        get_enrichment_strategy, etc. to eliminate repeated candidate iteration.

        Args:
            extension_key: Cache key for the extension.
            suffix: Module suffix to resolve candidates for.
            class_name: Optional explicit class name (auto-generated if None).

        Returns:
            Extension instance or None if not found.
        """
        candidate_paths = cls._find_available_candidates(suffix)
        if not candidate_paths:
            return None

        last_error: Optional[Exception] = None
        for module_path in candidate_paths:
            try:
                return cls._get_extension_factory(
                    extension_key,
                    module_path,
                    class_name,
                )
            except (ImportError, AttributeError) as exc:
                last_error = exc

        if last_error:
            raise last_error
        return None

    @classmethod
    def _resolve_class_or_factory_extension(
        cls,
        extension_key: str,
        suffix: str,
        class_name: Optional[str] = None,
    ) -> Optional[Any]:
        """Common pattern: find candidates → try direct class import → fallback to factory.

        Used by get_mode_config_provider, get_rl_config_provider, get_rl_hooks,
        get_capability_provider, get_service_provider, etc.

        Args:
            extension_key: Cache key for the extension.
            suffix: Module suffix to resolve candidates for.
            class_name: Explicit class name to look for. If None, auto-generated.

        Returns:
            Extension instance or None if not found.
        """
        if class_name is None:
            class_name = cls._module_resolver.auto_generate_class_name(cls.__name__, extension_key)
        candidate_paths = cls._find_available_candidates(suffix)
        if not candidate_paths:
            return None

        last_error: Optional[Exception] = None
        # Try direct class import first.
        #
        # ``importlib.import_module()`` handles nested synthetic/runtime package
        # layouts more reliably than ``__import__`` during mixed-mode migration
        # tests, while preserving the same observable behavior for real packages.
        for module_path in candidate_paths:
            try:
                module = importlib.import_module(module_path)
                provider_cls = getattr(module, class_name, None)
                if provider_cls is not None:
                    return provider_cls()
            except (ImportError, AttributeError) as exc:
                last_error = exc

        # Fallback to extension factory
        for module_path in candidate_paths:
            try:
                return cls._get_extension_factory(
                    extension_key,
                    module_path,
                    class_name,
                )
            except (ImportError, AttributeError) as exc:
                last_error = exc

        if last_error:
            raise last_error
        return None

    @classmethod
    def configure_extension_loader_pressure(
        cls,
        *,
        warn_queue_threshold: Optional[int] = None,
        error_queue_threshold: Optional[int] = None,
        warn_in_flight_threshold: Optional[int] = None,
        error_in_flight_threshold: Optional[int] = None,
        cooldown_seconds: Optional[float] = None,
        emit_events: Optional[bool] = None,
    ) -> None:
        """Configure queue-pressure warning/error thresholds for extension loading."""
        VerticalExtensionLoader._pressure_monitor.configure(
            warn_queue_threshold=warn_queue_threshold,
            error_queue_threshold=error_queue_threshold,
            warn_in_flight_threshold=warn_in_flight_threshold,
            error_in_flight_threshold=error_in_flight_threshold,
            cooldown_seconds=cooldown_seconds,
            emit_events=emit_events,
        )

    @classmethod
    def _pressure_level(cls, *, queued: int, in_flight: int) -> str:
        """Return pressure level for current loader queue and in-flight counts."""
        return VerticalExtensionLoader._pressure_monitor.pressure_level(
            queued=queued, in_flight=in_flight
        )

    @classmethod
    def _emit_pressure_event(cls, level: str, snapshot: Dict[str, Any], reason: str) -> None:
        """Emit queue-pressure signal for extension loader saturation."""
        VerticalExtensionLoader._pressure_monitor.emit_pressure_event(level, snapshot, reason)

    @classmethod
    def _check_pressure(cls, *, reason: str) -> None:
        """Check queue pressure and optionally emit warning/error diagnostics."""
        base = VerticalExtensionLoader
        base._pressure_monitor.check_pressure(
            reason=reason,
            max_workers=base._extension_executor_max_workers,
            queue_limit=base._extension_executor_queue_limit,
        )

    @classmethod
    def emit_extension_loader_metrics_event(
        cls,
        *,
        event_bus: Optional[Any] = None,
        topic: str = _EXTENSION_LOADER_METRICS_TOPIC,
        source: str = "VerticalExtensionLoader",
        reset_after_emit: bool = False,
    ) -> Dict[str, Any]:
        """Emit current extension-loader metrics as an observability event."""
        base = VerticalExtensionLoader
        return base._pressure_monitor.emit_metrics_event(
            max_workers=base._extension_executor_max_workers,
            queue_limit=base._extension_executor_queue_limit,
            event_bus=event_bus,
            topic=topic,
            source=source,
            reset_after_emit=reset_after_emit,
        )

    # =========================================================================
    # Extension Caching Infrastructure
    # =========================================================================

    @classmethod
    def _get_cached_extension(cls, key: str, factory: callable) -> Any:
        """Get extension from cache or create and cache it.

        This helper enables fine-grained caching of individual extension
        instances, avoiding repeated object creation when extensions are
        accessed multiple times.

        The cache uses a composite key of (class_name, extension_key) to
        ensure proper isolation between different vertical subclasses.

        Args:
            key: Unique key for this extension type (e.g., "middleware",
                 "safety_extension", "workflow_provider")
            factory: Zero-argument callable that creates the extension instance.
                     Only called if the extension is not already cached.

        Returns:
            Cached or newly created extension instance.

        Example:
            @classmethod
            def get_middleware(cls) -> List[MiddlewareProtocol]:
                def _create():
                    from myvertical.middleware import MyMiddleware
                    return [MyMiddleware()]
                return cls._get_cached_extension("middleware", _create)
        """
        return cls._cache_manager.get_or_create(cls._cache_namespace(), key, factory)

    @classmethod
    def _get_cached_extension_value(cls, key: str) -> tuple[bool, Any]:
        """Return a cached extension value without caching misses."""
        return cls._cache_manager.get_if_cached(cls._cache_namespace(), key)

    @classmethod
    def _load_cached_optional_extension(
        cls,
        extension_key: str,
        loader: Callable[[], Optional[Any]],
    ) -> Optional[Any]:
        """Load an optional extension while caching hits but not misses."""
        return cls._cache_manager.load_optional(cls._cache_namespace(), extension_key, loader)

    @classmethod
    def _load_named_entry_point_extension(
        cls,
        extension_key: str,
        group: str,
    ) -> Optional[Any]:
        """Resolve an optional runtime extension from an explicit entry-point group."""
        try:
            from victor.framework.entry_point_loader import (
                load_runtime_extension_from_entry_points,
            )
        except ImportError:
            return None

        return cls._load_cached_optional_extension(
            extension_key,
            lambda: load_runtime_extension_from_entry_points(cls.name, group),
        )

    @classmethod
    def _get_extension_factory(
        cls,
        extension_key: str,
        import_path: str,
        attribute_name: Optional[str] = None,
    ) -> Any:
        """Generic factory for lazy-loading and caching extensions.

        Eliminates boilerplate across all verticals by providing a single
        implementation of the lazy import + create + cache pattern.

        Args:
            extension_key: Cache key (e.g., "safety_extension", "prompt_contributor")
            import_path: Full Python import path (e.g., "victor.coding.safety")
            attribute_name: Class name to import. If None, auto-generates from vertical name
                          (e.g., "CodingSafetyExtension" for CodingAssistant)

        Returns:
            Cached or newly created extension instance

        Example:
            # Before (13 lines, in external vertical package)
            def get_safety_extension(cls):
                def _create():
                    # This import lives in the external vertical, not core
                    from my_vertical.safety import MySafetyExtension
                    return MySafetyExtension()
                return cls._get_cached_extension("safety_extension", _create)

            # After (3 lines)
            def get_safety_extension(cls):
                return cls._get_extension_factory(
                    "safety_extension",
                    "victor.coding.safety",
                )
        """

        resolver = cls._module_resolver

        def _create():
            # Determine the class name to import
            if attribute_name is None:
                class_name = resolver.auto_generate_class_name(cls.__name__, extension_key)
            else:
                class_name = attribute_name

            # Lazy import + retrieve attribute, then instantiate
            return resolver.load_attribute(import_path, class_name)()

        # Use existing caching infrastructure
        return cls._get_cached_extension(extension_key, _create)

    # =========================================================================
    # Extension Protocol Methods (Optional)
    # =========================================================================
    # These methods enable verticals to provide framework extensions.
    # Override them to integrate with the framework's middleware, safety,
    # prompt, and configuration systems.

    @classmethod
    def get_middleware(cls) -> List[Any]:
        """Get middleware implementations for this vertical."""
        from victor.core.verticals.extension_handlers.middleware import (
            MiddlewareHandler,
        )

        return MiddlewareHandler.load(cls)

    @classmethod
    def get_safety_extension(cls) -> Optional[Any]:
        """Get safety extension for this vertical."""
        from victor.core.verticals.extension_handlers.safety import SafetyHandler

        return SafetyHandler.load(cls)

    @classmethod
    def get_prompt_contributor(cls) -> Optional[Any]:
        """Get prompt contributor for this vertical."""
        from victor.core.verticals.extension_handlers.prompt import PromptHandler

        return PromptHandler.load(cls)

    @classmethod
    def get_mode_config_provider(cls) -> Optional[Any]:
        """Get mode configuration provider for this vertical."""
        from victor.core.verticals.extension_handlers.mode_config import (
            ModeConfigHandler,
        )

        return ModeConfigHandler.load(cls)

    @classmethod
    def get_mode_config(cls) -> Dict[str, Any]:
        """Get default mode configurations (fast, thorough, explore)."""
        from victor.core.verticals.extension_handlers.mode_config import (
            ModeConfigHandler,
        )

        return ModeConfigHandler.load_defaults(cls)

    @classmethod
    def get_task_type_hints(cls) -> Dict[str, Any]:
        """Get task-type-specific prompt hints."""
        from victor.core.verticals.extension_handlers.mode_config import (
            ModeConfigHandler,
        )

        return ModeConfigHandler.load_task_type_hints(cls)

    @classmethod
    def get_tool_dependency_provider(cls) -> Optional[Any]:
        """Get tool dependency provider for this vertical."""
        from victor.core.verticals.extension_handlers.tool_deps import ToolDepsHandler

        return ToolDepsHandler.load(cls)

    @classmethod
    def get_tool_graph(cls) -> Optional[Any]:
        """Get tool execution graph for this vertical."""
        from victor.core.verticals.extension_handlers.tool_deps import ToolDepsHandler

        return ToolDepsHandler.load_tool_graph(cls)

    @classmethod
    def get_tiered_tool_config(cls) -> Optional["TieredToolConfig"]:
        """Get tiered tool configuration for this vertical."""
        from victor.core.verticals.extension_handlers.tool_deps import ToolDepsHandler

        return ToolDepsHandler.load_tiered_tool_config(cls)

    # get_tiered_tools() removed (E5 M3) — use get_tiered_tool_config() instead

    @classmethod
    def get_rl_config_provider(cls) -> Optional[Any]:
        """Get RL configuration provider for this vertical."""
        from victor.core.verticals.extension_handlers.rl import RLConfigHandler

        return RLConfigHandler.load(cls)

    @classmethod
    def get_rl_hooks(cls) -> Optional[Any]:
        """Get RL hooks for outcome recording."""
        from victor.core.verticals.extension_handlers.rl import RLHooksHandler

        return RLHooksHandler.load(cls)

    @classmethod
    def get_team_spec_provider(cls) -> Optional[Any]:
        """Get team specification provider for this vertical."""
        from victor.core.verticals.extension_handlers.team import TeamHandler

        return TeamHandler.load(cls)

    @classmethod
    def get_team_specs(cls) -> Dict[str, Any]:
        """Get team specifications for this vertical."""
        from victor.core.verticals.extension_handlers.team import TeamHandler

        return TeamHandler.load_team_specs(cls)

    @classmethod
    def get_capability_provider(cls) -> Optional[Any]:
        """Get capability provider for this vertical."""
        from victor.core.verticals.extension_handlers.service import CapabilityHandler

        return CapabilityHandler.load(cls)

    @classmethod
    def get_service_provider(cls) -> Optional[Any]:
        """Get service provider for this vertical."""
        from victor.core.verticals.extension_handlers.service import ServiceHandler

        return ServiceHandler.load(cls)

    @classmethod
    def get_composed_chains(cls) -> Dict[str, Any]:
        """Get composed tool chains for this vertical."""
        from victor.core.verticals.extension_handlers.chains import ChainsHandler

        return ChainsHandler.load(cls)

    @classmethod
    def get_personas(cls) -> Dict[str, Any]:
        """Get vertical personas from runtime team modules."""
        from victor.core.verticals.extension_handlers.personas import PersonasHandler

        return PersonasHandler.load(cls)

    @classmethod
    def get_enrichment_strategy(cls) -> Optional[Any]:
        """Get vertical-specific enrichment strategy."""
        from victor.core.verticals.extension_handlers.enrichment import (
            EnrichmentHandler,
        )

        return EnrichmentHandler.load(cls)

    @classmethod
    def get_extensions(
        cls,
        *,
        use_cache: bool = True,
        strict: Optional[bool] = None,
    ) -> "VerticalExtensions":
        """Get all extensions for this vertical with strict error handling.

        Aggregates all extension implementations for framework integration.
        Override for custom extension aggregation.

        LSP Compliance: This method ALWAYS returns a valid VerticalExtensions
        object, never None. Even on exceptions (in non-strict mode), it returns
        a VerticalExtensions with successfully loaded extensions.

        Error Handling Modes:
        - strict=True: Raises ExtensionLoadError on ANY extension failure
        - strict=False: Collects errors, logs warnings, returns partial extensions
        - strict=None: Uses class-level strict_extension_loading setting

        Required Extensions:
        Even when strict=False, extensions listed in required_extensions will
        raise ExtensionLoadError if they fail to load.

        Args:
            use_cache: If True (default), return cached extensions if available.
                       Set to False to force rebuild.
            strict: Override the class-level strict_extension_loading setting.
                    If None (default), uses cls.strict_extension_loading.

        Returns:
            VerticalExtensions containing all vertical extensions (never None)

        Raises:
            ExtensionLoadError: In strict mode or when a required extension fails
        """
        from victor.core.errors import ExtensionLoadError
        from victor.core.verticals.protocols import VerticalExtensions

        cache_key = cls._cache_namespace()

        # Return cached extensions if available and caching enabled
        if use_cache:
            with cls._cache_manager._lock:
                cached = cls._cache_manager._cache.get(cache_key)
            if cached is not None:
                return cached

        # Determine strict mode
        is_strict = strict if strict is not None else cls.strict_extension_loading

        # Collect errors for reporting
        errors: List["ExtensionLoadError"] = []

        def _load_extension(
            extension_type: str,
            loader: callable,
            is_list: bool = False,
        ) -> Any:
            """Load an extension with error handling.

            Args:
                extension_type: Type name for error reporting
                loader: Callable that loads the extension
                is_list: If True, the extension should be a list

            Returns:
                The loaded extension, or default value on error
            """
            try:
                result = loader()
                return result
            except Exception as e:
                is_required = extension_type in cls.required_extensions
                error = ExtensionLoadError(
                    message=f"Failed to load '{extension_type}' extension for vertical '{cls.name}': {e}",
                    extension_type=extension_type,
                    vertical_name=cls.name,
                    original_error=e,
                    is_required=is_required,
                )
                errors.append(error)

                # Log the error with appropriate severity
                if is_strict or is_required:
                    logger.error(
                        f"[{error.correlation_id}] {extension_type} extension failed to load "
                        f"for vertical '{cls.name}': {e}",
                        exc_info=True,
                    )
                else:
                    logger.warning(
                        f"[{error.correlation_id}] {extension_type} extension failed to load "
                        f"for vertical '{cls.name}': {e}"
                    )

                # Return default value
                return [] if is_list else None

        # Build lazy factories — extensions are loaded on first access,
        # not at construction time.  This eliminates the synchronous
        # import storm for extensions that are never used.
        def _make_list_factory(ext_type: str, loader: callable):
            """Create a factory for list-valued extensions."""

            def factory():
                result = _load_extension(ext_type, loader, is_list=True)
                return result if result else []

            return factory

        def _make_single_factory(ext_type: str, loader: callable):
            """Create a factory for optional single-valued extensions."""

            def factory():
                return _load_extension(ext_type, loader)

            return factory

        def _make_wrapped_list_factory(ext_type: str, loader: callable):
            """Create a factory that wraps a single result in a list."""

            def factory():
                result = _load_extension(ext_type, loader)
                return [result] if result else []

            return factory

        extensions = VerticalExtensions(
            middleware=_make_list_factory("middleware", cls.get_middleware),
            safety_extensions=_make_wrapped_list_factory("safety", cls.get_safety_extension),
            prompt_contributors=_make_wrapped_list_factory("prompt", cls.get_prompt_contributor),
            mode_config_provider=_make_single_factory("mode_config", cls.get_mode_config_provider),
            tool_dependency_provider=_make_single_factory(
                "tool_deps", cls.get_tool_dependency_provider
            ),
            workflow_provider=_make_single_factory("workflow", cls.get_workflow_provider),
            service_provider=_make_single_factory("service", cls.get_service_provider),
            rl_config_provider=_make_single_factory("rl_config", cls.get_rl_config_provider),
            team_spec_provider=_make_single_factory("team_spec", cls.get_team_spec_provider),
            enrichment_strategy=_make_single_factory("enrichment", cls.get_enrichment_strategy),
            tiered_tool_config=_make_single_factory("tiered_tools", cls.get_tiered_tool_config),
        )

        # Cache the extensions
        with cls._cache_manager._lock:
            cls._cache_manager._cache[cache_key] = extensions
        return extensions

    @classmethod
    def _get_cached_extensions(cls, use_cache: bool) -> Optional["VerticalExtensions"]:
        """Return cached VerticalExtensions if available, else None."""
        if not use_cache:
            return None
        cache_key = cls._cache_namespace()
        with cls._cache_manager._lock:
            return cls._cache_manager._cache.get(cache_key)

    @classmethod
    def _resolve_strict_mode(cls, strict: Optional[bool]) -> bool:
        """Resolve effective strict mode from explicit arg or class default."""
        return strict if strict is not None else cls.strict_extension_loading

    @classmethod
    async def _submit_extension_tasks(
        cls,
        load_fn: callable,
    ) -> Dict[str, Any]:
        """Submit all extension loads in parallel with bounded concurrency.

        Args:
            load_fn: A callable(extension_type, loader, is_list) that loads
                     a single extension with error handling.

        Returns:
            Dict mapping extension type keys to loaded results.
        """
        loop = asyncio.get_running_loop()
        executor = cls._get_shared_extension_executor()
        semaphore = cls._get_extension_load_semaphore()

        async def _submit_with_limits(
            extension_type: str,
            loader: callable,
            is_list: bool = False,
        ) -> Any:
            cls._increment_loader_metric("submitted")
            cls._increment_loader_metric("queued")
            queued_now = cls._pressure_monitor.get_metric("queued")
            cls._update_loader_peak_metric("max_queued", queued_now)

            if semaphore.locked():
                cls._increment_loader_metric("queue_waits")

            acquired = False
            try:
                await semaphore.acquire()
                acquired = True

                cls._increment_loader_metric("queued", -1)
                cls._increment_loader_metric("in_flight")
                in_flight_now = cls._pressure_monitor.get_metric("in_flight")
                cls._update_loader_peak_metric("max_in_flight", in_flight_now)
                cls._check_pressure(reason="extension_task_started")

                future = loop.run_in_executor(
                    executor,
                    lambda: load_fn(extension_type, loader, is_list),
                )
                timeout = cls._extension_load_timeout
                return await asyncio.wait_for(future, timeout=timeout)
            except asyncio.TimeoutError:
                logger.error(
                    "Extension '%s' timed out after %.0fs for vertical '%s'",
                    extension_type,
                    cls._extension_load_timeout,
                    cls.name,
                )
                cls._increment_loader_metric("failed")
                return [] if is_list else None
            except Exception:
                cls._increment_loader_metric("failed")
                raise
            finally:
                if acquired:
                    cls._increment_loader_metric("in_flight", -1)
                    cls._increment_loader_metric("completed")
                    semaphore.release()
                else:
                    cls._increment_loader_metric("queued", -1)

        futures = {
            "middleware": asyncio.create_task(
                _submit_with_limits("middleware", cls.get_middleware, True)
            ),
            "safety": asyncio.create_task(_submit_with_limits("safety", cls.get_safety_extension)),
            "prompt": asyncio.create_task(
                _submit_with_limits("prompt", cls.get_prompt_contributor)
            ),
            "mode_config": asyncio.create_task(
                _submit_with_limits("mode_config", cls.get_mode_config_provider)
            ),
            "tool_deps": asyncio.create_task(
                _submit_with_limits("tool_deps", cls.get_tool_dependency_provider)
            ),
            "workflow": asyncio.create_task(
                _submit_with_limits("workflow", cls.get_workflow_provider)
            ),
            "service": asyncio.create_task(
                _submit_with_limits("service", cls.get_service_provider)
            ),
            "rl_config": asyncio.create_task(
                _submit_with_limits("rl_config", cls.get_rl_config_provider)
            ),
            "team_spec": asyncio.create_task(
                _submit_with_limits("team_spec", cls.get_team_spec_provider)
            ),
            "enrichment": asyncio.create_task(
                _submit_with_limits("enrichment", cls.get_enrichment_strategy)
            ),
            "tiered_tools": asyncio.create_task(
                _submit_with_limits("tiered_tools", cls.get_tiered_tool_config)
            ),
        }

        results = {}
        for key, task in futures.items():
            results[key] = await task
        return results

    @classmethod
    def _validate_extension_errors(
        cls,
        errors: List[Any],
        is_strict: bool,
    ) -> None:
        """Check for critical extension errors and raise or log as appropriate."""
        critical_errors = [e for e in errors if is_strict or e.is_required]
        if critical_errors:
            raise critical_errors[0]

        if errors:
            logger.warning(
                f"Vertical '{cls.name}' loaded with {len(errors)} extension error(s). "
                f"Affected extensions: {', '.join(e.extension_type for e in errors)}"
            )

    @classmethod
    def _assemble_extensions(cls, results: Dict[str, Any]) -> "VerticalExtensions":
        """Build a VerticalExtensions from a results dict."""
        from victor.core.verticals.protocols import VerticalExtensions

        return VerticalExtensions(
            middleware=results["middleware"] if results["middleware"] else [],
            safety_extensions=[results["safety"]] if results["safety"] else [],
            prompt_contributors=[results["prompt"]] if results["prompt"] else [],
            mode_config_provider=results["mode_config"],
            tool_dependency_provider=results["tool_deps"],
            workflow_provider=results["workflow"],
            service_provider=results["service"],
            rl_config_provider=results["rl_config"],
            team_spec_provider=results["team_spec"],
            enrichment_strategy=results["enrichment"],
            tiered_tool_config=results["tiered_tools"],
        )

    @classmethod
    def _cache_extensions(cls, extensions: "VerticalExtensions") -> None:
        """Store extensions in the shared cache."""
        cache_key = cls._cache_namespace()
        with cls._cache_manager._lock:
            cls._cache_manager._cache[cache_key] = extensions

    @classmethod
    async def get_extensions_async(
        cls,
        *,
        use_cache: bool = True,
        strict: Optional[bool] = None,
    ) -> "VerticalExtensions":
        """Async version of get_extensions that loads extensions in parallel.

        Uses a thread pool executor to load all extensions concurrently,
        providing faster initialization when extensions involve I/O.

        Shares the same cache as the synchronous get_extensions() method.

        Args:
            use_cache: If True (default), return cached extensions if available.
            strict: Override the class-level strict_extension_loading setting.

        Returns:
            VerticalExtensions containing all vertical extensions (never None)

        Raises:
            ExtensionLoadError: In strict mode or when a required extension fails
        """
        from victor.core.errors import ExtensionLoadError

        cached = cls._get_cached_extensions(use_cache)
        if cached is not None:
            return cached

        is_strict = cls._resolve_strict_mode(strict)

        errors: List[ExtensionLoadError] = []
        errors_lock = threading.Lock()

        def _load_extension(
            extension_type: str,
            loader: callable,
            is_list: bool = False,
        ) -> Any:
            """Load an extension with error handling (runs in thread pool)."""
            try:
                return loader()
            except Exception as e:
                cls._increment_loader_metric("failed")
                is_required = extension_type in cls.required_extensions
                error = ExtensionLoadError(
                    message=(
                        f"Failed to load '{extension_type}' extension "
                        f"for vertical '{cls.name}': {e}"
                    ),
                    extension_type=extension_type,
                    vertical_name=cls.name,
                    original_error=e,
                    is_required=is_required,
                )
                with errors_lock:
                    errors.append(error)

                if is_strict or is_required:
                    logger.error(
                        f"[{error.correlation_id}] {extension_type} extension failed "
                        f"for vertical '{cls.name}': {e}",
                        exc_info=True,
                    )
                else:
                    logger.warning(
                        f"[{error.correlation_id}] {extension_type} extension failed "
                        f"for vertical '{cls.name}': {e}"
                    )
                return [] if is_list else None

        results = await cls._submit_extension_tasks(_load_extension)
        cls._validate_extension_errors(errors, is_strict)
        extensions = cls._assemble_extensions(results)
        cls._cache_extensions(extensions)
        return extensions

    @classmethod
    def get_workflow_provider(cls) -> Optional[Any]:
        """Get workflow provider for this vertical."""
        from victor.core.verticals.extension_handlers.workflow import WorkflowHandler

        return WorkflowHandler.load(cls)

    @classmethod
    def clear_extension_cache(cls, *, clear_all: bool = False) -> None:
        """Clear the extension cache for this vertical.

        Args:
            clear_all: If True, clear cache for all verticals.
                       If False (default), clear only for this class.
        """
        if clear_all:
            with cls._cache_manager._lock:
                cls._cache_manager._cache.clear()
        else:
            namespaced_key = cls._cache_namespace()
            namespaced_prefix = f"{namespaced_key}:"
            legacy_key = cls.__name__
            legacy_prefix = f"{legacy_key}:"
            with cls._cache_manager._lock:
                # Clear namespaced cache entries
                cls._cache_manager._cache.pop(namespaced_key, None)
                namespaced_keys = [
                    k for k in cls._cache_manager._cache if k.startswith(namespaced_prefix)
                ]
                for key in namespaced_keys:
                    cls._cache_manager._cache.pop(key, None)

                # Backward compatibility: clear legacy class-name-only keys.
                cls._cache_manager._cache.pop(legacy_key, None)
                legacy_keys = [
                    k
                    for k in cls._cache_manager._cache
                    if k.startswith(legacy_prefix) and k.count(":") == 1
                ]
                for key in legacy_keys:
                    cls._cache_manager._cache.pop(key, None)


class ExtensionLoaderMetricsReporter:
    """Periodic publisher for extension-loader metrics snapshots."""

    def __init__(
        self,
        *,
        interval_seconds: float = 60.0,
        topic: str = _EXTENSION_LOADER_METRICS_TOPIC,
        source: str = "ExtensionLoaderMetricsReporter",
        reset_after_emit: bool = False,
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be > 0")

        self._interval_seconds = interval_seconds
        self._topic = topic
        self._source = source
        self._reset_after_emit = reset_after_emit
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None

    @property
    def is_running(self) -> bool:
        """Return True while periodic reporter thread is active."""
        with self._lock:
            return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        """Start periodic metrics emission if not already running."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._run_loop,
                name="ext-loader-metrics-reporter",
                daemon=True,
            )
            self._thread.start()

    def stop(self, *, timeout: float = 2.0) -> None:
        """Stop periodic metrics emission."""
        with self._lock:
            thread = self._thread
            if thread is None:
                return
            self._stop_event.set()
        thread.join(timeout=timeout)
        with self._lock:
            if self._thread is thread:
                self._thread = None

    def _run_loop(self) -> None:
        """Loop that emits extension-loader metrics at fixed intervals."""
        while not self._stop_event.wait(self._interval_seconds):
            try:
                VerticalExtensionLoader.emit_extension_loader_metrics_event(
                    topic=self._topic,
                    source=self._source,
                    reset_after_emit=self._reset_after_emit,
                )
            except Exception as e:
                logger.debug("ExtensionLoaderMetricsReporter loop error: %s", e)


def start_extension_loader_metrics_reporter(
    *,
    interval_seconds: float = 60.0,
    topic: str = _EXTENSION_LOADER_METRICS_TOPIC,
    source: str = "ExtensionLoaderMetricsReporter",
    reset_after_emit: bool = False,
) -> ExtensionLoaderMetricsReporter:
    """Start singleton periodic reporter for extension-loader metrics."""
    global _METRICS_REPORTER_SINGLETON
    with _METRICS_REPORTER_LOCK:
        if _METRICS_REPORTER_SINGLETON is None:
            _METRICS_REPORTER_SINGLETON = ExtensionLoaderMetricsReporter(
                interval_seconds=interval_seconds,
                topic=topic,
                source=source,
                reset_after_emit=reset_after_emit,
            )
        _METRICS_REPORTER_SINGLETON.start()
        return _METRICS_REPORTER_SINGLETON


def stop_extension_loader_metrics_reporter(*, timeout: float = 2.0) -> None:
    """Stop and clear singleton extension-loader metrics reporter."""
    global _METRICS_REPORTER_SINGLETON
    with _METRICS_REPORTER_LOCK:
        reporter = _METRICS_REPORTER_SINGLETON
        _METRICS_REPORTER_SINGLETON = None
    if reporter is not None:
        reporter.stop(timeout=timeout)
