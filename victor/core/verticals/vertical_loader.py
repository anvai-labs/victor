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

"""Vertical Loader for dynamic vertical activation.

This module provides functionality for loading, activating, and managing
verticals at runtime. It integrates with the DI container to register
vertical-specific services.

Supports plugin discovery via entry points:
- victor.plugins: Canonical entry point group for vertical packages
- victor.tools: Entry point group for tool plugins

Usage:
    from victor.core.verticals.vertical_loader import VerticalLoader

    # Load and activate a vertical
    loader = VerticalLoader()
    vertical = loader.load("coding")

    # Get extensions for framework integration
    extensions = loader.get_extensions()

    # Register services with DI container
    loader.register_services(container, settings)

    # Discover installed plugins
    verticals = loader.discover_verticals()
    tools = loader.discover_tools()
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple, Type

from victor.core.context import bind_active_vertical
from victor.core.events.emit_helper import emit_event_sync
from victor.core.verticals.adapters import ensure_runtime_vertical
from victor.core.verticals.base import VerticalBase, VerticalRegistry
from victor.core.verticals.compatibility_gate import VerticalCompatibilityGate
from victor.core.verticals.framework_version import get_framework_version
from victor.core.verticals.manifest_contract import (
    VerticalRuntimeProvenance,
    get_or_create_vertical_manifest,
    get_vertical_runtime_provenance,
    get_vertical_runtime_metadata,
    load_vertical_package_manifest_for_module,
)
from victor_contracts.discovery import collect_verticals_from_candidate

if TYPE_CHECKING:
    from victor.core.container import ServiceContainer
    from victor.config.settings import Settings
    from victor.core.verticals.dependency_graph import ExtensionDependencyGraph
    from victor.core.verticals.protocols import VerticalExtensions

logger = logging.getLogger(__name__)

# Per-name set so each contrib vertical warns exactly once per process (Pattern C).
_CONTRIB_VERTICAL_WARNED: Set[str] = set()


# --- Lazy victor.framework delegates -----------------------------------------
# victor.core must not import victor.framework at module import time:
# framework/__init__ eagerly pulls framework._api, which imports
# victor.core.verticals back — a cycle that was masked by the eager
# victor/__init__ and unmasked when it went lazy (PEP 562, #390). These thin
# module-level delegates keep the import deferred to call time while
# preserving the `victor.core.verticals.vertical_loader.<name>` seam that
# tests patch.


def get_entry_point_registry() -> Any:
    """Lazy delegate for :func:`victor.framework.entry_point_registry.get_entry_point_registry`."""
    from victor.framework.entry_point_registry import get_entry_point_registry as _impl

    return _impl()


def get_entry_point_values(group_name: str, *, force: bool = False) -> Dict[str, str]:
    """Lazy delegate for :func:`victor.framework.entry_point_registry.get_entry_point_values`."""
    from victor.framework.entry_point_registry import get_entry_point_values as _impl

    return _impl(group_name, force=force)


def get_entry_point_cache() -> Any:
    """Lazy delegate for :func:`victor.framework.module_loader.get_entry_point_cache`."""
    from victor.framework.module_loader import get_entry_point_cache as _impl

    return _impl()


@dataclass(frozen=True)
class VerticalActivationResult:
    """Result of activating a vertical and registering its services."""

    vertical_name: str
    previous_vertical: Optional[str]
    activated: bool
    services_registered: bool


class VerticalLoader:
    """Loader for dynamic vertical activation and management.

    Handles loading verticals by name, activating them, and integrating
    their extensions with the framework. Supports plugin discovery via
    Python entry points for verticals and tools.

    Entry Point Groups:
        - victor.plugins: Vertical plugins (e.g., coding, research)
        - victor.tools: Tool plugins (e.g., code_search, refactor)

    Attributes:
        _active_vertical: Currently active vertical class
        _extensions: Cached extensions from active vertical
        _discovered_verticals: Cache of discovered vertical entry points
        _discovered_tools: Cache of discovered tool entry points
    """

    def __init__(self) -> None:
        """Initialize the vertical loader."""
        self._lock = threading.RLock()
        self._active_vertical: Optional[Type[VerticalBase]] = None
        self._extensions: Optional["VerticalExtensions"] = None
        self._registered_services: bool = False
        self._discovered_verticals: Optional[Dict[str, Type[VerticalBase]]] = None
        self._discovered_vertical_entry_points: Optional[Dict[str, str]] = None
        self._discovered_tools: Optional[Dict[str, Type]] = None
        # Discovery telemetry counters (for diagnostics and observability).
        self._vertical_discovery_calls: int = 0
        self._vertical_discovery_cache_hits: int = 0
        self._vertical_discovery_scans: int = 0
        self._vertical_last_discovery_ms: float = 0.0
        self._tool_discovery_calls: int = 0
        self._tool_discovery_cache_hits: int = 0
        self._tool_discovery_scans: int = 0
        self._tool_last_discovery_ms: float = 0.0
        self._plugin_refresh_count: int = 0
        self._plugin_refresh_last_ms: float = 0.0

        # Dependency graph for resolving load order
        from victor.core.verticals.dependency_graph import ExtensionDependencyGraph

        self._dependency_graph = ExtensionDependencyGraph()

    @property
    def active_vertical(self) -> Optional[Type[VerticalBase]]:
        """Get the currently active vertical."""
        return self._active_vertical

    @property
    def active_vertical_name(self) -> Optional[str]:
        """Get the name of the currently active vertical."""
        return self._active_vertical.name if self._active_vertical else None

    def load(self, name: str) -> Type[VerticalBase]:
        """Load and activate a vertical by name.

        Searches for verticals in this order:
        1. Global VerticalRegistry (includes built-ins registered on import)
        2. Entry point plugins (victor.plugins group)

        Args:
            name: Vertical name (e.g., "coding", "research")

        Returns:
            Loaded vertical class

        Raises:
            ValueError: If vertical not found
        """
        with self._lock:
            vertical = self.resolve(name)

            # Error with available names
            if vertical is None:
                available = self._get_available_names()
                raise ValueError(f"Vertical '{name}' not found. Available: {', '.join(available)}")

            runtime_vertical = vertical
            runtime_metadata = get_vertical_runtime_metadata(runtime_vertical)

            with bind_active_vertical(
                runtime_metadata["vertical_name"],
                manifest_version=runtime_metadata["vertical_manifest_version"],
                namespace=runtime_metadata["vertical_plugin_namespace"],
            ):
                # Capability negotiation: validate manifest before activation
                self._negotiate_manifest(runtime_vertical)

                # Dependency validation: check dependencies before activation
                self._validate_dependencies(runtime_vertical)

                self._activate(runtime_vertical)
            return runtime_vertical

    def resolve(self, name: str) -> Optional[Type[VerticalBase]]:
        """Resolve a vertical class without activating it.

        Prefers a registered class and otherwise imports only the requested
        entry point instead of scanning and importing every external vertical.
        """
        with self._lock:
            vertical = VerticalRegistry.get(name)
            ep_vertical = None
            if self._should_consider_entry_point(vertical):
                ep_vertical = self._import_from_entrypoint(name)

            if vertical is not None and ep_vertical is not None and vertical is not ep_vertical:
                reg_module = getattr(vertical, "__module__", "")
                ep_module = getattr(ep_vertical, "__module__", "")
                reg_is_contrib = (
                    get_vertical_runtime_provenance(vertical) is VerticalRuntimeProvenance.CONTRIB
                )
                ep_is_contrib = (
                    get_vertical_runtime_provenance(ep_vertical)
                    is VerticalRuntimeProvenance.CONTRIB
                )
                is_expected_override = reg_is_contrib != ep_is_contrib
                if not is_expected_override:
                    logger.warning(
                        "Vertical '%s' registered via both VerticalRegistry (%s) and "
                        "entry point (%s). Using registry version. To use the entry point "
                        "version, unregister the built-in first.",
                        name,
                        f"{vertical.__module__}.{vertical.__qualname__}",
                        f"{ep_vertical.__module__}.{ep_vertical.__qualname__}",
                    )
                elif reg_is_contrib and not ep_is_contrib:
                    vertical = ep_vertical
                elif not reg_is_contrib and ep_is_contrib:
                    logger.debug(
                        "Retaining non-contrib vertical '%s' from %s over contrib entry point %s.",
                        name,
                        reg_module,
                        ep_module,
                    )

            if vertical is None:
                vertical = ep_vertical

            return ensure_runtime_vertical(vertical) if vertical is not None else None

    def _should_consider_entry_point(
        self,
        registered_vertical: Optional[Type[VerticalBase]],
    ) -> bool:
        """Return True when entry-point resolution can still affect the outcome."""

        if registered_vertical is None:
            return True
        return (
            get_vertical_runtime_provenance(registered_vertical)
            is VerticalRuntimeProvenance.CONTRIB
        )

    def _import_from_entrypoint(self, name: str) -> Optional[Type[VerticalBase]]:
        """Import a vertical from entry points.

        Args:
            name: Vertical name

        Returns:
            Vertical class or None
        """
        if self._discovered_verticals:
            cached = self._discovered_verticals.get(name)
            if cached is not None:
                return cached

        entry_match = self._get_matching_vertical_entry_point(name)
        if entry_match is None:
            return None

        entry_name, value = entry_match
        self._preflight_entry_point_manifest(entry_name, value)
        candidate = self._load_entry_point(entry_name, value)
        try:
            discovered_verticals = self._collect_validated_verticals(candidate, entry_name)
        except TypeError:
            return None
        if discovered_verticals:
            if self._discovered_verticals is None:
                self._discovered_verticals = {}
            self._discovered_verticals.update(discovered_verticals)
            for vertical_cls in discovered_verticals.values():
                if VerticalRegistry.get(vertical_cls.name) is None:
                    VerticalRegistry.register(vertical_cls)
            requested = discovered_verticals.get(name)
            if requested is None:
                requested = next(
                    (
                        vertical_cls
                        for vertical_name, vertical_cls in discovered_verticals.items()
                        if vertical_name.lower() == name.lower()
                    ),
                    None,
                )
            return requested
        return None

    def _get_matching_vertical_entry_point(self, name: str) -> Optional[Tuple[str, str]]:
        """Return the canonical entry-point name/value pair for *name*."""

        ep_entries = self._get_vertical_entry_points()
        entry_name = name if name in ep_entries else None
        if entry_name is None:
            entry_name = next((key for key in ep_entries if key.lower() == name.lower()), None)
        if entry_name is None:
            return None
        return entry_name, ep_entries[entry_name]

    def _preflight_entry_point_manifest(self, name: str, value: str) -> None:
        """Validate package-level manifest metadata before importing a plugin module."""

        module_name = self._parse_entry_point_module_name(value)
        if module_name is None:
            return

        manifest = load_vertical_package_manifest_for_module(module_name)
        if manifest is None:
            return

        report = VerticalCompatibilityGate().assess_manifest(manifest)
        for warning in report.warnings:
            logger.warning(
                "Package manifest negotiation warning for entry point '%s': %s",
                name,
                warning,
            )
        report.raise_if_incompatible()

    def _parse_entry_point_module_name(self, value: str) -> Optional[str]:
        """Extract the module portion from a setuptools entry-point value."""

        if ":" in value:
            module_name, _attr_name = value.split(":", 1)
            return module_name or None
        if "." in value:
            module_name, _attr_name = value.rsplit(".", 1)
            return module_name or None
        return None

    def _get_vertical_entry_points(self, force_refresh: bool = False) -> Dict[str, str]:
        """Return cached raw vertical entry-point values without importing them."""

        with self._lock:
            if self._discovered_vertical_entry_points is not None and not force_refresh:
                return self._discovered_vertical_entry_points

            registry = get_entry_point_registry()
            if force_refresh:
                registry.invalidate()
            group = registry.get_group("victor.plugins")
            entries = (
                {
                    name: entry_point_tuple[0].value
                    for name, entry_point_tuple in group.entry_points.items()
                }
                if group is not None
                else {}
            )
            self._discovered_vertical_entry_points = entries
            return entries

    def discover_vertical_names(self, force_refresh: bool = False) -> List[str]:
        """Discover installed vertical names without importing their modules."""

        return sorted(self._get_vertical_entry_points(force_refresh=force_refresh).keys())

    def _emit_observability_event(self, topic: str, data: Dict[str, Any]) -> None:
        """Emit loader observability event from sync contexts."""
        try:
            from victor.core.events import get_observability_bus

            bus = get_observability_bus()
            if bus:
                emit_event_sync(
                    bus,
                    topic,
                    data,
                    source="VerticalLoader",
                    use_background_loop=True,
                )
        except Exception as e:
            logger.debug("Failed to emit %s event: %s", topic, e)

    async def _emit_observability_event_async(self, topic: str, data: Dict[str, Any]) -> None:
        """Emit loader observability event from async contexts."""
        try:
            from victor.core.events import get_observability_bus

            bus = get_observability_bus()
            if bus:
                await bus.emit(
                    topic=topic,
                    data=data,
                    source="VerticalLoader",
                )
        except Exception as e:
            logger.debug("Failed to emit %s event (async): %s", topic, e)

    def _build_discovery_event_payload(
        self,
        *,
        kind: str,
        count: int,
        duration_ms: float,
        cache_hit: bool,
        force_refresh: bool,
    ) -> Dict[str, Any]:
        """Build discovery observability payload."""
        return {
            "kind": kind,
            "count": count,
            "duration_ms": duration_ms,
            "cache_hit": cache_hit,
            "force_refresh": force_refresh,
            "stats": self.get_discovery_stats(),
        }

    def _log_discovery_telemetry(
        self,
        *,
        event: str,
        kind: str,
        payload: Dict[str, Any],
    ) -> None:
        """Emit structured discovery logging with cache/discovery context."""
        stats = payload.get("stats", {})
        kind_stats = stats.get("vertical" if kind == "vertical" else "tools", {})
        entry_point_cache = stats.get("entry_point_cache", {})
        entry_point_group = "victor.plugins" if kind == "vertical" else "victor.tools"
        entry_point_group_stats: Dict[str, Any] = {}
        if isinstance(entry_point_cache, dict):
            groups = entry_point_cache.get("groups", {})
            if isinstance(groups, dict):
                raw_group_stats = groups.get(entry_point_group, {})
                if isinstance(raw_group_stats, dict):
                    entry_point_group_stats = raw_group_stats

        level = (
            logging.DEBUG
            if payload.get("cache_hit") and not payload.get("force_refresh")
            else logging.INFO
        )
        logger.log(
            level,
            "%s kind=%s count=%s cache_hit=%s force_refresh=%s duration_ms=%.2f",
            event,
            kind,
            payload.get("count", 0),
            payload.get("cache_hit", False),
            payload.get("force_refresh", False),
            float(payload.get("duration_ms", 0.0) or 0.0),
            extra={
                "event": event,
                "discovery_kind": kind,
                "discovered_count": int(payload.get("count", 0) or 0),
                "cache_hit": bool(payload.get("cache_hit", False)),
                "force_refresh": bool(payload.get("force_refresh", False)),
                "duration_ms": float(payload.get("duration_ms", 0.0) or 0.0),
                "loader_stats": kind_stats,
                "entry_point_cache_group": entry_point_group,
                "entry_point_cache_group_stats": entry_point_group_stats,
                "entry_point_groups_cached": (
                    int(entry_point_cache.get("groups_cached", 0) or 0)
                    if isinstance(entry_point_cache, dict)
                    else 0
                ),
            },
        )

    def _log_refresh_telemetry(self) -> None:
        """Emit structured logging after plugin refresh/invalidation."""
        stats = self.get_discovery_stats()
        refresh_stats = stats.get("refresh", {})
        entry_point_cache = stats.get("entry_point_cache", {})
        logger.info(
            "VERTICAL_PLUGIN_REFRESH count=%s duration_ms=%.2f",
            int(refresh_stats.get("count", 0) or 0),
            float(refresh_stats.get("last_refresh_ms", 0.0) or 0.0),
            extra={
                "event": "VERTICAL_PLUGIN_REFRESH",
                "refresh_count": int(refresh_stats.get("count", 0) or 0),
                "duration_ms": float(refresh_stats.get("last_refresh_ms", 0.0) or 0.0),
                "refresh_stats": refresh_stats,
                "entry_point_groups_cached": (
                    int(entry_point_cache.get("groups_cached", 0) or 0)
                    if isinstance(entry_point_cache, dict)
                    else 0
                ),
                "entry_point_cache": entry_point_cache,
            },
        )

    def discover_verticals(
        self,
        force_refresh: bool = False,
        emit_event: bool = True,
    ) -> Dict[str, Type[VerticalBase]]:
        """Discover verticals from installed packages via entry points.

        Scans the canonical 'victor.plugins' entry point group for installed
        vertical packages. Results are cached for performance using
        EntryPointCache for fast startup.

        Args:
            force_refresh: Force re-scan of entry points (bypass cache)
            emit_event: Internal flag to suppress sync event emission when
                async callers emit from the event-loop context.

        Returns:
            Dictionary mapping vertical names to their classes

        Example:
            # In victor-coding's pyproject.toml:
            # [project.entry-points."victor.plugins"]
            # coding = "victor_coding.plugin:plugin"

            loader = VerticalLoader()
            verticals = loader.discover_verticals()
            # {'coding': <class 'victor_coding.CodingVertical'>}
        """
        discovered, cache_hit, duration_ms = self._discover_verticals_internal(force_refresh)
        payload = self._build_discovery_event_payload(
            kind="vertical",
            count=len(discovered),
            duration_ms=duration_ms,
            cache_hit=cache_hit,
            force_refresh=force_refresh,
        )
        if emit_event:
            self._emit_observability_event(
                topic="vertical.plugins.discovered",
                data=payload,
            )
        self._log_discovery_telemetry(event="VERTICAL_DISCOVERY", kind="vertical", payload=payload)
        return discovered

    # CONSOLIDATION: plugin-vertical unification — see memory plugin_vertical_consolidation.md
    def _discover_verticals_internal(
        self,
        force_refresh: bool = False,
    ) -> Tuple[Dict[str, Type[VerticalBase]], bool, float]:
        """Discover verticals and return result with cache/duration metadata.

        Prefers PluginRegistry.get_vertical_classes() as the single source of
        truth — one capture per process shared with bootstrap capability
        discovery and registry_manager. Falls back to a direct entry-point
        scan if PluginRegistry has not initialized yet (e.g., in isolated
        tests).
        """
        with self._lock:
            self._vertical_discovery_calls += 1
            if self._discovered_verticals is not None and not force_refresh:
                self._vertical_discovery_cache_hits += 1
                return self._discovered_verticals, True, 0.0

            start = time.perf_counter()
            self._vertical_discovery_scans += 1
            self._discovered_verticals = {}

            if self._try_populate_from_plugin_registry(force_refresh=force_refresh):
                self._vertical_last_discovery_ms = max(
                    0.0,
                    (time.perf_counter() - start) * 1000.0,
                )
                return (
                    self._discovered_verticals,
                    False,
                    self._vertical_last_discovery_ms,
                )

            try:
                # Fallback: direct entry-point scan.
                ep_entries = self._get_vertical_entry_points(force_refresh=force_refresh)
                self._load_vertical_entries(ep_entries)
            except Exception as e:
                logger.warning("Failed to discover vertical entry points: %s", e)

            self._vertical_last_discovery_ms = max(
                0.0,
                (time.perf_counter() - start) * 1000.0,
            )

            return self._discovered_verticals, False, self._vertical_last_discovery_ms

    def _try_populate_from_plugin_registry(self, *, force_refresh: bool) -> bool:
        """Populate ``self._discovered_verticals`` from PluginRegistry if possible.

        Returns True when the population succeeded (caller should not fall
        back to an independent entry-point scan). Returns False when the
        PluginRegistry has not discovered yet, when it is unavailable, or
        when a forced refresh is requested — in which case callers perform
        their own scan.
        """
        if force_refresh:
            return False

        try:
            from victor.core.plugins.registry import PluginRegistry
        except ImportError:
            return False

        try:
            plugin_registry = PluginRegistry.get_instance()
        except Exception:
            return False

        if not plugin_registry.is_discovered:
            return False

        try:
            vertical_classes = plugin_registry.get_vertical_classes()
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("PluginRegistry.get_vertical_classes failed: %s", exc)
            return False

        if not vertical_classes:
            # Plugin registry ran but captured no verticals; signal fallback.
            return False

        for vertical_name, vertical_cls in vertical_classes.items():
            if not VerticalRegistry._validate_external_vertical(vertical_cls, vertical_name):
                continue
            existing = VerticalRegistry.get(vertical_cls.name)
            if existing is not None and existing is not vertical_cls:
                existing_module = getattr(existing, "__module__", "")
                existing_is_contrib = (
                    get_vertical_runtime_provenance(existing) is VerticalRuntimeProvenance.CONTRIB
                )
                new_is_contrib = (
                    get_vertical_runtime_provenance(vertical_cls)
                    is VerticalRuntimeProvenance.CONTRIB
                )
                if not existing_is_contrib and new_is_contrib:
                    continue
                if existing_is_contrib == new_is_contrib:
                    logger.warning(
                        "Vertical '%s' has name '%s' which conflicts with "
                        "registered vertical %s (from %s). Skipping.",
                        vertical_name,
                        vertical_cls.name,
                        existing.__name__,
                        existing_module,
                    )
                    continue
            self._discovered_verticals[vertical_name] = vertical_cls
            VerticalRegistry.register(vertical_cls)

        return True

    async def discover_verticals_async(
        self,
        force_refresh: bool = False,
    ) -> Dict[str, Type[VerticalBase]]:
        """Discover verticals asynchronously (non-blocking).

        Async version of discover_verticals() that offloads entry point
        scanning to a thread pool to avoid blocking the event loop.

        Args:
            force_refresh: Force re-scan of entry points (bypass cache)
            emit_event: Internal flag to suppress sync event emission when
                async callers emit from the event-loop context.

        Returns:
            Dictionary mapping vertical names to their classes
        """
        discovered, cache_hit, duration_ms = await asyncio.to_thread(
            self._discover_verticals_internal,
            force_refresh,
        )

        payload = self._build_discovery_event_payload(
            kind="vertical",
            count=len(discovered),
            duration_ms=duration_ms,
            cache_hit=cache_hit,
            force_refresh=force_refresh,
        )
        await self._emit_observability_event_async(
            topic="vertical.plugins.discovered",
            data=payload,
        )
        self._log_discovery_telemetry(event="VERTICAL_DISCOVERY", kind="vertical", payload=payload)
        return discovered

    def _load_vertical_entries(self, ep_entries: Dict[str, str]) -> None:
        """Load vertical classes from entry point entries.

        Args:
            ep_entries: Dictionary of name -> module:attr strings
        """
        for name, value in ep_entries.items():
            try:
                self._preflight_entry_point_manifest(name, value)
                candidate = self._load_entry_point(name, value)
                discovered_verticals = self._collect_validated_verticals(candidate, name)
                if not discovered_verticals:
                    logger.warning("Entry point '%s' did not register any valid verticals", name)
                    continue

                for vertical_name, vertical_cls in discovered_verticals.items():
                    existing = VerticalRegistry.get(vertical_cls.name)
                    if existing is not None and existing is not vertical_cls:
                        existing_module = getattr(existing, "__module__", "")
                        existing_is_contrib = (
                            get_vertical_runtime_provenance(existing)
                            is VerticalRuntimeProvenance.CONTRIB
                        )
                        new_is_contrib = (
                            get_vertical_runtime_provenance(vertical_cls)
                            is VerticalRuntimeProvenance.CONTRIB
                        )
                        if not existing_is_contrib and new_is_contrib:
                            # External already registered; skip contrib
                            continue
                        if existing_is_contrib == new_is_contrib:
                            # Genuine collision (both external or both contrib)
                            logger.warning(
                                "Vertical '%s' has name '%s' which conflicts with "
                                "registered vertical %s (from %s). Skipping.",
                                name,
                                vertical_cls.name,
                                existing.__name__,
                                existing_module,
                            )
                            continue
                        # External overriding contrib — let register() handle it
                    self._discovered_verticals[vertical_name] = vertical_cls
                    VerticalRegistry.register(vertical_cls)
                    logger.debug("Discovered vertical plugin: %s -> %s", name, vertical_name)
            except Exception as e:
                logger.warning("Failed to load vertical entry point '%s': %s", name, e)

    def _collect_validated_verticals(
        self,
        candidate: Any,
        entry_point_name: str,
    ) -> Dict[str, Type[VerticalBase]]:
        """Collect and validate vertical classes using the shared SDK helper."""

        discovered: Dict[str, Type[VerticalBase]] = {}
        for vertical_cls in collect_verticals_from_candidate(candidate).values():
            if VerticalRegistry._validate_external_vertical(vertical_cls, entry_point_name):
                discovered[vertical_cls.name] = vertical_cls
        return discovered

    def _load_entry_point(self, name: str, value: str) -> Type:
        """Load an entry point by its value string.

        Args:
            name: Entry point name
            value: Entry point value (module:attr format)

        Returns:
            Loaded class/object
        """
        import importlib

        if ":" in value:
            module_name, attr_name = value.split(":", 1)
        else:
            # Handle "module.Class" format
            module_name, attr_name = value.rsplit(".", 1)

        module = importlib.import_module(module_name)
        return getattr(module, attr_name)

    def discover_tools(
        self,
        force_refresh: bool = False,
        emit_event: bool = True,
        cancel_event: Optional[threading.Event] = None,
    ) -> Dict[str, Type]:
        """Discover tools from installed packages via entry points.

        Scans the 'victor.tools' entry point group for installed
        tool plugins. Results are cached for performance using
        EntryPointCache for fast startup.

        Args:
            force_refresh: Force re-scan of entry points (bypass cache)
            emit_event: Whether to emit an observability discovery event
            cancel_event: Optional ``threading.Event`` used to cooperatively
                cancel discovery. When set before/during discovery, the
                internal scan is aborted early and a cached or empty result
                may be returned. Propagated down to ``_discover_tools_internal``.

        Returns:
            Dictionary mapping tool names to their classes

        Example:
            # In victor-coding's pyproject.toml:
            # [project.entry-points."victor.tools"]
            # code_search = "victor_coding.tools:CodeSearchTool"

            loader = VerticalLoader()
            tools = loader.discover_tools()
            # {'code_search': <class 'victor_coding.tools.CodeSearchTool'>}
        """
        discovered, cache_hit, duration_ms = self._discover_tools_internal(
            force_refresh,
            cancel_event=cancel_event,
        )
        payload = self._build_discovery_event_payload(
            kind="tools",
            count=len(discovered),
            duration_ms=duration_ms,
            cache_hit=cache_hit,
            force_refresh=force_refresh,
        )
        if emit_event:
            self._emit_observability_event(
                topic="vertical.plugins.discovered",
                data=payload,
            )
        self._log_discovery_telemetry(event="TOOL_DISCOVERY", kind="tools", payload=payload)
        return discovered

    def _discover_tools_internal(
        self,
        force_refresh: bool = False,
        *,
        cancel_event: Optional[threading.Event] = None,
    ) -> Tuple[Dict[str, Type], bool, float]:
        """Discover tools and return result with cache/duration metadata.

        Args:
            force_refresh: Force re-scan of entry points (bypass cache)
            cancel_event: Optional ``threading.Event`` for cooperative
                cancellation. When set, discovery aborts early: if a cached
                result exists it is returned (cache hit), otherwise the
                current (possibly empty) discovery result is returned.
        """
        with self._lock:
            self._tool_discovery_calls += 1
            if self._discovered_tools is not None and not force_refresh:
                self._tool_discovery_cache_hits += 1
                return self._discovered_tools, True, 0.0

            start = time.perf_counter()
            self._tool_discovery_scans += 1
            self._discovered_tools = {}

            # Bail out early if cancellation was requested before the scan.
            if cancel_event is not None and cancel_event.is_set():
                logger.debug("Tool discovery cancelled before scan")
                self._tool_last_discovery_ms = max(
                    0.0,
                    (time.perf_counter() - start) * 1000.0,
                )
                return self._discovered_tools, False, self._tool_last_discovery_ms

            try:
                ep_entries = get_entry_point_values("victor.tools", force=force_refresh)
                # Abort after the (potentially slow) entry point scan if the
                # caller requested cancellation. Return whatever we have so far.
                if cancel_event is not None and cancel_event.is_set():
                    logger.debug("Tool discovery cancelled after entry point scan")
                    return self._discovered_tools, False, self._tool_last_discovery_ms
                self._load_tool_entries(ep_entries)
            except Exception as e:
                logger.warning("Failed to discover tool entry points: %s", e)
            finally:
                self._tool_last_discovery_ms = max(
                    0.0,
                    (time.perf_counter() - start) * 1000.0,
                )

            return self._discovered_tools, False, self._tool_last_discovery_ms

    async def discover_tools_async(
        self,
        force_refresh: bool = False,
        cancel_event: Optional[threading.Event] = None,
    ) -> Dict[str, Type]:
        """Discover tools asynchronously (non-blocking).

        Async version of discover_tools() that offloads entry point
        scanning to a thread pool to avoid blocking the event loop.

        Args:
            force_refresh: Force re-scan of entry points (bypass cache)
            cancel_event: Optional ``threading.Event`` for cooperative
                cancellation, propagated down to ``_discover_tools_internal``.

        Returns:
            Dictionary mapping tool names to their classes
        """
        discovered, cache_hit, duration_ms = await asyncio.to_thread(
            self._discover_tools_internal,
            force_refresh,
            cancel_event=cancel_event,
        )

        payload = self._build_discovery_event_payload(
            kind="tools",
            count=len(discovered),
            duration_ms=duration_ms,
            cache_hit=cache_hit,
            force_refresh=force_refresh,
        )
        await self._emit_observability_event_async(
            topic="vertical.plugins.discovered",
            data=payload,
        )
        self._log_discovery_telemetry(event="TOOL_DISCOVERY", kind="tools", payload=payload)
        return discovered

    def _load_tool_entries(self, ep_entries: Dict[str, str]) -> None:
        """Load tool classes from entry point entries.

        Args:
            ep_entries: Dictionary of name -> module:attr strings
        """
        for name, value in ep_entries.items():
            try:
                tool_cls = self._load_entry_point(name, value)
                self._discovered_tools[name] = tool_cls
                logger.debug("Discovered tool plugin: %s", name)
            except Exception as e:
                logger.warning("Failed to load tool entry point '%s': %s", name, e)

    def refresh_plugins(self) -> None:
        """Refresh the cached plugin discovery.

        Call this after installing new packages to re-scan entry points.
        Invalidates both the local cache and the global EntryPointCache.
        Also clears the extension cache for consistency.
        """
        with self._lock:
            refresh_start = time.perf_counter()
            self._plugin_refresh_count += 1
            self._discovered_verticals = None
            self._discovered_vertical_entry_points = None
            self._discovered_tools = None
            # Reset loader-level extension/service state to avoid stale plugin config
            self._extensions = None
            self._registered_services = False

            # Also invalidate the entry point cache
            cache = get_entry_point_cache()
            cache.invalidate("victor.plugins")
            cache.invalidate("victor.tools")
            try:
                get_entry_point_registry().invalidate()
            except Exception as e:
                logger.debug("Failed invalidating unified entry-point registry: %s", e)

            # Clear extension cache for consistency (Phase 3.3 fix)
            from victor.core.verticals.extension_loader import VerticalExtensionLoader

            VerticalExtensionLoader.clear_extension_cache(clear_all=True)

            # Clear framework integration cache to avoid stale vertical metadata.
            try:
                from victor.framework.vertical_service import (
                    clear_vertical_integration_pipeline_cache,
                )

                clear_vertical_integration_pipeline_cache()
            except Exception as e:
                logger.debug("Failed clearing framework vertical integration cache: %s", e)

            try:
                from victor.framework.entry_point_loader import (
                    clear_entry_point_loader_cache,
                )

                clear_entry_point_loader_cache()
            except Exception as e:
                logger.debug("Failed clearing framework entry-point loader cache: %s", e)

            try:
                from victor.core.tool_dependency_loader import (
                    clear_tool_dependency_entry_point_cache,
                    clear_vertical_tool_dependency_provider_cache,
                )

                clear_tool_dependency_entry_point_cache()
                clear_vertical_tool_dependency_provider_cache()
            except Exception as e:
                logger.debug("Failed clearing tool dependency entry-point cache: %s", e)

            self._plugin_refresh_last_ms = max(
                0.0,
                (time.perf_counter() - refresh_start) * 1000.0,
            )
            self._emit_observability_event(
                topic="vertical.plugins.refreshed",
                data={
                    "refresh_count": self._plugin_refresh_count,
                    "duration_ms": self._plugin_refresh_last_ms,
                    "stats": self.get_discovery_stats(),
                },
            )
            self._log_refresh_telemetry()

    def reset_discovery_state(self) -> None:
        """Reset local discovery state without global cache side effects.

        This is intended for tests or explicit registry resets where we need
        the loader to re-run discovery logic without emitting refresh events
        or touching unrelated extension/service state.
        """
        with self._lock:
            self._discovered_verticals = None
            self._discovered_vertical_entry_points = None
            self._discovered_tools = None
            self._vertical_last_discovery_ms = 0.0
            self._tool_last_discovery_ms = 0.0

    def _build_dependency_graph(self) -> None:
        """Build dependency graph from discovered verticals.

        Constructs the dependency graph by collecting manifests from all
        discovered verticals and adding their dependencies to the graph.
        """
        from victor.core.verticals.dependency_graph import ExtensionDependencyGraph

        # Create fresh graph
        self._dependency_graph = ExtensionDependencyGraph()

        # Get all discovered verticals
        discovered = self.discover_verticals()

        # Add each vertical to the graph
        for vertical_name, vertical_class in discovered.items():
            try:
                manifest = get_or_create_vertical_manifest(vertical_class)
                if manifest is None:
                    raise AttributeError("manifest unavailable")
                self._dependency_graph.add_vertical(
                    vertical_name,
                    manifest.version,
                    manifest,
                    manifest.load_priority,
                )
            except (AttributeError, NotImplementedError):
                # Vertical doesn't have manifest support
                self._dependency_graph.add_vertical(
                    vertical_name,
                    getattr(vertical_class, "version", "1.0.0"),
                    None,
                    0,
                )

        # Add dependency relationships
        for vertical_name, vertical_class in discovered.items():
            try:
                manifest = get_or_create_vertical_manifest(vertical_class)
                if manifest is None:
                    raise AttributeError("manifest unavailable")
                for dep in manifest.extension_dependencies:
                    try:
                        self._dependency_graph.add_dependency(
                            vertical_name,
                            dep.extension_name,
                            required=not dep.optional,
                        )
                    except ValueError as e:
                        # Dependency not in graph
                        if not dep.optional:
                            logger.warning(
                                f"Required dependency '{dep.extension_name}' "
                                f"not found for vertical '{vertical_name}': {e}"
                            )
            except (AttributeError, NotImplementedError):
                pass

        logger.debug(f"Built dependency graph with {len(discovered)} verticals")

    def get_dependency_graph(self) -> "ExtensionDependencyGraph":
        """Get the dependency graph.

        Returns:
            ExtensionDependencyGraph instance
        """
        return self._dependency_graph

    def get_dependency_graph_depth(self) -> int:
        """Get the depth of the dependency graph.

        Returns:
            Maximum depth of dependency chains
        """
        return self._dependency_graph.get_graph_depth()

    def _negotiate_manifest(self, vertical: Type[VerticalBase]) -> None:
        """Run capability negotiation on the vertical's manifest.

        Logs warnings for degraded features and raises ValueError on
        incompatible manifests.
        """
        try:
            manifest = get_or_create_vertical_manifest(vertical)
            if manifest is None:
                return
        except (ImportError, AttributeError, NotImplementedError) as exc:
            logger.debug("Vertical manifest not available: %s", exc)
            return

        report = VerticalCompatibilityGate().assess_manifest(manifest)
        for warning in report.warnings:
            logger.warning("Manifest negotiation warning for '%s': %s", manifest.name, warning)
        report.raise_if_incompatible()

    def _validate_dependencies(self, vertical: Type[VerticalBase]) -> None:
        """Validate vertical dependencies before activation.

        Checks that:
        1. All required dependencies are available
        2. No circular dependencies exist
        3. Load order can be resolved

        Args:
            vertical: Vertical class to validate

        Raises:
            ValueError: If dependencies cannot be satisfied
        """
        try:
            manifest = get_or_create_vertical_manifest(vertical)
            if manifest is None:
                return
        except (ImportError, AttributeError, NotImplementedError):
            # No manifest - skip dependency validation
            return

        if not manifest.extension_dependencies:
            # No dependencies - nothing to validate
            return

        # Rebuild dependency graph to ensure it's up-to-date
        self._build_dependency_graph()

        # Check for circular dependencies
        try:
            load_sequence = self._dependency_graph.get_load_sequence(manifest.name)

            # Check if all dependencies in sequence are available
            discovered = self.discover_verticals()
            missing = set(load_sequence) - set(discovered.keys())

            if missing:
                # Filter out optional dependencies
                required_missing = set()
                for dep_name in missing:
                    for dep in manifest.extension_dependencies:
                        if dep.extension_name == dep_name and not dep.optional:
                            required_missing.add(dep_name)

                if required_missing:
                    raise ValueError(
                        f"Vertical '{manifest.name}' requires dependencies "
                        f"that are not available: {', '.join(sorted(required_missing))}"
                    )

                # Warn about optional missing dependencies
                optional_missing = missing - required_missing
                if optional_missing:
                    logger.warning(
                        "Vertical '%s' has optional dependencies that are not available: %s",
                        manifest.name,
                        ", ".join(sorted(optional_missing)),
                    )

        except Exception as exc:
            from victor.core.verticals.dependency_graph import DependencyCycleError

            if isinstance(exc, DependencyCycleError):
                raise ValueError(f"Vertical '{manifest.name}' has circular dependencies: {exc}")
            else:
                raise

    def _fire_plugin_lifecycle(self, hook: str, vertical_name: str) -> None:
        """Fire a lifecycle hook on the plugin associated with a vertical name."""
        try:
            from victor.core.plugins.registry import PluginRegistry, call_lifecycle_hook

            registry = PluginRegistry.get_instance()
            plugin = registry.get_plugin(vertical_name)
            if plugin is not None:
                vertical_cls = VerticalRegistry.get(vertical_name)
                runtime_metadata = (
                    get_vertical_runtime_metadata(vertical_cls)
                    if vertical_cls is not None
                    else {
                        "vertical_name": vertical_name,
                        "vertical_manifest_version": "",
                        "vertical_plugin_namespace": "",
                    }
                )
                with bind_active_vertical(
                    runtime_metadata["vertical_name"],
                    manifest_version=runtime_metadata["vertical_manifest_version"],
                    namespace=runtime_metadata["vertical_plugin_namespace"],
                ):
                    call_lifecycle_hook(plugin, hook)
        except Exception as e:
            logger.debug(
                "Failed to fire lifecycle hook '%s' for vertical '%s': %s",
                hook,
                vertical_name,
                e,
            )

    def _activate(self, vertical: Type[VerticalBase]) -> None:
        """Activate a vertical.

        Args:
            vertical: Vertical class to activate
        """
        with self._lock:
            # Warn once per process when falling back to a built-in contrib vertical.
            provenance = get_vertical_runtime_provenance(vertical)
            if (
                provenance is VerticalRuntimeProvenance.CONTRIB
                and vertical.name not in _CONTRIB_VERTICAL_WARNED
            ):
                _CONTRIB_VERTICAL_WARNED.add(vertical.name)
                warnings.warn(
                    f"Vertical '{vertical.name}' was loaded from the built-in contrib path. "
                    f"Install the external package (e.g. 'pip install victor-{vertical.name}') "
                    f"and declare it under 'victor.plugins' for the production-supported path. "
                    f"Contrib verticals are deprecated since v0.7.0 and will be removed in "
                    f"v1.0.0 (2027-06-30). "
                    f"See victor-contracts/MIGRATION_GUIDE.md for migration guidance.",
                    DeprecationWarning,
                    stacklevel=3,
                )

            previous_vertical = self._active_vertical

            # Fire on_deactivate for outgoing vertical's plugin
            if previous_vertical is not None:
                self._fire_plugin_lifecycle("on_deactivate", previous_vertical.name)

            self._active_vertical = vertical
            self._extensions = None  # Clear cached extensions
            self._registered_services = False

            # Fire on_activate for incoming vertical's plugin
            self._fire_plugin_lifecycle("on_activate", vertical.name)

            logger.info("Activated vertical: %s", vertical.name)

    def _get_available_names(self) -> List[str]:
        """Get list of available vertical names.

        Includes:
        - Registered verticals (includes built-ins registered on import)
        - Entry point plugin verticals

        Returns:
            List of vertical names
        """
        names = set(VerticalRegistry.list_names())
        names.update(self.discover_vertical_names())
        return sorted(names)

    def get_extensions(self) -> Optional["VerticalExtensions"]:
        """Get extensions from the active vertical.

        Returns:
            VerticalExtensions or None if no vertical active
        """
        with self._lock:
            if self._active_vertical is None:
                return None

            if self._extensions is None:
                self._extensions = self._active_vertical.get_extensions()

            return self._extensions

    def register_services(
        self,
        container: "ServiceContainer",
        settings: "Settings",
    ) -> bool:
        """Register vertical-specific services with DI container.

        Args:
            container: DI container
            settings: Application settings

        Returns:
            True if services were newly registered, False if already registered
            or unavailable for this vertical.
        """
        with self._lock:
            if self._registered_services:
                logger.debug("Vertical services already registered")
                return False

            extensions = self.get_extensions()
            if extensions is None or extensions.service_provider is None:
                logger.debug("No service provider for active vertical")
                return False

            try:
                extensions.service_provider.register_services(container, settings)
                self._registered_services = True
                logger.info(
                    "Registered services for vertical: %s",
                    self.active_vertical_name,
                )
                return True
            except Exception as e:
                logger.error("Failed to register vertical services: %s", e)
                return False

    def get_config(self):
        """Get configuration from active vertical.

        Returns:
            VerticalConfig or None
        """
        with self._lock:
            if self._active_vertical is None:
                return None
            return self._active_vertical.get_config()

    def get_tools(self) -> List[str]:
        """Get tools from active vertical.

        Returns:
            List of tool names
        """
        with self._lock:
            if self._active_vertical is None:
                return []
            return self._active_vertical.get_tools()

    def get_system_prompt(self) -> str:
        """Get system prompt from active vertical.

        Returns:
            System prompt string
        """
        with self._lock:
            if self._active_vertical is None:
                return ""
            return self._active_vertical.get_system_prompt()

    def reset(self) -> None:
        """Reset the loader, deactivating current vertical."""
        with self._lock:
            self._active_vertical = None
            self._extensions = None
            self._registered_services = False

    def get_discovery_stats(self) -> Dict[str, Any]:
        """Get vertical/tool discovery telemetry snapshot."""
        tool_dependency_stats: Dict[str, Any] = {}
        framework_entry_point_stats: Dict[str, Any] = {}
        entry_point_cache_stats: Dict[str, Any] = {}
        try:
            from victor.core.tool_dependency_loader import (
                get_tool_dependency_resolution_stats,
            )

            tool_dependency_stats = get_tool_dependency_resolution_stats()
        except Exception as e:
            tool_dependency_stats = {"error": str(e)}

        try:
            from victor.framework.entry_point_loader import get_entry_point_loader_stats

            framework_entry_point_stats = get_entry_point_loader_stats()
        except Exception as e:
            framework_entry_point_stats = {"error": str(e)}

        try:
            entry_point_cache_stats = get_entry_point_cache().get_cache_stats()
        except Exception as e:
            entry_point_cache_stats = {"error": str(e)}

        with self._lock:
            return {
                "vertical": {
                    "calls": self._vertical_discovery_calls,
                    "cache_hits": self._vertical_discovery_cache_hits,
                    "scans": self._vertical_discovery_scans,
                    "last_discovery_ms": self._vertical_last_discovery_ms,
                },
                "tools": {
                    "calls": self._tool_discovery_calls,
                    "cache_hits": self._tool_discovery_cache_hits,
                    "scans": self._tool_discovery_scans,
                    "last_discovery_ms": self._tool_last_discovery_ms,
                },
                "refresh": {
                    "count": self._plugin_refresh_count,
                    "last_refresh_ms": self._plugin_refresh_last_ms,
                },
                "tool_dependency_resolution": tool_dependency_stats,
                "framework_entry_point_loader": framework_entry_point_stats,
                "entry_point_cache": entry_point_cache_stats,
            }


# Global loader instance
_loader: Optional[VerticalLoader] = None
_loader_lock = threading.Lock()


def get_vertical_loader() -> VerticalLoader:
    """Get the global vertical loader instance.

    Returns:
        Global VerticalLoader instance
    """
    global _loader
    if _loader is None:
        with _loader_lock:
            if _loader is None:
                _loader = VerticalLoader()
    return _loader


def load_vertical(name: str) -> Type[VerticalBase]:
    """Load a vertical by name (convenience function).

    Args:
        name: Vertical name

    Returns:
        Loaded vertical class
    """
    return get_vertical_loader().load(name)


def get_active_vertical() -> Optional[Type[VerticalBase]]:
    """Get the currently active vertical (convenience function).

    Returns:
        Active vertical class or None
    """
    return get_vertical_loader().active_vertical


def get_vertical_extensions() -> Optional["VerticalExtensions"]:
    """Get extensions from active vertical (convenience function).

    Returns:
        VerticalExtensions or None
    """
    return get_vertical_loader().get_extensions()


def activate_vertical_services(
    container: "ServiceContainer",
    settings: "Settings",
    vertical_name: str,
) -> VerticalActivationResult:
    """Activate a vertical and ensure its services are registered.

    This is the canonical activation path used by bootstrap and framework
    step handlers to avoid diverging registration behavior.

    Args:
        container: DI container.
        settings: Application settings.
        vertical_name: Vertical name to activate.

    Returns:
        Activation result with activation and service registration details.
    """
    loader = get_vertical_loader()
    previous_vertical = loader.active_vertical_name
    activated = previous_vertical != vertical_name or loader.active_vertical is None

    if activated:
        loader.load(vertical_name)

    services_registered = loader.register_services(container, settings)
    return VerticalActivationResult(
        vertical_name=vertical_name,
        previous_vertical=previous_vertical,
        activated=activated,
        services_registered=services_registered,
    )


def discover_vertical_plugins() -> Dict[str, Type[VerticalBase]]:
    """Discover vertical plugins from entry points (convenience function).

    Returns:
        Dictionary mapping vertical names to their classes
    """
    return get_vertical_loader().discover_verticals()


def discover_tool_plugins(
    cancel_event: Optional[threading.Event] = None,
) -> Dict[str, Type]:
    """Discover tool plugins from entry points (convenience function).

    Args:
        cancel_event: Optional ``threading.Event`` for cooperative
            cancellation, propagated down to ``VerticalLoader.discover_tools``.

    Returns:
        Dictionary mapping tool names to their classes
    """
    return get_vertical_loader().discover_tools(cancel_event=cancel_event)


async def discover_vertical_plugins_async() -> Dict[str, Type[VerticalBase]]:
    """Discover vertical plugins asynchronously (convenience function).

    Non-blocking version that offloads entry point scanning to thread pool.

    Returns:
        Dictionary mapping vertical names to their classes
    """
    return await get_vertical_loader().discover_verticals_async()


async def discover_tool_plugins_async(
    cancel_event: Optional[threading.Event] = None,
) -> Dict[str, Type]:
    """Discover tool plugins asynchronously (convenience function).

    Non-blocking version that offloads entry point scanning to thread pool.

    Args:
        cancel_event: Optional ``threading.Event`` for cooperative
            cancellation, propagated down to ``VerticalLoader.discover_tools_async``.

    Returns:
        Dictionary mapping tool names to their classes
    """
    return await get_vertical_loader().discover_tools_async(cancel_event=cancel_event)


__all__ = [
    "VerticalActivationResult",
    "VerticalLoader",
    "activate_vertical_services",
    "get_vertical_loader",
    "load_vertical",
    "get_active_vertical",
    "get_vertical_extensions",
    "discover_vertical_plugins",
    "discover_tool_plugins",
    "discover_vertical_plugins_async",
    "discover_tool_plugins_async",
]
