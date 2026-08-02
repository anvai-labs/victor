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

"""Entry point loader for framework capabilities.

This module provides utilities for loading vertical-specific capabilities
via entry points, enabling the framework to discover and use features
from external vertical packages without hardcoded dependencies.

Design Principles:
    - Framework does not depend on specific verticals
    - Verticals register capabilities via entry points
    - Graceful fallback when verticals are not installed
    - Clear separation between framework and vertical code
    - Single-pass scanning for performance (via UnifiedEntryPointRegistry)

Entry Point Groups:
    victor.tool_dependencies  - Tool dependency provider factories
    victor.safety_rules        - Safety rule registration functions
    victor.rl_configs          - RL configuration provider factories
    victor.prompt_contributors - Prompt contributor factories/classes
    victor.mode_configs        - Mode config provider factories/classes
    victor.workflow_providers  - Workflow provider factories/classes
    victor.team_spec_providers - Team provider factories/classes
    victor.capability_providers - Capability provider factories/classes
    victor.service_providers   - Service provider factories/classes
    victor.escape_hatches     - Escape hatch registration functions
    victor.commands            - CLI command registration functions
"""

from __future__ import annotations

import functools
import logging
import threading
import warnings
from typing import Any, Callable, Dict, List, Optional, TypeVar

from victor.framework.config import SafetyEnforcer

# Import unified registry for single-pass scanning
from victor.framework.entry_point_registry import (
    get_entry_point,
    get_entry_point_objects,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")

PROMPT_CONTRIBUTORS_ENTRY_POINT_GROUP = "victor.prompt_contributors"
MODE_CONFIGS_ENTRY_POINT_GROUP = "victor.mode_configs"
WORKFLOW_PROVIDERS_ENTRY_POINT_GROUP = "victor.workflow_providers"
TEAM_SPEC_PROVIDERS_ENTRY_POINT_GROUP = "victor.team_spec_providers"
CAPABILITY_PROVIDERS_ENTRY_POINT_GROUP = "victor.capability_providers"
SERVICE_PROVIDERS_ENTRY_POINT_GROUP = "victor.service_providers"

_ENTRY_POINT_LOADER_STATS: Dict[str, int] = {
    "safety_calls": 0,
    "safety_loaded": 0,
    "safety_failures": 0,
    "tool_dependency_calls": 0,
    "tool_dependency_entry_point_resolutions": 0,
    "tool_dependency_fallback_resolutions": 0,
    "tool_dependency_none_returns": 0,
    "tool_dependency_failures": 0,
    "rl_config_calls": 0,
    "rl_config_hits": 0,
    "rl_config_failures": 0,
    "escape_hatch_calls": 0,
    "escape_hatch_loaded": 0,
    "escape_hatch_failures": 0,
    "command_calls": 0,
    "command_loaded": 0,
    "command_failures": 0,
    "cache_clears": 0,
}
_ENTRY_POINT_LOADER_STATS_LOCK = threading.Lock()
_WARNED_LEGACY_VERTICAL_LISTINGS = False


def _increment_loader_stat(name: str) -> None:
    """Increment an entry-point loader telemetry counter."""
    with _ENTRY_POINT_LOADER_STATS_LOCK:
        _ENTRY_POINT_LOADER_STATS[name] = _ENTRY_POINT_LOADER_STATS.get(name, 0) + 1


def get_entry_point_loader_stats() -> Dict[str, int]:
    """Get entry-point loader telemetry counters and cache stats."""
    with _ENTRY_POINT_LOADER_STATS_LOCK:
        stats = dict(_ENTRY_POINT_LOADER_STATS)

    cache_info = _cached_entry_points.cache_info()
    stats.update(
        {
            "cache_hits": cache_info.hits,
            "cache_misses": cache_info.misses,
            "cache_maxsize": cache_info.maxsize or 0,
            "cache_currsize": cache_info.currsize,
        }
    )
    return stats


def reset_entry_point_loader_stats(clear_cache: bool = False) -> None:
    """Reset entry-point loader telemetry counters and optionally clear cache."""
    global _WARNED_LEGACY_VERTICAL_LISTINGS
    if clear_cache:
        _cached_entry_points.cache_clear()
    with _ENTRY_POINT_LOADER_STATS_LOCK:
        for key in _ENTRY_POINT_LOADER_STATS:
            _ENTRY_POINT_LOADER_STATS[key] = 0
    _WARNED_LEGACY_VERTICAL_LISTINGS = False


def _get_normalize_fn() -> Callable[[str], str]:
    """Lazy import normalize_vertical_name to avoid circular imports."""
    from victor.core.verticals.import_resolver import normalize_vertical_name

    return normalize_vertical_name


def normalize_vertical_name(name: str) -> str:
    """Normalize a vertical name (lazy-imported to avoid circular imports)."""
    return _get_normalize_fn()(name)


def _normalize_vertical_names(
    vertical_names: Optional[List[str]],
) -> Optional[set[str]]:
    """Normalize an optional list of vertical names for matching."""
    if vertical_names is None:
        return None
    fn = _get_normalize_fn()
    return {fn(name) for name in vertical_names}


def _entry_point_group_stat_prefix(group: str) -> str:
    """Return a stable telemetry prefix for an entry-point group."""
    if group.startswith("victor."):
        group = group[len("victor.") :]
    return group.replace(".", "_").replace("-", "_")


def _increment_group_loader_stat(group: str, suffix: str) -> None:
    """Increment a telemetry counter for a specific entry-point group."""
    _increment_loader_stat(f"{_entry_point_group_stat_prefix(group)}_{suffix}")


def _resolve_loaded_entry_point_target(target: Any) -> Any:
    """Instantiate zero-argument entry-point targets when needed."""
    if callable(target):
        return target()
    return target


@functools.lru_cache(maxsize=16)
def _cached_entry_points(group: str) -> tuple:
    """Get entry points for a group using unified registry.

    DEPRECATED: Use UnifiedEntryPointRegistry directly for better performance.
    This function maintains backward compatibility with existing code.

    Args:
        group: Entry point group name

    Returns:
        Tuple of entry point objects
    """
    logger.debug(f"_cached_entry_points('{group}') called - using shared entry-point discovery")
    return get_entry_point_objects(group)


def clear_entry_point_loader_cache() -> None:
    """Clear cached entry-point lookups for this module.

    Clears both the legacy LRU cache and the unified registry cache.
    """
    global _WARNED_LEGACY_VERTICAL_LISTINGS
    _cached_entry_points.cache_clear()
    _increment_loader_stat("cache_clears")
    _WARNED_LEGACY_VERTICAL_LISTINGS = False

    # Also clear the unified registry
    try:
        from victor.framework.entry_point_registry import get_entry_point_registry

        get_entry_point_registry().invalidate()
        logger.debug("Cleared unified entry point registry")
    except Exception as e:
        logger.warning(f"Failed to clear unified entry point registry: {e}")


def _resilient_load_entry_point(
    ep: Any,
    group: str,
) -> Optional[Any]:
    """Load a single entry point with failure isolation.

    Returns the loaded target or None on failure. Logs warnings for
    individual failures without aborting the entire group scan.
    """
    try:
        return ep.load()
    except Exception as e:
        _increment_group_loader_stat(group, "failures")
        logger.warning(
            "Failed to load entry point '%s' from group '%s': %s. "
            "Continuing with remaining entry points.",
            ep.name,
            group,
            e,
        )
        return None


def load_runtime_extension_from_entry_points(
    vertical: str,
    group: str,
) -> Optional[Any]:
    """Load a runtime extension/provider instance from an explicit entry-point group.

    The entry-point target may be a class, a zero-argument factory, or a pre-built
    instance. This helper performs only entry-point resolution; callers remain
    responsible for any compatibility fallbacks.

    Args:
        vertical: Vertical name (for example, ``"coding"``)
        group: Entry-point group name (for example, ``"victor.workflow_providers"``)

    Returns:
        Instantiated extension/provider object, or ``None`` when no matching entry
        point is available.
    """
    _increment_group_loader_stat(group, "calls")
    normalized_vertical = normalize_vertical_name(vertical)

    try:
        loaded_ep = get_entry_point(group, normalized_vertical)
    except Exception as e:
        _increment_group_loader_stat(group, "failures")
        logger.debug(
            "Failed to resolve entry point group '%s' for vertical '%s': %s",
            group,
            vertical,
            e,
        )
        return None

    if loaded_ep is not None:
        # Resolve loaded entry point target if needed
        resolved = _resolve_loaded_entry_point_target(loaded_ep)
        if resolved is not None:
            _increment_group_loader_stat(group, "hits")
            return resolved

    _increment_group_loader_stat(group, "none_returns")
    return None


def load_safety_rules_from_entry_points(
    enforcer: SafetyEnforcer,
    vertical_names: Optional[List[str]] = None,
) -> int:
    """Load safety rules from all installed vertical packages via entry points.

    This function discovers all safety rule registration functions via the
    victor.safety_rules entry point group and calls them to register rules
    with the provided enforcer.

    Args:
        enforcer: SafetyEnforcer to register rules with
        vertical_names: Optional list of specific verticals to load.
            If None, loads all available verticals.

    Returns:
        Number of verticals that successfully registered safety rules

    Example:
        from victor.framework.config import SafetyEnforcer, SafetyConfig, SafetyLevel
        from victor.framework.entry_point_loader import load_safety_rules_from_entry_points

        enforcer = SafetyEnforcer(config=SafetyConfig(level=SafetyLevel.HIGH))
        count = load_safety_rules_from_entry_points(enforcer)
        print(f"Loaded safety rules from {count} verticals")

        # Load specific verticals only
        count = load_safety_rules_from_entry_points(enforcer, vertical_names=["coding", "devops"])
    """
    _increment_loader_stat("safety_calls")
    count = 0
    normalized_verticals = _normalize_vertical_names(vertical_names)

    try:
        eps = _cached_entry_points("victor.safety_rules")
    except Exception as e:
        _increment_loader_stat("safety_failures")
        logger.warning("Failed to discover safety rule entry points: %s", e)
        return count

    for ep in eps:
        # Filter by vertical name if specified
        if (
            normalized_verticals is not None
            and normalize_vertical_name(ep.name) not in normalized_verticals
        ):
            continue

        register_func = _resilient_load_entry_point(ep, "victor.safety_rules")
        if register_func is None:
            continue

        try:
            register_func(enforcer)
            count += 1
            _increment_loader_stat("safety_loaded")
            logger.debug("Loaded safety rules from '%s' vertical", ep.name)
        except Exception as e:
            _increment_loader_stat("safety_failures")
            logger.warning(
                "Safety rule registration failed for '%s': %s. "
                "Other verticals' safety rules remain active.",
                ep.name,
                e,
            )

    return count


def load_tool_dependency_provider_from_entry_points(
    vertical: str,
) -> Optional[Any]:
    """Load a tool dependency provider for a specific vertical.

    Resolution order:
    1. ``victor.tool_dependencies`` entry points (preferred)
    2. Core compatibility loader fallbacks (module factory / package resources)

    Args:
        vertical: Vertical name (e.g., "coding", "devops", "research")

    Returns:
        Tool dependency provider instance, or None if not found

    Example:
        from victor.framework.entry_point_loader import load_tool_dependency_provider_from_entry_points

        provider = load_tool_dependency_provider_from_entry_points("coding")
        if provider:
            deps = provider.get_dependencies()
            sequences = provider.get_tool_sequences()
    """
    _increment_loader_stat("tool_dependency_calls")
    normalized_vertical = normalize_vertical_name(vertical)

    try:
        eps = _cached_entry_points("victor.tool_dependencies")
        for ep in eps:
            if normalize_vertical_name(ep.name) == normalized_vertical:
                provider_factory = ep.load()
                _increment_loader_stat("tool_dependency_entry_point_resolutions")
                return provider_factory()
    except Exception as e:
        _increment_loader_stat("tool_dependency_failures")
        logger.debug(f"No tool dependency provider found for '{vertical}' via entry points: {e}")

    # Compatibility fallback for extracted verticals (core + external packages).
    try:
        from victor.core.tool_dependency_loader import (
            create_vertical_tool_dependency_provider,
        )
        from victor.core.tool_types import EmptyToolDependencyProvider

        provider = create_vertical_tool_dependency_provider(normalized_vertical)
        if isinstance(provider, EmptyToolDependencyProvider):
            _increment_loader_stat("tool_dependency_none_returns")
            return None
        _increment_loader_stat("tool_dependency_fallback_resolutions")
        return provider
    except ValueError:
        _increment_loader_stat("tool_dependency_none_returns")
        logger.debug("Unknown vertical '%s' for tool dependency provider resolution", vertical)
    except Exception as e:
        _increment_loader_stat("tool_dependency_failures")
        logger.debug(
            "Fallback tool dependency provider resolution failed for '%s': %s",
            vertical,
            e,
        )

    _increment_loader_stat("tool_dependency_none_returns")
    return None


def load_rl_config_from_entry_points(vertical: str) -> Optional[Dict[str, Any]]:
    """Load an RL configuration for a specific vertical via entry points.

    Args:
        vertical: Vertical name (e.g., "coding", "devops")

    Returns:
        RL configuration dictionary, or None if not found

    Example:
        from victor.framework.entry_point_loader import load_rl_config_from_entry_points

        rl_config = load_rl_config_from_entry_points("coding")
        if rl_config:
            learning_rate = rl_config.get("learning_rate", 0.001)
    """
    _increment_loader_stat("rl_config_calls")
    provider = load_rl_config_provider_from_entry_points(vertical)
    if provider is None:
        return None

    if isinstance(provider, dict):
        _increment_loader_stat("rl_config_hits")
        return provider

    get_rl_config = getattr(provider, "get_rl_config", None)
    if callable(get_rl_config):
        try:
            config = get_rl_config()
        except Exception as e:
            _increment_loader_stat("rl_config_failures")
            logger.debug("Failed to resolve RL config for '%s': %s", vertical, e)
            return None
        _increment_loader_stat("rl_config_hits")
        return config

    _increment_loader_stat("rl_config_failures")
    logger.debug(
        "RL config entry point for '%s' did not expose get_rl_config() or a config dict",
        vertical,
    )
    return None


def load_rl_config_provider_from_entry_points(vertical: str) -> Optional[Any]:
    """Load an RL config provider instance for a specific vertical via entry points."""
    return load_runtime_extension_from_entry_points(vertical, "victor.rl_configs")


def register_escape_hatches_from_entry_points(
    registry: Any,
    vertical_names: Optional[List[str]] = None,
) -> int:
    """Register escape hatches from all installed vertical packages via entry points.

    Args:
        registry: Escape hatch registry to register hatches with
        vertical_names: Optional list of specific verticals to load.
            If None, loads all available verticals.

    Returns:
        Number of verticals that successfully registered escape hatches

    Example:
        from victor.framework.escape_hatch_registry import EscapeHatchRegistry
        from victor.framework.entry_point_loader import register_escape_hatches_from_entry_points

        registry = EscapeHatchRegistry()
        count = register_escape_hatches_from_entry_points(registry)
        print(f"Registered escape hatches from {count} verticals")
    """
    _increment_loader_stat("escape_hatch_calls")
    count = 0
    normalized_verticals = _normalize_vertical_names(vertical_names)
    try:
        eps = _cached_entry_points("victor.escape_hatches")
        for ep in eps:
            # Filter by vertical name if specified
            if (
                normalized_verticals is not None
                and normalize_vertical_name(ep.name) not in normalized_verticals
            ):
                continue

            try:
                register_func = ep.load()
                register_func(registry)
                count += 1
                _increment_loader_stat("escape_hatch_loaded")
                logger.debug(f"Registered escape hatches from '{ep.name}' vertical")
            except Exception as e:
                _increment_loader_stat("escape_hatch_failures")
                logger.warning(f"Failed to register escape hatches from '{ep.name}': {e}")
    except Exception as e:
        _increment_loader_stat("escape_hatch_failures")
        logger.debug(f"No escape hatch entry points found: {e}")

    return count


def register_commands_from_entry_points(
    app: Any,
    vertical_names: Optional[List[str]] = None,
) -> int:
    """Register CLI commands from all installed vertical packages via entry points.

    Args:
        app: Typer application to register commands with
        vertical_names: Optional list of specific verticals to load.
            If None, loads all available verticals.

    Returns:
        Number of verticals that successfully registered commands

    Example:
        import typer
        from victor.framework.entry_point_loader import register_commands_from_entry_points

        app = typer.Typer()
        count = register_commands_from_entry_points(app)
        print(f"Registered commands from {count} verticals")
    """
    _increment_loader_stat("command_calls")
    count = 0
    normalized_verticals = _normalize_vertical_names(vertical_names)
    try:
        eps = _cached_entry_points("victor.commands")
        for ep in eps:
            # Filter by vertical name if specified
            if (
                normalized_verticals is not None
                and normalize_vertical_name(ep.name) not in normalized_verticals
            ):
                continue

            try:
                register_func = ep.load()
                register_func(app)
                count += 1
                _increment_loader_stat("command_loaded")
                logger.debug(f"Registered commands from '{ep.name}' vertical")
            except Exception as e:
                _increment_loader_stat("command_failures")
                logger.warning(f"Failed to register commands from '{ep.name}': {e}")
    except Exception as e:
        _increment_loader_stat("command_failures")
        logger.debug(f"No command entry points found: {e}")

    return count


def list_installed_verticals() -> List[str]:
    """List all installed vertical packages via entry points.

    Returns:
        List of vertical names that have registered entry points

    Example:
        from victor.framework.entry_point_loader import list_installed_verticals

        verticals = list_installed_verticals()
        print(f"Installed verticals: {', '.join(verticals)}")
    """
    global _WARNED_LEGACY_VERTICAL_LISTINGS
    verticals = set()
    legacy_verticals = set()
    try:
        for group in ("victor.plugins", "victor.verticals"):
            eps = _cached_entry_points(group)
            for ep in eps:
                verticals.add(ep.name)
                if group == "victor.verticals":
                    legacy_verticals.add(ep.name)
    except Exception:
        pass

    if legacy_verticals and not _WARNED_LEGACY_VERTICAL_LISTINGS:
        message = (
            "Installed vertical inventory includes deprecated raw entry points from "
            "'victor.verticals'. Migrate these packages to 'victor.plugins'. "
            f"Observed legacy entries: {', '.join(sorted(legacy_verticals))}"
        )
        logger.warning(message)
        _WARNED_LEGACY_VERTICAL_LISTINGS = True

    return sorted(list(verticals))


__all__ = [
    "PROMPT_CONTRIBUTORS_ENTRY_POINT_GROUP",
    "MODE_CONFIGS_ENTRY_POINT_GROUP",
    "WORKFLOW_PROVIDERS_ENTRY_POINT_GROUP",
    "TEAM_SPEC_PROVIDERS_ENTRY_POINT_GROUP",
    "CAPABILITY_PROVIDERS_ENTRY_POINT_GROUP",
    "SERVICE_PROVIDERS_ENTRY_POINT_GROUP",
    "load_runtime_extension_from_entry_points",
    "load_safety_rules_from_entry_points",
    "load_tool_dependency_provider_from_entry_points",
    "load_rl_config_from_entry_points",
    "load_rl_config_provider_from_entry_points",
    "register_escape_hatches_from_entry_points",
    "register_commands_from_entry_points",
    "list_installed_verticals",
    "clear_entry_point_loader_cache",
    "get_entry_point_loader_stats",
    "reset_entry_point_loader_stats",
]
