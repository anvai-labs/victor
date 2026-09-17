# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
# SPDX-License-Identifier: Apache-2.0
"""Byte-stable ToolDefinition list builders shared by every tool-supply path.

The sorted → registry.get → tool_to_definition loop previously existed in four
places (ToolSelectionRuntime curated/all/read-core and ToolSelector curated),
each with its own cache and log line. The copies drifted once already (#353
split-brain: the curated short-circuit lived in only one of two call paths).
This module is the single implementation; callers keep thin delegates so their
public shapes — and the caches' home on the selector — stay stable.

Cache keys include ``id(registry)`` so a registry instance swap rebuilds
instead of serving stale definitions.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, List, Optional

logger = logging.getLogger(__name__)


def _registry_version(registry: Any) -> Any:
    """Registry-provided schema version for O(1) cache invalidation.

    ToolRegistry bumps _schema_cache_version on (re-)registration; unknown
    registries contribute a constant (name-set key still detects membership
    changes).
    """
    return getattr(registry, "_schema_cache_version", 0)


def normalize_registry_names(listed: Any) -> List[str]:
    """Normalize ``registry.list_tools()`` output (instances or names) to names."""
    return [
        str(getattr(t, "name", t))
        for t in (listed or [])
        if getattr(t, "name", None) or isinstance(t, str)
    ]


def build_stable_definitions(registry: Any, names: Iterable[str], schema_level: Any) -> List[Any]:
    """sorted(names) → registry.get → tool_to_definition; missing names skipped."""
    from victor.agent.tool_selection import tool_to_definition

    stable: List[Any] = []
    for name in sorted(names):
        tool = registry.get(name)
        if tool is not None:
            stable.append(tool_to_definition(tool, schema_level))
    return stable


def _cached_stable(owner: Any, cache_attr: str, cache_key: Any, builder: Any, label: str) -> Any:
    cached = getattr(owner, cache_attr, None)
    if cached and cached[0] == cache_key:
        return cached[1]
    stable = builder()
    if stable:
        setattr(owner, cache_attr, (cache_key, stable))
        # Rebuild-only logging: a hit is the happy path, not a signal.
        logger.info("[ToolSchema] %s rebuilt: %d tools", label, len(stable))
    return stable or None


def stable_curated_definitions(selector: Any) -> Optional[List[Any]]:
    """FULL-schema definitions for the selector's curated ``_enabled_tools`` set.

    Returns None when no curated set is active (the common auto-selected case).
    """
    enabled = getattr(selector, "_enabled_tools", None)
    if not enabled:
        return None
    registry = getattr(selector, "tools", None)
    if not registry:
        return None
    cache_key = (id(registry), _registry_version(registry), frozenset(enabled))

    def build() -> List[Any]:
        from victor.tools.enums import SchemaLevel

        return build_stable_definitions(registry, enabled, SchemaLevel.FULL)

    return _cached_stable(selector, "_stable_tools_cache", cache_key, build, "Stable curated")


def stable_all_definitions(selector: Any) -> Optional[List[Any]]:
    """FULL-schema definitions over the whole enabled registry (pruning off)."""
    registry = getattr(selector, "tools", None)
    if not registry:
        return None
    try:
        names = normalize_registry_names(registry.list_tools(only_enabled=True))
    except Exception:
        return None
    if not names:
        return None
    cache_key = (id(registry), _registry_version(registry), "all", frozenset(names))

    def build() -> List[Any]:
        from victor.tools.enums import SchemaLevel

        return build_stable_definitions(registry, names, SchemaLevel.FULL)

    return _cached_stable(selector, "_all_tools_cache", cache_key, build, "Pruning-off full supply")
