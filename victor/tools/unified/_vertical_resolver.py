# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
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

"""Vertical-tool delegation resolver for unified command tools.

Unified command-shell dispatchers (``git``, ``code``) own the canonical tool
names and *delegate* to sister-repo implementations when the optional vertical
package is available, falling back to a core implementation otherwise.

Resolution is **key-based** with two layers:

1. **Entry points** (preferred): sister repos register callables under the
   ``victor.tool_callables`` group. The entry-point *name* is a stable logical
   key (e.g. ``"git"``, ``"code_search"``) and the value is ``"module:attr"``.
   This decouples the framework from sister-repo internal module layouts — if a
   vertical reorganizes its modules it only updates its entry point.
2. **Direct import** (fallback): when no entry point is registered (older
   vertical versions), the dispatcher may pass ``fallback_module`` /
   ``fallback_attr`` for a direct import probe.

The resolver never imports a sister-repo package at module top level, so an
absent optional package degrades gracefully rather than breaking framework
import.
"""

from __future__ import annotations

import importlib
import logging
from typing import Any, Callable, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# Entry-point group: name = logical key, value = "module:attr".
TOOL_CALLABLES_GROUP = "victor.tool_callables"

_entry_point_cache: Optional[Dict[str, Any]] = None


def _entry_point_callables() -> Dict[str, Any]:
    """Return ``{key: EntryPoint}`` for the tool-callables group (cached).

    Entry points are fixed for a process lifetime, so the result is cached after
    first discovery. :func:`_reset_entry_point_cache` clears it for tests.
    """
    global _entry_point_cache
    if _entry_point_cache is not None:
        return _entry_point_cache
    cache: Dict[str, Any] = {}
    try:
        from importlib.metadata import entry_points

        eps = entry_points()
        group_eps = (
            eps.select(group=TOOL_CALLABLES_GROUP)
            if hasattr(eps, "select")
            else eps.get(TOOL_CALLABLES_GROUP, [])
        )
        cache = {ep.name: ep for ep in group_eps}
    except Exception as exc:  # pragma: no cover - environment dependent
        logger.debug("tool_callables entry-point discovery failed: %s", exc)
    _entry_point_cache = cache
    return cache


def _reset_entry_point_cache() -> None:
    """Clear the entry-point cache (for tests that add/remove entry points)."""
    global _entry_point_cache
    _entry_point_cache = None


def _import_callable(
    module_path: str, attribute: str
) -> Tuple[Optional[Callable[..., Any]], Optional[str]]:
    """Direct-import probe for ``(module_path, attribute)``; never raises."""
    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        logger.debug(
            "Vertical module '%s' not available; will use fallback: %s",
            module_path,
            exc,
        )
        return None, None
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("Vertical module '%s' failed to import: %s", module_path, exc)
        return None, None

    candidate = getattr(module, attribute, None)
    if not callable(candidate):
        logger.debug("Vertical module '%s' has no callable '%s'", module_path, attribute)
        return None, None
    return candidate, module_path


def resolve_vertical_callable(
    key: str,
    *,
    fallback_module: Optional[str] = None,
    fallback_attr: Optional[str] = None,
) -> Tuple[Optional[Callable[..., Any]], Optional[str]]:
    """Resolve a vertical callable by logical ``key``.

    Resolution order:
    1. ``victor.tool_callables`` entry point named ``key`` (preferred).
    2. Direct import of ``(fallback_module, fallback_attr)`` if provided.

    Args:
        key: Stable logical callable key (e.g. ``"git"``, ``"code_search"``).
        fallback_module: Dotted module path for the direct-import fallback.
        fallback_attr: Attribute name for the direct-import fallback.

    Returns:
        ``(callable, source)`` where ``source`` identifies the resolution path
        (``"entry-point:<key>"`` or the module path), else ``(None, None)``.
        Never raises.
    """
    ep = _entry_point_callables().get(key)
    if ep is not None:
        try:
            obj = ep.load()
        except Exception as exc:  # pragma: no cover - environment dependent
            logger.debug("entry-point '%s' load failed: %s", key, exc)
            obj = None
        if callable(obj):
            return obj, f"entry-point:{key}"
        logger.debug("entry-point '%s' did not yield a callable", key)

    if fallback_module and fallback_attr:
        return _import_callable(fallback_module, fallback_attr)

    return None, None


__all__ = [
    "TOOL_CALLABLES_GROUP",
    "resolve_vertical_callable",
    "_reset_entry_point_cache",
]
