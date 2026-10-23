"""Compatibility helpers for extracted coding-package integrations.

These helpers centralize the migration bridge from legacy in-tree coding
modules to the extracted ``victor-coding`` package. Discovery is done via
entry points and importlib — core never imports ``victor_coding`` directly.

Discovery chain for each capability:
1. Entry point: ``victor.extension.capabilities`` group, then legacy
   ``victor.sdk.capabilities`` group
2. Importlib: Try ``victor_coding.*`` via importlib.import_module (no static import)
3. Legacy: Try ``victor.verticals.contrib.coding.*`` fallback
4. Raise ImportError with helpful message
"""

from __future__ import annotations

import importlib
import logging
from types import ModuleType
from typing import Callable, TypeVar, cast

logger = logging.getLogger(__name__)
T = TypeVar("T")


def _try_import(module_path: str) -> ModuleType:
    """Import a module by dotted path. Raises ImportError on failure."""
    return importlib.import_module(module_path)


def _try_entry_point(group: str, name: str) -> object:
    """Load a named entry point from a group. Returns None if not found."""
    try:
        from victor.core.entry_point_registry import get_entry_point

        return get_entry_point(group, name)
    except Exception:
        return None


def _try_capability_entry_point(name: str) -> object:
    """Load a capability provider from preferred and legacy entry-point groups."""
    try:
        from victor.core.entry_point_registry import CAPABILITY_ENTRY_POINT_GROUPS
    except Exception:
        groups = ("victor.extension.capabilities", "victor.sdk.capabilities")
    else:
        groups = CAPABILITY_ENTRY_POINT_GROUPS

    for group in groups:
        ep = _try_entry_point(group, name)
        if ep is not None:
            return ep
    return None


# Migration bridge: importlib fallback paths tried after entry points.
# These are runtime importlib.import_module() calls, NOT static imports.
# Remove once all deployments use the victor-coding entry point package.
_FALLBACK_MODULES: dict[str, tuple[str, ...]] = {
    "codebase_analyzer": (
        "victor_coding.codebase_analyzer",
        "victor.verticals.contrib.coding.codebase_analyzer",
    ),
    "tree_sitter": (
        "victor_coding.codebase.tree_sitter_manager",
        "victor.verticals.contrib.coding.codebase.tree_sitter_manager",
    ),
    "analyze": (
        "victor_coding.commands.analyze",
        "victor.verticals.contrib.coding.commands.analyze",
    ),
    # Extracted vertical tool modules. The dotted paths live here (a data
    # structure) rather than as string-literal import_module() calls so that
    # core never holds a static vertical import — satisfying the core/vertical
    # import boundary guard (tests/unit/contracts/test_core_vertical_import_boundary.py).
    "graph_tool": (
        "victor_coding.tools.graph_tool",
        "victor.verticals.contrib.coding.tools.graph_tool",
    ),
    "code_search_tool": (
        "victor_coding.tools.code_search_tool",
        "victor.verticals.contrib.coding.tools.code_search_tool",
    ),
    "code_executor_tool": (
        "victor_coding.tools.code_executor_tool",
        "victor.verticals.contrib.coding.tools.code_executor_tool",
    ),
}


def _load_fallback_module(capability: str, description: str) -> ModuleType:
    """Resolve a vertical tool module through the importlib fallback chain.

    The dotted module paths are stored in ``_FALLBACK_MODULES`` (a data
    structure) and resolved through ``_try_import`` using a *variable*, so
    core never holds a static string-literal import of an external vertical
    package. This is the sanctioned pattern for the core/vertical import
    boundary.
    """
    last_error: ImportError | None = None
    for module_path in _FALLBACK_MODULES[capability]:
        try:
            return _try_import(module_path)
        except ImportError as exc:
            last_error = exc
            continue
    raise ImportError(
        f"{description} requires the victor-coding package. "
        "Install with: pip install victor-coding"
    ) from last_error


def load_graph_tool_module() -> ModuleType:
    """Resolve the coding vertical's graph tool module (dynamic discovery)."""
    return _load_fallback_module("graph_tool", "graph tool")


def load_code_search_module() -> ModuleType:
    """Resolve the coding vertical's code search tool module (dynamic discovery)."""
    return _load_fallback_module("code_search_tool", "code search tool")


def load_code_executor_module() -> ModuleType:
    """Resolve the coding vertical's code executor tool module (dynamic discovery)."""
    return _load_fallback_module("code_executor_tool", "code executor tool")


# ---------------------------------------------------------------------------
# Codebase Analyzer
# ---------------------------------------------------------------------------


def load_codebase_analyzer_module() -> ModuleType:
    """Load the codebase analyzer module via discovery chain.

    Discovery order:
    1. Entry point ``victor.extension.capabilities::codebase_analyzer`` or
       legacy ``victor.sdk.capabilities::codebase_analyzer``
    2. ``victor_coding.codebase_analyzer`` via importlib
    3. Legacy ``victor.verticals.contrib.coding.codebase_analyzer``
    """
    # Try entry point first
    ep = _try_capability_entry_point("codebase_analyzer")
    if ep is not None and hasattr(ep, "__module__"):
        try:
            return importlib.import_module(ep.__module__)
        except ImportError:
            pass

    # Try module paths
    last_error: ImportError | None = None
    for module_path in _FALLBACK_MODULES["codebase_analyzer"]:
        try:
            return _try_import(module_path)
        except ImportError as exc:
            last_error = exc
            continue

    raise ImportError(
        "codebase_analyzer requires the victor-coding package. "
        "Install with: pip install victor-coding"
    ) from last_error


def load_codebase_analyzer_attr(name: str) -> object:
    """Resolve a single analyzer export and raise ImportError when unavailable."""
    module = load_codebase_analyzer_module()
    try:
        return getattr(module, name)
    except AttributeError as exc:
        raise ImportError(f"codebase_analyzer does not expose required symbol '{name}'") from exc


# ---------------------------------------------------------------------------
# Tree-Sitter Manager
# ---------------------------------------------------------------------------


def load_tree_sitter_get_parser() -> Callable[[str], object]:
    """Resolve the canonical get_parser function for tree-sitter support.

    Discovery order:
    1. Entry point ``victor.extension.capabilities::tree_sitter`` or
       legacy ``victor.sdk.capabilities::tree_sitter``
    2. ``victor_coding.codebase.tree_sitter_manager`` via importlib
    3. Legacy ``victor.verticals.contrib.coding.codebase.tree_sitter_manager``
    """
    # Try entry point first
    ep = _try_capability_entry_point("tree_sitter")
    if ep is not None:
        get_parser = getattr(ep, "get_parser", None)
        if callable(get_parser):
            return cast(Callable[[str], object], get_parser)

    # Try module paths
    last_error: ImportError | None = None
    for module_path in _FALLBACK_MODULES["tree_sitter"]:
        try:
            module = _try_import(module_path)
        except ImportError as exc:
            last_error = exc
            continue
        get_parser = getattr(module, "get_parser", None)
        if callable(get_parser):
            return cast(Callable[[str], object], get_parser)

    raise ImportError(
        "tree_sitter_manager requires the victor-coding package. "
        "Install with: pip install victor-coding"
    ) from last_error


# ---------------------------------------------------------------------------
# Coding Analyze CLI App
# ---------------------------------------------------------------------------


def load_coding_analyze_app() -> object:
    """Resolve the coding analyze CLI app.

    Discovery order:
    1. Entry point ``victor.commands::analyze``
    2. ``victor_coding.commands.analyze`` via importlib
    3. Legacy ``victor.verticals.contrib.coding.commands.analyze``
    """
    # Try entry point first
    ep = _try_entry_point("victor.commands", "analyze")
    if ep is not None:
        return ep

    # Try module paths
    last_error: ImportError | None = None
    for module_path in _FALLBACK_MODULES["analyze"]:
        try:
            module = _try_import(module_path)
        except ImportError as exc:
            last_error = exc
            continue
        app = getattr(module, "app", None)
        if app is not None:
            return app

    raise ImportError(
        "coding analyze command requires victor-coding. " "Install with: pip install victor-coding"
    ) from last_error


__all__ = [
    "load_codebase_analyzer_module",
    "load_codebase_analyzer_attr",
    "load_tree_sitter_get_parser",
    "load_coding_analyze_app",
    "load_graph_tool_module",
    "load_code_search_module",
    "load_code_executor_module",
]
