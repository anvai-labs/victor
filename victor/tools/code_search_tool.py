# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""STUB MODULE - Legacy code_search_tool functionality migrated.

The original code_search_tool module with LanceDB indexing has been removed.
Code search functionality is now provided by:
- victor.tools.unified.search_tool (name="code_search") - grep and file search
- Graph-based code search through victor.tools.graph_tool

This stub provides deprecated placeholder imports for test compatibility.
Tests importing from this module should be updated to skip or use the new tools.

TODO: Migrate tests to use the new code_search tool or mark as skipped.
"""

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set


# Stub for SearchFilters - used in tests
@dataclass
class SearchFilters:
    """Filters for code search (STUB - deprecated)."""

    file_pattern: Optional[str] = None
    extensions: Optional[Set[str]] = None
    exclude_patterns: Optional[Set[str]] = None
    top_k: int = 10
    mode: str = "text"
    semantic_threshold: float = 0.25

    def __post_init__(self):
        warnings.warn(
            "SearchFilters from code_search_tool is deprecated. "
            "Use the new code_search tool in victor.tools.unified.search_tool instead.",
            DeprecationWarning,
            stacklevel=2,
        )


# Stub functions - all raise NotImplementedError
def _normalize_search_filters(filters: Any) -> SearchFilters:
    """Normalize search filters (STUB - deprecated)."""
    warnings.warn(
        "_normalize_search_filters is deprecated. " "Use the new code_search tool instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    if isinstance(filters, dict):
        return SearchFilters(
            **{k: v for k, v in filters.items() if k in SearchFilters.__dataclass_fields__}
        )
    if isinstance(filters, SearchFilters):
        return filters
    return SearchFilters()


def _build_codebase_embedding_config(settings: Any) -> Dict[str, Any]:
    """Build embedding config (STUB - deprecated)."""
    warnings.warn(
        "_build_codebase_embedding_config is deprecated.",
        DeprecationWarning,
        stacklevel=2,
    )
    return {}


def _decorate_literal_fallback_result(result: Any, query: str) -> Dict[str, Any]:
    """Decorate literal search result (STUB - deprecated)."""
    warnings.warn(
        "_decorate_literal_fallback_result is deprecated.",
        DeprecationWarning,
        stacklevel=2,
    )
    return {"result": result, "query": query}


def _get_index_build_failure_cache() -> Optional[Any]:
    """Get index build failure cache (STUB - deprecated)."""
    warnings.warn(
        "_get_index_build_failure_cache is deprecated.",
        DeprecationWarning,
        stacklevel=2,
    )
    return None


def _get_or_build_index(settings: Any) -> Optional[Any]:
    """Get or build index (STUB - deprecated)."""
    warnings.warn(
        "_get_or_build_index is deprecated. " "Use victor.tools.unified.search_tool instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return None


def _literal_search(query: str, path: str, **kwargs) -> List[Dict[str, Any]]:
    """Literal search (STUB - deprecated)."""
    warnings.warn(
        "_literal_search is deprecated. " "Use victor.tools.unified.search_tool instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return []


def clear_index_cache() -> None:
    """Clear index cache (STUB - deprecated)."""
    warnings.warn("clear_index_cache is deprecated.", DeprecationWarning, stacklevel=2)


async def code_search(
    query: str,
    path: str = ".",
    mode: str = "text",
    filters: Optional[SearchFilters] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Code search (STUB - deprecated).

    Use victor.tools.unified.search_tool instead.
    """
    warnings.warn(
        "code_search is deprecated. "
        "Use victor.tools.unified.search_tool (name='code_search') instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return {
        "results": [],
        "query": query,
        "path": path,
        "mode": mode,
        "error": "Deprecated - use victor.tools.unified.search_tool",
    }


# Additional stub classes/functions for test compatibility
@dataclass
class IntegrityProbeOutcome:
    """Outcome of index integrity probe (STUB - deprecated)."""

    rebuilt: bool = False
    healthy: bool = True
    error: Optional[str] = None

    def __post_init__(self):
        warnings.warn("IntegrityProbeOutcome is deprecated.", DeprecationWarning, stacklevel=2)


async def _probe_index_integrity(index: Any) -> IntegrityProbeOutcome:
    """Probe index integrity (STUB - deprecated)."""
    warnings.warn("_probe_index_integrity is deprecated.", DeprecationWarning, stacklevel=2)
    return IntegrityProbeOutcome(healthy=True)


def _calculate_index_build_timeout(file_count: int, base_timeout: float = 300.0) -> float:
    """Calculate index build timeout (STUB - deprecated)."""
    warnings.warn(
        "_calculate_index_build_timeout is deprecated.",
        DeprecationWarning,
        stacklevel=2,
    )
    return base_timeout


def extract_skeleton(source: str, language: str = "python") -> str:
    """Extract code skeleton (STUB - deprecated).

    Use victor.tools.unified.search_tool or graph-based analysis instead.
    """
    warnings.warn(
        "extract_skeleton is deprecated. " "Use graph-based analysis or manual inspection instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    # Return a simple skeleton as placeholder
    lines = source.split("\n")
    skeleton = []
    for line in lines:
        stripped = line.strip()
        # Keep function/class definitions and important structural lines
        if stripped.startswith(("def ", "class ", "async def ", "import ", "from ")):
            skeleton.append(line)
        elif stripped and not stripped.startswith("#") and not stripped.startswith(('"""', "'''")):
            # Keep other non-empty, non-comment lines at a basic level
            if len(stripped) < 80:  # Short lines are likely structural
                skeleton.append(line)
    return "\n".join(skeleton)


# Export all stub items
__all__ = [
    "SearchFilters",
    "_normalize_search_filters",
    "_build_codebase_embedding_config",
    "_decorate_literal_fallback_result",
    "_get_index_build_failure_cache",
    "_get_or_build_index",
    "_literal_search",
    "clear_index_cache",
    "code_search",
    "IntegrityProbeOutcome",
    "_probe_index_integrity",
    "_calculate_index_build_timeout",
    "extract_skeleton",
]
