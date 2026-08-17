from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
import re
import threading
import time
from collections.abc import Iterable as IterableABC
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from typing import Any, DefaultDict, Dict, Iterable, List, Literal, Optional, Set

from victor.config.settings import get_project_paths, load_settings

from victor.native.python.graph_algo import (
    connected_components,
    pagerank,
    weighted_pagerank,
)
from victor.storage.graph.edge_types import EdgeType
from victor.storage.graph.protocol import GraphEdge, GraphNode, GraphStoreProtocol
from victor.storage.unified.protocol import UnifiedId
from victor.tools.base import AccessMode, DangerLevel, ExecutionCategory, Priority
from victor.tools.code_search_tool import _get_or_build_index
from victor.tools.context import ToolExecutionContext
from victor.tools.decorators import tool

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pre-flight dependency check (Layer 3 robustness)
# ---------------------------------------------------------------------------
# Validates ALL victor_contracts imports resolve before the tool attempts any
# operation. If any dependency is missing (bridge gap), the tool returns a
# clean error on the FIRST call (1 turn wasted, not 3+). Combined with the
# circuit breaker (3 calls → auto-disable), worst case is 3 turns total
# instead of 24-64+. Cached after first call.
_GRAPH_DEPENDENCIES_CHECKED: Optional[bool] = None

_GRAPH_DEPENDENCY_CHECKS = [
    ("victor_contracts.database_runtime", "get_project_database"),
    ("victor_contracts.database_runtime", "resolve_project_db_root"),
    ("victor_contracts.indexing_runtime", "GraphManager"),
    ("victor_contracts.capability_runtime", "get_capability_provider"),
]


def _check_graph_dependencies() -> bool:
    """Validate ALL victor_contracts imports resolve. Cached after first call.

    Returns False if any dependency is missing (bridge gap). The caller should
    return _graph_error_response immediately — no point attempting operations
    that will crash on import.
    """
    global _GRAPH_DEPENDENCIES_CHECKED
    if _GRAPH_DEPENDENCIES_CHECKED is not None:
        return _GRAPH_DEPENDENCIES_CHECKED
    import importlib as _importlib

    for mod_name, attr_name in _GRAPH_DEPENDENCY_CHECKS:
        try:
            # Module names come from the hardcoded _GRAPH_DEPENDENCY_CHECKS
            # list above, not from user input.
            # nosemgrep: python.lang.security.audit.non-literal-import.non-literal-import
            mod = _importlib.import_module(mod_name)  # nosemgrep
            getattr(mod, attr_name)
        except Exception:
            _GRAPH_DEPENDENCIES_CHECKED = False
            logger.warning(
                "Graph tool dependency check failed: %s.%s not available. "
                "The tool will return errors until the bridge is fixed.",
                mod_name,
                attr_name,
            )
            return False
    _GRAPH_DEPENDENCIES_CHECKED = True
    return True


_GRAPH_ANALYTICS_TIMEOUT_SECONDS = 90.0
_GRAPH_TOOL_TIMEOUT_SECONDS = _GRAPH_ANALYTICS_TIMEOUT_SECONDS
_GRAPH_ANALYTICS_CACHE_TTL_SECONDS = 300.0
_GRAPH_ANALYTICS_CACHE_MAX_ENTRIES = 64
_GRAPH_ANALYTICS_MAX_CONCURRENT = 2
_GRAPH_ANALYTICS_CACHE: Dict[tuple[Any, ...], tuple[float, Any]] = {}
_GRAPH_ANALYTICS_THREAD_SEMAPHORE = threading.BoundedSemaphore(_GRAPH_ANALYTICS_MAX_CONCURRENT)


# ---------------------------------------------------------------------------
# Graph result output bounds
# ---------------------------------------------------------------------------
# A single graph() call can otherwise materialize the entire node/edge set
# (e.g. `SELECT * FROM graph_edge`, a deep neighbor traversal, or "map all
# components recursively"), producing multi-megabyte payloads. That blows the
# context window and triggers aggressive compaction (an observed result drove
# estimated_output_tokens=646224). These caps bound any list-valued field in the
# result and enforce a total serialized-size ceiling, attaching a truncation
# note so the LLM knows the view is partial and how to narrow it.
def _graph_env_int(name: str, default: int) -> int:
    """Read a positive int from the environment, falling back to ``default``."""
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


GRAPH_MAX_LIST_ITEMS = _graph_env_int("VICTOR_GRAPH_MAX_LIST_ITEMS", 250)
GRAPH_MAX_RESULT_CHARS = _graph_env_int("VICTOR_GRAPH_MAX_RESULT_CHARS", 60_000)
# Hard row cap for raw `query` mode. The output guard above trims what reaches the
# LLM, but a `SELECT * FROM graph_edge` would still materialize every row into
# memory first; this bounds it at the database source. A user-supplied LIMIT is
# always honored as-is.
GRAPH_MAX_QUERY_ROWS = _graph_env_int("VICTOR_GRAPH_MAX_QUERY_ROWS", 1000)


def _ensure_sql_row_limit(sql: str, max_rows: int) -> tuple[str, bool]:
    """Append a defensive ``LIMIT`` to a SELECT that lacks one.

    Bounds how many rows the database materializes for raw ``query`` mode. If the
    query already specifies a ``LIMIT`` it is honored unchanged (the user took
    responsibility, and the output guard still caps what reaches the LLM). Returns
    the (possibly modified) SQL and whether a limit was injected.
    """
    stripped = sql.rstrip().rstrip(";").rstrip()
    if re.search(r"\blimit\b\s+\d+", stripped, re.IGNORECASE):
        return sql, False
    return f"{stripped} LIMIT {max_rows}", True


def _bound_graph_result(
    payload: Any,
    *,
    max_list_items: int = GRAPH_MAX_LIST_ITEMS,
    max_chars: int = GRAPH_MAX_RESULT_CHARS,
) -> Any:
    """Bound the size of a graph tool result before it enters LLM context.

    Recursively caps every list-valued field to ``max_list_items`` entries and,
    if the serialized result still exceeds ``max_chars``, re-applies the pass with
    progressively smaller caps. When anything is dropped, a ``truncated`` flag and
    an LLM-readable ``truncation_note`` are attached so the caller knows the result
    is partial and how to narrow the request (lower top_k/depth, add a SQL
    LIMIT/WHERE, or scope to a file/module).
    """
    if not isinstance(payload, (dict, list)):
        return payload

    def _measure(obj: Any) -> int:
        try:
            return len(json.dumps(obj, default=str))
        except (TypeError, ValueError):
            return len(str(obj))

    def _apply(obj: Any, cap: int, stats: Dict[str, int]) -> Any:
        if isinstance(obj, dict):
            return {key: _apply(value, cap, stats) for key, value in obj.items()}
        if isinstance(obj, list):
            if len(obj) > cap:
                stats["dropped"] += len(obj) - cap
                stats["lists"] += 1
                obj = obj[:cap]
            return [_apply(item, cap, stats) for item in obj]
        return obj

    cap = max_list_items
    bounded: Any = payload
    dropped = 0
    truncated_lists = 0
    # Shrink the per-list cap until the serialized payload fits the char budget
    # (or we hit a sensible floor — a small partial result beats a context blowup).
    for candidate in (max_list_items, max_list_items // 2, max_list_items // 5, 25, 10):
        cap = max(1, candidate)
        stats = {"dropped": 0, "lists": 0}
        bounded = _apply(payload, cap, stats)
        dropped = stats["dropped"]
        truncated_lists = stats["lists"]
        if _measure(bounded) <= max_chars:
            break

    if (dropped or truncated_lists) and isinstance(bounded, dict):
        bounded["truncated"] = True
        bounded["truncation_note"] = (
            f"Result bounded for context safety: {dropped} item(s) across "
            f"{truncated_lists} list(s) were dropped (per-list cap {cap}). This is a "
            "PARTIAL view — narrow the request to see more (lower top_k/depth, add a "
            "SQL LIMIT/WHERE clause, or scope to a specific file/module)."
        )
    return bounded


# Graph size thresholds for adaptive timeout
_SMALL_GRAPH_MAX_NODES = 50_000  # Nodes under this use base timeout
_MEDIUM_GRAPH_MAX_NODES = 200_000  # Nodes up to this get 1.5x timeout
_LARGE_GRAPH_MAX_NODES = 500_000  # Nodes up to this get 2x timeout
_TIMEOUT_SCALING_FACTOR = 2.0  # Maximum scaling factor for very large graphs


def _compute_adaptive_timeout(
    mode: str,
    node_count: int,
    base_timeout: float = _GRAPH_ANALYTICS_TIMEOUT_SECONDS,
) -> float:
    """Compute adaptive timeout based on graph size and operation complexity.

    Args:
        mode: Graph operation mode (centrality, patterns, pagerank, etc.)
        node_count: Number of nodes in the graph
        base_timeout: Base timeout in seconds

    Returns:
        Adaptive timeout in seconds
    """
    # Modes that benefit from extra time on large graphs
    expensive_modes = {
        "centrality",
        "patterns",
        "pagerank",
        "module_pagerank",
        "module_centrality",
    }

    if mode not in expensive_modes or node_count < _SMALL_GRAPH_MAX_NODES:
        return base_timeout

    # Scale timeout based on graph size
    if node_count < _MEDIUM_GRAPH_MAX_NODES:
        return base_timeout * 1.5
    elif node_count < _LARGE_GRAPH_MAX_NODES:
        return base_timeout * 2.0
    else:
        # For very large graphs, use maximum scaling factor
        return base_timeout * _TIMEOUT_SCALING_FACTOR


# =============================================================================
# Enums for Graph Operations
# =============================================================================


class GraphDirection(str, Enum):
    """Direction for graph traversal.

    Determines which edges to follow:
    - OUT: Follow outgoing edges (forward)
    - IN: Follow incoming edges (backward)
    - BOTH: Follow both directions (undirected)
    """

    OUT = "out"  # Follow outgoing edges
    IN = "in"  # Follow incoming edges
    BOTH = "both"  # Follow both directions


class GraphMode(str, Enum):
    """Mode for graph analysis operations.

    Determines what type of analysis to perform:
    - find: Find nodes by query
    - neighbors: Get neighboring nodes
    - pagerank: Calculate PageRank scores
    - centrality: Calculate centrality measures
    - path: Find paths between nodes
    - impact: Analyze impact/dependencies
    - clusters: Find connected components
    - stats: Calculate graph statistics
    - subgraph: Extract subgraph
    - file_deps: File dependency analysis
    - patterns: Find code patterns
    - module_pagerank: Module-level PageRank
    - module_centrality: Module-level centrality
    - call_flow: Call flow analysis
    - callers: Find callers of a node
    - callees: Find callees of a node
    - trace: Trace execution paths
    """

    FIND = "find"  # Find nodes by query
    SEARCH = "search"  # Alias for find
    OVERVIEW = "overview"  # High-level graph summary
    NEIGHBORS = "neighbors"  # Get neighboring nodes
    PAGERANK = "pagerank"  # Calculate PageRank scores
    CENTRALITY = "centrality"  # Calculate centrality measures
    PATH = "path"  # Find paths between nodes
    IMPACT = "impact"  # Analyze impact/dependencies
    CLUSTERS = "clusters"  # Find connected components
    STATS = "stats"  # Calculate graph statistics
    SUBGRAPH = "subgraph"  # Extract subgraph
    FILE_DEPS = "file_deps"  # File dependency analysis
    PATTERNS = "patterns"  # Find code patterns
    MODULE_PAGERANK = "module_pagerank"  # Module-level PageRank
    MODULE_CENTRALITY = "module_centrality"  # Module-level centrality
    CALL_FLOW = "call_flow"  # Call flow analysis
    CALLERS = "callers"  # Find callers of a node
    CALLEES = "callees"  # Find callees of a node
    TRACE = "trace"  # Trace execution paths
    SEMANTIC = "semantic"  # Semantic relationship discovery
    QUERY = "query"  # Direct SQL query mode
    SCHEMA = "schema"  # Discover supported graph modes and relationship filters
    # NOTE: do not add a mode here without a corresponding handler in the dispatch
    # chain below. `dead_code` and `dynamic_imports` were advertised (in the schema +
    # "supported modes" error) but had no dispatch handler, so calling them passed
    # validation then hit the "Unsupported graph mode" fallback. A dynamic-import
    # tracker exists (victor/tools/graph_dynamic_import_tracker.py) and could be wired
    # as a real mode in a follow-up; dead-code analysis is unimplemented.


_GRAPH_MODE_ALIAS_NOTES = {
    "capabilities": "schema",
    "components": "clusters",
    "connected_components": "clusters",
    "connectedComponents": "clusters",
    "help": "schema",
    "hub_analysis": "overview",
    "top_k": "search (when query/node/file is provided) or pagerank",
}

_FILE_FALLBACK_DIRECTIONS: Dict[str, GraphDirection] = {
    "callers": "in",
    "callees": "out",
    "trace": "out",
    "call_flow": "out",
    "neighbors": "both",
    "impact": "both",
    "subgraph": "both",
}

ALL_EDGE_TYPES = sorted({edge_type.value for edge_type in EdgeType} | {"COMPOSED_OF"})

_EDGE_TYPE_ALIASES = {
    "call": EdgeType.CALLS.value,
    "calls": EdgeType.CALLS.value,
    "invoke": EdgeType.CALLS.value,
    "invokes": EdgeType.CALLS.value,
    "invocation": EdgeType.CALLS.value,
    "references": EdgeType.REFERENCES.value,
    "ref": EdgeType.REFERENCES.value,
    "refs": EdgeType.REFERENCES.value,
    "contains": EdgeType.CONTAINS.value,
    "containment": EdgeType.CONTAINS.value,
    "inherits": EdgeType.INHERITS.value,
    "extends": EdgeType.INHERITS.value,
    "subclasses": EdgeType.INHERITS.value,
    "implements": EdgeType.IMPLEMENTS.value,
    "imports": EdgeType.IMPORTS.value,
    "instantiates": EdgeType.INSTANTIATES.value,
    "composition": "COMPOSED_OF",
    "composed_of": "COMPOSED_OF",
    "composes": "COMPOSED_OF",
    "has_a": EdgeType.HAS_A.value,
    "isa": EdgeType.IS_A.value,
    "is_a": EdgeType.IS_A.value,
}

_RUNTIME_EDGE_TYPES = {
    EdgeType.CALLS.value,
    EdgeType.REFERENCES.value,
    EdgeType.INHERITS.value,
    EdgeType.IMPLEMENTS.value,
    EdgeType.INSTANTIATES.value,
    "COMPOSED_OF",
    EdgeType.HAS_A.value,
}

_EDGE_GROUPS: Dict[str, Set[str]] = {
    "call_flow": {EdgeType.CALLS.value},
    "runtime": _RUNTIME_EDGE_TYPES,
    "references": {EdgeType.REFERENCES.value},
    "imports": {EdgeType.IMPORTS.value},
    "type_hierarchy": {
        EdgeType.INHERITS.value,
        EdgeType.IMPLEMENTS.value,
        EdgeType.IS_A.value,
    },
    "composition": {EdgeType.CONTAINS.value, "COMPOSED_OF", EdgeType.HAS_A.value},
    "structure": {
        EdgeType.CONTAINS.value,
        EdgeType.INHERITS.value,
        EdgeType.IMPLEMENTS.value,
        EdgeType.IMPORTS.value,
        "COMPOSED_OF",
        EdgeType.HAS_A.value,
    },
    "dependencies": {
        EdgeType.CALLS.value,
        EdgeType.REFERENCES.value,
        EdgeType.IMPORTS.value,
        EdgeType.INHERITS.value,
        EdgeType.IMPLEMENTS.value,
        EdgeType.INSTANTIATES.value,
        "COMPOSED_OF",
        EdgeType.HAS_A.value,
    },
    "control_flow": EdgeType.get_cfg_edge_types(),
    "control_dependence": EdgeType.get_cdg_edge_types(),
    "data_flow": EdgeType.get_ddg_edge_types(),
    "ccg": EdgeType.get_ccg_edge_types(),
    "semantic": {
        EdgeType.SEMANTIC_SIMILAR.value,
        EdgeType.STRUCTURAL_SIMILAR.value,
        EdgeType.FUNCTIONAL_SIMILAR.value,
        EdgeType.IS_A.value,
        EdgeType.HAS_A.value,
    },
    "requirements": {
        EdgeType.SATISFIES.value,
        EdgeType.TESTS.value,
        EdgeType.DERIVES_FROM.value,
        EdgeType.REFINES.value,
        EdgeType.CONTRADICTS.value,
        EdgeType.COVERS.value,
    },
}

_EDGE_GROUP_ALIASES = {
    "calls": "call_flow",
    "callflow": "call_flow",
    "invoke": "call_flow",
    "invocation": "call_flow",
    "inheritance": "type_hierarchy",
    "hierarchy": "type_hierarchy",
    "types": "type_hierarchy",
    "containment": "composition",
    "composed_of": "composition",
    "control": "control_flow",
    "cfg": "control_flow",
    "cdg": "control_dependence",
    "ddg": "data_flow",
    "data_dependencies": "data_flow",
    "requirement": "requirements",
    "reqs": "requirements",
}
_SYMBOL_TYPES = {
    "function",
    "method",
    "class",
    "interface",
    "struct",
    "trait",
    "enum",
    "module",
}
_SYMBOL_IDENTITY_BASIS = "unique node_id (repo-relative path-qualified symbols)"


def _ctx_value(exec_ctx: Optional[Dict[str, Any]], key: str, default: Any = None) -> Any:
    if exec_ctx is None:
        return default
    if isinstance(exec_ctx, ToolExecutionContext):
        return getattr(exec_ctx, key, default)
    return exec_ctx.get(key, default)


def _build_node_required_error(mode: str, path: str = ".") -> str:
    """Build a helpful error message when node parameter is required but missing.

    Provides mode-specific usage examples to guide users.
    """
    examples = {
        "call_flow": (
            f"  graph(mode='call_flow', node='MyFunction', path='{path}')\n"
            f"  graph(mode='call_flow', node='victor_coding/my_file.py:MyClass.my_method', path='{path}')\n"
            f"  # Use node='ClassName', node='function_name', or 'path/to/file.py:symbol'"
        ),
        "callers": (
            f"  graph(mode='callers', node='MyFunction', path='{path}')\n"
            f"  graph(mode='callers', node='victor_coding/assistant.py:main', path='{path}')\n"
            f"  # Find all functions/classes that call the specified node"
        ),
        "callees": (
            f"  graph(mode='callees', node='MyFunction', path='{path}')\n"
            f"  graph(mode='callees', node='victor_coding/assistant.py:main', path='{path}')\n"
            f"  # Find all functions/classes called by the specified node"
        ),
        "trace": (
            f"  graph(mode='trace', node='MyFunction', path='{path}', depth=3)\n"
            f"  graph(mode='trace', node='victor_coding/plugin.py:entry_point', path='{path}', depth=2)\n"
            f"  # Trace execution paths from the specified node"
        ),
        "neighbors": (
            f"  graph(mode='neighbors', node='MyClass', path='{path}')\n"
            f"  graph(mode='neighbors', node='victor_coding/safety.py', path='{path}')\n"
            f"  # Find all neighbors of the specified node"
        ),
        "impact": (
            f"  graph(mode='impact', node='MyFunction', path='{path}')\n"
            f"  graph(mode='impact', node='victor_coding/codebase/indexer.py:CodebaseIndex', path='{path}')\n"
            f"  # Analyze downstream impact of changing the specified node"
        ),
        "subgraph": (
            f"  graph(mode='subgraph', node='MyModule', path='{path}', depth=2)\n"
            f"  graph(mode='subgraph', node='victor_coding', path='{path}', depth=1)\n"
            f"  # Extract a subgraph around the specified node"
        ),
        "semantic": (
            f"  graph(mode='semantic', node='MyFunction', path='{path}')\n"
            f"  graph(mode='semantic', node='victor_coding/safety.py:CodingSafetyRules', path='{path}')\n"
            f"  # Find semantically similar code to the specified node"
        ),
    }

    mode_examples = examples.get(mode, f"  graph(mode='{mode}', node='SymbolName', path='{path}')")

    return (
        f"The '{mode}' mode requires a 'node' parameter to identify the starting point.\n\n"
        f"Usage examples:\n{mode_examples}\n\n"
        f"Tips:\n"
        f"  - Use symbol names directly: node='MyClass', node='my_function'\n"
        f"  - Use file-qualified symbols: node='path/to/file.py:ClassName'\n"
        f"  - Use 'find' mode to search: graph(mode='find', query='manager', path='{path}')\n"
        f"  - Use 'overview' mode to explore: graph(mode='overview', path='{path}')"
    )


def _normalize_relpath(file_path: str) -> str:
    return file_path.replace("\\", "/").strip()


# Directory-module index filenames whose parent dir names the module
# (e.g. ``src/network/mod.rs`` -> module ``src/network``).
_MODULE_INDEX_STEMS = ("/mod", "/__init__", "/index")


def _module_path_stem(file_path: str) -> str:
    """Return a file path with its extension and directory-module suffix removed.

    Used to match Python/dotted-style module references (``src.network``) against
    real file nodes regardless of language extension:
      ``src/network/multi_server.rs`` -> ``src/network/multi_server``
      ``src/network/mod.rs``          -> ``src/network``
      ``src/lib.rs``                  -> ``src/lib``
    """
    normalized = _normalize_relpath(file_path)
    suffix = Path(normalized).suffix
    if suffix:
        normalized = normalized[: -len(suffix)]
    for tail in _MODULE_INDEX_STEMS:
        if normalized.endswith(tail):
            return normalized[: -len(tail)]
    return normalized


def _project_graph_watch_daemon_active(root_path: Path) -> bool:
    """Return True when a project-scoped graph-watch daemon is alive."""
    pid_file = get_project_paths(root_path).project_victor_dir / "graph-watch.pid"
    try:
        pid = int(pid_file.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return False

    try:
        os.kill(pid, 0)
        return True
    except PermissionError:
        return True
    except ProcessLookupError:
        return False


def _graph_mode_value(mode: str | GraphMode) -> str:
    """Return the string value for a graph mode enum or raw string."""
    if isinstance(mode, GraphMode):
        return mode.value
    return str(mode).strip()


def _normalize_graph_mode_alias(
    mode: str | GraphMode,
    *,
    node: Optional[str] = None,
    source: Optional[str] = None,
    target: Optional[str] = None,
    file: Optional[str] = None,
    query: Optional[str] = None,
) -> str:
    """Map common model-invented graph mode aliases to canonical modes."""
    raw_mode = _graph_mode_value(mode).strip().lower()
    if not raw_mode:
        return GraphMode.NEIGHBORS.value

    if "|" in raw_mode:
        normalized_parts = [
            _normalize_graph_mode_alias(
                part,
                node=node,
                source=source,
                target=target,
                file=file,
                query=query,
            )
            for part in raw_mode.split("|")
            if part.strip()
        ]
        return "|".join(normalized_parts)

    if raw_mode == "hub_analysis":
        return GraphMode.OVERVIEW.value
    if raw_mode in {"schema", "capabilities", "help", "usage"}:
        return GraphMode.SCHEMA.value
    if raw_mode in {
        "components",
        "component",
        "connected_components",
        "connectedcomponents",
    }:
        return GraphMode.CLUSTERS.value
    if raw_mode == "top_k":
        if query or node or file:
            return GraphMode.SEARCH.value
        return GraphMode.PAGERANK.value

    return raw_mode


def _normalize_edge_token(value: Any) -> str:
    """Normalize a user/model-provided edge token to snake-ish lowercase."""
    if isinstance(value, Enum):
        value = value.value
    return str(value).strip().replace("-", "_").replace(" ", "_").lower()


def _normalize_edge_group_name(edge_group: Optional[str]) -> Optional[str]:
    """Normalize relationship group aliases to canonical edge group names."""
    if edge_group is None:
        return None
    token = _normalize_edge_token(edge_group)
    if not token:
        return None
    if token in {"all", "any", "*"}:
        return "all"
    canonical = _EDGE_GROUP_ALIASES.get(token, token)
    if canonical not in _EDGE_GROUPS:
        supported = ", ".join(sorted(_EDGE_GROUPS))
        raise ValueError(f"Unsupported edge_group: {edge_group}. Supported: all, {supported}")
    return canonical


def _edge_group_types(edge_group: Optional[str]) -> Optional[List[str]]:
    """Resolve an edge group name to concrete edge types.

    Returns None for no filter or the explicit all/any group.
    """
    canonical = _normalize_edge_group_name(edge_group)
    if canonical is None or canonical == "all":
        return None
    return sorted(_EDGE_GROUPS[canonical])


def _coerce_edge_type_items(edge_types: Optional[Any]) -> List[Any]:
    """Coerce edge_types from list or model-generated scalar/comma-separated forms."""
    if edge_types is None:
        return []
    if isinstance(edge_types, str):
        separators = "|", ","
        items = [edge_types]
        for separator in separators:
            items = [part for item in items for part in item.split(separator)]
        return items
    if isinstance(edge_types, IterableABC):
        return list(edge_types)
    return [edge_types]


def _normalize_edge_types(edge_types: Optional[Any]) -> Optional[List[str]]:
    """Normalize edge type aliases and embedded group names to concrete edge types."""
    normalized: Set[str] = set()
    for item in _coerce_edge_type_items(edge_types):
        token = _normalize_edge_token(item)
        if not token:
            continue
        canonical_group = _EDGE_GROUP_ALIASES.get(token, token)
        if canonical_group in _EDGE_GROUPS:
            normalized.update(_EDGE_GROUPS[canonical_group])
            continue
        if token in {"all", "any", "*"}:
            return None
        edge_type = _EDGE_TYPE_ALIASES.get(token)
        if edge_type is None:
            upper_token = token.upper()
            edge_type = upper_token if upper_token in ALL_EDGE_TYPES else upper_token
        normalized.add(edge_type)
    return sorted(normalized) if normalized else None


def _resolve_effective_edge_types(
    *,
    edge_types: Optional[Any],
    edge_group: Optional[str],
    default_edge_types: Optional[List[str]],
) -> Optional[List[str]]:
    """Resolve explicit edge types, group presets, and mode defaults.

    Explicit edge_types are exact overrides. edge_group is the ergonomic preset.
    Mode defaults remain the fallback for modes like callers/callees/trace.
    """
    normalized_edge_types = _normalize_edge_types(edge_types)
    if normalized_edge_types is not None:
        return normalized_edge_types

    grouped_edge_types = _edge_group_types(edge_group)
    if grouped_edge_types is not None:
        return grouped_edge_types

    return _normalize_edge_types(default_edge_types)


def _format_tool_command_value(value: Any) -> str:
    """Format a tool argument value for a follow-up command string."""
    if isinstance(value, Enum):
        value = value.value
    if isinstance(value, str):
        return json.dumps(value)
    return repr(value)


def _build_tool_follow_up_command(tool_name: str, arguments: Dict[str, Any]) -> str:
    """Build a compact follow-up command string."""
    serialized_args = ", ".join(
        f"{key}={_format_tool_command_value(value)}"
        for key, value in arguments.items()
        if value is not None
    )
    return f"{tool_name}({serialized_args})"


def _build_follow_up_suggestion(
    tool_name: str,
    arguments: Dict[str, Any],
    description: str,
) -> Dict[str, Any]:
    """Create a normalized follow-up suggestion payload."""
    return {
        "tool": tool_name,
        "command": _build_tool_follow_up_command(tool_name, arguments),
        "arguments": arguments,
        "description": description,
        "reason": description,
    }


def _build_graph_error_follow_up_suggestions(
    *,
    path: str,
    requested_mode: str,
    normalized_mode: str,
    node: Optional[str],
    source: Optional[str],
    target: Optional[str],
    file: Optional[str],
    query: Optional[str],
    depth: int,
    top_k: int,
    unavailable: bool = False,
    unresolved_node: Optional[str] = None,
    empty_database: bool = False,
) -> List[Dict[str, Any]]:
    """Build structured follow-up suggestions for recoverable graph failures."""
    suggestions: List[Dict[str, Any]] = []
    search_seed = unresolved_node or query or node or source or target or file
    effective_top_k = max(1, min(top_k, 10))

    # Handle empty database - suggest reindex
    if empty_database:
        suggestions.append(
            _build_follow_up_suggestion(
                "graph",
                {
                    "mode": normalized_mode,
                    "path": path,
                    "reindex": True,
                    "top_k": top_k,
                },
                "Build the graph index first with reindex=True.",
            )
        )
        # Also suggest overview as an alternative
        suggestions.append(
            _build_follow_up_suggestion(
                "ls",
                {"path": path, "depth": 2},
                "List directory structure to understand the codebase layout.",
            )
        )
        return suggestions[:2]

    if unavailable:
        suggestions.append(
            _build_follow_up_suggestion(
                "project_overview",
                {"path": path, "max_depth": 2},
                "Inspect the project structure without relying on the graph index.",
            )
        )
        if search_seed:
            suggestions.append(
                _build_follow_up_suggestion(
                    "code_search",
                    {
                        "query": search_seed,
                        "path": path,
                        "mode": "text",
                        "k": effective_top_k,
                    },
                    f'Search code textually for "{search_seed}" while graph indexing is unavailable.',
                )
            )
        return suggestions

    if normalized_mode != requested_mode:
        canonical_args: Dict[str, Any] = {"mode": normalized_mode, "path": path}
        if normalized_mode in {"search", "find"}:
            canonical_args["query"] = search_seed
            canonical_args["top_k"] = effective_top_k
        elif normalized_mode in {
            "neighbors",
            "callers",
            "callees",
            "trace",
            "call_flow",
            "impact",
            "subgraph",
        }:
            canonical_args["node"] = node or source or target
            canonical_args["depth"] = depth
        elif normalized_mode == "file_deps":
            canonical_args["file"] = file
        elif normalized_mode not in {"stats"}:
            canonical_args["top_k"] = effective_top_k
        suggestions.append(
            _build_follow_up_suggestion(
                "graph",
                canonical_args,
                f'Use the supported graph mode "{normalized_mode}" instead of "{requested_mode}".',
            )
        )

    if search_seed:
        # Dotted module references (``src.network.multi_server``) rarely match
        # path-based file nodes; seed the search with a path-style variant so the
        # follow-up actually resolves on non-Python repos.
        search_query = search_seed
        if "." in search_seed and "/" not in search_seed and ":" not in search_seed:
            search_query = search_seed.replace(".", "/")
        suggestions.append(
            _build_follow_up_suggestion(
                "graph",
                {
                    "mode": "search",
                    "query": search_query,
                    "path": path,
                    "top_k": effective_top_k,
                },
                f'Search the graph index for matches to "{search_query}".',
            )
        )
    elif not suggestions:
        suggestions.append(
            _build_follow_up_suggestion(
                "graph",
                {"mode": "overview", "path": path, "top_k": effective_top_k},
                "Start with a supported graph overview of the codebase.",
            )
        )

    return suggestions[:2]


def _graph_error_response(
    *,
    requested_mode: str,
    mode: str,
    error: str,
    suggestions: Optional[List[Dict[str, Any]]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a structured graph error response."""
    response: Dict[str, Any] = {
        "success": False,
        "mode": mode,
        "requested_mode": requested_mode,
        "error": error,
    }
    if suggestions:
        response["metadata"] = {"follow_up_suggestions": suggestions}
    if extra:
        response.update(extra)
    return response


def _unsupported_graph_mode_error(mode: str) -> str:
    """Build a user/model-facing unsupported mode error with recovery hints."""
    supported_modes = ", ".join(sorted(graph_mode.value for graph_mode in GraphMode))
    alias_notes = "; ".join(
        f"{alias} -> {target}" for alias, target in sorted(_GRAPH_MODE_ALIAS_NOTES.items())
    )
    return (
        f"Unsupported graph mode: {mode}. "
        f"Supported modes: {supported_modes}. "
        f"Common aliases: {alias_notes}."
    )


def _build_graph_schema_result() -> Dict[str, Any]:
    """Return compact graph-tool capability metadata without loading a graph."""
    return {
        "modes": sorted(graph_mode.value for graph_mode in GraphMode),
        "mode_aliases": dict(sorted(_GRAPH_MODE_ALIAS_NOTES.items())),
        "edge_types": ALL_EDGE_TYPES,
        "edge_type_aliases": dict(sorted(_EDGE_TYPE_ALIASES.items())),
        "edge_groups": {
            group: sorted(edge_types) for group, edge_types in sorted(_EDGE_GROUPS.items())
        },
        "edge_group_aliases": dict(sorted(_EDGE_GROUP_ALIASES.items())),
        "precedence": "edge_types overrides edge_group; edge_group overrides mode defaults",
        "examples": [
            'graph(mode="search", query="AgentOrchestrator", top_k=5)',
            'graph(mode="subgraph", query="singleton", path="victor/agent", depth=2)',
            'graph(mode="connectedComponents", edge_group="type_hierarchy", top_k=10)',
            'graph(mode="neighbors", node="AgentOrchestrator", edge_group="call_flow", depth=2)',
            'graph(mode="path", source="start", target="finish", edge_types=["CALLS"])',
        ],
    }


def _module_name_from_file(file_path: str) -> str:
    normalized = _normalize_relpath(file_path)
    path = Path(normalized)
    if path.name == "__init__.py":
        parts = path.parent.parts
    else:
        parts = path.with_suffix("").parts
    return ".".join(part for part in parts if part and part != ".")


def _module_name_for_node(node: GraphNode) -> Optional[str]:
    if node.type == "module":
        return node.name
    if node.type == "stdlib_module":
        return None
    if node.file:
        return _module_name_from_file(node.file)
    return None


def _qualified_name_for_node(node: GraphNode) -> str:
    metadata_name = node.metadata.get("qualified_name")
    if isinstance(metadata_name, str) and metadata_name:
        return metadata_name

    try:
        unified_id = UnifiedId.from_string(node.node_id)
    except ValueError:
        unified_id = None

    if unified_id is not None:
        if unified_id.type == "file":
            return unified_id.path
        if unified_id.name:
            return f"{unified_id.path}:{unified_id.name}"
        if unified_id.path:
            return unified_id.path

    module_name = _module_name_for_node(node)
    if node.type in {"file", "module"}:
        return node.file or module_name or node.name
    if node.file:
        return f"{node.file}:{node.name}"
    return module_name or node.name


def _node_payload(node: GraphNode, **extra: Any) -> Dict[str, Any]:
    payload = {
        "node_id": node.node_id,
        "name": node.name,
        "qualified_name": _qualified_name_for_node(node),
        "type": node.type,
        "file": node.file,
        "module": _module_name_for_node(node),
        "line": node.line,
        "end_line": node.end_line,
        "lang": node.lang,
    }
    payload.update(extra)
    return payload


@dataclass
class LoadedGraph:
    root_path: Path
    index: Any
    graph_store: GraphStoreProtocol
    analyzer: "GraphAnalyzer"
    rebuilt: bool


class GraphAnalyzer:
    """In-memory graph analytics helper backed by GraphNode/GraphEdge objects."""

    def __init__(self) -> None:
        self.nodes: Dict[str, GraphNode] = {}
        self.outgoing: DefaultDict[str, List[GraphEdge]] = defaultdict(list)
        self.incoming: DefaultDict[str, List[GraphEdge]] = defaultdict(list)
        self._name_index: DefaultDict[str, List[str]] = defaultdict(list)
        self._file_index: DefaultDict[str, List[str]] = defaultdict(list)

    def add_node(self, node: GraphNode) -> None:
        self.nodes[node.node_id] = node
        self._name_index[node.name.lower()].append(node.node_id)
        self._file_index[_normalize_relpath(node.file)].append(node.node_id)

    def add_edge(self, edge: GraphEdge) -> None:
        self.outgoing[edge.src].append(edge)
        self.incoming[edge.dst].append(edge)

    def resolve_node_id(
        self,
        reference: str,
        *,
        preferred_types: Optional[Set[str]] = None,
    ) -> Optional[str]:
        if reference in self.nodes:
            return reference

        normalized = reference.strip().strip("`'\"")
        lowered = normalized.lower()
        compound_matches = self._resolve_compound_reference(
            normalized,
            lowered=lowered,
            preferred_types=preferred_types,
        )
        if compound_matches:
            return compound_matches[0]
        candidates = list(self._name_index.get(lowered, ()))

        normalized_file = _normalize_relpath(normalized)
        if not candidates:
            candidates.extend(self._file_index.get(normalized_file, ()))
            if not candidates:
                for file_path, node_ids in self._file_index.items():
                    if file_path.endswith(normalized_file):
                        candidates.extend(node_ids)

        if not candidates:
            for node in self.nodes.values():
                if lowered in node.name.lower() or normalized_file in _normalize_relpath(node.file):
                    candidates.append(node.node_id)

        if not candidates:
            candidates.extend(self._resolve_dotted_reference(normalized, lowered=lowered))

        if not candidates:
            return None

        def _score(node_id: str) -> tuple[int, int, str]:
            node = self.nodes[node_id]
            preferred = 1 if preferred_types and node.type in preferred_types else 0
            exact_name = 1 if node.name.lower() == lowered else 0
            return (-preferred, -exact_name, node.node_id)

        candidates = sorted(dict.fromkeys(candidates), key=_score)
        return candidates[0]

    def _resolve_compound_reference(
        self,
        normalized: str,
        *,
        lowered: str,
        preferred_types: Optional[Set[str]] = None,
    ) -> List[str]:
        """Resolve references like ``path/to/file.py:SymbolName``."""
        if ":" not in normalized:
            return []

        file_ref, _, symbol_ref = normalized.rpartition(":")
        if not file_ref or not symbol_ref:
            return []

        normalized_file = _normalize_relpath(file_ref)
        symbol_lower = symbol_ref.strip().lower()
        if not symbol_lower:
            return []

        matches: List[str] = []
        basename_matches: List[str] = []
        file_base = Path(normalized_file).name
        for node in self.nodes.values():
            if node.name.lower() != symbol_lower:
                continue
            node_file = _normalize_relpath(node.file)
            if node_file == normalized_file or node_file.endswith(normalized_file):
                matches.append(node.node_id)
            elif file_base and node_file and Path(node_file).name == file_base:
                basename_matches.append(node.node_id)

        # Fall back to a basename-only match (correct symbol, differently-formatted file
        # path) only when it is unambiguous — otherwise we'd risk resolving to a
        # same-named symbol in a different directory.
        if not matches and len(basename_matches) == 1:
            matches = basename_matches

        if preferred_types:
            matches.sort(
                key=lambda node_id: (
                    0 if self.nodes[node_id].type in preferred_types else 1,
                    node_id,
                )
            )
        else:
            matches.sort()
        return matches

    def _resolve_dotted_reference(self, normalized: str, *, lowered: str) -> List[str]:
        """Resolve Python/dotted module references against path-based file nodes.

        Models frequently address files with dotted module syntax
        (``src.network.multi_server``) even in non-Python repos where the graph
        stores path-based file nodes (``src/network/multi_server.rs``). Convert
        the dotted reference to a slashed stem and match file nodes
        language-agnostically; fall back to the last segment as a symbol name.
        """
        if "." not in normalized or "/" in normalized or ":" in normalized:
            return []

        slashed = _normalize_relpath(normalized.replace(".", "/"))
        if not slashed:
            return []

        candidates: List[str] = []
        for file_path, node_ids in self._file_index.items():
            stem = _module_path_stem(file_path)
            if stem == slashed or stem.endswith("/" + slashed):
                candidates.extend(node_ids)

        if not candidates:
            last_segment = normalized.rsplit(".", 1)[-1].strip().lower()
            if last_segment and last_segment != lowered:
                candidates.extend(self._name_index.get(last_segment, ()))

        return candidates

    def search(
        self,
        query: str,
        *,
        node_types: Optional[Set[str]] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        lowered = query.lower()
        results: List[Dict[str, Any]] = []
        for node in self.nodes.values():
            if node_types is not None and node.type not in node_types:
                continue
            score = 0
            if node.name.lower() == lowered:
                score += 5
            elif lowered in node.name.lower():
                score += 3
            if lowered in _normalize_relpath(node.file):
                score += 2
            if score == 0:
                continue
            results.append(_node_payload(node, score=score))
        results.sort(key=lambda item: (-item["score"], item["file"], item["line"] or 0))
        return results[:limit]

    def _iter_edges(
        self,
        node_id: str,
        *,
        direction: GraphDirection,
        edge_types: Optional[Set[str]],
    ) -> Iterable[tuple[GraphEdge, str, GraphDirection]]:
        if direction in {"out", "both"}:
            for edge in self.outgoing.get(node_id, ()):
                if edge_types is None or edge.type in edge_types:
                    yield edge, edge.dst, "out"
        if direction in {"in", "both"}:
            for edge in self.incoming.get(node_id, ()):
                if edge_types is None or edge.type in edge_types:
                    yield edge, edge.src, "in"

    def get_neighbors(
        self,
        node_id: str,
        *,
        direction: GraphDirection = GraphDirection.OUT,
        edge_types: Optional[Iterable[str]] = None,
        max_depth: int = 1,
        node_types: Optional[Set[str]] = None,
    ) -> Dict[str, Any]:
        if node_id not in self.nodes:
            return {
                "source": node_id,
                "direction": direction,
                "max_depth": max_depth,
                "total_neighbors": 0,
                "neighbors_by_depth": {},
                "message": f"Node '{node_id}' not found",
            }

        allowed_edges = set(edge_types) if edge_types else None
        queue: deque[tuple[str, int]] = deque([(node_id, 0)])
        visited = {node_id}
        neighbors_by_depth: Dict[int, List[Dict[str, Any]]] = {}

        while queue:
            current, depth = queue.popleft()
            if depth >= max_depth:
                continue

            for edge, neighbor_id, edge_direction in self._iter_edges(
                current,
                direction=direction,
                edge_types=allowed_edges,
            ):
                if neighbor_id in visited:
                    continue
                neighbor = self.nodes.get(neighbor_id)
                if neighbor is None:
                    continue
                if node_types is not None and neighbor.type not in node_types:
                    continue
                visited.add(neighbor_id)
                queue.append((neighbor_id, depth + 1))
                neighbors_by_depth.setdefault(depth + 1, []).append(
                    _node_payload(
                        neighbor,
                        edge_type=edge.type,
                        direction=edge_direction,
                        via=current,
                    )
                )

        return {
            "source": node_id,
            "direction": direction,
            "max_depth": max_depth,
            "total_neighbors": sum(len(items) for items in neighbors_by_depth.values()),
            "neighbors_by_depth": neighbors_by_depth,
        }

    def pagerank(
        self,
        *,
        edge_types: Optional[Iterable[str]] = None,
        iterations: int = 20,
        top_k: int = 10,
        node_types: Optional[Set[str]] = None,
    ) -> List[Dict[str, Any]]:
        if not self.nodes:
            return []

        allowed_edges = set(edge_types) if edge_types else None
        allowed_nodes = {
            node_id
            for node_id, node in self.nodes.items()
            if node_types is None or node.type in node_types
        }
        if not allowed_nodes:
            return []

        adjacency = {node_id: [] for node_id in allowed_nodes}
        for node_id in allowed_nodes:
            for edge in self.outgoing.get(node_id, ()):
                if allowed_edges is not None and edge.type not in allowed_edges:
                    continue
                if edge.dst in allowed_nodes:
                    adjacency[node_id].append(edge.dst)

        scores = pagerank(adjacency, iterations=iterations)
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:top_k]
        results: List[Dict[str, Any]] = []
        for rank, (node_id, score) in enumerate(ranked, start=1):
            node = self.nodes[node_id]
            results.append(_node_payload(node, rank=rank, score=score))
        return results

    def degree_centrality(
        self,
        *,
        top_k: int = 10,
        edge_types: Optional[Iterable[str]] = None,
        node_types: Optional[Set[str]] = None,
    ) -> List[Dict[str, Any]]:
        if not self.nodes:
            return []

        allowed_edges = set(edge_types) if edge_types else None
        allowed_nodes = {
            node_id
            for node_id, node in self.nodes.items()
            if node_types is None or node.type in node_types
        }
        if not allowed_nodes:
            return []

        results: List[Dict[str, Any]] = []
        for node_id in allowed_nodes:
            node = self.nodes[node_id]
            out_degree = sum(
                1
                for edge in self.outgoing.get(node_id, ())
                if (allowed_edges is None or edge.type in allowed_edges)
                and edge.dst in allowed_nodes
            )
            in_degree = sum(
                1
                for edge in self.incoming.get(node_id, ())
                if (allowed_edges is None or edge.type in allowed_edges)
                and edge.src in allowed_nodes
            )
            results.append(
                _node_payload(
                    node,
                    degree=in_degree + out_degree,
                    in_degree=in_degree,
                    out_degree=out_degree,
                )
            )

        results.sort(key=lambda item: (-item["degree"], item["node_id"]))
        for rank, item in enumerate(results[:top_k], start=1):
            item["rank"] = rank
        return results[:top_k]

    def shortest_path(
        self,
        source: str,
        target: str,
        *,
        edge_types: Optional[Iterable[str]] = None,
    ) -> Dict[str, Any]:
        if source not in self.nodes or target not in self.nodes:
            return {
                "found": False,
                "source": source,
                "target": target,
                "message": "Source or target node not found",
            }

        allowed_edges = set(edge_types) if edge_types else None
        queue: deque[str] = deque([source])
        previous: Dict[str, Optional[str]] = {source: None}

        while queue:
            current = queue.popleft()
            if current == target:
                break
            for edge in self.outgoing.get(current, ()):
                if allowed_edges is not None and edge.type not in allowed_edges:
                    continue
                if edge.dst in previous:
                    continue
                previous[edge.dst] = current
                queue.append(edge.dst)

        if target not in previous:
            return {
                "found": False,
                "source": source,
                "target": target,
                "message": f"No path from '{source}' to '{target}'",
            }

        path_ids: List[str] = []
        current: Optional[str] = target
        while current is not None:
            path_ids.append(current)
            current = previous[current]
        path_ids.reverse()

        return {
            "found": True,
            "source": source,
            "target": target,
            "path": [_node_payload(self.nodes[node_id]) for node_id in path_ids],
        }


def _resolve_root_path(path: str) -> Path:
    search_root = path
    if not search_root or search_root == ".":
        search_root = str(get_project_paths().project_root)
    root_path = Path(search_root).resolve()
    if root_path.is_file():
        return root_path.parent
    return root_path


def _current_project_root() -> Path:
    return Path(get_project_paths().project_root).resolve()


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _should_reuse_project_graph_root(
    *,
    requested_path: str,
    requested_mode: str,
    node: Optional[str],
    source: Optional[str],
    target: Optional[str],
    file: Optional[str],
    query: Optional[str],
) -> bool:
    if requested_mode in {GraphMode.OVERVIEW.value, GraphMode.FILE_DEPS.value}:
        return False
    if file:
        return False

    project_root = _current_project_root()
    requested_subject = _resolve_requested_subject_path(requested_path)
    if requested_subject == project_root or not _is_relative_to(requested_subject, project_root):
        return False

    if requested_subject.is_file():
        return True

    return any(value for value in (node, source, target, query))


def _has_enhanced_codebase_index_provider() -> bool:
    from victor_contracts.capability_runtime import (
        CodebaseIndexFactoryProtocol,
        get_capability_provider,
        is_capability_enhanced,
    )

    factory = get_capability_provider(CodebaseIndexFactoryProtocol)
    return factory is not None and is_capability_enhanced(CodebaseIndexFactoryProtocol)


def _resolved_graph_backend(root_path: Path) -> str:
    """The repo's graph backend per its .victor/graph_backend marker."""
    try:
        from victor.storage.graph.registry import resolve_graph_backend

        return resolve_graph_backend(root_path, default="sqlite")
    except Exception:  # pragma: no cover - victor-ai absent
        return "sqlite"


def _project_graph_has_data(root_path: Path) -> bool:
    # A repo flipped to a non-sqlite backend stores its graph THERE — probing
    # project.db's sqlite tables would report "empty" for a fully-indexed
    # repo and silently drop the graph tool from the LLM schema. Advertise
    # the tool and let the load path construct the real store (an actually
    # empty store raises the same self-help reindex message either way).
    if _resolved_graph_backend(root_path) != "sqlite":
        return True

    from victor_contracts.database_runtime import get_project_database

    project_db = get_project_database(root_path)
    try:
        _ensure_project_graph_tables(project_db)
    except RuntimeError:
        return False

    node_row = project_db.query_one("SELECT COUNT(*) FROM graph_node")
    edge_row = project_db.query_one("SELECT COUNT(*) FROM graph_edge")
    node_count = int(node_row[0]) if node_row is not None else 0
    edge_count = int(edge_row[0]) if edge_row is not None else 0
    return node_count > 0 or edge_count > 0


def _ensure_project_graph_tables(project_db: Any) -> None:
    """Ensure project graph tables exist, raise error if not.

    Args:
        project_db: Project database instance

    Raises:
        RuntimeError: If graph_node or graph_edge tables don't exist
    """
    if not project_db.table_exists("graph_node") or not project_db.table_exists("graph_edge"):
        raise RuntimeError("Project graph tables are unavailable")


def _graph_tool_is_available() -> bool:
    try:
        if _has_enhanced_codebase_index_provider():
            return True
    except Exception:
        logger.debug("[graph] Provider availability check failed", exc_info=True)

    try:
        return _project_graph_has_data(_resolve_root_path("."))
    except Exception:
        return False


def _resolve_requested_subject_path(path: str) -> Path:
    requested = Path(path)
    if requested.is_absolute():
        return requested.resolve()
    return (Path(get_project_paths().project_root) / requested).resolve()


def _recover_file_deps_without_file(
    *,
    loaded: LoadedGraph,
    requested_path: str,
    requested_mode: str,
    direction: GraphDirection,
    structured: bool,
    include_modules: bool,
    top_k: int,
    effective_edge_types: Optional[List[str]],
    only_runtime: bool,
    include_callsites: bool,
    max_callsites: int,
) -> tuple[str, Dict[str, Any]]:
    """Recover file dependency requests when models overload `path` as the subject."""
    if not requested_path or requested_path == ".":
        result = _build_overview(
            loaded,
            top_k=top_k,
            effective_edge_types=effective_edge_types,
            only_runtime=only_runtime,
            include_callsites=include_callsites,
            max_callsites=max_callsites,
        )
        result["recovered_from_mode"] = requested_mode
        result["recovered_from_path"] = requested_path or "."
        result["recovery_reason"] = "file_deps_without_file"
        return "overview", result

    requested_location = _resolve_requested_subject_path(requested_path)
    requested_ref = Path(requested_path)
    if requested_ref.suffix:
        file_ref = requested_path
        if requested_location.is_relative_to(loaded.root_path):
            file_ref = requested_location.relative_to(loaded.root_path).as_posix()
        else:
            parent_ref = requested_ref.parent.as_posix()
            if parent_ref and loaded.root_path.as_posix().endswith(parent_ref):
                file_ref = requested_ref.name

        result = _build_file_dependency_result(
            loaded,
            file_ref,
            direction=direction,
            structured=structured,
            include_modules=include_modules,
        )
        result["recovered_from_mode"] = requested_mode
        result["recovered_from_path"] = requested_path
        return "file_deps", result

    result = _build_overview(
        loaded,
        top_k=top_k,
        effective_edge_types=effective_edge_types,
        only_runtime=only_runtime,
        include_callsites=include_callsites,
        max_callsites=max_callsites,
    )
    result["recovered_from_mode"] = requested_mode
    result["recovered_from_path"] = requested_path
    return "overview", result


async def _materialize_loaded_graph(
    root_path: Path,
    *,
    index: Any,
    graph_store: GraphStoreProtocol,
    rebuilt: bool,
) -> LoadedGraph:
    get_all_nodes = getattr(graph_store, "get_all_nodes", None)
    if callable(get_all_nodes):
        nodes = await get_all_nodes()
    else:
        nodes = await graph_store.find_nodes()
    edges = await graph_store.get_all_edges()

    analyzer = GraphAnalyzer()
    for node in nodes:
        analyzer.add_node(node)
    for edge in edges:
        analyzer.add_edge(edge)

    return LoadedGraph(
        root_path=root_path,
        index=index,
        graph_store=graph_store,
        analyzer=analyzer,
        rebuilt=rebuilt,
    )


async def _load_graph_from_project_store(root_path: Path) -> LoadedGraph:
    from victor_contracts.database_runtime import get_project_database
    from victor.storage.graph.sqlite_store import SqliteGraphStore

    # Non-sqlite backends: the project store IS the marker-resolved store —
    # project.db's sqlite tables are not where this repo's graph lives.
    if _resolved_graph_backend(root_path) != "sqlite":
        from victor_coding.codebase.graph.registry import (
            create_graph_store as _vc_create_graph_store,
        )

        graph_store = _vc_create_graph_store("auto", str(root_path))
        if graph_store is None:
            raise RuntimeError(
                f"Graph backend for '{root_path}' could not be constructed "
                "(victor-ai not installed?)"
            )
        await graph_store.initialize()
        nodes = await graph_store.get_all_nodes()
        if not nodes:
            path_str = str(root_path)
            raise RuntimeError(
                f"Project graph store is empty for path '{path_str}'. "
                f"To build the index: graph(mode='stats', path='{path_str}', reindex=True). "
                f"Or use ls(path='{path_str}', depth=2) for file operations."
            )
        fallback_index = SimpleNamespace(graph_store=graph_store, files={})
        return await _materialize_loaded_graph(
            root_path,
            index=fallback_index,
            graph_store=graph_store,
            rebuilt=False,
        )

    project_db = get_project_database(root_path)
    _ensure_project_graph_tables(project_db)

    node_row = project_db.query_one("SELECT COUNT(*) FROM graph_node")
    edge_row = project_db.query_one("SELECT COUNT(*) FROM graph_edge")
    node_count = int(node_row[0]) if node_row is not None else 0
    edge_count = int(edge_row[0]) if edge_row is not None else 0
    if node_count == 0 and edge_count == 0:
        path_str = str(root_path)
        raise RuntimeError(
            f"Project graph database is empty for path '{path_str}'. "
            f"To build the index: graph(mode='stats', path='{path_str}', reindex=True). "
            f"Or use ls(path='{path_str}', depth=2) for file operations."
        )

    graph_store = SqliteGraphStore(root_path)
    fallback_index = SimpleNamespace(graph_store=graph_store, files={})
    return await _materialize_loaded_graph(
        root_path,
        index=fallback_index,
        graph_store=graph_store,
        rebuilt=False,
    )


async def _load_graph(
    path: str = ".",
    *,
    reindex: bool = False,
    exec_ctx: Optional[Dict[str, Any]] = None,
) -> LoadedGraph:
    """Load graph data with fast-path: use project database immediately, rebuild index asynchronously.

    Priority order:
    1. Project graph database (fast, non-blocking)
    2. CodebaseIndex with graph_store (if available)
    3. Fallback to project database (if index unavailable)

    Index rebuilds happen asynchronously in the background and don't block queries.
    """
    root_path = _resolve_root_path(path)
    settings = _ctx_value(exec_ctx, "settings")
    if settings is None:
        settings = load_settings()

    # Fast path: Check if project graph database exists and has data
    # This is non-blocking and serves from existing structural bridge data
    if not reindex and _project_graph_has_data(root_path):
        try:
            logger.info("[graph] Using existing project graph database (fast path)")
            return await _load_graph_from_project_store(root_path)
        except Exception as exc:
            logger.debug("[graph] Project database load failed, falling back to index: %s", exc)

    # Secondary path: Try CodebaseIndex if available
    # Only blocks if reindex=True or project database doesn't exist
    try:
        index, rebuilt = await _get_or_build_index(
            root_path,
            settings,
            force_reindex=reindex,
            exec_ctx=exec_ctx,
        )
        graph_store = getattr(index, "graph_store", None)
        if graph_store is not None:
            return await _materialize_loaded_graph(
                root_path,
                index=index,
                graph_store=graph_store,
                rebuilt=rebuilt,
            )
    except (ImportError, RuntimeError, ValueError) as exc:
        logger.debug("[graph] CodebaseIndex unavailable, trying project database: %s", exc)

    # Fallback: Project database (even if empty, better than nothing)
    try:
        loaded = await _load_graph_from_project_store(root_path)
        logger.info("[graph] Loaded persisted project graph for %s (fallback)", root_path)
        return loaded
    except Exception as exc:
        raise ValueError(
            f"Graph data unavailable. CodebaseIndex failed and project database is empty: {exc}"
        ) from exc


def _default_edge_types(mode: GraphMode, *, only_runtime: bool) -> Optional[List[str]]:
    if mode in {"callers", "callees", "trace", "call_flow"}:
        return ["CALLS"]
    if only_runtime:
        return sorted(_RUNTIME_EDGE_TYPES)
    return None


def _node_type_filter(
    mode: GraphMode,
    *,
    files_only: bool,
    modules_only: bool,
) -> Optional[Set[str]]:
    if files_only:
        return {"file"}
    if modules_only:
        return {"module"}
    if mode in {"pagerank", "centrality"}:
        return set(_SYMBOL_TYPES)
    return None


def _flatten_neighbors(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for neighbors in result.get("neighbors_by_depth", {}).values():
        items.extend(neighbors)
    return items


def _build_structured_neighbors(
    analyzer: GraphAnalyzer,
    focus_id: str,
    base_result: Dict[str, Any],
    *,
    include_modules: bool,
    include_symbols: bool,
    include_calls: bool,
    include_refs: bool,
    include_callsites: bool,
    max_callsites: int,
) -> Dict[str, Any]:
    focus = analyzer.nodes[focus_id]
    flattened = _flatten_neighbors(base_result)

    structured = {
        "focus": _node_payload(focus),
        "neighborhood": base_result,
    }
    if include_modules:
        modules = {
            module_name
            for item in [structured["focus"], *flattened]
            for module_name in [_module_name_from_file(item["file"])]
            if module_name
        }
        structured["modules"] = sorted(modules)
    if include_symbols:
        structured["symbols"] = [
            item for item in flattened if item["type"] not in {"file", "module", "stdlib_module"}
        ]
    if include_calls or include_callsites:
        structured["calls"] = [item for item in flattened if item["edge_type"] == "CALLS"][
            :max_callsites
        ]
    if include_refs:
        structured["references"] = [
            item for item in flattened if item["edge_type"] == "REFERENCES"
        ][:max_callsites]
    return structured


def _project_module_adjacency(
    analyzer: GraphAnalyzer,
    *,
    only_runtime: bool,
    include_callsites: bool,
    max_callsites: int,
) -> Dict[str, Any]:
    adjacency: Dict[str, Dict[str, int]] = {}
    callsites: Dict[tuple[str, str], List[Dict[str, Any]]] = {}

    for node in analyzer.nodes.values():
        module_name = _module_name_for_node(node)
        if module_name:
            adjacency.setdefault(module_name, {})

    for edges in analyzer.outgoing.values():
        for edge in edges:
            if only_runtime and edge.type not in _RUNTIME_EDGE_TYPES:
                continue
            src_node = analyzer.nodes.get(edge.src)
            dst_node = analyzer.nodes.get(edge.dst)
            if src_node is None or dst_node is None:
                continue
            src_module = _module_name_for_node(src_node)
            dst_module = _module_name_for_node(dst_node)
            if not src_module or not dst_module or src_module == dst_module:
                continue
            adjacency.setdefault(src_module, {})
            adjacency.setdefault(dst_module, {})
            adjacency[src_module][dst_module] = adjacency[src_module].get(dst_module, 0) + 1
            if include_callsites:
                callsites.setdefault((src_module, dst_module), []).append(
                    {
                        "src_node": edge.src,
                        "dst_node": edge.dst,
                        "edge_type": edge.type,
                        "src_file": src_node.file,
                        "dst_file": dst_node.file,
                    }
                )

    projected = {"adjacency": adjacency}
    if include_callsites:
        projected["callsites"] = {
            f"{src}->{dst}": samples[:max_callsites] for (src, dst), samples in callsites.items()
        }
    return projected


def _rank_projected_modules(
    adjacency: Dict[str, Dict[str, int]],
    *,
    mode: Literal["pagerank", "centrality"],
    top_k: int,
) -> List[Dict[str, Any]]:
    if not adjacency:
        return []

    if mode == "pagerank":
        scores = weighted_pagerank(adjacency, iterations=25)
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:top_k]
        return [
            {"rank": rank, "module": module, "score": score}
            for rank, (module, score) in enumerate(ranked, start=1)
        ]

    degree_rows: List[Dict[str, Any]] = []
    incoming: Dict[str, int] = dict.fromkeys(adjacency, 0)
    for src, neighbors in adjacency.items():
        for dst, weight in neighbors.items():
            incoming[dst] = incoming.get(dst, 0) + weight
            incoming.setdefault(src, incoming.get(src, 0))

    for module, neighbors in adjacency.items():
        out_degree = sum(neighbors.values())
        in_degree = incoming.get(module, 0)
        degree_rows.append(
            {
                "module": module,
                "degree": in_degree + out_degree,
                "in_degree": in_degree,
                "out_degree": out_degree,
            }
        )

    degree_rows.sort(key=lambda item: (-item["degree"], item["module"]))
    for rank, row in enumerate(degree_rows[:top_k], start=1):
        row["rank"] = rank
    return degree_rows[:top_k]


def _resolve_file_metadata(index: Any, file_path: str) -> tuple[str, Optional[Any]]:
    normalized = _normalize_relpath(file_path)
    files = getattr(index, "files", None) or {}
    if normalized in files:
        return normalized, files[normalized]

    for candidate, metadata in files.items():
        candidate_norm = _normalize_relpath(candidate)
        if candidate_norm == normalized or candidate_norm.endswith(normalized):
            return candidate_norm, metadata
    return normalized, None


def _build_file_dependency_result(
    loaded: LoadedGraph,
    file_path: str,
    *,
    direction: GraphDirection,
    structured: bool,
    include_modules: bool,
) -> Dict[str, Any]:
    normalized, metadata = _resolve_file_metadata(loaded.index, file_path)
    dependencies: List[str] = []
    dependents: List[str] = []

    if metadata is not None:
        dependencies = sorted(dict.fromkeys(getattr(metadata, "dependencies", []) or []))
        files = getattr(loaded.index, "files", None) or {}
        dependents = sorted(
            candidate
            for candidate, candidate_meta in files.items()
            if normalized in (getattr(candidate_meta, "dependencies", []) or [])
        )
    else:
        file_node_id = loaded.analyzer.resolve_node_id(normalized, preferred_types={"file"})
        if file_node_id is not None:
            file_neighbors = loaded.analyzer.get_neighbors(
                file_node_id,
                direction="out",
                edge_types=["IMPORTS"],
                max_depth=1,
                node_types={"module"},
            )
            dependencies = sorted(item["name"] for item in _flatten_neighbors(file_neighbors))

    result: Dict[str, Any] = {"file": normalized}
    if direction in {"out", "both"}:
        result["dependencies"] = dependencies
    if direction in {"in", "both"}:
        result["dependents"] = dependents
    if structured:
        summary = {"focus_module": _module_name_from_file(normalized)}
        if include_modules:
            summary["modules"] = sorted(
                {
                    _module_name_from_file(normalized),
                    *[
                        _module_name_from_file(dep)
                        for dep in dependencies
                        if "." in dep or "/" in dep
                    ],
                    *[
                        _module_name_from_file(dep)
                        for dep in dependents
                        if "." in dep or "/" in dep
                    ],
                }
            )
        result["summary"] = summary
    return result


async def _run_graph_sql_query(
    loaded: LoadedGraph,
    sql: str,
) -> Dict[str, Any]:
    """Execute a raw SQL query against the graph database.

    Enables 'Big Picture' metrics by leveraging SQLite's aggregation capabilities.
    For safety, only SELECT statements are allowed.

    Args:
        loaded: LoadedGraph instance
        sql: SQL SELECT query to execute

    Returns:
        Dict with query results and metadata
    """
    return await _run_graph_sql_query_for_root(loaded.root_path, sql)


async def _run_graph_sql_query_for_root(
    root_path: Path,
    sql: str,
) -> Dict[str, Any]:
    """Execute a raw SQL query against the persisted project graph database."""
    from victor_contracts.database_runtime import get_project_database

    # Security: strictly enforce read-only SELECT queries
    # Strip whitespace and check prefix
    clean_sql = sql.strip()
    if not clean_sql.upper().startswith("SELECT"):
        return {
            "error": "Only SELECT queries are allowed for security reasons.",
            "success": False,
        }

    # Block common malicious patterns
    forbidden = [
        "DELETE",
        "UPDATE",
        "INSERT",
        "DROP",
        "ALTER",
        "CREATE",
        "REPLACE",
        "ATTACH",
    ]
    for word in forbidden:
        if f" {word} " in clean_sql.upper() or clean_sql.upper().startswith(f"{word} "):
            return {
                "error": f"Forbidden keyword detected in query: {word}",
                "success": False,
            }

    try:
        project_db = get_project_database(root_path)

        # Bound DB materialization for unbounded queries (e.g. SELECT * FROM ...)
        bounded_sql, row_limit_applied = _ensure_sql_row_limit(clean_sql, GRAPH_MAX_QUERY_ROWS)

        # Execute query
        start_time = time.perf_counter()
        rows = project_db.query(bounded_sql)
        duration = time.perf_counter() - start_time

        # Convert Row objects to serializable dicts
        results = [dict(row) for row in rows]

        result: Dict[str, Any] = {
            "results": results,
            "row_count": len(results),
            "execution_time_sec": round(duration, 4),
            "query": clean_sql,
            "success": True,
        }
        if row_limit_applied:
            result["row_limit_applied"] = GRAPH_MAX_QUERY_ROWS
            if len(results) >= GRAPH_MAX_QUERY_ROWS:
                result["row_limit_note"] = (
                    f"Auto-applied LIMIT {GRAPH_MAX_QUERY_ROWS}; more rows may exist. "
                    "Add an explicit LIMIT/WHERE clause or aggregate to refine."
                )
        return result
    except Exception as e:
        error_str = str(e)
        logger.warning(f"[graph] SQL query failed: {e}")

        # Add helpful context for common column errors
        available_columns = "node_id, type, name, file, line, end_line, lang, signature, docstring, parent_id, embedding_ref, metadata"
        edge_columns = "src, dst, type, weight, metadata, file (if edge file tracking is active)"
        if "no such column" in error_str.lower() or "does not exist" in error_str.lower():
            return {
                "error": f"SQL execution failed: {error_str}\n\nAvailable columns in graph_node: {available_columns}\nAvailable columns in graph_edge: {edge_columns}\n\nNOTE: Some edge rows may carry 'file' directly (newer edge schemas); for stable behavior with all versions, use JOIN with graph_node: JOIN graph_node n1 ON e.src = n1.node_id",
                "success": False,
                "available_columns": {
                    "graph_node": available_columns,
                    "graph_edge": edge_columns,
                },
            }
        return {
            "error": f"SQL execution failed: {error_str}",
            "success": False,
        }


def _graph_store_fingerprint(root_path: Path) -> float:
    """Return a cheap graph freshness fingerprint for analytics cache invalidation."""
    try:
        from victor_contracts.database_runtime import get_project_database

        db_path = Path(get_project_database(root_path).db_path)
        return db_path.stat().st_mtime
    except Exception:
        return 0.0


def _copy_cached_graph_result(result: Any) -> Any:
    """Return a detached copy of a JSON-like graph result."""
    try:
        return json.loads(json.dumps(result))
    except Exception:
        return dict(result) if isinstance(result, dict) else result


def _graph_cache_key(
    *,
    root_path: Path,
    mode: str,
    top_k: int,
    depth: int,
    direction: GraphDirection,
    edge_types: Optional[List[str]],
    edge_group: Optional[str],
    only_runtime: bool,
    files_only: bool,
    modules_only: bool,
    include_callsites: bool,
    max_callsites: int,
    file: Optional[str],
    query: Optional[str],
) -> tuple[Any, ...]:
    return (
        str(root_path),
        _graph_store_fingerprint(root_path),
        mode,
        top_k,
        depth,
        str(direction),
        tuple(edge_types or ()),
        edge_group,
        only_runtime,
        files_only,
        modules_only,
        include_callsites,
        max_callsites,
        file,
        query,
    )


async def _run_expensive_graph_analysis(
    *,
    cache_key: tuple[Any, ...],
    mode: str,
    compute: Any,
    node_count: int = 0,
) -> Any:
    """Run expensive graph analytics off the event loop with bounded wait and cache."""
    now = time.monotonic()
    cached = _GRAPH_ANALYTICS_CACHE.get(cache_key)
    if cached is not None:
        cached_at, cached_result = cached
        if now - cached_at <= _GRAPH_ANALYTICS_CACHE_TTL_SECONDS:
            result = _copy_cached_graph_result(cached_result)
            if isinstance(result, dict):
                result.setdefault("_meta", {})["cache_hit"] = True
            return result
        _GRAPH_ANALYTICS_CACHE.pop(cache_key, None)

    start = time.monotonic()

    def _compute_with_slot() -> Any:
        with _GRAPH_ANALYTICS_THREAD_SEMAPHORE:
            return compute()

    # Compute adaptive timeout based on graph size and mode
    timeout = _compute_adaptive_timeout(
        mode, node_count, base_timeout=_GRAPH_ANALYTICS_TIMEOUT_SECONDS
    )

    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(_compute_with_slot),
            timeout=timeout,
        )
    except asyncio.TimeoutError as exc:
        raise TimeoutError(
            f"graph mode '{mode}' exceeded {timeout:.0f}s budget; "
            "retry with smaller top_k, narrower path, files_only/modules_only, "
            "or a more specific mode"
        ) from exc

    if isinstance(result, dict):
        elapsed_ms = int((time.monotonic() - start) * 1000)
        result.setdefault("_meta", {})
        result["_meta"].update(
            {
                "cache_hit": False,
                "elapsed_ms": elapsed_ms,
                "timeout_budget_seconds": timeout,
            }
        )
    _GRAPH_ANALYTICS_CACHE[cache_key] = (
        time.monotonic(),
        _copy_cached_graph_result(result),
    )
    if len(_GRAPH_ANALYTICS_CACHE) > _GRAPH_ANALYTICS_CACHE_MAX_ENTRIES:
        oldest_key = min(_GRAPH_ANALYTICS_CACHE, key=lambda key: _GRAPH_ANALYTICS_CACHE[key][0])
        _GRAPH_ANALYTICS_CACHE.pop(oldest_key, None)
    return result


def _build_stats_from_project_store(root_path: Path) -> Dict[str, Any]:
    """Build graph stats directly from persisted SQLite graph tables."""
    from victor_contracts.database_runtime import get_project_database

    project_db = get_project_database(root_path)
    _ensure_project_graph_tables(project_db)

    node_row = project_db.query_one("SELECT COUNT(*) FROM graph_node")
    edge_row = project_db.query_one("SELECT COUNT(*) FROM graph_edge")
    node_types = {
        str(row["type"]): int(row["count"])
        for row in project_db.query(
            "SELECT type, COUNT(*) AS count FROM graph_node GROUP BY type ORDER BY type"
        )
    }
    edge_types = {
        str(row["type"]): int(row["count"])
        for row in project_db.query(
            "SELECT type, COUNT(*) AS count FROM graph_edge GROUP BY type ORDER BY type"
        )
    }

    return {
        "nodes": int(node_row[0]) if node_row is not None else 0,
        "edges": int(edge_row[0]) if edge_row is not None else 0,
        "node_types": node_types,
        "edge_types": edge_types,
        "root_path": str(root_path),
        "rebuilt": False,
    }


def _row_to_rank_item(row: Any, score_key: str) -> Dict[str, Any]:
    file_path = row["file"]
    module = Path(str(file_path)).with_suffix("").as_posix().replace("/", ".") if file_path else ""
    return {
        "node_id": row["node_id"],
        "name": row["name"],
        "qualified_name": (f"{file_path}:{row['name']}" if file_path else str(row["name"])),
        "type": row["type"],
        "file": file_path,
        "module": module,
        "score": float(row[score_key] or 0),
    }


def _build_degree_centrality_from_project_store(
    root_path: Path,
    *,
    top_k: int,
    edge_types: Optional[List[str]] = None,
    node_types: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    """Build degree centrality ranking directly from persisted SQLite graph tables.

    Fast SQL-based approach that avoids materializing the entire graph in memory.
    Computes total degree (in_degree + out_degree) for nodes.
    """
    from victor_contracts.database_runtime import get_project_database

    project_db = get_project_database(root_path)
    _ensure_project_graph_tables(project_db)

    limit = max(1, min(int(top_k or 10), 100))

    # Build separate filters for node and edge clauses
    node_where_clauses = []
    edge_where_clauses = []
    edge_params: List[str] = []
    node_params: List[str] = []

    if node_types:
        placeholders = ",".join("?" for _ in node_types)
        node_where_clauses.append(f"n.type IN ({placeholders})")
        node_params.extend(node_types)

    scope = _project_relative_scope(root_path)
    if scope:
        node_where_clauses.append("n.file LIKE ?")
        node_params.append(f"{scope}/%")

    if edge_types:
        placeholders = ",".join("?" for _ in edge_types)
        edge_where_clauses.append(f"e.type IN ({placeholders})")
        edge_params.extend(edge_types)

    # Build WHERE clauses for different contexts
    edge_where_sql = f"WHERE {' AND '.join(edge_where_clauses)}" if edge_where_clauses else ""
    node_where_sql = f"WHERE {' AND '.join(node_where_clauses)}" if node_where_clauses else ""

    # Bind parameters in the exact order placeholders appear in the query below:
    # the edge filter is interpolated twice (outgoing + incoming subqueries), so
    # its params must be supplied twice, then the outer node filter, then LIMIT.
    params: List[Any] = [*edge_params, *edge_params, *node_params]

    # Compute degree centrality using SQL aggregation
    # Count both outgoing (src) and incoming (dst) edges
    query = f"""
        SELECT
            n.node_id,
            n.name,
            n.type,
            n.file,
            n.line,
            COALESCE(outgoing.out_degree, 0) + COALESCE(incoming.in_degree, 0) AS degree,
            COALESCE(incoming.in_degree, 0) AS in_degree,
            COALESCE(outgoing.out_degree, 0) AS out_degree
        FROM graph_node n
        LEFT JOIN (
            SELECT src, COUNT(*) AS out_degree
            FROM graph_edge e
            {edge_where_sql}
            GROUP BY src
        ) outgoing ON n.node_id = outgoing.src
        LEFT JOIN (
            SELECT dst, COUNT(*) AS in_degree
            FROM graph_edge e
            {edge_where_sql}
            GROUP BY dst
        ) incoming ON n.node_id = incoming.dst
        {node_where_sql}
        ORDER BY degree DESC, n.file ASC, n.name ASC
        LIMIT ?
    """

    # Add limit parameter
    params.append(limit)

    rows = project_db.query(query, tuple(params))

    ranked = []
    for rank, row in enumerate(rows, start=1):
        file_path = row["file"]
        module = (
            Path(str(file_path)).with_suffix("").as_posix().replace("/", ".") if file_path else ""
        )
        ranked.append(
            {
                "rank": rank,
                "node_id": row["node_id"],
                "name": row["name"],
                "qualified_name": (f"{file_path}:{row['name']}" if file_path else str(row["name"])),
                "type": row["type"],
                "file": file_path,
                "module": module,
                "degree": int(row["degree"] or 0),
                "in_degree": int(row["in_degree"] or 0),
                "out_degree": int(row["out_degree"] or 0),
            }
        )

    return {
        "nodes": ranked,
        "_meta": {
            "degraded": True,
            "degradation_reason": "project_db_sql_aggregation",
            "timeout_budget_seconds": _GRAPH_TOOL_TIMEOUT_SECONDS,
        },
    }


def _project_relative_scope(root_path: Path) -> Optional[str]:
    """Return the POSIX subpath of ``root_path`` within its project, or None.

    Broad graph modes query the single project-level database (whole repo). When
    the caller scopes a request to a subdirectory (e.g. ``src/network``), this
    returns ``"src/network"`` so the SQL can be filtered to that subtree via a
    ``n.file LIKE scope || '/%'`` predicate. Returns ``None`` when the request is
    at (or above) the project root, i.e. genuinely repo-wide.
    """
    from victor_contracts.database_runtime import resolve_project_db_root

    project_root = resolve_project_db_root(root_path)  # noqa: guaranteed by bridge
    try:
        rel = Path(root_path).resolve().relative_to(project_root)
    except ValueError:
        return None
    rel_str = rel.as_posix()
    if rel_str in ("", "."):
        return None
    return rel_str


def _build_cheap_overview_from_project_store(
    root_path: Path,
    *,
    top_k: int,
) -> Dict[str, Any]:
    """Build a bounded graph overview using SQL counts only.

    This avoids materializing very large graphs and never triggers semantic
    index rebuilds. It is intentionally degree-based rather than full PageRank.
    When ``root_path`` is a subdirectory of the project, results are scoped to
    that subtree via ``n.file`` prefix matching.
    """
    from victor_contracts.database_runtime import get_project_database

    project_db = get_project_database(root_path)
    _ensure_project_graph_tables(project_db)

    limit = max(1, min(int(top_k or 10), 25))
    scope = _project_relative_scope(root_path)
    scope_like = f"{scope}/%" if scope else None
    scope_sql = " AND n.file LIKE ?" if scope_like else ""
    symbol_types = tuple(sorted(_SYMBOL_TYPES))
    placeholders = ",".join("?" for _ in symbol_types)
    symbol_params: List[Any] = [*symbol_types]
    if scope_like:
        symbol_params.append(scope_like)
    symbol_params.append(limit)
    symbol_rows = project_db.query(
        f"""
        SELECT n.node_id, n.name, n.type, n.file, COUNT(e.src) AS degree
        FROM graph_node n
        LEFT JOIN graph_edge e ON e.src = n.node_id OR e.dst = n.node_id
        WHERE n.type IN ({placeholders}){scope_sql}
        GROUP BY n.node_id, n.name, n.type, n.file
        ORDER BY degree DESC, n.file ASC, n.name ASC
        LIMIT ?
        """,
        tuple(symbol_params),
    )
    module_params: List[Any] = []
    if scope_like:
        module_params.append(scope_like)
    module_params.append(limit)
    module_rows = project_db.query(
        f"""
        SELECT n.file AS module_path, COUNT(e.src) AS degree
        FROM graph_node n
        LEFT JOIN graph_edge e ON e.src = n.node_id OR e.dst = n.node_id
        WHERE n.file IS NOT NULL AND n.file != ''{scope_sql}
        GROUP BY n.file
        ORDER BY degree DESC, n.file ASC
        LIMIT ?
        """,
        tuple(module_params),
    )
    modules = [
        {
            "module": Path(str(row["module_path"])).with_suffix("").as_posix().replace("/", "."),
            "file": row["module_path"],
            "score": float(row["degree"] or 0),
        }
        for row in module_rows
    ]
    ranked_symbols = [_row_to_rank_item(row, "degree") for row in symbol_rows]
    return {
        "stats": _build_stats_from_project_store(root_path),
        "symbol_identity_basis": _SYMBOL_IDENTITY_BASIS,
        "important_symbols": ranked_symbols,
        "hub_symbols": ranked_symbols,
        "important_modules": modules,
        "hub_modules": modules,
        "_meta": {
            "degraded": True,
            "degradation_reason": "project_db_degree_summary",
            "timeout_budget_seconds": _GRAPH_TOOL_TIMEOUT_SECONDS,
        },
    }


def _build_cheap_module_rank_from_project_store(
    root_path: Path,
    *,
    top_k: int,
    mode: str,
) -> Dict[str, Any]:
    """Build module-level rank cheaply from persisted graph degree counts."""
    overview = _build_cheap_overview_from_project_store(root_path, top_k=top_k)
    return {
        "modules": (
            overview["important_modules"]
            if mode == GraphMode.MODULE_PAGERANK.value
            else overview["hub_modules"]
        ),
        "_meta": overview["_meta"],
    }


async def _find_semantic_relationships(
    loaded: LoadedGraph,
    node_id: str,
    *,
    threshold: float = 0.5,
    limit: int = 10,
    node_types: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    """Find components semantically similar to a given node.

    Uses vector embeddings to discover 'Potential Architectural Relationships'
    between components that lack explicit code-level links.

    Args:
        loaded: LoadedGraph instance with index and analyzer
        node_id: Target node identifier
        threshold: Minimum similarity score (0.0 to 1.0)
        limit: Maximum number of relationships to return
        node_types: Optional set of node types to filter results

    Returns:
        Dict with discovered potential relationships
    """
    target_node = loaded.analyzer.nodes.get(node_id)
    if not target_node:
        return {"error": f"Node '{node_id}' not found in graph analyzer"}

    # 1. Create semantic anchor for the node
    # Combine name, type, and docstring to capture the node's intent
    parts = [target_node.name, f"type:{target_node.type}"]
    if target_node.signature:
        parts.append(target_node.signature)
    if target_node.docstring:
        # Take first 250 chars of docstring to focus on intent, not details
        parts.append(target_node.docstring[:250])

    anchor_text = " ".join(parts)
    logger.info(f"[graph] Searching for semantic matches for '{target_node.name}'")

    # 2. Search for similar nodes using CodebaseIndex
    semantic_search = getattr(loaded.index, "semantic_search", None)
    if not callable(semantic_search):
        return {
            "focus_node": _node_payload(target_node),
            "potential_relationships": [],
            "threshold_used": threshold,
            "total_discovered": 0,
            "semantic_search_available": False,
        }

    try:
        # Use existing semantic search infrastructure
        search_results = await semantic_search(
            query=anchor_text,
            max_results=limit * 3,  # Over-fetch for filtering
            similarity_threshold=threshold,
        )
    except Exception as e:
        logger.warning(f"[graph] Semantic search failed during discovery: {e}")
        return {"error": f"Semantic search failed: {str(e)}"}

    # 3. Filter and rank results
    # Get existing structural neighbors (OUT only for dependency-like relationships)
    neighbors = loaded.analyzer.get_neighbors(node_id, direction=GraphDirection.OUT, max_depth=1)
    existing_neighbor_ids = {
        item["node_id"]
        for depth_items in neighbors["neighbors_by_depth"].values()
        for item in depth_items
    }
    # Also ignore self
    existing_neighbor_ids.add(node_id)

    relationships: List[Dict[str, Any]] = []
    for result in search_results:
        # Match search result back to a graph node if possible
        # Result typically contains 'file_path', 'symbol_name'
        res_file = result.get("file_path")
        res_name = result.get("symbol_name")

        if not res_file or not res_name:
            continue

        # Find corresponding node ID in graph analyzer
        match_id = loaded.analyzer.resolve_node_id(
            res_name, preferred_types={result.get("symbol_type") or "function"}
        )
        if not match_id or match_id in existing_neighbor_ids:
            continue

        # Double check it's the right file to avoid collisions with same-named symbols
        match_node = loaded.analyzer.nodes[match_id]
        if _normalize_relpath(match_node.file) != _normalize_relpath(res_file):
            continue

        if node_types and match_node.type not in node_types:
            continue

        relationships.append(
            _node_payload(
                match_node,
                similarity=round(float(result.get("score") or 0.0), 3),
                discovery_reason="semantic_similarity",
            )
        )

        if len(relationships) >= limit:
            break

    return {
        "focus_node": _node_payload(target_node),
        "potential_relationships": sorted(
            relationships, key=lambda x: x["similarity"], reverse=True
        ),
        "threshold_used": threshold,
        "total_discovered": len(relationships),
        "semantic_search_available": True,
    }


def _find_similar_node_names(
    analyzer: GraphAnalyzer, node_ref: str, max_suggestions: int = 5
) -> List[str]:
    """Find similar node names for error suggestions.

    Args:
        analyzer: GraphAnalyzer instance
        node_ref: The node reference that failed to resolve
        max_suggestions: Maximum number of suggestions to return

    Returns:
        List of similar node names
    """
    # Use the analyzer's search method to find similar nodes
    results = analyzer.search(node_ref, limit=max_suggestions)
    names: List[str] = [r["name"] for r in results]

    # For compound `file:Symbol` references (a common shape) the symbol is often
    # misremembered or hallucinated — e.g. `tool_selection.py:ToolSelectionService`
    # when the file actually defines `ToolSelector`. Surface the symbols that ARE
    # defined in that file so the caller can self-correct.
    if ":" in node_ref:
        file_ref = node_ref.rpartition(":")[0]
        for name in _symbols_in_file(analyzer, file_ref, limit=max_suggestions):
            if name not in names:
                names.append(name)

    return names[:max_suggestions]


def _symbols_in_file(analyzer: GraphAnalyzer, file_ref: str, *, limit: int = 8) -> List[str]:
    """Return symbol names defined in ``file_ref`` (for "did you mean" suggestions).

    Resolves the file via the analyzer's file index (exact then suffix match) and
    returns the names of symbol-typed nodes in it.
    """
    normalized_file = _normalize_relpath(file_ref)
    if not normalized_file:
        return []

    file_index = getattr(analyzer, "_file_index", {}) or {}
    node_ids = list(file_index.get(normalized_file, ()))
    if not node_ids:
        for indexed_file, ids in file_index.items():
            if indexed_file.endswith(normalized_file):
                node_ids = list(ids)
                break

    names: List[str] = []
    for node_id in node_ids:
        node = analyzer.nodes.get(node_id)
        if node is None or getattr(node, "type", None) not in _SYMBOL_TYPES:
            continue
        name = getattr(node, "name", None)
        if name and name not in names:
            names.append(name)
            if len(names) >= limit:
                break
    return names


def _resolve_query_match_for_node_mode(
    analyzer: GraphAnalyzer,
    query: Optional[str],
    *,
    node_types: Optional[Set[str]],
) -> Optional[Dict[str, Any]]:
    """Resolve a model-provided query to the best node for node-oriented modes."""
    if not query:
        return None
    matches = analyzer.search(query, node_types=node_types, limit=1)
    return matches[0] if matches else None


def _build_stats(loaded: LoadedGraph) -> Dict[str, Any]:
    node_types: Dict[str, int] = {}
    edge_types: Dict[str, int] = {}
    for node in loaded.analyzer.nodes.values():
        node_types[node.type] = node_types.get(node.type, 0) + 1
    for edges in loaded.analyzer.outgoing.values():
        for edge in edges:
            edge_types[edge.type] = edge_types.get(edge.type, 0) + 1

    return {
        "nodes": len(loaded.analyzer.nodes),
        "edges": sum(len(edges) for edges in loaded.analyzer.outgoing.values()),
        "node_types": node_types,
        "edge_types": edge_types,
        "root_path": str(loaded.root_path),
        "rebuilt": loaded.rebuilt,
    }


def _build_overview(
    loaded: LoadedGraph,
    *,
    top_k: int,
    effective_edge_types: Optional[List[str]],
    only_runtime: bool,
    include_callsites: bool,
    max_callsites: int,
) -> Dict[str, Any]:
    """Build a high-level graph overview for exploratory requests."""
    projected = _project_module_adjacency(
        loaded.analyzer,
        only_runtime=only_runtime,
        include_callsites=include_callsites,
        max_callsites=max_callsites,
    )
    return {
        "stats": _build_stats(loaded),
        "symbol_identity_basis": _SYMBOL_IDENTITY_BASIS,
        "important_symbols": loaded.analyzer.pagerank(
            edge_types=effective_edge_types,
            top_k=min(top_k, 10),
            node_types=set(_SYMBOL_TYPES),
        ),
        "hub_symbols": loaded.analyzer.degree_centrality(
            top_k=min(top_k, 10),
            edge_types=effective_edge_types,
            node_types=set(_SYMBOL_TYPES),
        ),
        "important_modules": _rank_projected_modules(
            projected["adjacency"],
            mode="pagerank",
            top_k=min(top_k, 10),
        ),
        "hub_modules": _rank_projected_modules(
            projected["adjacency"],
            mode="centrality",
            top_k=min(top_k, 10),
        ),
    }


async def _handle_multi_mode(
    loaded: Any,
    mode_str: str,
    path: str,
    node: Optional[str],
    source: Optional[str],
    target: Optional[str],
    file: Optional[str],
    query: Optional[str],
    depth: int,
    top_k: int,
    direction: GraphDirection,
    edge_types: Optional[List[str]],
    edge_group: Optional[str],
    only_runtime: bool,
    files_only: bool,
    modules_only: bool,
    structured: bool,
    include_modules: bool,
    include_symbols: bool,
    include_calls: bool,
    include_refs: bool,
    include_callsites: bool,
    max_callsites: int,
    _exec_ctx: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Handle pipe-separated modes by expanding and combining results.

    Example: mode="callers|callees" executes both modes and merges results.

    Returns:
        Combined results with mode-specific sections
    """
    modes = [m.strip() for m in mode_str.split("|")]
    logger.info(f"[graph] Expanding multi-mode request: {mode_str} -> {modes}")

    combined_results = {}

    for single_mode in modes:
        try:
            # Execute each mode directly without recursion
            # We replicate the graph function logic here for single modes
            default_edge_types = _default_edge_types(single_mode, only_runtime=only_runtime)
            effective_edge_types = _resolve_effective_edge_types(
                edge_types=edge_types,
                edge_group=edge_group,
                default_edge_types=default_edge_types,
            )
            node_types = _node_type_filter(
                single_mode, files_only=files_only, modules_only=modules_only
            )

            if single_mode in {
                "callers",
                "callees",
                "trace",
                "call_flow",
                "neighbors",
                "impact",
                "subgraph",
            }:
                target_ref = node or source
                if not target_ref:
                    raise ValueError(_build_node_required_error(single_mode, path))

                preferred_types = (
                    {"file"} if files_only else {"module"} if modules_only else _SYMBOL_TYPES
                )
                resolved_id = loaded.analyzer.resolve_node_id(
                    target_ref, preferred_types=preferred_types
                )
                if resolved_id is None:
                    suggestions = _find_similar_node_names(loaded.analyzer, target_ref)
                    error_msg = f"Could not resolve graph node '{target_ref}'"
                    if suggestions:
                        error_msg += "\n\nDid you mean one of these?\n  - " + "\n  - ".join(
                            suggestions[:5]
                        )
                    raise ValueError(error_msg)

                effective_direction = direction
                if single_mode == "callers":
                    effective_direction = "in"
                elif single_mode in {"callees", "trace", "call_flow"}:
                    effective_direction = "out"
                elif single_mode in {"impact", "subgraph"}:
                    effective_direction = "both"

                base_result = loaded.analyzer.get_neighbors(
                    resolved_id,
                    direction=effective_direction,
                    edge_types=effective_edge_types,
                    max_depth=max(1, depth),
                    node_types=node_types,
                )

                if structured:
                    result = _build_structured_neighbors(
                        loaded.analyzer,
                        resolved_id,
                        base_result,
                        include_modules=include_modules,
                        include_symbols=include_symbols,
                        include_calls=include_calls,
                        include_refs=include_refs,
                        include_callsites=include_callsites,
                        max_callsites=max_callsites,
                    )
                else:
                    result = base_result

                combined_results[single_mode] = result

            else:
                # For other modes, set error (not implemented in multi-mode yet)
                combined_results[single_mode] = {
                    "error": f"Mode '{single_mode}' not supported in multi-mode queries",
                    "mode": single_mode,
                }

        except Exception as e:
            logger.warning(f"[graph] Mode {single_mode} failed: {e}")
            combined_results[single_mode] = {
                "error": str(e),
                "mode": single_mode,
            }

    # Return combined results with metadata
    return {
        "success": True,
        "mode": mode_str,
        "modes_expanded": modes,
        "results": combined_results,
        "node": node,
    }


# Tool registration options, shared by the public ``graph`` wrapper defined
# after ``_graph_impl``. The wrapper applies ``_bound_graph_result`` to every
# return path so no single mode (present or future) can emit an unbounded payload.
_GRAPH_TOOL_OPTS: Dict[str, Any] = {
    "category": "search",
    "priority": Priority.HIGH,
    "access_mode": AccessMode.READONLY,
    "danger_level": DangerLevel.SAFE,
    "execution_category": ExecutionCategory.READ_ONLY,
    "keywords": [
        "graph",
        "callers",
        "callees",
        "trace",
        "dependencies",
        "impact",
        "pagerank",
        "centrality",
        "architecture",
        "neighbors",
    ],
    "aliases": ["graph_tool"],
    "availability_check": _graph_tool_is_available,
    "timeout": _GRAPH_TOOL_TIMEOUT_SECONDS,
}


async def _graph_impl(
    mode: GraphMode = GraphMode.NEIGHBORS,
    path: str = ".",
    node: Optional[str] = None,
    source: Optional[str] = None,
    target: Optional[str] = None,
    file: Optional[str] = None,
    query: Optional[str] = None,
    depth: int = 2,
    top_k: int = 10,
    threshold: float = 0.5,
    direction: GraphDirection = "out",
    edge_types: Optional[List[str]] = None,
    edge_group: Optional[str] = None,
    reindex: bool = False,
    only_runtime: bool = False,
    files_only: bool = False,
    modules_only: bool = False,
    structured: bool = False,
    include_modules: bool = False,
    include_symbols: bool = False,
    include_calls: bool = False,
    include_refs: bool = False,
    include_callsites: bool = False,
    max_callsites: int = 3,
    _exec_ctx: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Analyze the indexed code graph using the active SQLite + LanceDB code index.

    Valid modes include: overview, stats, search/find, neighbors, callers, callees,
    trace, path, pagerank, centrality, patterns, clusters, semantic, query, and
    module/file dependency variants. Use schema/capabilities/help for a cheap
    metadata response describing supported modes and relationship filters.

    Common alias recovery is supported for model-generated variants such as:
    - hub_analysis -> overview
    - top_k -> search (when a query/node/file is supplied) or pagerank
    - connected_components / connectedComponents / components -> clusters

    Relationship filtering:
    - edge_types is the exact filter, e.g. ["CALLS", "INHERITS"].
    - edge_group is an ergonomic preset, e.g. call_flow, type_hierarchy,
      composition, imports, control_flow, data_flow, ccg, semantic.
    - edge_types overrides edge_group when both are provided.

    Enhanced with file watching for automatic cache invalidation.
    """
    requested_mode = _graph_mode_value(mode)
    normalized_mode = _normalize_graph_mode_alias(
        mode,
        node=node,
        source=source,
        target=target,
        file=file,
        query=query,
    )
    root_path = _resolve_root_path(path)
    if normalized_mode == GraphMode.SCHEMA.value:
        return {
            "success": True,
            "mode": GraphMode.SCHEMA.value,
            "requested_mode": requested_mode,
            "root_path": str(root_path),
            "rebuilt": False,
            "result": _build_graph_schema_result(),
        }

    if _should_reuse_project_graph_root(
        requested_path=path,
        requested_mode=normalized_mode,
        node=node,
        source=source,
        target=target,
        file=file,
        query=query,
    ):
        project_root = _current_project_root()
        if root_path != project_root:
            logger.info(
                "[graph] Reusing project graph root %s for nested request scoped to %s",
                project_root,
                path,
            )
            root_path = project_root

    # Pre-flight check: if any contracts-bridge dependency is missing, return
    # immediately with an actionable error. This prevents ImportError crashes
    # deep in helper functions (resolve_project_db_root, GraphManager, etc.)
    # and saves the 2-3 wasted turns the circuit breaker would otherwise need.
    if not _check_graph_dependencies():
        return _graph_error_response(
            requested_mode=requested_mode,
            mode=normalized_mode,
            error="Graph tool dependencies not available (contracts bridge gap). "
            "Use `code search` or `read` instead.",
            suggestions=[],
            extra={
                "suggestion": (
                    "A required function is missing from the victor-contracts "
                    "bridge. Use code_search, read, or ls for code exploration."
                ),
                "unavailable": True,
            },
        )

    # Subscribe to file watcher for automatic cache invalidation (only once per root).
    # The watcher + incremental refresh operate on the *project* graph DB, so they
    # must run against the canonical project root — not a scoped subpath like
    # ``src/network``. Passing the subpath here indexes a stray subdirectory DB
    # (and litters a ``.victor/`` dir) instead of refreshing the repo's project.db.
    from victor_contracts.database_runtime import resolve_project_db_root
    from victor_contracts.indexing_runtime import GraphManager

    refresh_root = resolve_project_db_root(root_path)

    # Subscribe GraphManager to file watcher for the project root
    if not reindex and not _project_graph_watch_daemon_active(refresh_root):
        try:
            manager = GraphManager.get_instance()
            await manager.ensure_background_refresh(refresh_root, exec_ctx=_exec_ctx)
        except Exception as e:
            # Non-fatal: log warning but continue
            logger.warning(f"[graph] Failed to subscribe to file watcher: {e}")

    try:
        if not reindex and normalized_mode == "stats" and _project_graph_has_data(root_path):
            return {
                "success": True,
                "mode": "stats",
                "requested_mode": requested_mode,
                "root_path": str(root_path),
                "rebuilt": False,
                "result": _build_stats_from_project_store(root_path),
            }

        if (
            not reindex
            and normalized_mode in {"overview", "module_pagerank", "module_centrality"}
            and _project_graph_has_data(root_path)
        ):
            result = (
                _build_cheap_overview_from_project_store(root_path, top_k=top_k)
                if normalized_mode == "overview"
                else _build_cheap_module_rank_from_project_store(
                    root_path,
                    top_k=top_k,
                    mode=normalized_mode,
                )
            )
            return {
                "success": True,
                "mode": normalized_mode,
                "requested_mode": requested_mode,
                "root_path": str(root_path),
                "rebuilt": False,
                "edge_group": _normalize_edge_group_name(edge_group),
                "effective_edge_types": _resolve_effective_edge_types(
                    edge_types=edge_types,
                    edge_group=edge_group,
                    default_edge_types=_default_edge_types(
                        normalized_mode, only_runtime=only_runtime
                    ),
                ),
                "result": result,
            }

        # Fast path: degree centrality using SQL aggregation for large graphs
        if not reindex and normalized_mode == "centrality" and _project_graph_has_data(root_path):
            # Resolve effective edge types for the SQL query
            default_edge_types = _default_edge_types(normalized_mode, only_runtime=only_runtime)
            effective_edge_types = _resolve_effective_edge_types(
                edge_types=edge_types,
                edge_group=edge_group,
                default_edge_types=default_edge_types,
            )
            node_types = _node_type_filter(
                normalized_mode, files_only=files_only, modules_only=modules_only
            )
            result = _build_degree_centrality_from_project_store(
                root_path,
                top_k=top_k,
                edge_types=effective_edge_types,
                node_types=node_types,
            )
            return {
                "success": True,
                "mode": normalized_mode,
                "requested_mode": requested_mode,
                "root_path": str(root_path),
                "rebuilt": False,
                "edge_group": _normalize_edge_group_name(edge_group),
                "effective_edge_types": effective_edge_types,
                "result": result,
            }

        # Fast path: patterns mode using SQL aggregation for large graphs
        if not reindex and normalized_mode == "patterns" and _project_graph_has_data(root_path):
            default_edge_types = _default_edge_types(normalized_mode, only_runtime=only_runtime)
            effective_edge_types = _resolve_effective_edge_types(
                edge_types=edge_types,
                edge_group=edge_group,
                default_edge_types=default_edge_types,
            )
            node_types = _node_type_filter(
                normalized_mode, files_only=files_only, modules_only=modules_only
            )
            patterns_top_k = min(top_k, 20)  # Limit results for patterns mode

            result = {
                "symbol_identity_basis": _SYMBOL_IDENTITY_BASIS,
                "important_symbols": _build_degree_centrality_from_project_store(
                    root_path,
                    top_k=patterns_top_k,
                    edge_types=effective_edge_types,
                    node_types=node_types or set(_SYMBOL_TYPES),
                )["nodes"],
                "hub_symbols": _build_degree_centrality_from_project_store(
                    root_path,
                    top_k=patterns_top_k,
                    edge_types=effective_edge_types,
                    node_types=node_types or set(_SYMBOL_TYPES),
                )["nodes"],
            }
            return {
                "success": True,
                "mode": normalized_mode,
                "requested_mode": requested_mode,
                "root_path": str(root_path),
                "rebuilt": False,
                "edge_group": _normalize_edge_group_name(edge_group),
                "effective_edge_types": effective_edge_types,
                "result": result,
            }

        if (
            not reindex
            and normalized_mode == GraphMode.QUERY
            and _project_graph_has_data(root_path)
        ):
            if not query:
                raise ValueError("query mode requires a SQL SELECT statement")
            return {
                "success": True,
                "mode": GraphMode.QUERY,
                "requested_mode": requested_mode,
                "root_path": str(root_path),
                "rebuilt": False,
                "result": await _run_graph_sql_query_for_root(root_path, query),
            }

        # Fast path: file_deps mode can work with CodebaseIndex even without project database
        # Try to use CodebaseIndex files metadata which contains dependency information
        if (
            not reindex
            and normalized_mode == "file_deps"
            and not _project_graph_has_data(root_path)
        ):
            try:
                index, rebuilt = await _get_or_build_index(
                    root_path,
                    load_settings(),
                    force_reindex=False,
                    exec_ctx=_exec_ctx,
                )
                # Create a minimal LoadedGraph-like object for file_deps processing
                from types import SimpleNamespace

                fake_loaded = SimpleNamespace(
                    root_path=root_path,
                    index=index,
                    rebuilt=rebuilt,
                    analyzer=None,  # Not needed for file_deps mode
                )

                # Process file_deps using the index
                if file:
                    result = _build_file_dependency_result(
                        fake_loaded,
                        file,
                        direction=direction,
                        structured=structured,
                        include_modules=include_modules,
                    )
                    return {
                        "success": True,
                        "mode": normalized_mode,
                        "requested_mode": requested_mode,
                        "root_path": str(root_path),
                        "rebuilt": rebuilt,
                        "result": result,
                    }
            except (ImportError, RuntimeError) as exc:
                # CodebaseIndex not available, fall through to standard error path
                logger.debug("[graph] CodebaseIndex unavailable for file_deps fallback: %s", exc)

        if (
            not reindex
            and normalized_mode in {"overview", "module_pagerank", "module_centrality"}
            and not _project_graph_has_data(root_path)
        ):
            settings = _ctx_value(_exec_ctx, "settings")
            is_memory_graph_test = getattr(settings, "codebase_graph_store", None) == "memory"
            if not is_memory_graph_test:
                return {
                    "success": False,
                    "mode": normalized_mode,
                    "requested_mode": requested_mode,
                    "root_path": str(root_path),
                    "rebuilt": False,
                    "error": (
                        "Project graph database is unavailable for broad graph analytics. "
                        "Run a project graph index first, pass reindex=True explicitly, "
                        "or use a narrower search/find query."
                    ),
                    "metadata": {
                        "semantic_rebuild_skipped": True,
                        "reason": "broad_graph_mode_requires_project_db_or_explicit_reindex",
                    },
                }

        loaded = await _load_graph(str(root_path), reindex=reindex, exec_ctx=_exec_ctx)

        # Handle pipe-separated modes (e.g., "callers|callees")
        # Expand into multiple results combined
        if "|" in normalized_mode:
            return await _handle_multi_mode(
                loaded=loaded,
                mode_str=normalized_mode,
                path=path,
                node=node,
                source=source,
                target=target,
                file=file,
                query=query,
                depth=depth,
                top_k=top_k,
                direction=direction,
                edge_types=edge_types,
                edge_group=edge_group,
                only_runtime=only_runtime,
                files_only=files_only,
                modules_only=modules_only,
                structured=structured,
                include_modules=include_modules,
                include_symbols=include_symbols,
                include_calls=include_calls,
                include_refs=include_refs,
                include_callsites=include_callsites,
                max_callsites=max_callsites,
                _exec_ctx=_exec_ctx,
            )

        mode = normalized_mode

        default_edge_types = _default_edge_types(mode, only_runtime=only_runtime)
        effective_edge_types = _resolve_effective_edge_types(
            edge_types=edge_types,
            edge_group=edge_group,
            default_edge_types=default_edge_types,
        )
        node_types = _node_type_filter(mode, files_only=files_only, modules_only=modules_only)
        analytics_cache_key = _graph_cache_key(
            root_path=loaded.root_path,
            mode=mode,
            top_k=top_k,
            depth=depth,
            direction=direction,
            edge_types=effective_edge_types,
            edge_group=edge_group,
            only_runtime=only_runtime,
            files_only=files_only,
            modules_only=modules_only,
            include_callsites=include_callsites,
            max_callsites=max_callsites,
            file=file,
            query=query,
        )

        if mode == "overview":
            result = await _run_expensive_graph_analysis(
                cache_key=analytics_cache_key,
                mode=mode,
                compute=lambda: _build_overview(
                    loaded=loaded,
                    top_k=top_k,
                    effective_edge_types=effective_edge_types,
                    only_runtime=only_runtime,
                    include_callsites=include_callsites,
                    max_callsites=max_callsites,
                ),
            )
        elif mode == "stats":
            result = _build_stats(loaded)
        elif mode == "search":
            # Alias for 'find' mode - more intuitive for users
            search_query = query or node or file
            if not search_query:
                raise ValueError("search mode requires query, node, or file")
            result = {
                "matches": loaded.analyzer.search(search_query, node_types=node_types, limit=top_k)
            }
        elif mode == "find":
            search_query = query or node or file
            if not search_query:
                raise ValueError("find mode requires query, node, or file")
            result = {
                "matches": loaded.analyzer.search(search_query, node_types=node_types, limit=top_k)
            }
        elif mode in {
            "callers",
            "callees",
            "trace",
            "call_flow",
            "neighbors",
            "impact",
            "subgraph",
        }:
            target_ref = node or source
            recovered_query_match = None
            preferred_types = (
                {"file"} if files_only else {"module"} if modules_only else _SYMBOL_TYPES
            )
            if not target_ref and query:
                recovered_query_match = _resolve_query_match_for_node_mode(
                    loaded.analyzer,
                    query,
                    node_types=node_types or preferred_types,
                )
                if recovered_query_match:
                    target_ref = recovered_query_match["node_id"]
            if not target_ref:
                if file:
                    mode = "file_deps"
                    fallback_direction = _FILE_FALLBACK_DIRECTIONS.get(
                        requested_mode,
                        direction if isinstance(direction, str) else direction.value,
                    )
                    result = _build_file_dependency_result(
                        loaded,
                        file,
                        direction=fallback_direction,
                        structured=structured,
                        include_modules=include_modules,
                    )
                    result["recovered_from_mode"] = requested_mode
                else:
                    raise ValueError(_build_node_required_error(mode, path))
            else:
                resolved_id = loaded.analyzer.resolve_node_id(
                    target_ref, preferred_types=preferred_types
                )
                if resolved_id is None:
                    # Suggest similar node names
                    suggestions = _find_similar_node_names(loaded.analyzer, target_ref)
                    error_msg = f"Could not resolve graph node '{target_ref}'"
                    if suggestions:
                        error_msg += "\n\nDid you mean one of these?\n  - " + "\n  - ".join(
                            suggestions[:5]
                        )
                    raise ValueError(error_msg)

                effective_direction = direction
                if mode == "callers":
                    effective_direction = "in"
                elif mode in {"callees", "trace", "call_flow"}:
                    effective_direction = "out"
                elif mode in {"impact", "subgraph"}:
                    effective_direction = "both"

                base_result = loaded.analyzer.get_neighbors(
                    resolved_id,
                    direction=effective_direction,
                    edge_types=effective_edge_types,
                    max_depth=max(1, depth),
                    node_types=node_types,
                )
                if structured:
                    result = _build_structured_neighbors(
                        loaded.analyzer,
                        resolved_id,
                        base_result,
                        include_modules=include_modules,
                        include_symbols=include_symbols,
                        include_calls=include_calls,
                        include_refs=include_refs,
                        include_callsites=include_callsites,
                        max_callsites=max_callsites,
                    )
                else:
                    result = base_result
                if recovered_query_match:
                    result["recovered_from_query"] = query
                    result["resolved_query_match"] = recovered_query_match
        elif mode == "path":
            if not source or not target:
                raise ValueError("path mode requires source and target")
            resolved_source = loaded.analyzer.resolve_node_id(source, preferred_types=_SYMBOL_TYPES)
            resolved_target = loaded.analyzer.resolve_node_id(target, preferred_types=_SYMBOL_TYPES)
            if resolved_source is None or resolved_target is None:
                # Determine which node failed and provide suggestions
                error_parts = []
                if resolved_source is None:
                    source_suggestions = _find_similar_node_names(loaded.analyzer, source)
                    error_parts.append(f"Source '{source}' not found")
                    if source_suggestions:
                        error_parts.append("  Did you mean?")
                        error_parts.extend(f"    - {s}" for s in source_suggestions[:3])
                if resolved_target is None:
                    target_suggestions = _find_similar_node_names(loaded.analyzer, target)
                    error_parts.append(f"Target '{target}' not found")
                    if target_suggestions:
                        error_parts.append("  Did you mean?")
                        error_parts.extend(f"    - {s}" for s in target_suggestions[:3])
                raise ValueError("\n".join(error_parts))
            result = loaded.analyzer.shortest_path(
                resolved_source,
                resolved_target,
                edge_types=effective_edge_types,
            )
        elif mode in {"pagerank", "centrality"}:
            node_count = len(loaded.analyzer.nodes)
            if mode == "pagerank":
                result = await _run_expensive_graph_analysis(
                    cache_key=analytics_cache_key,
                    mode=mode,
                    compute=lambda: loaded.analyzer.pagerank(
                        edge_types=effective_edge_types,
                        top_k=top_k,
                        node_types=node_types,
                    ),
                    node_count=node_count,
                )
            else:
                result = await _run_expensive_graph_analysis(
                    cache_key=analytics_cache_key,
                    mode=mode,
                    compute=lambda: loaded.analyzer.degree_centrality(
                        top_k=top_k,
                        edge_types=effective_edge_types,
                        node_types=node_types,
                    ),
                    node_count=node_count,
                )
        elif mode in {"module_pagerank", "module_centrality"}:
            node_count = len(loaded.analyzer.nodes)

            def _compute_module_rank() -> Dict[str, Any]:
                projected = _project_module_adjacency(
                    loaded.analyzer,
                    only_runtime=only_runtime,
                    include_callsites=include_callsites,
                    max_callsites=max_callsites,
                )
                module_result: Dict[str, Any] = {
                    "modules": _rank_projected_modules(
                        projected["adjacency"],
                        mode="pagerank" if mode == "module_pagerank" else "centrality",
                        top_k=top_k,
                    )
                }
                if include_callsites and "callsites" in projected:
                    module_result["callsites"] = projected["callsites"]
                return module_result

            result = await _run_expensive_graph_analysis(
                cache_key=analytics_cache_key,
                mode=mode,
                compute=_compute_module_rank,
                node_count=node_count,
            )
        elif mode == "file_deps":
            if not file:
                mode, result = _recover_file_deps_without_file(
                    loaded=loaded,
                    requested_path=path,
                    requested_mode=requested_mode,
                    direction=direction,
                    structured=structured,
                    include_modules=include_modules,
                    top_k=top_k,
                    effective_edge_types=effective_edge_types,
                    only_runtime=only_runtime,
                    include_callsites=include_callsites,
                    max_callsites=max_callsites,
                )
            else:
                result = _build_file_dependency_result(
                    loaded,
                    file,
                    direction=direction,
                    structured=structured,
                    include_modules=include_modules,
                )
        elif mode == "clusters":
            node_count = len(loaded.analyzer.nodes)

            def _compute_clusters() -> Dict[str, Any]:
                adjacency = {
                    node_id: [
                        edge.dst
                        for edge in edges
                        if effective_edge_types is None or edge.type in set(effective_edge_types)
                    ]
                    for node_id, edges in loaded.analyzer.outgoing.items()
                }
                components = connected_components(adjacency)
                ranked = sorted(components, key=lambda component: (-len(component), component))
                return {
                    "edge_group": _normalize_edge_group_name(edge_group),
                    "edge_types": effective_edge_types,
                    "components": [
                        {"size": len(component), "nodes": component} for component in ranked[:top_k]
                    ],
                }

            result = await _run_expensive_graph_analysis(
                cache_key=analytics_cache_key,
                mode=mode,
                compute=_compute_clusters,
                node_count=node_count,
            )
        elif mode == "patterns":
            node_count = len(loaded.analyzer.nodes)

            def _compute_patterns() -> Dict[str, Any]:
                projected = _project_module_adjacency(
                    loaded.analyzer,
                    only_runtime=only_runtime,
                    include_callsites=False,
                    max_callsites=max_callsites,
                )
                return {
                    "symbol_identity_basis": _SYMBOL_IDENTITY_BASIS,
                    "important_symbols": loaded.analyzer.pagerank(
                        edge_types=effective_edge_types,
                        top_k=min(top_k, 5),
                        node_types=set(_SYMBOL_TYPES),
                    ),
                    "hub_symbols": loaded.analyzer.degree_centrality(
                        top_k=min(top_k, 5),
                        edge_types=effective_edge_types,
                        node_types=set(_SYMBOL_TYPES),
                    ),
                    "important_modules": _rank_projected_modules(
                        projected["adjacency"],
                        mode="pagerank",
                        top_k=min(top_k, 5),
                    ),
                    "hub_modules": _rank_projected_modules(
                        projected["adjacency"],
                        mode="centrality",
                        top_k=min(top_k, 5),
                    ),
                }

            result = await _run_expensive_graph_analysis(
                cache_key=analytics_cache_key,
                mode=mode,
                compute=_compute_patterns,
                node_count=node_count,
            )
        elif mode == GraphMode.SEMANTIC:
            target_ref = node or source
            if not target_ref:
                raise ValueError(_build_node_required_error("semantic", path))

            preferred_types = (
                {"file"} if files_only else {"module"} if modules_only else _SYMBOL_TYPES
            )
            resolved_id = loaded.analyzer.resolve_node_id(
                target_ref, preferred_types=preferred_types
            )
            if resolved_id is None:
                suggestions = _find_similar_node_names(loaded.analyzer, target_ref)
                error_msg = f"Could not resolve graph node '{target_ref}'"
                if suggestions:
                    error_msg += "\n\nDid you mean one of these?\n  - " + "\n  - ".join(
                        suggestions[:5]
                    )
                raise ValueError(error_msg)

            result = await _find_semantic_relationships(
                loaded,
                resolved_id,
                threshold=threshold,
                limit=top_k,
                node_types=node_types,
            )
        elif mode == GraphMode.QUERY:
            if not query:
                raise ValueError("query mode requires a SQL SELECT statement")
            result = await _run_graph_sql_query(loaded, query)
        else:
            raise ValueError(_unsupported_graph_mode_error(str(mode)))

        graph_result = {
            "success": True,
            "mode": mode,
            "requested_mode": requested_mode,
            "root_path": str(loaded.root_path),
            "rebuilt": loaded.rebuilt,
            "result": result,
        }
        if edge_group is not None:
            graph_result["edge_group"] = _normalize_edge_group_name(edge_group)
        if effective_edge_types is not None:
            graph_result["effective_edge_types"] = effective_edge_types

        return graph_result
    except ImportError as exc:
        # CodebaseIndex provider not installed — permanent unavailability
        return _graph_error_response(
            requested_mode=requested_mode,
            mode=normalized_mode,
            error=str(exc),
            suggestions=_build_graph_error_follow_up_suggestions(
                path=path,
                requested_mode=requested_mode,
                normalized_mode=normalized_mode,
                node=node,
                source=source,
                target=target,
                file=file,
                query=query,
                depth=depth,
                top_k=top_k,
                unavailable=True,
            ),
            extra={
                "suggestion": (
                    "The graph tool requires a codebase indexing provider. "
                    "Use code_search, overview, ls, or read tools for code exploration instead. "
                    "To enable graph analysis: pip install victor-coding"
                ),
                "unavailable": True,
            },
        )
    except Exception as exc:
        error_msg = str(exc)
        # Only treat specific ImportError-related messages as permanent unavailability
        # Temporary issues like empty database, missing tables, or SQL errors should NOT disable the tool
        permanent_unavailable_keywords = [
            "No module named",
            "cannot import",
            "not installed",
        ]
        is_permanent_unavailable = any(
            keyword in error_msg for keyword in permanent_unavailable_keywords
        )

        # Check for empty database errors to provide helpful reindex suggestion
        empty_database_keywords = [
            "database is empty",
            "database is unavailable",
            "project database is empty",
        ]
        is_empty_database = any(keyword in error_msg.lower() for keyword in empty_database_keywords)

        if is_permanent_unavailable:
            return _graph_error_response(
                requested_mode=requested_mode,
                mode=normalized_mode,
                error=error_msg,
                suggestions=_build_graph_error_follow_up_suggestions(
                    path=path,
                    requested_mode=requested_mode,
                    normalized_mode=normalized_mode,
                    node=node,
                    source=source,
                    target=target,
                    file=file,
                    query=query,
                    depth=depth,
                    top_k=top_k,
                    unavailable=True,
                    empty_database=is_empty_database,
                ),
                extra={
                    "suggestion": (
                        "The graph tool requires a codebase indexing provider. "
                        "Use code_search, overview, ls, or read tools for code exploration instead. "
                        "To enable graph analysis: pip install victor-coding"
                    ),
                    "unavailable": True,
                },
            )

        unresolved_node = None
        if error_msg.startswith("Could not resolve graph node '"):
            unresolved_node = error_msg.split("Could not resolve graph node '", 1)[1].split(
                "'",
                1,
            )[0]

        # For empty database errors, provide a more helpful error message
        if is_empty_database:
            error_msg = (
                f"Project graph database is unavailable/empty. Run: graph(mode='{normalized_mode}', path='{path}', reindex=True) "
                f"to build the graph index first."
            )

        return _graph_error_response(
            requested_mode=requested_mode,
            mode=normalized_mode,
            error=error_msg,
            suggestions=_build_graph_error_follow_up_suggestions(
                path=path,
                requested_mode=requested_mode,
                normalized_mode=normalized_mode,
                node=node,
                source=source,
                target=target,
                file=file,
                query=query,
                depth=depth,
                top_k=top_k,
                unresolved_node=unresolved_node,
                empty_database=is_empty_database,
            ),
        )


async def _graph_tool_entry(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """Public ``graph`` tool entry point.

    Single seam over every ``_graph_impl`` return path: it bounds the result via
    ``_bound_graph_result`` so no mode can emit an unbounded payload into context.
    """
    result = await _graph_impl(*args, **kwargs)
    return _bound_graph_result(result)


# Present the wrapper to the @tool decorator (and the LLM schema) exactly as the
# original ``graph`` function: same name, docstring, and signature. The decorator
# introspects these via inspect.signature, which honors the explicit __signature__.
_graph_tool_entry.__name__ = "graph"
_graph_tool_entry.__qualname__ = "graph"
_graph_tool_entry.__doc__ = _graph_impl.__doc__
_graph_tool_entry.__wrapped__ = _graph_impl  # type: ignore[attr-defined]
_graph_tool_entry.__signature__ = inspect.signature(_graph_impl)  # type: ignore[attr-defined]

graph = tool(**_GRAPH_TOOL_OPTS)(_graph_tool_entry)


# =============================================================================
# Focused Graph Tools (Split from monolithic graph function)
# =============================================================================


@tool(
    category="search",
    priority=Priority.HIGH,
    access_mode=AccessMode.READONLY,
    danger_level=DangerLevel.SAFE,
    execution_category=ExecutionCategory.READ_ONLY,
    keywords=["graph", "search", "find", "query"],
    timeout=180.0,
)
async def graph_search(
    query: str,
    path: str = ".",
    top_k: int = 10,
    _exec_ctx: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Find nodes by query in code graph.

    Fast search for symbols, files, and modules by name or pattern.
    Use for: Quick lookups when you know what you're looking for.

    Args:
        query: Search query (symbol name, file path, or pattern)
        path: Path to codebase root (default: ".")
        top_k: Maximum number of results to return

    Returns:
        Dict with matches list containing node IDs and metadata
    """
    return await graph(
        mode="search",
        query=query,
        path=path,
        top_k=top_k,
        _exec_ctx=_exec_ctx,
    )


@tool(
    category="search",
    priority=Priority.HIGH,
    access_mode=AccessMode.READONLY,
    danger_level=DangerLevel.SAFE,
    execution_category=ExecutionCategory.READ_ONLY,
    keywords=["graph", "neighbors", "callers", "callees", "relationships"],
    timeout=180.0,
)
async def graph_neighbors(
    node: str,
    path: str = ".",
    depth: int = 2,
    direction: GraphDirection = "out",
    edge_types: Optional[List[str]] = None,
    edge_group: Optional[str] = None,
    include_callsites: int = 3,
    _exec_ctx: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Get neighboring nodes in code graph.

    Explore relationships: callers, callees, references, etc.
    Use for: Understanding what calls or is called by a symbol.

    Args:
        node: Node identifier (symbol, file, or module)
        path: Path to codebase root (default: ".")
        depth: How many hops to explore (default: 2)
        direction: "out" (callees), "in" (callers), or "both"
        edge_types: Filter to specific edge types (None = all)
        edge_group: Relationship preset such as call_flow, type_hierarchy, or composition
        include_callsites: Number of call sites to include (default: 3)

    Returns:
        Dict with neighbors list and relationship details
    """
    return await graph(
        mode="neighbors",
        node=node,
        path=path,
        depth=depth,
        direction=direction,
        edge_types=edge_types,
        edge_group=edge_group,
        include_callsites=include_callsites,
        _exec_ctx=_exec_ctx,
    )


@tool(
    category="analysis",
    priority=Priority.MEDIUM,
    access_mode=AccessMode.READONLY,
    danger_level=DangerLevel.SAFE,
    execution_category=ExecutionCategory.READ_ONLY,
    keywords=["graph", "analytics", "metrics", "pagerank", "centrality", "stats"],
    timeout=180.0,
)
async def graph_analytics(
    path: str = ".",
    reindex: bool = False,
    only_runtime: bool = False,
    files_only: bool = False,
    modules_only: bool = False,
    top_k: int = 10,
    _exec_ctx: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Calculate graph metrics and analytics.

    PageRank, centrality, stats, clusters, etc.
    Use for: Understanding codebase structure and importance.

    Args:
        path: Path to codebase root (default: ".")
        reindex: Force rebuild of graph index
        only_runtime: Only include runtime dependencies
        files_only: Analyze files only (not symbols)
        modules_only: Analyze modules only (not files/symbols)
        top_k: Top K results to return

    Returns:
        Dict with metrics: pagerank, centrality, stats, etc.
    """
    return await graph(
        mode="stats",
        path=path,
        reindex=reindex,
        only_runtime=only_runtime,
        files_only=files_only,
        modules_only=modules_only,
        top_k=top_k,
        _exec_ctx=_exec_ctx,
    )


@tool(
    category="search",
    priority=Priority.MEDIUM,
    access_mode=AccessMode.READONLY,
    danger_level=DangerLevel.SAFE,
    execution_category=ExecutionCategory.READ_ONLY,
    keywords=["graph", "path", "trace", "flow", "dependencies"],
    timeout=180.0,
)
async def graph_path(
    source: str,
    target: str,
    path: str = ".",
    max_depth: int = 5,
    _exec_ctx: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Find paths between nodes in code graph.

    Discover how code connects: call chains, data flow, etc.
    Use for: Tracing execution paths or data flow.

    Args:
        source: Source node identifier
        target: Target node identifier
        path: Path to codebase root (default: ".")
        max_depth: Maximum path length to search

    Returns:
        Dict with paths list and path details
    """
    return await graph(
        mode="path",
        source=source,
        target=target,
        path=path,
        depth=max_depth,
        _exec_ctx=_exec_ctx,
    )


@tool(
    category="analysis",
    priority=Priority.HIGH,
    access_mode=AccessMode.READONLY,
    danger_level=DangerLevel.SAFE,
    execution_category=ExecutionCategory.READ_ONLY,
    keywords=["graph", "semantic", "discovery", "similarity", "architectural patterns"],
    timeout=180.0,
)
async def graph_semantic(
    node: str,
    path: str = ".",
    threshold: float = 0.5,
    top_k: int = 5,
    _exec_ctx: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Discover potential architectural relationships based on semantic similarity.

    Identifies components that are semantically related but lack explicit code-level links.
    Use for: Finding members of a registry, related service implementations, or pattern participants.

    Args:
        node: Node identifier (symbol, file, or module) to analyze
        path: Path to codebase root (default: ".")
        threshold: Minimum similarity score (default: 0.5)
        top_k: Maximum number of results to return (default: 5)

    Returns:
        Dict with potential relationships and similarity scores
    """
    return await graph(
        mode="semantic",
        node=node,
        path=path,
        top_k=top_k,
        threshold=threshold,
        _exec_ctx=_exec_ctx,
    )


@tool(
    category="analysis",
    priority=Priority.MEDIUM,
    access_mode=AccessMode.READONLY,
    danger_level=DangerLevel.SAFE,
    execution_category=ExecutionCategory.READ_ONLY,
    keywords=["graph", "sql", "query", "aggregate", "metrics", "coupling"],
    timeout=30.0,
)
async def graph_query(
    query: str,
    path: str = ".",
    _exec_ctx: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Execute a raw SQL SELECT query against the code graph database.

    Enables 'Big Picture' metrics by leveraging SQLite's aggregation capabilities.
    For security, only SELECT statements are allowed.

    Tables available:
    - graph_node (node_id, type, name, file, line, end_line, lang, signature, docstring, parent_id, metadata)
    - graph_edge (src, dst, type, weight, metadata, file)
    - graph_file_mtime (file, mtime, indexed_at)

    IMPORTANT — column names are exact. Common mistakes:
    - There is NO 'module' column. graph_analytics returns a computed 'module' field that
      does NOT exist as a SQL column. To extract the top-level package name (e.g. 'victor'
      from 'victor/core/foo.py') use: SUBSTR(file, 1, INSTR(file || '/', '/') - 1)
      Note: this returns the first path component, not the full dotted module path.
      Top-level files with no slash (e.g. 'setup.py') return the full filename.
    - graph_edge can include an optional 'file' column (newer versions); for portability, use
      joins through graph_node on src/dst when you need file-aware filtering.

    Examples:
    - Find most imported modules: "SELECT count(*) as count, dst FROM graph_edge WHERE type='IMPORTS' GROUP BY dst ORDER BY count DESC LIMIT 10"
    - Count symbols by type: "SELECT type, count(*) as count FROM graph_node GROUP BY type ORDER BY count DESC"
    - Find files with most symbols: "SELECT file, count(*) as count FROM graph_node GROUP BY file ORDER BY count DESC LIMIT 10"
    - Group by top-level package: "SELECT SUBSTR(file, 1, INSTR(file || '/', '/') - 1) as pkg, count(*) as cnt FROM graph_node GROUP BY pkg ORDER BY cnt DESC LIMIT 20"
    - Cross-package coupling (imports between packages): "SELECT SUBSTR(n1.file, 1, INSTR(n1.file || '/', '/') - 1) as src_pkg, SUBSTR(n2.file, 1, INSTR(n2.file || '/', '/') - 1) as dst_pkg, count(*) as edges FROM graph_edge e JOIN graph_node n1 ON e.src = n1.node_id JOIN graph_node n2 ON e.dst = n2.node_id WHERE e.type='IMPORTS' GROUP BY src_pkg, dst_pkg ORDER BY edges DESC LIMIT 20"

    Args:
        query: SQL SELECT query to execute
        path: Path to codebase root (default: ".")

    Returns:
        Dict with query results and execution metadata
    """
    return await graph(
        mode="query",
        query=query,
        path=path,
        _exec_ctx=_exec_ctx,
    )


@tool(
    category="analysis",
    priority=Priority.MEDIUM,
    access_mode=AccessMode.READONLY,
    danger_level=DangerLevel.SAFE,
    execution_category=ExecutionCategory.READ_ONLY,
    keywords=["graph", "dependencies", "file", "imports", "impact"],
    timeout=180.0,
)
async def graph_dependencies(
    path: str = ".",
    reindex: bool = False,
    top_k: int = 10,
    _exec_ctx: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Analyze file dependencies in codebase.

    Understand import relationships and file dependencies.
    Use for: Impact analysis, dependency mapping.

    Args:
        path: Path to codebase root (default: ".")
        reindex: Force rebuild of graph index
        top_k: Top K files to return

    Returns:
        Dict with dependency information for each file
    """
    return await graph(
        mode="file_deps",
        path=path,
        reindex=reindex,
        top_k=top_k,
        _exec_ctx=_exec_ctx,
    )


@tool(
    category="search",
    priority=Priority.LOW,
    access_mode=AccessMode.READONLY,
    danger_level=DangerLevel.SAFE,
    execution_category=ExecutionCategory.READ_ONLY,
    keywords=["graph", "patterns", "anti-patterns", "structure"],
    timeout=180.0,
)
async def graph_patterns(
    query: str,
    path: str = ".",
    top_k: int = 10,
    _exec_ctx: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Find code patterns in graph.

    Discover structural patterns, anti-patterns, and code smells.
    Use for: Finding design patterns, architectural patterns.

    Args:
        query: Pattern description or query
        path: Path to codebase root (default: ".")
        top_k: Top K patterns to return

    Returns:
        Dict with pattern matches and analysis
    """
    return await graph(
        mode="patterns",
        query=query,
        path=path,
        top_k=top_k,
        _exec_ctx=_exec_ctx,
    )


# Legacy alias for backward compatibility
graph_tool = graph

__all__ = [
    "ALL_EDGE_TYPES",
    "GraphAnalyzer",
    "GraphMode",
    "_load_graph",
    "graph",
    "graph_search",
    "graph_neighbors",
    "graph_analytics",
    "graph_path",
    "graph_dependencies",
    "graph_patterns",
    "graph_tool",
]
