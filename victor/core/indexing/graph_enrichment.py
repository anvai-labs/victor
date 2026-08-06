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

"""Post-index graph enrichment for synthetic architecture edges.

Reads and writes go through ``GraphStoreProtocol``, not raw SQL. This module
used to query and INSERT against SQLite's ``graph_node``/``graph_edge`` tables
directly, which meant it silently did nothing on any other backend: a
ProximaDB-backed ``victor init`` left those tables empty, so enrichment bailed at
the node-count guard and the run produced no IMPLEMENTS edges at all — with no
error and no log line, because "no nodes" and "wrong backend" looked identical.

Enrichment *state* (version + last-seen mtime) stays in the project database.
That is per-run bookkeeping about the enrichment pass, not graph content, so it
belongs with the project's other metadata regardless of graph backend.
"""

from __future__ import annotations

import ast
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from victor.core.async_utils import run_sync
from victor.core.database import get_project_database

logger = logging.getLogger(__name__)

_ENRICHMENT_VERSION = "1"
_VERSION_KEY = "graph_enrichment.version"
_LATEST_MTIME_KEY = "graph_enrichment.latest_mtime"

_TOOL_DECORATOR_NODE_ID = "symbol:victor/tools/decorators.py:tool"
_TOOL_METADATA_REGISTRY_NODE_ID = "symbol:victor/tools/metadata_registry.py:ToolMetadataRegistry"


@dataclass(frozen=True)
class GraphEnrichmentStats:
    """Summary of synthetic edges added to the persisted project graph."""

    implements_edges: int = 0
    decorates_edges: int = 0
    registers_edges: int = 0
    skipped: bool = False

    @property
    def total_edges(self) -> int:
        return self.implements_edges + self.decorates_edges + self.registers_edges


def ensure_project_graph_enriched(
    root_path: Path | str,
    *,
    latest_mtime: Optional[float] = None,
    force: bool = False,
    graph_store: Any = None,
) -> GraphEnrichmentStats:
    """Ensure project graph includes synthetic architecture edges.

    Args:
        root_path: Project root.
        latest_mtime: Newest source mtime seen by the caller; used to skip when
            enrichment already covers this state.
        force: Run even when the recorded state looks current.
        graph_store: Optional pre-built store. When omitted, one is resolved via
            ``create_graph_store("auto", …)`` so the per-repo
            ``.victor/graph_backend`` marker is honored — the same resolution the
            indexing pipeline and query tools use.
    """

    root = Path(root_path)
    project_db = get_project_database(root)

    if not force and _is_enrichment_current(project_db, latest_mtime):
        return GraphEnrichmentStats(skipped=True)

    stats = run_sync(_enrich_via_store(root, graph_store=graph_store))
    if stats.skipped:
        return stats

    with project_db.transaction() as conn:
        _record_enrichment_state(conn, latest_mtime)

    repo_root = root.resolve()
    if stats.total_edges:
        logger.info(
            "[graph-enrichment] Added %d synthetic edges for %s "
            "(IMPLEMENTS=%d, DECORATES=%d, REGISTERS=%d)",
            stats.total_edges,
            repo_root,
            stats.implements_edges,
            stats.decorates_edges,
            stats.registers_edges,
        )
    else:
        logger.debug("[graph-enrichment] No synthetic edges added for %s", repo_root)

    return stats


async def _enrich_via_store(root: Path, *, graph_store: Any = None) -> GraphEnrichmentStats:
    """Compute and persist synthetic edges through the graph store."""
    from victor.storage.graph.registry import create_graph_store

    store = graph_store
    owns_store = store is None
    if owns_store:
        store = create_graph_store("auto", project_path=root)
        await store.initialize()

    try:
        nodes = await store.get_all_nodes()
        if not nodes:
            return GraphEnrichmentStats(skipped=True)
        existing_edges = await store.get_all_edges()

        # Only edges that are not already present count as inserted. The SQL
        # version relied on INSERT OR IGNORE's rowcount for this; deduping
        # against a read is the backend-agnostic equivalent, and it keeps the
        # pass idempotent across reruns.
        seen = {(e.src, e.dst, str(getattr(e.type, "value", e.type))) for e in existing_edges}

        new_edges: List[Any] = []
        implements = _protocol_implementation_edges(nodes, existing_edges, seen, new_edges)
        decorates, registers = _tool_registration_edges(nodes, root.resolve(), seen, new_edges)

        if new_edges:
            await store.upsert_edges(new_edges)

        return GraphEnrichmentStats(
            implements_edges=implements,
            decorates_edges=decorates,
            registers_edges=registers,
            skipped=False,
        )
    finally:
        if owns_store:
            await store.close()


def _make_edge(src: str, dst: str, edge_type: str, metadata: Dict[str, object]) -> Any:
    from victor.storage.graph import GraphEdge

    return GraphEdge(src=src, dst=dst, type=edge_type, metadata=dict(metadata))


def _add_edge(
    src: str,
    dst: str,
    edge_type: str,
    metadata: Dict[str, object],
    seen: Set[Tuple[str, str, str]],
    out: List[Any],
) -> int:
    """Queue an edge unless it already exists. Returns 1 if queued."""
    key = (src, dst, edge_type)
    if key in seen:
        return 0
    seen.add(key)
    out.append(_make_edge(src, dst, edge_type, metadata))
    return 1


def _protocol_implementation_edges(
    nodes: Sequence[Any],
    edges: Sequence[Any],
    seen: Set[Tuple[str, str, str]],
    out: List[Any],
) -> int:
    """A class INHERITing a ``*Protocol`` class also IMPLEMENTS it."""
    protocol_ids = {
        n.node_id for n in nodes if n.type == "class" and (n.name or "").endswith("Protocol")
    }
    if not protocol_ids:
        return 0

    metadata = {
        "synthetic": True,
        "inferred_from": "INHERITS",
        "rule": "protocol_suffix_target",
    }
    inserted = 0
    for edge in edges:
        if str(getattr(edge.type, "value", edge.type)) != "INHERITS":
            continue
        if edge.dst not in protocol_ids:
            continue
        inserted += _add_edge(edge.src, edge.dst, "IMPLEMENTS", metadata, seen, out)
    return inserted


def _tool_registration_edges(
    nodes: Sequence[Any],
    repo_root: Path,
    seen: Set[Tuple[str, str, str]],
    out: List[Any],
) -> Tuple[int, int]:
    """``@tool``-decorated symbols are DECORATES targets and REGISTERS sources."""
    node_ids = {n.node_id for n in nodes}
    if _TOOL_DECORATOR_NODE_ID not in node_ids:
        return 0, 0
    has_registry = _TOOL_METADATA_REGISTRY_NODE_ID in node_ids

    node_lookup: Dict[str, Dict[Tuple[str, int], str]] = {}
    for node in nodes:
        if node.type not in ("function", "class"):
            continue
        if not node.file or not str(node.file).endswith(".py") or node.line is None:
            continue
        node_lookup.setdefault(str(node.file), {})[(str(node.name), int(node.line))] = str(
            node.node_id
        )

    decorates_inserted = 0
    registers_inserted = 0
    for rel_path, nodes_by_name_line in node_lookup.items():
        abs_path = repo_root / rel_path
        if not abs_path.exists():
            continue
        try:
            content = abs_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if "@tool" not in content or "victor.tools.decorators" not in content:
            continue

        discovered = _find_tool_decorated_nodes(content, rel_path, nodes_by_name_line)
        for target_node_id in discovered:
            decorates_inserted += _add_edge(
                _TOOL_DECORATOR_NODE_ID,
                target_node_id,
                "DECORATES",
                {"synthetic": True, "inferred_from": "python_ast", "decorator": "tool"},
                seen,
                out,
            )
            if has_registry:
                registers_inserted += _add_edge(
                    target_node_id,
                    _TOOL_METADATA_REGISTRY_NODE_ID,
                    "REGISTERS",
                    {
                        "synthetic": True,
                        "inferred_from": "python_ast",
                        "via_decorator": "tool",
                        "registry": "ToolMetadataRegistry",
                    },
                    seen,
                    out,
                )

    return decorates_inserted, registers_inserted


def _is_enrichment_current(project_db: object, latest_mtime: Optional[float]) -> bool:
    version_row = project_db.query_one(
        "SELECT value FROM _project_metadata WHERE key = ?",
        (_VERSION_KEY,),
    )
    if version_row is None or str(version_row[0]) != _ENRICHMENT_VERSION:
        return False

    if latest_mtime is None:
        return True

    mtime_row = project_db.query_one(
        "SELECT value FROM _project_metadata WHERE key = ?",
        (_LATEST_MTIME_KEY,),
    )
    if mtime_row is None:
        return False

    try:
        recorded = float(mtime_row[0])
    except (TypeError, ValueError):
        return False
    return recorded >= float(latest_mtime)


def _record_enrichment_state(conn: object, latest_mtime: Optional[float]) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO _project_metadata (key, value, updated_at)
        VALUES (?, ?, datetime('now'))
        """,
        (_VERSION_KEY, _ENRICHMENT_VERSION),
    )
    if latest_mtime is not None:
        conn.execute(
            """
            INSERT OR REPLACE INTO _project_metadata (key, value, updated_at)
            VALUES (?, ?, datetime('now'))
            """,
            (_LATEST_MTIME_KEY, str(float(latest_mtime))),
        )


def _find_tool_decorated_nodes(
    content: str,
    rel_path: str,
    nodes_by_name_line: Dict[Tuple[str, int], str],
) -> List[str]:
    try:
        tree = ast.parse(content, filename=rel_path)
    except SyntaxError:
        return []

    direct_tool_aliases: Set[str] = set()
    module_tool_aliases: Set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "victor.tools.decorators":
            for alias in node.names:
                if alias.name == "tool":
                    direct_tool_aliases.add(alias.asname or alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "victor.tools.decorators":
                    module_tool_aliases.add(alias.asname or alias.name)

    decorated: List[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if not _has_tool_decorator(node.decorator_list, direct_tool_aliases, module_tool_aliases):
            continue
        node_id = nodes_by_name_line.get((node.name, int(node.lineno)))
        if node_id:
            decorated.append(node_id)
    return decorated


def _has_tool_decorator(
    decorators: Iterable[ast.expr],
    direct_tool_aliases: Set[str],
    module_tool_aliases: Set[str],
) -> bool:
    return any(
        _resolve_tool_decorator(decorator, direct_tool_aliases, module_tool_aliases)
        for decorator in decorators
    )


def _resolve_tool_decorator(
    decorator: ast.expr,
    direct_tool_aliases: Set[str],
    module_tool_aliases: Set[str],
) -> bool:
    target = decorator.func if isinstance(decorator, ast.Call) else decorator
    if isinstance(target, ast.Name):
        return target.id in direct_tool_aliases
    if isinstance(target, ast.Attribute):
        dotted = _attribute_path(target)
        if dotted == ("victor", "tools", "decorators", "tool"):
            return True
        if len(dotted) == 2 and dotted[0] in module_tool_aliases and dotted[1] == "tool":
            return True
    return False


def _attribute_path(node: ast.Attribute) -> Tuple[str, ...]:
    parts: List[str] = []
    current: ast.expr = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
        parts.reverse()
        return tuple(parts)
    return ()


__all__ = ["GraphEnrichmentStats", "ensure_project_graph_enriched"]
