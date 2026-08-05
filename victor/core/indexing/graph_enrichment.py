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

"""Post-index graph enrichment for synthetic architecture edges."""

from __future__ import annotations

import ast
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

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
) -> GraphEnrichmentStats:
    """Ensure project graph includes synthetic architecture edges."""

    root = Path(root_path)
    project_db = get_project_database(root)
    if not project_db.table_exists("graph_node") or not project_db.table_exists("graph_edge"):
        return GraphEnrichmentStats(skipped=True)

    node_count_row = project_db.query_one("SELECT COUNT(*) FROM graph_node")
    if node_count_row is None or int(node_count_row[0]) == 0:
        return GraphEnrichmentStats(skipped=True)

    if not force and _is_enrichment_current(project_db, latest_mtime):
        return GraphEnrichmentStats(skipped=True)

    stats = GraphEnrichmentStats()
    repo_root = root.resolve()

    with project_db.transaction() as conn:
        implements_edges = _infer_protocol_implementation_edges(conn)
        decorates_edges, registers_edges = _infer_tool_registration_edges(conn, repo_root)
        stats = GraphEnrichmentStats(
            implements_edges=implements_edges,
            decorates_edges=decorates_edges,
            registers_edges=registers_edges,
            skipped=False,
        )
        _record_enrichment_state(conn, latest_mtime)

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


def _infer_protocol_implementation_edges(conn: object) -> int:
    rows = conn.execute("""
        SELECT e.src, e.dst
        FROM graph_edge e
        JOIN graph_node dst ON dst.node_id = e.dst
        WHERE e.type = 'INHERITS'
          AND dst.type = 'class'
          AND dst.name LIKE '%Protocol'
        """).fetchall()
    if not rows:
        return 0

    metadata = {
        "synthetic": True,
        "inferred_from": "INHERITS",
        "rule": "protocol_suffix_target",
    }
    inserted = 0
    for src, dst in rows:
        inserted += _insert_synthetic_edge(
            conn,
            src=str(src),
            dst=str(dst),
            edge_type="IMPLEMENTS",
            metadata=metadata,
        )
    return inserted


def _infer_tool_registration_edges(conn: object, repo_root: Path) -> Tuple[int, int]:
    available_nodes = {
        row[0]
        for row in conn.execute(
            "SELECT node_id FROM graph_node WHERE node_id IN (?, ?)",
            (_TOOL_DECORATOR_NODE_ID, _TOOL_METADATA_REGISTRY_NODE_ID),
        ).fetchall()
    }
    if _TOOL_DECORATOR_NODE_ID not in available_nodes:
        return 0, 0

    node_rows = conn.execute("""
        SELECT node_id, file, name, line
        FROM graph_node
        WHERE file LIKE '%.py'
          AND line IS NOT NULL
          AND type IN ('function', 'class')
        """).fetchall()
    node_lookup: Dict[str, Dict[Tuple[str, int], str]] = {}
    for node_id, file_path, name, line in node_rows:
        node_lookup.setdefault(str(file_path), {})[(str(name), int(line))] = str(node_id)

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
        if not discovered:
            continue

        for target_node_id in discovered:
            decorates_inserted += _insert_synthetic_edge(
                conn,
                src=_TOOL_DECORATOR_NODE_ID,
                dst=target_node_id,
                edge_type="DECORATES",
                metadata={
                    "synthetic": True,
                    "inferred_from": "python_ast",
                    "decorator": "tool",
                },
            )
            if _TOOL_METADATA_REGISTRY_NODE_ID in available_nodes:
                registers_inserted += _insert_synthetic_edge(
                    conn,
                    src=target_node_id,
                    dst=_TOOL_METADATA_REGISTRY_NODE_ID,
                    edge_type="REGISTERS",
                    metadata={
                        "synthetic": True,
                        "inferred_from": "python_ast",
                        "via_decorator": "tool",
                        "registry": "ToolMetadataRegistry",
                    },
                )

    return decorates_inserted, registers_inserted


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


def _insert_synthetic_edge(
    conn: object,
    *,
    src: str,
    dst: str,
    edge_type: str,
    metadata: Dict[str, object],
) -> int:
    cursor = conn.execute(
        """
        INSERT OR IGNORE INTO graph_edge (src, dst, type, weight, metadata)
        VALUES (?, ?, ?, NULL, ?)
        """,
        (src, dst, edge_type, json.dumps(metadata, sort_keys=True)),
    )
    return int(cursor.rowcount or 0)


__all__ = ["GraphEnrichmentStats", "ensure_project_graph_enriched"]
