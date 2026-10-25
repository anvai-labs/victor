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

"""JSON graph queries for the interactive explorer (ADR-033 phase 1).

The mermaid layer (``visualize.py``) renders static snapshots; this module
answers the SAME aggregations as Cytoscape-ready element lists (the shape
``graph_export.py`` established for workflow graphs) so the web explorer
can navigate package -> module -> class as zoom states of one graph.
Deterministic ordering everywhere (weight desc, name asc) — golden tests
pin it. Same scope as visualize.py: the SQLite project store; the store
protocol cannot express the GROUP-BY aggregations.
"""

from __future__ import annotations

import sqlite3
from collections import defaultdict
from typing import Any, Dict, List, Optional

from victor.core.graph_rag.visualize import _package_of

_EDGE_TYPES = ("CALLS", "REFERENCES")
# Statement nodes are CCG machinery (97% of upserts, FTS-skipped): the
# explorer shows symbols, not statement soup.
_SELECT_NODE_TYPES = ("class", "function", "method", "module", "struct", "interface", "enum")


def _connect(db_path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def _elements(
    nodes: Dict[str, Dict[str, Any]],
    edges: Dict[tuple[str, str, str], Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    """Render the internal maps to the Cytoscape elements shape."""
    return {
        "nodes": [nodes[key] for key in sorted(nodes)],
        "edges": [edges[key] for key in sorted(edges, key=lambda e: (e[0], e[1], e[2]))],
    }


def _node_entry(node_id: str, label: str, **data: Any) -> Dict[str, Any]:
    return {"data": {"id": node_id, "label": label, **data}}


def _edge_entry(src: str, dst: str, kind: str, weight: int) -> Dict[str, Any]:
    return {
        "data": {
            "id": f"{src}->{dst}:{kind}",
            "source": src,
            "target": dst,
            "kind": kind,
            "weight": weight,
        }
    }


def package_graph(db_path, top_packages: int = 12) -> Dict[str, List[Dict[str, Any]]]:
    """Cross-package coupling as Cytoscape elements (weight = CALLS/REFERENCES
    count). Nodes carry degree; edges carry weight."""
    conn = _connect(db_path)
    try:
        weights: Dict[tuple[str, str], int] = defaultdict(int)
        for row in conn.execute(
            f"""
            SELECT ns.file AS src_file, nd.file AS dst_file
            FROM graph_edge e
            JOIN graph_node ns ON ns.node_id = e.src
            JOIN graph_node nd ON nd.node_id = e.dst
            WHERE e.type IN ({", ".join("?" * len(_EDGE_TYPES))})
              AND ns.file != nd.file
            """,
            _EDGE_TYPES,
        ):
            src = _package_of(row["src_file"])
            dst = _package_of(row["dst_file"])
            if src and dst and src != dst:
                weights[(src, dst)] += 1
    finally:
        conn.close()

    ranked = sorted(weights.items(), key=lambda kv: (-kv[1], kv[0]))
    packages: List[str] = []
    for (src, dst), _ in ranked:
        for candidate in (src, dst):
            if candidate not in packages:
                packages.append(candidate)
        if len(packages) >= top_packages:
            break
    top = set(packages)

    degree: Dict[str, int] = defaultdict(int)
    nodes: Dict[str, Dict[str, Any]] = {
        p: _node_entry(p, p.rsplit("/", 1)[-1], kind="package") for p in packages
    }
    edges: Dict[tuple[str, str, str], Dict[str, Any]] = {}
    for (src, dst), weight in ranked:
        if src in top and dst in top:
            degree[src] += weight
            degree[dst] += weight
            edges[(src, dst, "coupling")] = _edge_entry(src, dst, "coupling", weight)
    for package, deg in degree.items():
        nodes[package]["data"]["degree"] = deg
    return _elements(nodes, edges)


def module_graph(db_path, package: str, top_modules: int = 15) -> Dict[str, List[Dict[str, Any]]]:
    """File-level coupling inside one package as Cytoscape elements."""
    conn = _connect(db_path)
    try:
        weights: Dict[tuple[str, str], int] = defaultdict(int)
        node_rows = conn.execute(
            "SELECT DISTINCT file FROM graph_node WHERE file LIKE ?",
            (f"{package}/%",),
        ).fetchall()
        file_kinds: Dict[str, str] = {row["file"]: "module" for row in node_rows}
        for row in conn.execute(
            f"""
            SELECT ns.file AS src_file, nd.file AS dst_file
            FROM graph_edge e
            JOIN graph_node ns ON ns.node_id = e.src
            JOIN graph_node nd ON nd.node_id = e.dst
            WHERE e.type IN ({", ".join("?" * len(_EDGE_TYPES))})
              AND ns.file != nd.file
              AND ns.file LIKE ?
              AND nd.file LIKE ?
            """,
            (*_EDGE_TYPES, f"{package}/%", f"{package}/%"),
        ):
            weights[(row["src_file"], row["dst_file"])] += 1
    finally:
        conn.close()

    ranked = sorted(weights.items(), key=lambda kv: (-kv[1], kv[0]))
    modules: List[str] = []
    for (src, dst), _ in ranked:
        for candidate in (src, dst):
            if candidate not in modules:
                modules.append(candidate)
        if len(modules) >= top_modules:
            break
    top = set(modules)

    nodes: Dict[str, Dict[str, Any]] = {}
    for file_path in sorted(file_kinds):
        if file_path in top or not top:
            nodes[file_path] = _node_entry(file_path, file_path.rsplit("/", 1)[-1], kind="module")
    edges: Dict[tuple[str, str, str], Dict[str, Any]] = {}
    for (src, dst), weight in ranked:
        if src in top and dst in top:
            edges[(src, dst, "coupling")] = _edge_entry(src, dst, "coupling", weight)
    return _elements(nodes, edges)


def class_graph(
    db_path, module_file: str, top_symbols: int = 40
) -> Dict[str, List[Dict[str, Any]]]:
    """Symbol-level elements for one module (symbols + their intra-module
    edges), statement nodes excluded."""
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            f"""
            SELECT node_id, name, type, line, metadata
            FROM graph_node
            WHERE file = ?
              AND node_id IN (
                SELECT node_id FROM graph_node
                WHERE file = ? AND type IN ({", ".join("?" * len(_SELECT_NODE_TYPES))})
              )
            ORDER BY line, name
            """,
            (module_file, module_file, *_SELECT_NODE_TYPES),
        ).fetchall()
        node_ids = {row["node_id"] for row in rows}
        edges: Dict[tuple[str, str, str], Dict[str, Any]] = {}
        for row in conn.execute(
            f"""
            SELECT e.src, e.dst, e.type, e.metadata
            FROM graph_edge e
            JOIN graph_node ns ON ns.node_id = e.src
            JOIN graph_node nd ON nd.node_id = e.dst
            WHERE e.type IN ({", ".join("?" * len(_EDGE_TYPES))})
              AND ns.file = ? AND nd.file = ?
            """,
            (*_EDGE_TYPES, module_file, module_file),
        ):
            if row["src"] in node_ids and row["dst"] in node_ids:
                key = (row["src"], row["dst"], row["type"])
                edges[key] = _edge_entry(row["src"], row["dst"], row["type"], 1)
    finally:
        conn.close()

    # Degree-ranked cap: keep the top symbols by edge involvement.
    degree: Dict[str, int] = defaultdict(int)
    for src, dst, _ in edges:
        degree[src] += 1
        degree[dst] += 1
    keep = {nid for nid, _ in sorted(degree.items(), key=lambda kv: (-kv[1], kv[0]))[:top_symbols]}
    keep.update(list(node_ids)[: max(0, top_symbols - len(keep))])

    nodes: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        if row["node_id"] in keep:
            nodes[row["node_id"]] = _node_entry(
                row["node_id"], row["name"], kind=row["type"], line=row["line"]
            )
    edges = {key: entry for key, entry in edges.items() if key[0] in keep and key[1] in keep}
    return _elements(nodes, edges)


def search(db_path, query: str, limit: int = 25) -> List[Dict[str, Any]]:
    """Symbol search results as node entries (name match, statement-free)."""
    conn = _connect(db_path)
    try:
        like = f"%{query}%"
        placeholders = ",".join("?" for _ in _SELECT_NODE_TYPES)
        rows = conn.execute(
            f"""
            SELECT node_id, name, type, file, line
            FROM graph_node
            WHERE name LIKE ?
              AND type IN ({placeholders})
            ORDER BY name, file, line
            LIMIT ?
            """,
            (like, *_SELECT_NODE_TYPES, limit),
        ).fetchall()
    finally:
        conn.close()
    return [
        {
            "data": {
                "id": row["node_id"],
                "label": row["name"],
                "kind": row["type"],
                "file": row["file"],
                "line": row["line"],
            }
        }
        for row in rows
    ]
