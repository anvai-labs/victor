# DuckDB-backed GraphStoreProtocol implementation (optional dependency).
from __future__ import annotations

import asyncio
from pathlib import Path

from victor.core.json_utils import json_dumps, json_loads
from typing import Any, Dict, Iterable, List, Optional

from victor.storage.graph.protocol import (
    GraphEdge,
    GraphNode,
    GraphStoreProtocol,
    GraphTraversalDirection,
)


class DuckDBGraphStore(GraphStoreProtocol):
    """Embedded DuckDB graph store for per-repo symbol graphs.

    Uses a single-file DuckDB database with simple tables for nodes and edges.
    """

    def __init__(self, db_path: Path) -> None:
        try:
            import duckdb  # type: ignore
        except Exception as exc:  # pragma: no cover - import guard
            raise RuntimeError("duckdb must be installed to use DuckDBGraphStore") from exc

        self.duckdb = duckdb
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()
        self._ensure_schema()

    def _connect(self):
        return self.duckdb.connect(str(self.db_path))

    def _ensure_schema(self) -> None:
        conn = self._connect()
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS nodes (
                    node_id TEXT PRIMARY KEY,
                    type TEXT,
                    name TEXT,
                    file TEXT,
                    line BIGINT,
                    lang TEXT,
                    embedding_ref TEXT,
                    metadata TEXT
                );
                CREATE TABLE IF NOT EXISTS edges (
                    src TEXT,
                    dst TEXT,
                    type TEXT,
                    weight DOUBLE,
                    metadata TEXT,
                    PRIMARY KEY (src, dst, type)
                );
                CREATE INDEX IF NOT EXISTS idx_nodes_type_name ON nodes(type, name);
                CREATE INDEX IF NOT EXISTS idx_nodes_file ON nodes(file);
                CREATE INDEX IF NOT EXISTS idx_edges_src_type ON edges(src, type);
                CREATE INDEX IF NOT EXISTS idx_edges_dst_type ON edges(dst, type);
                """)
        finally:
            conn.close()

    async def upsert_nodes(self, nodes: Iterable[GraphNode]) -> None:
        rows = [
            (
                n.node_id,
                n.type,
                n.name,
                n.file,
                n.line,
                n.lang,
                n.embedding_ref,
                json_dumps(n.metadata),
            )
            for n in nodes
        ]
        if not rows:
            return
        async with self._lock:
            conn = self._connect()
            try:
                conn.executemany(
                    """
                    INSERT INTO nodes(node_id, type, name, file, line, lang, embedding_ref, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(node_id) DO UPDATE SET
                        type=excluded.type,
                        name=excluded.name,
                        file=excluded.file,
                        line=excluded.line,
                        lang=excluded.lang,
                        embedding_ref=excluded.embedding_ref,
                        metadata=excluded.metadata
                    """,
                    rows,
                )
            finally:
                conn.close()

    async def upsert_edges(self, edges: Iterable[GraphEdge]) -> None:
        rows = [
            (
                e.src,
                e.dst,
                e.type,
                e.weight,
                json_dumps(e.metadata),
            )
            for e in edges
        ]
        if not rows:
            return
        async with self._lock:
            conn = self._connect()
            try:
                conn.executemany(
                    """
                    INSERT INTO edges(src, dst, type, weight, metadata)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(src, dst, type) DO UPDATE SET
                        weight=excluded.weight,
                        metadata=excluded.metadata
                    """,
                    rows,
                )
            finally:
                conn.close()

    async def get_neighbors(
        self,
        node_id: str,
        edge_types: Optional[Iterable[str]] = None,
        *,
        direction: GraphTraversalDirection = "both",
        max_depth: int = 1,
    ) -> List[GraphEdge]:
        if direction not in {"out", "in", "both"}:
            raise ValueError(f"Unsupported graph traversal direction: {direction}")
        if max_depth < 1:
            return []

        allowed_types = list(edge_types) if edge_types else []
        frontier = {node_id}
        visited_nodes = {node_id}
        seen_edges: Dict[tuple[str, str, str], GraphEdge] = {}

        async with self._lock:
            conn = self._connect()
            try:
                for _depth in range(max_depth):
                    traversed_edges: List[tuple[GraphEdge, str]] = []

                    if direction in {"out", "both"}:
                        traversed_edges.extend(
                            self._select_frontier_edges(
                                conn,
                                frontier=frontier,
                                edge_types=allowed_types,
                                node_column="src",
                                neighbor_column="dst",
                            )
                        )
                    if direction in {"in", "both"}:
                        traversed_edges.extend(
                            self._select_frontier_edges(
                                conn,
                                frontier=frontier,
                                edge_types=allowed_types,
                                node_column="dst",
                                neighbor_column="src",
                            )
                        )

                    next_frontier: set[str] = set()
                    for edge, neighbor_id in traversed_edges:
                        seen_edges[(edge.src, edge.dst, edge.type)] = edge
                        next_frontier.add(neighbor_id)

                    next_frontier -= visited_nodes
                    if not next_frontier:
                        break
                    visited_nodes.update(next_frontier)
                    frontier = next_frontier
            finally:
                conn.close()

        return sorted(seen_edges.values(), key=lambda edge: (edge.src, edge.dst, edge.type))

    def _select_frontier_edges(
        self,
        conn: Any,
        *,
        frontier: set[str],
        edge_types: list[str],
        node_column: str,
        neighbor_column: str,
    ) -> List[tuple[GraphEdge, str]]:
        if not frontier:
            return []

        frontier_params = list(frontier)
        frontier_placeholders = ",".join("?" for _ in frontier_params)
        params: list[Any] = frontier_params
        type_clause = ""

        if edge_types:
            type_placeholders = ",".join("?" for _ in edge_types)
            type_clause = f" AND type IN ({type_placeholders})"
            params.extend(edge_types)

        query = f"""
            SELECT src, dst, type, weight, metadata, {neighbor_column}
            FROM edges
            WHERE {node_column} IN ({frontier_placeholders}){type_clause}
        """
        cur = conn.execute(query, params)
        return [
            (
                GraphEdge(
                    src=row[0],
                    dst=row[1],
                    type=row[2],
                    weight=row[3],
                    metadata=json_loads(row[4]) if row[4] else {},
                ),
                row[5],
            )
            for row in cur.fetchall()
        ]

    async def find_nodes(
        self,
        *,
        name: str | None = None,
        type: str | None = None,
        file: str | None = None,
    ) -> List[GraphNode]:
        clauses = []
        params: list[Any] = []
        if name:
            clauses.append("name = ?")
            params.append(name)
        if type:
            clauses.append("type = ?")
            params.append(type)
        if file:
            clauses.append("file = ?")
            params.append(file)
        where = " AND ".join(clauses) if clauses else "1=1"
        query = f"SELECT node_id, type, name, file, line, lang, embedding_ref, metadata FROM nodes WHERE {where}"
        async with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute(query, params)
                return [
                    GraphNode(
                        node_id=row[0],
                        type=row[1],
                        name=row[2],
                        file=row[3],
                        line=row[4],
                        lang=row[5],
                        embedding_ref=row[6],
                        metadata=json_loads(row[7]) if row[7] else {},
                    )
                    for row in cur.fetchall()
                ]
            finally:
                conn.close()

    async def delete_by_repo(self, clear_embeddings: bool = False) -> None:
        del clear_embeddings  # This backend has no external embedding collection.
        async with self._lock:
            conn = self._connect()
            try:
                conn.execute("DELETE FROM edges")
                conn.execute("DELETE FROM nodes")
            finally:
                conn.close()

    async def stats(self) -> Dict[str, Any]:
        async with self._lock:
            conn = self._connect()
            try:
                node_count = conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
                edge_count = conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
                return {
                    "nodes": node_count,
                    "edges": edge_count,
                    "path": str(self.db_path),
                    "backend": "duckdb",
                }
            finally:
                conn.close()
