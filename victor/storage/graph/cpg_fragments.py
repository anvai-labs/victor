# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
# SPDX-License-Identifier: Apache-2.0

"""Durable, on-demand storage for Tier-B intra-procedural CPG fragments.

The hot semantic graph must stay small enough for traversal. Statement nodes and
CFG/CDG/DDG edges therefore live in this focused SQLite index, partitioned by
file and scope and read only for explicit dataflow drilldowns. The class is kept
behind the Proxima graph adapter so callers retain the GraphStoreProtocol API and
the cold representation can later move to a columnar/PAX implementation without
changing the indexing pipeline.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path
from typing import Any, AsyncIterator, Iterable, List, Protocol

from victor.storage.graph.protocol import (
    GraphEdge,
    GraphNode,
    GraphTraversalDirection,
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS cpg_fragment_node (
    node_id TEXT PRIMARY KEY,
    type TEXT NOT NULL,
    name TEXT NOT NULL,
    file TEXT NOT NULL,
    line INTEGER,
    end_line INTEGER,
    lang TEXT,
    signature TEXT,
    docstring TEXT,
    parent_id TEXT,
    embedding_ref TEXT,
    metadata TEXT NOT NULL DEFAULT '{}',
    ast_kind TEXT,
    scope_id TEXT,
    statement_type TEXT,
    requirement_id TEXT,
    visibility TEXT
);

CREATE INDEX IF NOT EXISTS idx_cpg_fragment_node_file
    ON cpg_fragment_node(file, line);
CREATE INDEX IF NOT EXISTS idx_cpg_fragment_node_scope
    ON cpg_fragment_node(scope_id, file, line);
CREATE INDEX IF NOT EXISTS idx_cpg_fragment_node_statement
    ON cpg_fragment_node(statement_type, file, line);
CREATE INDEX IF NOT EXISTS idx_cpg_fragment_node_requirement
    ON cpg_fragment_node(requirement_id);

CREATE TABLE IF NOT EXISTS cpg_fragment_edge (
    src TEXT NOT NULL,
    dst TEXT NOT NULL,
    type TEXT NOT NULL,
    weight REAL,
    metadata TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY (src, dst, type)
);

CREATE INDEX IF NOT EXISTS idx_cpg_fragment_edge_src
    ON cpg_fragment_edge(src, type);
CREATE INDEX IF NOT EXISTS idx_cpg_fragment_edge_dst
    ON cpg_fragment_edge(dst, type);
"""

_NODE_COLUMNS = """
node_id, type, name, file, line, end_line, lang, signature, docstring,
parent_id, embedding_ref, metadata, ast_kind, scope_id, statement_type,
requirement_id, visibility
""".replace("\n", " ")


class CpgFragmentStoreProtocol(Protocol):
    """Narrow replacement seam for local or service-backed Tier-B fragments."""

    async def close(self) -> None: ...

    async def upsert_nodes(self, nodes: Iterable[GraphNode]) -> None: ...

    async def upsert_edges(self, edges: Iterable[GraphEdge]) -> None: ...

    async def find_nodes(
        self, *, name: str | None = None, file: str | None = None
    ) -> List[GraphNode]: ...

    async def get_node_by_id(self, node_id: str) -> GraphNode | None: ...

    async def get_nodes_by_file(self, file: str) -> List[GraphNode]: ...

    async def get_nodes_by_scope(self, scope_id: str) -> List[GraphNode]: ...

    async def get_nodes_by_statement_type(
        self, statement_type: str, *, file: str | None = None
    ) -> List[GraphNode]: ...

    async def get_nodes_by_requirement(self, requirement_id: str) -> List[GraphNode]: ...

    async def get_neighbors(
        self,
        node_id: str,
        edge_types: Iterable[str] | None = None,
        *,
        direction: GraphTraversalDirection = "both",
        max_depth: int = 1,
    ) -> List[GraphEdge]: ...

    def iter_edges(
        self,
        *,
        batch_size: int = 100,
        edge_types: Iterable[str] | None = None,
    ) -> AsyncIterator[List[GraphEdge]]: ...

    async def delete_by_file(self, file: str) -> None: ...

    async def clear(self) -> None: ...

    async def stats(self) -> dict[str, int]: ...


class CpgFragmentStore:
    """A durable local Tier-B store indexed for file/scope drilldown."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._connection: sqlite3.Connection | None = None
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        if self._connection is not None:
            return
        async with self._lock:
            if self._connection is not None:
                return
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._connection = await asyncio.to_thread(self._open)

    def _open(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, check_same_thread=False)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        connection.execute("PRAGMA foreign_keys=ON")
        connection.executescript(_SCHEMA)
        connection.commit()
        return connection

    async def close(self) -> None:
        async with self._lock:
            connection = self._connection
            self._connection = None
            if connection is not None:
                await asyncio.to_thread(connection.close)

    async def _ready(self) -> sqlite3.Connection:
        await self.initialize()
        assert self._connection is not None
        return self._connection

    @staticmethod
    def _node_row(node: GraphNode) -> tuple[Any, ...]:
        return (
            node.node_id,
            node.type,
            node.name,
            node.file,
            node.line,
            node.end_line,
            node.lang,
            node.signature,
            node.docstring,
            node.parent_id,
            node.embedding_ref,
            json.dumps(node.metadata),
            node.ast_kind,
            node.scope_id,
            node.statement_type,
            node.requirement_id,
            node.visibility,
        )

    @staticmethod
    def _row_to_node(row: sqlite3.Row) -> GraphNode:
        return GraphNode(
            node_id=row["node_id"],
            type=row["type"],
            name=row["name"],
            file=row["file"],
            line=row["line"],
            end_line=row["end_line"],
            lang=row["lang"],
            signature=row["signature"],
            docstring=row["docstring"],
            parent_id=row["parent_id"],
            embedding_ref=row["embedding_ref"],
            metadata=json.loads(row["metadata"] or "{}"),
            ast_kind=row["ast_kind"],
            scope_id=row["scope_id"],
            statement_type=row["statement_type"],
            requirement_id=row["requirement_id"],
            visibility=row["visibility"],
        )

    @staticmethod
    def _row_to_edge(row: sqlite3.Row) -> GraphEdge:
        return GraphEdge(
            src=row["src"],
            dst=row["dst"],
            type=row["type"],
            weight=row["weight"],
            metadata=json.loads(row["metadata"] or "{}"),
        )

    async def upsert_nodes(self, nodes: Iterable[GraphNode]) -> None:
        rows = [self._node_row(node) for node in nodes]
        if not rows:
            return
        connection = await self._ready()
        async with self._lock:
            await asyncio.to_thread(self._upsert_nodes, connection, rows)

    @staticmethod
    def _upsert_nodes(connection: sqlite3.Connection, rows: List[tuple[Any, ...]]) -> None:
        connection.executemany(
            """
            INSERT INTO cpg_fragment_node (
                node_id, type, name, file, line, end_line, lang, signature,
                docstring, parent_id, embedding_ref, metadata, ast_kind, scope_id,
                statement_type, requirement_id, visibility
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(node_id) DO UPDATE SET
                type=excluded.type, name=excluded.name, file=excluded.file,
                line=excluded.line, end_line=excluded.end_line, lang=excluded.lang,
                signature=excluded.signature, docstring=excluded.docstring,
                parent_id=excluded.parent_id, embedding_ref=excluded.embedding_ref,
                metadata=excluded.metadata, ast_kind=excluded.ast_kind,
                scope_id=excluded.scope_id, statement_type=excluded.statement_type,
                requirement_id=excluded.requirement_id, visibility=excluded.visibility
            """,
            rows,
        )
        connection.commit()

    async def upsert_edges(self, edges: Iterable[GraphEdge]) -> None:
        rows = [
            (edge.src, edge.dst, edge.type, edge.weight, json.dumps(edge.metadata))
            for edge in edges
        ]
        if not rows:
            return
        connection = await self._ready()
        async with self._lock:
            await asyncio.to_thread(self._upsert_edges, connection, rows)

    @staticmethod
    def _upsert_edges(connection: sqlite3.Connection, rows: List[tuple[Any, ...]]) -> None:
        connection.executemany(
            """
            INSERT INTO cpg_fragment_edge (src, dst, type, weight, metadata)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(src, dst, type) DO UPDATE SET
                weight=excluded.weight, metadata=excluded.metadata
            """,
            rows,
        )
        connection.commit()

    async def get_node_by_id(self, node_id: str) -> GraphNode | None:
        rows = await self._select_nodes("node_id = ?", [node_id])
        return rows[0] if rows else None

    async def get_nodes_by_file(self, file: str) -> List[GraphNode]:
        return await self._select_nodes("file = ?", [file])

    async def find_nodes(
        self,
        *,
        name: str | None = None,
        file: str | None = None,
    ) -> List[GraphNode]:
        clauses: List[str] = []
        params: List[Any] = []
        if name is not None:
            clauses.append("name = ?")
            params.append(name)
        if file is not None:
            clauses.append("file = ?")
            params.append(file)
        return await self._select_nodes(" AND ".join(clauses) or "1 = 1", params)

    async def get_nodes_by_scope(self, scope_id: str) -> List[GraphNode]:
        return await self._select_nodes("scope_id = ?", [scope_id])

    async def get_nodes_by_statement_type(
        self, statement_type: str, *, file: str | None = None
    ) -> List[GraphNode]:
        where = "statement_type = ?"
        params: List[Any] = [statement_type]
        if file is not None:
            where += " AND file = ?"
            params.append(file)
        return await self._select_nodes(where, params)

    async def get_nodes_by_requirement(self, requirement_id: str) -> List[GraphNode]:
        return await self._select_nodes("requirement_id = ?", [requirement_id])

    async def _select_nodes(self, where: str, params: List[Any]) -> List[GraphNode]:
        connection = await self._ready()
        async with self._lock:
            rows = await asyncio.to_thread(
                lambda: connection.execute(
                    f"SELECT {_NODE_COLUMNS} FROM cpg_fragment_node "
                    f"WHERE {where} ORDER BY file, COALESCE(line, 0), name, node_id",
                    params,
                ).fetchall()
            )
        return [self._row_to_node(row) for row in rows]

    async def get_neighbors(
        self,
        node_id: str,
        edge_types: Iterable[str] | None = None,
        *,
        direction: GraphTraversalDirection = "both",
        max_depth: int = 1,
    ) -> List[GraphEdge]:
        if direction not in {"out", "in", "both"}:
            raise ValueError(f"Unsupported graph traversal direction: {direction}")
        if max_depth < 1:
            return []
        connection = await self._ready()
        allowed = list(edge_types) if edge_types else []
        async with self._lock:
            return await asyncio.to_thread(
                self._get_neighbors,
                connection,
                node_id,
                allowed,
                direction,
                max_depth,
            )

    @classmethod
    def _get_neighbors(
        cls,
        connection: sqlite3.Connection,
        node_id: str,
        edge_types: List[str],
        direction: GraphTraversalDirection,
        max_depth: int,
    ) -> List[GraphEdge]:
        frontier = {node_id}
        visited = {node_id}
        found: dict[tuple[str, str, str], GraphEdge] = {}
        for _ in range(max_depth):
            next_frontier: set[str] = set()
            for column, neighbor_column in cls._directions(direction):
                if not frontier:
                    continue
                placeholders = ",".join("?" for _ in frontier)
                params: List[Any] = list(frontier)
                type_clause = ""
                if edge_types:
                    type_placeholders = ",".join("?" for _ in edge_types)
                    type_clause = f" AND type IN ({type_placeholders})"
                    params.extend(edge_types)
                rows = connection.execute(
                    "SELECT src, dst, type, weight, metadata "
                    f"FROM cpg_fragment_edge WHERE {column} IN ({placeholders})"
                    f"{type_clause}",
                    params,
                ).fetchall()
                for row in rows:
                    edge = cls._row_to_edge(row)
                    found[(edge.src, edge.dst, edge.type)] = edge
                    next_frontier.add(row[neighbor_column])
            next_frontier -= visited
            if not next_frontier:
                break
            visited.update(next_frontier)
            frontier = next_frontier
        return sorted(found.values(), key=lambda edge: (edge.src, edge.dst, edge.type))

    async def iter_edges(
        self,
        *,
        batch_size: int = 100,
        edge_types: Iterable[str] | None = None,
    ) -> AsyncIterator[List[GraphEdge]]:
        """Stream cold edges in bounded batches for explicit CPG analytics."""
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        connection = await self._ready()
        allowed = list(edge_types) if edge_types else []
        offset = 0
        while True:
            params: List[Any] = []
            where = ""
            if allowed:
                placeholders = ",".join("?" for _ in allowed)
                where = f"WHERE type IN ({placeholders})"
                params.extend(allowed)
            params.extend([batch_size, offset])
            async with self._lock:
                rows = await asyncio.to_thread(
                    lambda: connection.execute(
                        "SELECT src, dst, type, weight, metadata "
                        f"FROM cpg_fragment_edge {where} "
                        "ORDER BY src, dst, type LIMIT ? OFFSET ?",
                        params,
                    ).fetchall()
                )
            if not rows:
                break
            yield [self._row_to_edge(row) for row in rows]
            offset += len(rows)

    @staticmethod
    def _directions(
        direction: GraphTraversalDirection,
    ) -> List[tuple[str, str]]:
        result: List[tuple[str, str]] = []
        if direction in {"out", "both"}:
            result.append(("src", "dst"))
        if direction in {"in", "both"}:
            result.append(("dst", "src"))
        return result

    async def delete_by_file(self, file: str) -> None:
        connection = await self._ready()
        async with self._lock:
            await asyncio.to_thread(self._delete_by_file, connection, file)

    @staticmethod
    def _delete_by_file(connection: sqlite3.Connection, file: str) -> None:
        node_ids = [
            row[0]
            for row in connection.execute(
                "SELECT node_id FROM cpg_fragment_node WHERE file = ?", (file,)
            ).fetchall()
        ]
        if node_ids:
            placeholders = ",".join("?" for _ in node_ids)
            connection.execute(
                f"DELETE FROM cpg_fragment_edge WHERE src IN ({placeholders}) "
                f"OR dst IN ({placeholders})",
                [*node_ids, *node_ids],
            )
            connection.execute(
                f"DELETE FROM cpg_fragment_node WHERE node_id IN ({placeholders})", node_ids
            )
        connection.commit()

    async def clear(self) -> None:
        connection = await self._ready()
        async with self._lock:
            await asyncio.to_thread(self._clear, connection)

    @staticmethod
    def _clear(connection: sqlite3.Connection) -> None:
        connection.execute("DELETE FROM cpg_fragment_edge")
        connection.execute("DELETE FROM cpg_fragment_node")
        connection.commit()

    async def stats(self) -> dict[str, int]:
        connection = await self._ready()
        async with self._lock:
            rows = await asyncio.to_thread(
                lambda: (
                    connection.execute("SELECT COUNT(*) FROM cpg_fragment_node").fetchone()[0],
                    connection.execute("SELECT COUNT(*) FROM cpg_fragment_edge").fetchone()[0],
                    connection.execute(
                        "SELECT COUNT(DISTINCT file) FROM cpg_fragment_node"
                    ).fetchone()[0],
                )
            )
        return {"nodes": rows[0], "edges": rows[1], "files": rows[2]}


__all__ = ["CpgFragmentStore", "CpgFragmentStoreProtocol"]
