# SQLite-backed GraphStoreProtocol implementation.
from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
from contextlib import asynccontextmanager
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Dict,
    Iterable,
    List,
    Optional,
    Set,
)

from victor.core.schema import Tables
from victor.storage.graph.protocol import (
    GraphEdge,
    GraphNode,
    GraphQueryResult,
    GraphStoreProtocol,
    GraphTraversalDirection,
)

if TYPE_CHECKING:
    from victor.core.database import ProjectDatabaseManager

logger = logging.getLogger(__name__)

# Table names from centralized schema
_NODE_TABLE = Tables.GRAPH_NODE
_EDGE_TABLE = Tables.GRAPH_EDGE
_MTIME_TABLE = Tables.GRAPH_FILE_MTIME
_FTS_TABLE = f"{_NODE_TABLE}_fts"

# Schema for consolidated database
SCHEMA = f"""
PRAGMA journal_mode=WAL;

-- Core symbol/node table (body read from file via line numbers, not stored here)
CREATE TABLE IF NOT EXISTS {_NODE_TABLE} (
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
    metadata TEXT,
    ast_kind TEXT,
    scope_id TEXT,
    statement_type TEXT,
    requirement_id TEXT,
    visibility TEXT,
    FOREIGN KEY (parent_id) REFERENCES {_NODE_TABLE}(node_id)
);

-- Graph edges table
CREATE TABLE IF NOT EXISTS {_EDGE_TABLE} (
    src TEXT NOT NULL,
    dst TEXT NOT NULL,
    type TEXT NOT NULL,
    weight REAL,
    file TEXT,
    metadata TEXT,
    PRIMARY KEY (src, dst, type)
);

-- File modification times for staleness tracking
CREATE TABLE IF NOT EXISTS {_MTIME_TABLE} (
    file TEXT PRIMARY KEY,
    mtime REAL NOT NULL,
    indexed_at REAL NOT NULL
);

-- Indexes for fast lookups
CREATE INDEX IF NOT EXISTS idx_{_NODE_TABLE}_type_name ON {_NODE_TABLE}(type, name);
CREATE INDEX IF NOT EXISTS idx_{_NODE_TABLE}_file ON {_NODE_TABLE}(file);
CREATE INDEX IF NOT EXISTS idx_{_NODE_TABLE}_parent ON {_NODE_TABLE}(parent_id);
CREATE INDEX IF NOT EXISTS idx_{_EDGE_TABLE}_src_type ON {_EDGE_TABLE}(src, type);
CREATE INDEX IF NOT EXISTS idx_{_EDGE_TABLE}_dst_type ON {_EDGE_TABLE}(dst, type);
CREATE INDEX IF NOT EXISTS idx_{_MTIME_TABLE}_mtime ON {_MTIME_TABLE}(mtime);
"""

# FTS5 schema for full-text search on symbols
FTS_SCHEMA = f"""
CREATE VIRTUAL TABLE IF NOT EXISTS {_FTS_TABLE} USING fts5(
    node_id,
    name,
    signature,
    docstring,
    content='{_NODE_TABLE}',
    content_rowid='rowid'
);

-- Triggers to keep FTS index in sync
CREATE TRIGGER IF NOT EXISTS {_NODE_TABLE}_ai AFTER INSERT ON {_NODE_TABLE} BEGIN
    INSERT INTO {_FTS_TABLE}(rowid, node_id, name, signature, docstring)
    VALUES (new.rowid, new.node_id, new.name, new.signature, new.docstring);
END;

CREATE TRIGGER IF NOT EXISTS {_NODE_TABLE}_ad AFTER DELETE ON {_NODE_TABLE} BEGIN
    INSERT INTO {_FTS_TABLE}({_FTS_TABLE}, rowid, node_id, name, signature, docstring)
    VALUES ('delete', old.rowid, old.node_id, old.name, old.signature, old.docstring);
END;

CREATE TRIGGER IF NOT EXISTS {_NODE_TABLE}_au AFTER UPDATE ON {_NODE_TABLE} BEGIN
    INSERT INTO {_FTS_TABLE}({_FTS_TABLE}, rowid, node_id, name, signature, docstring)
    VALUES ('delete', old.rowid, old.node_id, old.name, old.signature, old.docstring);
    INSERT INTO {_FTS_TABLE}(rowid, node_id, name, signature, docstring)
    VALUES (new.rowid, new.node_id, new.name, new.signature, new.docstring);
END;
"""


class SqliteGraphStore(GraphStoreProtocol):
    """Embedded SQLite graph store using consolidated project.db."""

    def __init__(self, project_path: Optional[Path] = None) -> None:
        """Initialize graph store.

        Args:
            project_path: Path to project root. If None, uses current directory.
        """
        from victor.core.database import get_project_database

        self._db = get_project_database(project_path)
        self.db_path = self._db.db_path
        self._project_root = Path(self._db.project_root).resolve()
        self._ensure_schema()
        self._record_project_metadata()
        self._lock = asyncio.Lock()
        self._write_batch_conn: sqlite3.Connection | None = None
        self._write_batch_owner: asyncio.Task[Any] | None = None
        self._write_batch_depth = 0
        self._edge_has_file_column: bool | None = None

    async def initialize(self) -> None:
        """Ensure schema exists for compatibility with higher-level stores."""
        self._ensure_schema()

    async def close(self) -> None:
        """Release store resources without closing the shared project database.

        ``ProjectDatabaseManager`` is a shared singleton per project path.
        Closing it here invalidates other live components on the same thread.
        Test isolation and process shutdown already clean up the shared manager.
        """
        logger.debug("SqliteGraphStore close skipped shared project database shutdown")

    @property
    def repo_root(self) -> Path:
        """Resolved project root this store indexes (used for cache scoping)."""
        return self._project_root

    def _connect(self) -> sqlite3.Connection:
        """Get database connection."""
        return self._db.get_connection()

    def _get_active_write_batch_connection(self) -> sqlite3.Connection | None:
        """Return the active write-batch connection for the current task, if any."""
        current_task = asyncio.current_task()
        if (
            current_task is not None
            and current_task is self._write_batch_owner
            and self._write_batch_depth > 0
        ):
            return self._write_batch_conn
        return None

    def _canonical_file_path(self, file: str | Path) -> str:
        """Normalize project-local file paths to repo-relative graph keys."""
        path = Path(file).expanduser()
        if not path.is_absolute():
            path = self._project_root / path
        try:
            resolved = path.resolve(strict=False)
        except OSError:
            resolved = path.absolute()
        try:
            return resolved.relative_to(self._project_root).as_posix()
        except ValueError:
            return str(resolved)

    def _file_path_variants(self, file: str | Path) -> List[str]:
        """Return raw plus canonical path forms for compatibility lookups."""
        raw_path = str(file)
        canonical_path = self._canonical_file_path(raw_path)
        variants = [raw_path, canonical_path]
        raw_candidate = Path(raw_path).expanduser()
        if not raw_candidate.is_absolute():
            variants.append(str((self._project_root / raw_candidate).resolve(strict=False)))
        return list(dict.fromkeys(variants))

    def _ensure_schema(self) -> None:
        conn = self._connect()
        conn.executescript(SCHEMA)
        # Add new columns if upgrading from older schema
        self._migrate_schema(conn)
        self._edge_has_file_column = self._has_table_column(conn, _EDGE_TABLE, "file")
        # Create FTS5 table separately (handles IF NOT EXISTS)
        try:
            conn.executescript(FTS_SCHEMA)
        except sqlite3.OperationalError:
            # FTS5 might not be available on all SQLite builds
            pass
        conn.commit()

    def _record_project_metadata(self) -> None:
        """Record graph path identity metadata in the project database."""
        conn = self._connect()
        conn.execute(
            """
            INSERT OR REPLACE INTO _project_metadata (key, value, updated_at)
            VALUES (?, ?, datetime('now'))
            """,
            ("project_root", str(self._project_root)),
        )
        conn.execute(
            """
            INSERT OR REPLACE INTO _project_metadata (key, value, updated_at)
            VALUES (?, ?, datetime('now'))
            """,
            ("graph_file_path_identity", "repo_relative"),
        )
        conn.commit()

    def _enable_bulk_load_mode(self, conn: sqlite3.Connection) -> None:
        """Enable optimizations for bulk data loading.

        These settings significantly speed up bulk INSERT operations during
        force rebuilds by reducing fsync overhead and increasing cache sizes.

        WARNING: These settings assume a transaction will be committed atomically.
        Always call _disable_bulk_load_mode after bulk operations complete.
        """
        # Increase cache size to 50MB (negative value = KB)
        conn.execute("PRAGMA cache_size=-50000")
        # Enable memory-mapped I/O for 256MB
        conn.execute("PRAGMA mmap_size=268435456")
        # Store temporary tables in memory (not on disk)
        conn.execute("PRAGMA temp_store=MEMORY")
        # Disable fsync during transaction (safe because we commit atomically).
        # Keep WAL journal mode intact: changing journal_mode is database-wide
        # and can require an exclusive lock while other project DB connections
        # are alive.
        conn.execute("PRAGMA synchronous=OFF")

    def _disable_bulk_load_mode(self, conn: sqlite3.Connection) -> None:
        """Restore normal settings after bulk load completes.

        This restores safe defaults for normal operation where durability
        is important.
        """
        # Restore normal synchronous mode (still fast with WAL)
        conn.execute("PRAGMA synchronous=NORMAL")

    def _migrate_schema(self, conn: sqlite3.Connection) -> None:
        """Add new columns if upgrading from older schema."""
        cursor = conn.execute(f"PRAGMA table_info({_NODE_TABLE})")
        node_columns = {row[1] for row in cursor.fetchall()}
        edge_columns = {
            row[1] for row in conn.execute(f"PRAGMA table_info({_EDGE_TABLE})").fetchall()
        }

        # v4 columns (legacy)
        new_columns = [
            ("end_line", "INTEGER"),
            ("signature", "TEXT"),
            ("docstring", "TEXT"),
            ("parent_id", "TEXT"),
        ]
        # v5 columns (CCG and Graph RAG)
        new_columns.extend(
            [
                ("ast_kind", "TEXT"),
                ("scope_id", "TEXT"),
                ("statement_type", "TEXT"),
                ("requirement_id", "TEXT"),
                ("visibility", "TEXT"),
            ]
        )
        for col_name, col_type in new_columns:
            if col_name not in node_columns:
                try:
                    conn.execute(f"ALTER TABLE {_NODE_TABLE} ADD COLUMN {col_name} {col_type}")
                except sqlite3.OperationalError:
                    pass  # Column already exists

        # v5+ edge migration - add file column for direct file-aware deletes
        if "file" not in edge_columns:
            try:
                conn.execute(f"ALTER TABLE {_EDGE_TABLE} ADD COLUMN file TEXT")
            except sqlite3.OperationalError:
                pass

            # Backfill file using source/destination nodes when available.
            conn.execute(f"""
                UPDATE {_EDGE_TABLE}
                SET file = (
                    SELECT file FROM {_NODE_TABLE} n
                    WHERE n.node_id = {_EDGE_TABLE}.src
                )
                WHERE file IS NULL
                """)
            conn.execute(f"""
                UPDATE {_EDGE_TABLE}
                SET file = (
                    SELECT file FROM {_NODE_TABLE} n
                    WHERE n.node_id = {_EDGE_TABLE}.dst
                )
                WHERE file IS NULL
                """)

        # Keep file lookup fast when populated
        conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{_EDGE_TABLE}_file ON {_EDGE_TABLE}(file)")

    def _has_table_column(
        self,
        conn: sqlite3.Connection,
        table_name: str,
        column_name: str,
    ) -> bool:
        cursor = conn.execute(f"PRAGMA table_info({table_name})")
        return any(row[1] == column_name for row in cursor.fetchall())

    def _get_edge_file_hint(self, edge: GraphEdge) -> str | None:
        """Try to preserve file hints embedded in edge metadata/attributes."""
        file_hint = getattr(edge, "file", None)
        if file_hint:
            return self._canonical_file_path(file_hint)

        metadata = getattr(edge, "metadata", None)
        if isinstance(metadata, dict):
            file_hint = str(metadata.get("file") or metadata.get("file_path") or "") or None
            return self._canonical_file_path(file_hint) if file_hint else None

        return None

    @asynccontextmanager
    async def write_batch(self) -> AsyncIterator[None]:
        """Batch multiple graph writes into a single SQLite transaction."""
        current_task = asyncio.current_task()
        if current_task is None:
            raise RuntimeError("write_batch() requires an active asyncio task")

        if (
            current_task is self._write_batch_owner
            and self._write_batch_conn is not None
            and self._write_batch_depth > 0
        ):
            self._write_batch_depth += 1
            try:
                yield
            finally:
                self._write_batch_depth -= 1
            return

        async with self._lock:
            conn = self._connect()
            self._write_batch_owner = current_task
            self._write_batch_conn = conn
            self._write_batch_depth = 1
            try:
                conn.execute("BEGIN IMMEDIATE")
                yield
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                self._write_batch_owner = None
                self._write_batch_conn = None
                self._write_batch_depth = 0

    def _upsert_nodes_rows(
        self,
        conn: sqlite3.Connection,
        rows: List[tuple[Any, ...]],
    ) -> None:
        """Write node rows using the provided connection."""
        conn.executemany(
            f"""
            INSERT INTO {_NODE_TABLE}(node_id, type, name, file, line, end_line, lang,
                             signature, docstring, parent_id, embedding_ref, metadata,
                             ast_kind, scope_id, statement_type, requirement_id, visibility)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(node_id) DO UPDATE SET
                type=excluded.type,
                name=excluded.name,
                file=excluded.file,
                line=excluded.line,
                end_line=excluded.end_line,
                lang=excluded.lang,
                signature=excluded.signature,
                docstring=excluded.docstring,
                parent_id=excluded.parent_id,
                embedding_ref=excluded.embedding_ref,
                metadata=excluded.metadata,
                ast_kind=excluded.ast_kind,
                scope_id=excluded.scope_id,
                statement_type=excluded.statement_type,
                requirement_id=excluded.requirement_id,
                visibility=excluded.visibility
            """,
            rows,
        )

    def _upsert_edges_rows(
        self,
        conn: sqlite3.Connection,
        rows: List[tuple[Any, ...]],
    ) -> None:
        """Write edge rows using the provided connection."""
        has_file_column = self._edge_has_file_column
        if has_file_column is None:
            has_file_column = self._has_table_column(conn, _EDGE_TABLE, "file")
            self._edge_has_file_column = has_file_column

        if has_file_column:
            conn.executemany(
                f"""
                INSERT INTO {_EDGE_TABLE}(src, dst, type, weight, file, metadata)
                VALUES (?, ?, ?, ?, COALESCE(?,
                    (SELECT file FROM {_NODE_TABLE} n WHERE n.node_id = ?),
                    (SELECT file FROM {_NODE_TABLE} n WHERE n.node_id = ?)
                ), ?)
                ON CONFLICT(src, dst, type) DO UPDATE SET
                    file=COALESCE(excluded.file, {_EDGE_TABLE}.file),
                    weight=excluded.weight,
                    metadata=excluded.metadata
                """,
                rows,
            )
            return

        conn.executemany(
            f"""
            INSERT INTO {_EDGE_TABLE}(src, dst, type, weight, metadata)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(src, dst, type) DO UPDATE SET
                weight=excluded.weight,
                metadata=excluded.metadata
            """,
            rows,
        )

    def _update_file_mtime_conn(self, conn: sqlite3.Connection, file: str, mtime: float) -> None:
        """Record file modification time using the provided connection."""
        import time

        conn.execute(
            f"""
            INSERT INTO {_MTIME_TABLE}(file, mtime, indexed_at)
            VALUES (?, ?, ?)
            ON CONFLICT(file) DO UPDATE SET
                mtime=excluded.mtime,
                indexed_at=excluded.indexed_at
            """,
            (file, mtime, time.time()),
        )

    def _delete_by_file_conn(self, conn: sqlite3.Connection, file: str) -> None:
        """Delete all nodes, edges, and mtimes for a specific file."""
        file_variants = self._file_path_variants(file)
        file_placeholders = ",".join("?" for _ in file_variants)

        has_file_column = self._edge_has_file_column
        if has_file_column is None:
            has_file_column = self._has_table_column(conn, _EDGE_TABLE, "file")
            self._edge_has_file_column = has_file_column

        if has_file_column:
            conn.execute(
                f"DELETE FROM {_EDGE_TABLE} WHERE file IN ({file_placeholders})",
                file_variants,
            )

        # Get all node_ids for nodes in this file
        cur = conn.execute(
            f"SELECT node_id FROM {_NODE_TABLE} WHERE file IN ({file_placeholders})",
            file_variants,
        )
        node_ids = [row[0] for row in cur.fetchall()]

        if node_ids:
            placeholders = ",".join("?" for _ in node_ids)
            # Delete all edges connected to these nodes
            conn.execute(f"DELETE FROM {_EDGE_TABLE} WHERE src IN ({placeholders})", node_ids)
            conn.execute(f"DELETE FROM {_EDGE_TABLE} WHERE dst IN ({placeholders})", node_ids)
            # Delete the nodes themselves
            conn.execute(
                f"DELETE FROM {_NODE_TABLE} WHERE node_id IN ({placeholders})",
                node_ids,
            )

        # Also delete from mtime table
        conn.execute(
            f"DELETE FROM {_MTIME_TABLE} WHERE file IN ({file_placeholders})",
            file_variants,
        )

    def _delete_by_repo_conn(self, conn: sqlite3.Connection) -> None:
        """Clear all repo graph tables using DROP + CREATE (O(1) vs O(n) DELETE FROM).

        SQLite's WAL mode journals every deleted row, making DELETE FROM on millions of
        rows extremely slow. DROP TABLE + CREATE TABLE is the SQLite TRUNCATE equivalent:
        it rewrites only the root page, completing in milliseconds regardless of row count.
        """
        # Drop in dependency order: triggers first, then FTS, then edge (FK ref), then node, then mtime.
        # Triggers are automatically dropped with their table, but FTS must be dropped before graph_node
        # because it holds a content= reference to it.
        conn.executescript(f"""
            DROP TABLE IF EXISTS {_FTS_TABLE};
            DROP TRIGGER IF EXISTS {_NODE_TABLE}_ai;
            DROP TRIGGER IF EXISTS {_NODE_TABLE}_ad;
            DROP TRIGGER IF EXISTS {_NODE_TABLE}_au;
            DROP TABLE IF EXISTS {_EDGE_TABLE};
            DROP TABLE IF EXISTS {_NODE_TABLE};
            DROP TABLE IF EXISTS {_MTIME_TABLE};
            """)
        # Recreate all tables, indexes, FTS virtual table, and sync triggers.
        conn.executescript(SCHEMA)
        conn.executescript(FTS_SCHEMA)
        # Re-apply column migrations: SCHEMA is the base definition without newer columns
        # (e.g. ast_kind added in v5). _migrate_schema() adds them idempotently.
        self._migrate_schema(conn)

    def _build_embedding_config(self) -> Any:
        """Build vector-store config from settings, falling back to defaults."""
        from victor.storage.vector_stores.base import EmbeddingConfig

        kwargs: Dict[str, Any] = {}
        try:
            from victor.config.settings import get_project_paths, load_settings

            settings = load_settings()
            paths = get_project_paths(self._project_root)
            kwargs = {
                "vector_store": getattr(settings, "codebase_vector_store", "lancedb"),
                "embedding_model_type": getattr(
                    settings, "codebase_embedding_provider", "sentence-transformers"
                ),
                "embedding_model_name": getattr(
                    settings, "codebase_embedding_model", "BAAI/bge-small-en-v1.5"
                ),
                "persist_directory": str(paths.embeddings_dir),
            }
        except Exception as e:
            logger.debug(f"Settings unavailable for embedding config, using defaults: {e}")
        return EmbeddingConfig(**kwargs)

    async def _delete_embeddings_for_file(self, file: str, node_ids: List[str]) -> None:
        """Delete embeddings for nodes from vector store.

        Gracefully handles missing vector store or failures. This ensures
        that when graph nodes are deleted, their corresponding embeddings
        are also cleaned up to prevent orphaned data.

        Args:
            file: File path being deleted
            node_ids: List of node IDs being deleted (for potential per-node cleanup)
        """
        try:
            from victor.storage.vector_stores.registry import EmbeddingRegistry

            provider = EmbeddingRegistry.create(self._build_embedding_config())

            # Vector stores index repo-relative paths; use the canonical form so
            # the delete predicate matches what was stored at index time.
            canonical = self._canonical_file_path(file)
            deleted_count = await provider.delete_by_file(canonical)
            if deleted_count:
                logger.info(f"Deleted {deleted_count} embeddings for {canonical}")

        except ImportError:
            logger.debug("Vector store not available, skipping embedding cleanup")
        except Exception as e:
            logger.warning(f"Failed to delete embeddings for {file}: {e}")

    async def upsert_nodes(self, nodes: Iterable[GraphNode]) -> None:
        rows = [
            (
                n.node_id,
                n.type,
                n.name,
                self._canonical_file_path(n.file),
                n.line,
                n.end_line,
                n.lang,
                n.signature,
                n.docstring,
                n.parent_id,
                n.embedding_ref,
                json.dumps(n.metadata),
                # v5 CCG fields
                n.ast_kind,
                n.scope_id,
                n.statement_type,
                n.requirement_id,
                n.visibility,
            )
            for n in nodes
        ]
        if not rows:
            return
        batch_conn = self._get_active_write_batch_connection()
        if batch_conn is not None:
            self._upsert_nodes_rows(batch_conn, rows)
            return
        async with self._lock:
            conn = self._connect()
            self._upsert_nodes_rows(conn, rows)
            conn.commit()

    async def upsert_edges(self, edges: Iterable[GraphEdge]) -> None:
        has_file_column = self._edge_has_file_column
        if has_file_column is None:
            conn = self._connect()
            has_file_column = self._has_table_column(conn, _EDGE_TABLE, "file")
            self._edge_has_file_column = has_file_column

        if has_file_column:
            rows: List[tuple[Any, ...]] = [
                (
                    e.src,
                    e.dst,
                    e.type,
                    e.weight,
                    self._get_edge_file_hint(e),
                    e.src,
                    e.dst,
                    json.dumps(e.metadata),
                )
                for e in edges
            ]
        else:
            rows = [
                (
                    e.src,
                    e.dst,
                    e.type,
                    e.weight,
                    json.dumps(e.metadata),
                )
                for e in edges
            ]
        if not rows:
            return
        batch_conn = self._get_active_write_batch_connection()
        if batch_conn is not None:
            self._upsert_edges_rows(batch_conn, rows)
            return
        async with self._lock:
            conn = self._connect()
            self._upsert_edges_rows(conn, rows)
            conn.commit()

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

        return sorted(seen_edges.values(), key=lambda edge: (edge.src, edge.dst, edge.type))

    def _select_frontier_edges(
        self,
        conn: sqlite3.Connection,
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
            FROM {_EDGE_TABLE}
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
                    metadata=json.loads(row[4]) if row[4] else {},
                ),
                row[5],
            )
            for row in cur.fetchall()
        ]

    def _row_to_node(self, row: tuple) -> GraphNode:
        """Convert a database row to a GraphNode."""
        # Handle both v4 (12 columns) and v5 (17 columns) schemas
        row_len = len(row)
        node = GraphNode(
            node_id=row[0],
            type=row[1],
            name=row[2],
            file=row[3],
            line=row[4],
            end_line=row[5],
            lang=row[6],
            signature=row[7],
            docstring=row[8],
            parent_id=row[9],
            embedding_ref=row[10],
            metadata=json.loads(row[11]) if row_len > 11 and row[11] else {},
        )
        # v5 CCG fields (optional for backward compatibility)
        if row_len > 12:
            node.ast_kind = row[12]
        if row_len > 13:
            node.scope_id = row[13]
        if row_len > 14:
            node.statement_type = row[14]
        if row_len > 15:
            node.requirement_id = row[15]
        if row_len > 16:
            node.visibility = row[16]
        return node

    _NODE_COLS = "node_id, type, name, file, line, end_line, lang, signature, docstring, parent_id, embedding_ref, metadata, ast_kind, scope_id, statement_type, requirement_id, visibility"

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
            file_variants = self._file_path_variants(file)
            if len(file_variants) == 1:
                clauses.append("file = ?")
            else:
                placeholders = ",".join("?" for _ in file_variants)
                clauses.append(f"file IN ({placeholders})")
            params.extend(file_variants)
        where = " AND ".join(clauses) if clauses else "1=1"
        query = f"SELECT {self._NODE_COLS} FROM {_NODE_TABLE} WHERE {where}"
        async with self._lock:
            conn = self._connect()
            cur = conn.execute(query, params)
            return [self._row_to_node(row) for row in cur.fetchall()]

    async def delete_by_repo(self, clear_embeddings: bool = False) -> None:
        """Clear all nodes, edges, and file mtimes for this repo (full rebuild).

        Args:
            clear_embeddings: If True, also clear all embeddings from vector store.
                This is useful for force rebuilds where embeddings need to be regenerated.

        Note:
            Uses bulk load PRAGMA optimizations for faster truncate operation.
            These are safe within a single transaction that commits atomically.
        """
        batch_conn = self._get_active_write_batch_connection()
        if batch_conn is not None:
            self._delete_by_repo_conn(batch_conn)
        else:
            async with self._lock:
                conn = self._connect()
                self._enable_bulk_load_mode(conn)
                try:
                    self._delete_by_repo_conn(conn)
                    conn.commit()
                finally:
                    self._disable_bulk_load_mode(conn)

        if clear_embeddings:
            await self._clear_all_embeddings()

    async def _clear_all_embeddings(self) -> None:
        """Clear all embeddings from the vector store.

        This ensures orphaned embeddings are removed during force rebuilds.
        Failures are logged but don't prevent the graph rebuild from completing.
        """
        try:
            from victor.storage.vector_stores.registry import EmbeddingRegistry

            provider = EmbeddingRegistry.create(self._build_embedding_config())
            await provider.clear_index()
            logger.info("Cleared all embeddings from vector store")
        except ImportError:
            logger.debug("Vector store not available, skipping embedding cleanup")
        except Exception as e:
            logger.warning(f"Failed to clear embedding index: {e}")

    async def stats(self) -> Dict[str, Any]:
        async with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute(f"SELECT COUNT(*) FROM {_NODE_TABLE}")
                node_count = cur.fetchone()[0]
                cur = conn.execute(f"SELECT COUNT(*) FROM {_EDGE_TABLE}")
                edge_count = cur.fetchone()[0]
                cur = conn.execute(f"SELECT COUNT(*) FROM {_MTIME_TABLE}")
                file_count = cur.fetchone()[0]
                return {
                    "nodes": node_count,
                    "edges": edge_count,
                    "indexed_files": file_count,
                    "path": str(self.db_path),
                }
            except sqlite3.OperationalError:
                return {
                    "nodes": node_count,
                    "edges": edge_count,
                    "path": str(self.db_path),
                }

    async def search_symbols(
        self,
        query: str,
        *,
        limit: int = 20,
        symbol_types: Optional[Iterable[str]] = None,
    ) -> List[GraphNode]:
        """Full-text search across symbol names, signatures, and docstrings."""
        async with self._lock:
            conn = self._connect()
            # Try FTS5 first
            try:
                type_clause = ""
                params: list[Any] = [query, limit]
                if symbol_types:
                    types = list(symbol_types)
                    type_clause = f"AND n.type IN ({','.join('?' for _ in types)})"
                    params = [query] + types + [limit]

                fts_query = f"""
                    SELECT n.node_id, n.type, n.name, n.file, n.line, n.end_line, n.lang,
                           n.signature, n.docstring, n.parent_id, n.embedding_ref, n.metadata
                    FROM {_FTS_TABLE} fts
                    JOIN {_NODE_TABLE} n ON fts.node_id = n.node_id
                    WHERE {_FTS_TABLE} MATCH ?
                    {type_clause}
                    ORDER BY rank
                    LIMIT ?
                """
                cur = conn.execute(fts_query, params)
                return [self._row_to_node(row) for row in cur.fetchall()]
            except sqlite3.OperationalError:
                pass

            # Fallback: LIKE search
            like_pattern = f"%{query}%"
            type_clause = ""
            params = [like_pattern, like_pattern, like_pattern, limit]
            if symbol_types:
                types = list(symbol_types)
                type_clause = f"AND type IN ({','.join('?' for _ in types)})"
                params = [like_pattern, like_pattern, like_pattern] + types + [limit]

            like_query = f"""
                SELECT {self._NODE_COLS}
                FROM {_NODE_TABLE}
                WHERE (name LIKE ? OR signature LIKE ? OR docstring LIKE ?)
                {type_clause}
                LIMIT ?
            """
            cur = conn.execute(like_query, params)
            return [self._row_to_node(row) for row in cur.fetchall()]

    async def get_node_by_id(self, node_id: str) -> Optional[GraphNode]:
        """Get a single node by its ID."""
        async with self._lock:
            conn = self._connect()
            cur = conn.execute(
                f"SELECT {self._NODE_COLS} FROM {_NODE_TABLE} WHERE node_id = ?",
                (node_id,),
            )
            row = cur.fetchone()
            return self._row_to_node(row) if row else None

    async def get_all_nodes(self) -> List[GraphNode]:
        """Get all nodes in the graph."""
        async with self._lock:
            conn = self._connect()
            cur = conn.execute(
                f"SELECT {self._NODE_COLS} FROM {_NODE_TABLE} ORDER BY file, line, name"
            )
            return [self._row_to_node(row) for row in cur.fetchall()]

    async def get_nodes_by_file(self, file: str) -> List[GraphNode]:
        """Get all symbols in a specific file."""
        file_variants = self._file_path_variants(file)
        placeholders = ",".join("?" for _ in file_variants)
        async with self._lock:
            conn = self._connect()
            cur = conn.execute(
                f"""
                SELECT {self._NODE_COLS}
                FROM {_NODE_TABLE}
                WHERE file IN ({placeholders})
                ORDER BY line
                """,
                file_variants,
            )
            return [self._row_to_node(row) for row in cur.fetchall()]

    async def update_node_metadata(self, node_id: str, metadata: Dict[str, Any]) -> None:
        """Merge ``metadata`` into a node's metadata JSON (read-modify-write).

        ``embedding_ref`` in the patch lands in its dedicated column. No-op for
        unknown node ids. Signature mirrors ProximaGraphStore.update_node_metadata.
        """

        def _apply(conn: sqlite3.Connection) -> None:
            row = conn.execute(
                f"SELECT metadata FROM {_NODE_TABLE} WHERE node_id = ?", (node_id,)
            ).fetchone()
            if row is None:
                return
            merged = json.loads(row[0]) if row[0] else {}
            patch = dict(metadata or {})
            embedding_ref = patch.pop("embedding_ref", None)
            merged.update(patch)
            if embedding_ref is not None:
                conn.execute(
                    f"UPDATE {_NODE_TABLE} SET metadata = ?, embedding_ref = ? WHERE node_id = ?",
                    (json.dumps(merged), embedding_ref, node_id),
                )
            else:
                conn.execute(
                    f"UPDATE {_NODE_TABLE} SET metadata = ? WHERE node_id = ?",
                    (json.dumps(merged), node_id),
                )

        batch_conn = self._get_active_write_batch_connection()
        if batch_conn is not None:
            _apply(batch_conn)
            return
        async with self._lock:
            conn = self._connect()
            _apply(conn)
            conn.commit()

    async def update_file_mtime(self, file: str, mtime: float) -> None:
        """Record file modification time for staleness tracking."""
        file = self._canonical_file_path(file)
        batch_conn = self._get_active_write_batch_connection()
        if batch_conn is not None:
            self._update_file_mtime_conn(batch_conn, file, mtime)
            return
        async with self._lock:
            conn = self._connect()
            self._update_file_mtime_conn(conn, file, mtime)
            conn.commit()

    async def get_stale_files(self, file_mtimes: Dict[str, float]) -> List[str]:
        """Get files that have changed since last index."""
        stale = []
        async with self._lock:
            conn = self._connect()
            for file, current_mtime in file_mtimes.items():
                file_variants = self._file_path_variants(file)
                placeholders = ",".join("?" for _ in file_variants)
                cur = conn.execute(
                    f"SELECT MAX(mtime) FROM {_MTIME_TABLE} WHERE file IN ({placeholders})",
                    file_variants,
                )
                row = cur.fetchone()
                recorded_mtime = None if row is None else row[0]
                if recorded_mtime is None or recorded_mtime < current_mtime:
                    stale.append(file)
            return stale

    async def get_indexed_files(self) -> List[str]:
        """Get the set of files currently tracked for graph staleness."""
        async with self._lock:
            conn = self._connect()
            cur = conn.execute(f"SELECT file FROM {_MTIME_TABLE} ORDER BY file")
            return [str(row[0]) for row in cur.fetchall()]

    async def delete_by_file(self, file: str) -> None:
        """Delete all nodes, edges, and embeddings for a specific file (for incremental reindex)."""
        # Get nodes before deletion for embedding cleanup
        nodes = await self.get_nodes_by_file(file)
        node_ids = [n.node_id for n in nodes]

        # Delete graph nodes and edges
        batch_conn = self._get_active_write_batch_connection()
        if batch_conn is not None:
            self._delete_by_file_conn(batch_conn, file)
        else:
            async with self._lock:
                conn = self._connect()
                self._delete_by_file_conn(conn, file)
                conn.commit()

        # Clean up embeddings from vector store
        await self._delete_embeddings_for_file(file, node_ids)

    async def get_all_edges(self) -> List[GraphEdge]:
        """Get all edges in the graph (bulk retrieval for loading into memory)."""
        async with self._lock:
            conn = self._connect()
            cur = conn.execute(f"SELECT src, dst, type, weight, metadata FROM {_EDGE_TABLE}")
            return [
                GraphEdge(
                    src=row[0],
                    dst=row[1],
                    type=row[2],
                    weight=row[3],
                    metadata=json.loads(row[4]) if row[4] else {},
                )
                for row in cur.fetchall()
            ]

    # ===========================================
    # v5: Lazy loading methods (PH4-006)
    # ===========================================

    async def iter_nodes(
        self,
        *,
        batch_size: int = 100,
        name: str | None = None,
        type: str | None = None,
        file: str | None = None,
    ) -> AsyncIterator[List[GraphNode]]:
        """Iterate over nodes in batches for memory-efficient processing."""
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
        query = (
            f"SELECT {self._NODE_COLS} FROM {_NODE_TABLE} WHERE {where} ORDER BY file, line, name"
        )

        async with self._lock:
            conn = self._connect()
            cur = conn.execute(query, params)

            while True:
                rows = cur.fetchmany(batch_size)
                if not rows:
                    break
                yield [self._row_to_node(row) for row in rows]

    async def iter_edges(
        self,
        *,
        batch_size: int = 100,
        edge_types: Iterable[str] | None = None,
    ) -> AsyncIterator[List[GraphEdge]]:
        """Iterate over edges in batches for memory-efficient processing."""
        query = f"SELECT src, dst, type, weight, metadata FROM {_EDGE_TABLE}"
        params: list[Any] = []

        if edge_types:
            types = list(edge_types)
            placeholders = ",".join("?" for _ in types)
            query += f" WHERE type IN ({placeholders})"
            params.extend(types)

        query += " ORDER BY src, dst, type"

        async with self._lock:
            conn = self._connect()
            cur = conn.execute(query, params)

            while True:
                rows = cur.fetchmany(batch_size)
                if not rows:
                    break
                yield [
                    GraphEdge(
                        src=row[0],
                        dst=row[1],
                        type=row[2],
                        weight=row[3],
                        metadata=json.loads(row[4]) if row[4] else {},
                    )
                    for row in rows
                ]

    async def iter_neighbors(
        self,
        node_id: str,
        *,
        batch_size: int = 50,
        edge_types: Iterable[str] | None = None,
        direction: GraphTraversalDirection = "out",
    ) -> AsyncIterator[List[GraphEdge]]:
        """Iterate over neighbors in batches for memory-efficient traversal."""
        if direction not in {"out", "in", "both"}:
            raise ValueError(f"Unsupported graph traversal direction: {direction}")

        # Build query for single-hop neighbors
        if direction == "out":
            node_column = "src"
            neighbor_column = "dst"
        elif direction == "in":
            node_column = "dst"
            neighbor_column = "src"
        else:  # both
            # For both directions, we'll use two queries
            async for batch in self._iter_neighbors_direction(
                node_id, batch_size, edge_types, "src", "dst"
            ):
                yield batch
            async for batch in self._iter_neighbors_direction(
                node_id, batch_size, edge_types, "dst", "src"
            ):
                yield batch
            return

        async for batch in self._iter_neighbors_direction(
            node_id, batch_size, edge_types, node_column, neighbor_column
        ):
            yield batch

    async def _iter_neighbors_direction(
        self,
        node_id: str,
        batch_size: int,
        edge_types: Iterable[str] | None,
        node_column: str,
        neighbor_column: str,
    ) -> AsyncIterator[List[GraphEdge]]:
        """Helper for iterating neighbors in a specific direction."""
        query = f"""
            SELECT src, dst, type, weight, metadata, {neighbor_column}
            FROM {_EDGE_TABLE}
            WHERE {node_column} = ?
        """
        params: list[Any] = [node_id]

        if edge_types:
            types = list(edge_types)
            placeholders = ",".join("?" for _ in types)
            query += f" AND type IN ({placeholders})"
            params.extend(types)

        query += f" ORDER BY {neighbor_column}, type"

        async with self._lock:
            conn = self._connect()
            cur = conn.execute(query, params)

            while True:
                rows = cur.fetchmany(batch_size)
                if not rows:
                    break
                yield [
                    GraphEdge(
                        src=row[0],
                        dst=row[1],
                        type=row[2],
                        weight=row[3],
                        metadata=json.loads(row[4]) if row[4] else {},
                    )
                    for row in rows
                ]

    # ===========================================
    # v5: Parallel traversal methods (PH4-007)
    # ===========================================

    async def get_neighbors_batch(
        self,
        node_ids: List[str],
        *,
        edge_types: Iterable[str] | None = None,
        direction: GraphTraversalDirection = "out",
    ) -> Dict[str, List[GraphEdge]]:
        """Get neighbors for multiple nodes in parallel.

        Uses asyncio.gather() to fetch neighbors concurrently.

        Args:
            node_ids: List of node IDs to get neighbors for
            edge_types: Optional filter by edge types
            direction: Traversal direction

        Returns:
            Dict mapping node_id to list of neighboring edges
        """
        if not node_ids:
            return {}

        # Create tasks for each node
        tasks = [
            self.get_neighbors(
                node_id,
                edge_types=edge_types,
                direction=direction,
                max_depth=1,
            )
            for node_id in node_ids
        ]

        # Execute in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Build result dict
        neighbor_map: Dict[str, List[GraphEdge]] = {}
        for node_id, result in zip(node_ids, results):
            if isinstance(result, Exception):
                logger.warning(f"Error getting neighbors for {node_id}: {result}")
                neighbor_map[node_id] = []
            else:
                neighbor_map[node_id] = result

        return neighbor_map

    async def multi_hop_traverse_parallel(
        self,
        start_node_ids: List[str],
        max_hops: int = 2,
        edge_types: Iterable[str] | None = None,
        max_nodes: int = 100,
        max_workers: int = 4,
    ) -> GraphQueryResult:
        """Parallel multi-hop traversal from multiple start nodes.

        Performs BFS traversal but processes multiple frontier nodes
        in parallel using asyncio.gather().

        Args:
            start_node_ids: Multiple starting node IDs
            max_hops: Maximum hop distance
            edge_types: Optional filter by edge types
            max_nodes: Maximum total nodes to return
            max_workers: Maximum parallel workers

        Returns:
            GraphQueryResult with traversed nodes and edges
        """
        import time
        from victor.storage.graph.protocol import GraphQueryResult

        start_time = time.time()

        if not start_node_ids:
            return GraphQueryResult(nodes=[], edges=[], query="parallel_traversal")

        # Initialize BFS state
        visited: Set[str] = set(start_node_ids)
        frontier: List[str] = list(start_node_ids)
        all_edges: Dict[tuple[str, str, str], GraphEdge] = {}
        all_nodes: Dict[str, GraphNode] = {}

        # Get initial nodes
        for node_id in start_node_ids:
            node = await self.get_node_by_id(node_id)
            if node:
                all_nodes[node_id] = node

        # Parallel BFS
        for hop in range(max_hops):
            if not frontier or len(all_nodes) >= max_nodes:
                break

            # Process frontier in parallel batches
            batch_size = max_workers
            results: Dict[str, List[GraphEdge]] = {}

            for i in range(0, len(frontier), batch_size):
                batch = frontier[i : i + batch_size]

                # Fetch neighbors for this batch in parallel
                batch_results = await self.get_neighbors_batch(
                    batch,
                    edge_types=edge_types,
                    direction="out",
                )

                results.update(batch_results)

            # Process results
            next_frontier: Set[str] = set()

            for node_id, neighbors in results.items():
                for edge in neighbors:
                    # Store edge
                    edge_key = (edge.src, edge.dst, edge.type)
                    if edge_key not in all_edges:
                        all_edges[edge_key] = edge

                    # Track neighbor
                    neighbor_id = edge.dst
                    if neighbor_id not in visited:
                        visited.add(neighbor_id)
                        next_frontier.add(neighbor_id)

                        # Get neighbor node details
                        if len(all_nodes) < max_nodes:
                            node = await self.get_node_by_id(neighbor_id)
                            if node:
                                all_nodes[neighbor_id] = node

                        # Stop if we've reached max_nodes
                        if len(all_nodes) >= max_nodes:
                            break

                if len(all_nodes) >= max_nodes:
                    break

            frontier = list(next_frontier)

        # Build result
        return GraphQueryResult(
            nodes=list(all_nodes.values()),
            edges=list(all_edges.values()),
            query=f"parallel_traverse_{len(start_node_ids)}_seeds",
            execution_time_ms=(time.time() - start_time) * 1000,
            metadata={
                "start_nodes": start_node_ids,
                "max_hops": max_hops,
                "hops_completed": hop + 1 if hop < max_hops else max_hops,
                "max_workers": max_workers,
            },
        )
