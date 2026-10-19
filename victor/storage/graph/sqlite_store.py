# SQLite-backed GraphStoreProtocol implementation.
from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
)

from victor.core.schema import Tables
from victor.core.migrations import apply_sql
from victor.storage.graph.protocol import (
    GraphEdge,
    GraphNode,
    GraphQueryResult,
    GraphStoreProtocol,
    GraphTraversalDirection,
)

if TYPE_CHECKING:
    from victor.core.database import ProjectDatabaseManager
    from victor.storage.graph.sqlite_publication import GraphReplacement

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Table names from centralized schema
_NODE_TABLE = Tables.GRAPH_NODE
_EDGE_TABLE = Tables.GRAPH_EDGE
_MTIME_TABLE = Tables.GRAPH_FILE_MTIME
_FTS_TABLE = f"{_NODE_TABLE}_fts"

# Schema for consolidated database
_TABLE_SCHEMA = f"""

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

-- File modification times for staleness tracking. content_hash follows the
-- victor-codegraph manifest contract (sha-256 of decoded source text) so an
-- mtime bump without a content change can be verified as not-stale.
CREATE TABLE IF NOT EXISTS {_MTIME_TABLE} (
    file TEXT PRIMARY KEY,
    mtime REAL NOT NULL,
    indexed_at REAL NOT NULL,
    content_hash TEXT
);

-- Indexes for fast lookups
CREATE INDEX IF NOT EXISTS idx_{_NODE_TABLE}_type_name ON {_NODE_TABLE}(type, name);
CREATE INDEX IF NOT EXISTS idx_{_NODE_TABLE}_file ON {_NODE_TABLE}(file);
CREATE INDEX IF NOT EXISTS idx_{_NODE_TABLE}_parent ON {_NODE_TABLE}(parent_id);
CREATE INDEX IF NOT EXISTS idx_{_EDGE_TABLE}_src_type ON {_EDGE_TABLE}(src, type);
CREATE INDEX IF NOT EXISTS idx_{_EDGE_TABLE}_dst_type ON {_EDGE_TABLE}(dst, type);
CREATE INDEX IF NOT EXISTS idx_{_MTIME_TABLE}_mtime ON {_MTIME_TABLE}(mtime);
"""

SCHEMA = "PRAGMA journal_mode=WAL;\n" + _TABLE_SCHEMA

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

    def __init__(
        self, project_path: Optional[Path] = None, *, _staging_path: Path | None = None
    ) -> None:
        """Initialize graph store.

        Args:
            project_path: Path to project root. If None, uses current directory.
        """
        from victor.core.database import get_project_database

        self._db = get_project_database(project_path)
        self.db_path = _staging_path if _staging_path is not None else self._db.db_path
        self._is_staging = _staging_path is not None
        self._sealed = False
        self._project_root = Path(self._db.project_root).resolve()
        self._conn: sqlite3.Connection | None = None
        self._ensure_schema()
        if not self._is_staging:
            self._record_project_metadata()
        self._lock = asyncio.Lock()
        self._write_batch_conn: sqlite3.Connection | None = None
        self._write_batch_owner: asyncio.Task[Any] | None = None
        self._write_batch_depth = 0
        self._edge_has_file_column: bool | None = None
        # Dedicated single-worker executor for all synchronous SQLite work
        # (co-design review item 20b). Single-worker, NOT the default shared
        # pool: this store's dedicated connection (see _open_dedicated_connection)
        # is operated from the worker thread, and a shared pool would give each
        # pool thread a different thread-local connection with no single-owner
        # story at all. One worker keeps every operation on one stable thread
        # and serializes anything the asyncio lock doesn't already gate.
        # Created lazily so merely constructing a store doesn't spawn a thread.
        self._db_executor: ThreadPoolExecutor | None = None

    def _open_dedicated_connection(self) -> sqlite3.Connection:
        """Open this store's dedicated connection (adversarial-review F1 fix).

        The project database manager's ``get_connection()`` returns one shared
        per-thread connection also used by unrelated components (GraphRAG
        query cache, other store instances). Offloading this store's work to a
        worker thread dissolves the event-loop serialization that accidentally
        protected a shared connection: a co-writer committing mid-``write_batch``
        would commit the batch's partial transaction out from under it. A
        dedicated connection — used only by this store and its single worker —
        degrades cross-writer interference to the standard WAL + busy_timeout
        model, where SQLite's own write lock (BEGIN IMMEDIATE) is the boundary.

        Settings mirror the manager's (row factory, ``check_same_thread=False``,
        30s connect timeout, and its ``_configure_connection`` pragmas) by
        reusing the manager's configurator so the values cannot drift.
        """
        conn = sqlite3.connect(
            str(self.db_path),
            check_same_thread=False,
            timeout=30.0,
        )
        conn.row_factory = sqlite3.Row
        self._db._configure_connection(conn)
        return conn

    def _get_db_executor(self) -> ThreadPoolExecutor:
        """Lazily create the dedicated single-worker SQLite executor."""
        if self._db_executor is None:
            self._db_executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="sqlite-graph",
            )
        return self._db_executor

    async def _run_db(self, fn: Callable[[], T]) -> T:
        """Run a synchronous database operation on the dedicated worker thread.

        The callable must be self-contained (its arguments, including any
        connection, are resolved on the event-loop thread before offloading —
        this is what keeps write_batch's task-identity connection routing
        intact: ``asyncio.current_task()`` is only meaningful on the loop).
        """
        loop = asyncio.get_running_loop()
        pending = loop.run_in_executor(self._get_db_executor(), fn)
        try:
            return await asyncio.shield(pending)
        except asyncio.CancelledError as cancelled:
            # Cancelling an asyncio waiter cannot stop a running SQLite call.
            # Keep admission until this exact call settles; never replay it.
            while not pending.done():
                try:
                    await asyncio.shield(pending)
                except asyncio.CancelledError:
                    continue
                except Exception:
                    break
            try:
                pending.result()
            except Exception as worker_error:
                logger.error("SQLite worker failed while cancellation was settling", exc_info=True)
                raise cancelled from worker_error
            raise

    def _reject_batch_lifecycle(self, operation: str) -> None:
        if self._get_active_write_batch_connection() is not None:
            raise RuntimeError(f"Cannot {operation} during an active write batch")

    async def initialize(self) -> None:
        """Ensure schema under admission; never implicitly commit an active batch."""
        self._reject_batch_lifecycle("initialize")
        async with self._lock:
            await self._run_db(self._ensure_schema)

    async def _discard_connection(self, conn: sqlite3.Connection) -> None:
        """Close an admitted connection on its worker, even during cancellation."""
        try:
            await self._run_db(conn.close)
        finally:
            # _run_db settles its callable before it propagates cancellation.
            # An uncertain/failed close must not make this connection reusable.
            if self._conn is conn:
                self._conn = None

    async def close(self) -> None:
        """Wait for admitted work and release only this store's connection.

        The shared project manager remains usable. Later store operations may
        lazily reopen, preserving the existing lifecycle contract.
        """
        self._reject_batch_lifecycle("close")
        if self._is_staging:
            self._sealed = True
        async with self._lock:
            try:
                if self._conn is not None:
                    await self._discard_connection(self._conn)
            finally:
                if self._db_executor is not None:
                    self._db_executor.shutdown(wait=False)
                    self._db_executor = None

    @property
    def repo_root(self) -> Path:
        """Resolved project root this store indexes (used for cache scoping)."""
        return self._project_root

    def _connect(self) -> sqlite3.Connection:
        """Get this store's dedicated database connection (lazily reopened
        after close(), preserving the previous ops-after-close behavior)."""
        if self._sealed:
            raise RuntimeError("Staged graph is sealed; it cannot be reopened")
        if self._conn is None:
            self._conn = self._open_dedicated_connection()
        return self._conn

    def _stage_replacement(self) -> AbstractAsyncContextManager[GraphReplacement]:
        """Private SQLite publication foundation; not used by the index pipeline yet."""
        from victor.storage.graph.sqlite_publication import stage_replacement

        return stage_replacement(self)

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

        # Manifest-contract migration: content_hash on the mtime table lets the
        # pipeline verify that an mtime-changed file actually changed content.
        mtime_columns = {
            row[1] for row in conn.execute(f"PRAGMA table_info({_MTIME_TABLE})").fetchall()
        }
        if mtime_columns and "content_hash" not in mtime_columns:
            try:
                conn.execute(f"ALTER TABLE {_MTIME_TABLE} ADD COLUMN content_hash TEXT")
            except sqlite3.OperationalError:
                pass

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
            committing = False
            try:
                await self._run_db(lambda: conn.execute("BEGIN IMMEDIATE"))
                yield
                committing = True
                await self._run_db(conn.commit)
            except BaseException as error:
                if committing and isinstance(error, asyncio.CancelledError):
                    logger.warning(
                        "Graph batch cancelled during commit; commit settled and may have "
                        "succeeded. Reconcile durable state before retrying."
                    )
                try:
                    await self._run_db(conn.rollback)
                except BaseException:
                    # Failed/interrupted cleanup must not hand this connection
                    # to the next writer. Closing rolls back any remaining work.
                    await self._discard_connection(conn)
                    raise
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
        """Write node rows using the provided connection.

        ``parent_id`` carries a plain (NOT deferrable) foreign key checked
        statement-by-statement, so the batch is made safe for one executemany
        first: parent links that can never resolve are cleared and logged
        (``_clear_dangling_parent_links``, the node counterpart of the edge
        guard), and the rows are ordered parents-before-children
        (``_order_rows_parents_first``).
        """
        rows = self._order_rows_parents_first(self._clear_dangling_parent_links(conn, rows))
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

    # Column position of parent_id in the node row layout built by upsert_nodes
    # and written by _upsert_nodes_rows' INSERT column list.
    _NODE_ROW_PARENT = 9

    @staticmethod
    def _select_known_node_ids(conn: sqlite3.Connection, ids: Set[str]) -> Set[str]:
        """Return the subset of ``ids`` already present in the node table.

        One chunked SELECT over the batch's distinct ids keeps the check
        O(batch), not O(graph), and stays under SQLite's default
        host-parameter limit.
        """
        known: Set[str] = set()
        id_list = list(ids)
        chunk = 900
        for i in range(0, len(id_list), chunk):
            part = id_list[i : i + chunk]
            marks = ",".join("?" * len(part))
            cur = conn.execute(
                f"SELECT node_id FROM {_NODE_TABLE} WHERE node_id IN ({marks})",
                part,
            )
            known.update(row[0] for row in cur.fetchall())
        return known

    def _clear_dangling_parent_links(
        self,
        conn: sqlite3.Connection,
        rows: List[tuple[Any, ...]],
    ) -> List[tuple[Any, ...]]:
        """Clear parent links whose ids are in neither the batch nor the table.

        Node counterpart of ``_drop_dangling_edge_rows``: a ``parent_id`` that
        can never satisfy the ``nodes(parent_id)`` foreign key would fail the
        whole executemany, taking every other symbol in the batch down with
        it. The unresolvable link is cleared instead — the symbol itself still
        lands — and the loss is logged, never silent.
        """
        in_batch = {row[0] for row in rows}
        external = {
            row[self._NODE_ROW_PARENT]
            for row in rows
            if row[self._NODE_ROW_PARENT] is not None and row[self._NODE_ROW_PARENT] not in in_batch
        }
        if not external:
            return rows
        missing = external - self._select_known_node_ids(conn, external)
        if not missing:
            return rows
        staged: List[tuple[Any, ...]] = []
        cleared: List[str] = []
        for row in rows:
            parent = row[self._NODE_ROW_PARENT]
            if parent in missing:
                cleared.append(f"{row[0]}->{parent}")
                row = row[: self._NODE_ROW_PARENT] + (None,) + row[self._NODE_ROW_PARENT + 1 :]
            staged.append(row)
        logger.warning(
            "sqlite graph store: cleared %d parent link(s) referencing %d "
            "nonexistent node id(s); sample: %s",
            len(cleared),
            len(missing),
            cleared[:5],
        )
        return staged

    def _order_rows_parents_first(self, rows: List[tuple[Any, ...]]) -> List[tuple[Any, ...]]:
        """Emit node rows parents-before-children, one row per node id.

        The parent foreign key is checked per statement, so a child row written
        before its parent row fails the batch even though the parent arrives in
        the same executemany — graph emitters do not promise parents-first
        order (real-world Rust files emit impl members before their impl
        block), and the pipeline's per-file retry cannot fix an ordering
        problem that lives inside the single file batch. Ordering here makes
        the store independent of emitter ordering without schema changes or a
        second write pass.

        Duplicate node ids keep the last row, matching the final state of
        executemany's sequential upserts. A parent cycle — only possible when
        distinct symbols collide on one node id — has no valid order, so the
        minimal break clears ONE link per cycle (logged) and keeps every
        symbol and every other link intact.
        """
        # Dict: one row per node id, last occurrence winning, first-occurrence
        # order preserved for deterministic emission.
        staged: Dict[str, tuple[Any, ...]] = {row[0]: row for row in rows}
        children: Dict[str, List[str]] = {}
        ready: deque[str] = deque()
        for node_id, row in staged.items():
            parent = row[self._NODE_ROW_PARENT]
            if parent is not None and parent in staged:
                children.setdefault(parent, []).append(node_id)
            else:
                ready.append(node_id)

        ordered: List[tuple[Any, ...]] = []
        emitted: Set[str] = set()
        while True:
            while ready:
                node_id = ready.popleft()
                if node_id in emitted:
                    # A cycle break can re-enqueue an already-emitted node
                    # (the breaker is also some cycle member's child); emit
                    # every row exactly once.
                    continue
                ordered.append(staged[node_id])
                emitted.add(node_id)
                ready.extend(children.get(node_id, ()))
            if len(emitted) == len(staged):
                return ordered
            # Every remaining row waits on another remaining row: a parent
            # cycle. Walk parent pointers to one node on the cycle, clear its
            # link, and let the drain above emit it and its descendants with
            # their own links intact.
            walker = next(iter(staged.keys() - emitted))
            seen: Dict[str, int] = {}
            path: List[str] = []
            while walker not in seen:
                seen[walker] = len(path)
                path.append(walker)
                walker = staged[walker][self._NODE_ROW_PARENT]
            breaker = path[seen[walker]]
            row = staged[breaker]
            logger.warning(
                "sqlite graph store: cleared parent link %s->%s to break a parent "
                "cycle (colliding node ids?); %d row(s) were waiting on the cycle",
                breaker,
                row[self._NODE_ROW_PARENT],
                len(staged) - len(emitted),
            )
            staged[breaker] = (
                row[: self._NODE_ROW_PARENT] + (None,) + row[self._NODE_ROW_PARENT + 1 :]
            )
            ready.append(breaker)

    def _drop_dangling_edge_rows(
        self,
        conn: sqlite3.Connection,
        rows: List[tuple[Any, ...]],
    ) -> List[tuple[Any, ...]]:
        """Split off edge rows referencing nonexistent nodes; log what dropped.

        Row layout in both edge-table variants starts (src, dst, ...). Missing
        endpoints are resolved with one chunked SELECT over the batch's
        distinct ids, so the check is O(batch), not O(graph).
        """
        if not rows:
            return rows
        ids = {r[0] for r in rows} | {r[1] for r in rows}
        known = self._select_known_node_ids(conn, ids)
        missing = ids - known
        if not missing:
            return rows
        kept = [r for r in rows if r[0] in known and r[1] in known]
        dropped = len(rows) - len(kept)
        sample = [f"{r[0]}->{r[1]}" for r in rows if r[0] in missing or r[1] in missing][:5]
        logger.warning(
            "sqlite graph store: dropped %d dangling edge(s) referencing %d "
            "nonexistent node id(s); sample: %s",
            dropped,
            len(missing),
            sample,
        )
        return kept

    def _upsert_edges_rows(
        self,
        conn: sqlite3.Connection,
        rows: List[tuple[Any, ...]],
    ) -> None:
        """Write edge rows using the provided connection.

        Endpoint integrity is enforced here, at the same layer ORION enforces
        it: an edge whose src or dst is not a persisted node is SKIPPED and
        logged, and the rest of the batch still lands. Without this the SQLite
        backend silently accepted dangling edges (181 on the repo-scale
        corpus) that the ProximaDB backend rejected — the same write produced
        backend-dependent graphs, and resolver identity bugs reached disk
        invisibly on the default backend.
        """
        rows = self._drop_dangling_edge_rows(conn, rows)
        if not rows:
            return
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

    def _update_file_mtime_conn(
        self,
        conn: sqlite3.Connection,
        file: str,
        mtime: float,
        content_hash: str | None = None,
    ) -> None:
        """Record file modification time (and content hash) using the connection.

        A ``None`` hash overwrites any stored one: an unknown hash must never
        pass a later verified-unchanged check.
        """
        import time

        conn.execute(
            f"""
            INSERT INTO {_MTIME_TABLE}(file, mtime, indexed_at, content_hash)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(file) DO UPDATE SET
                mtime=excluded.mtime,
                indexed_at=excluded.indexed_at,
                content_hash=excluded.content_hash
            """,
            (file, mtime, time.time(), content_hash),
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
        """Recreate derived graph tables inside the caller's transaction.

        Do not use executescript: its implicit commit would publish partial
        cleanup and break rollback of both standalone and outer batch writes.
        """
        # Drop in dependency order: triggers first, then FTS, then edge (FK ref), then node, then mtime.
        # Triggers are automatically dropped with their table, but FTS must be dropped before graph_node
        # because it holds a content= reference to it.
        apply_sql(
            conn,
            f"""
            DROP TABLE IF EXISTS {_FTS_TABLE};
            DROP TRIGGER IF EXISTS {_NODE_TABLE}_ai;
            DROP TRIGGER IF EXISTS {_NODE_TABLE}_ad;
            DROP TRIGGER IF EXISTS {_NODE_TABLE}_au;
            DROP TABLE IF EXISTS {_EDGE_TABLE};
            DROP TABLE IF EXISTS {_NODE_TABLE};
            DROP TABLE IF EXISTS {_MTIME_TABLE};
            """,
        )
        # Recreate all tables, indexes, FTS virtual table, and sync triggers.
        apply_sql(conn, _TABLE_SCHEMA)
        apply_sql(conn, FTS_SCHEMA)
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
        if self._is_staging:
            raise RuntimeError("Staged graphs cannot mutate external embeddings")
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
            await self._run_db(lambda: self._upsert_nodes_rows(batch_conn, rows))
            return
        async with self._lock:
            conn = self._connect()

            def _work() -> None:
                self._upsert_nodes_rows(conn, rows)
                conn.commit()

            await self._run_db(_work)

    async def upsert_edges(self, edges: Iterable[GraphEdge]) -> None:
        has_file_column = self._edge_has_file_column
        if has_file_column is None:
            batch_conn = self._get_active_write_batch_connection()
            if batch_conn is not None:
                has_file_column = await self._run_db(
                    lambda: self._has_table_column(batch_conn, _EDGE_TABLE, "file")
                )
            else:
                async with self._lock:
                    conn = self._connect()
                    has_file_column = await self._run_db(
                        lambda: self._has_table_column(conn, _EDGE_TABLE, "file")
                    )
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
            await self._run_db(lambda: self._upsert_edges_rows(batch_conn, rows))
            return
        async with self._lock:
            conn = self._connect()

            def _work() -> None:
                self._upsert_edges_rows(conn, rows)
                conn.commit()

            await self._run_db(_work)

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
                        await self._run_db(
                            lambda: self._select_frontier_edges(
                                conn,
                                frontier=frontier,
                                edge_types=allowed_types,
                                node_column="src",
                                neighbor_column="dst",
                            )
                        )
                    )
                if direction in {"in", "both"}:
                    traversed_edges.extend(
                        await self._run_db(
                            lambda: self._select_frontier_edges(
                                conn,
                                frontier=frontier,
                                edge_types=allowed_types,
                                node_column="dst",
                                neighbor_column="src",
                            )
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

            def _work() -> List[GraphNode]:
                cur = conn.execute(query, params)
                return [self._row_to_node(row) for row in cur.fetchall()]

            return await self._run_db(_work)

    async def delete_by_repo(self, clear_embeddings: bool = False) -> None:
        """Clear all nodes, edges, and file mtimes for this repo (full rebuild).

        Args:
            clear_embeddings: If True, also clear all embeddings from vector store.
                This is useful for force rebuilds where embeddings need to be regenerated.

        Graph mutation reuses write_batch for transaction ownership, rollback
        and cancellation settlement. Connection durability settings are retained.
        Vector cleanup is separate and is not part of this SQLite transaction.
        """
        if clear_embeddings and self._is_staging:
            raise RuntimeError("Staged graphs cannot mutate external embeddings")
        async with self.write_batch():
            conn = self._get_active_write_batch_connection()
            assert conn is not None
            await self._run_db(lambda: self._delete_by_repo_conn(conn))

        if clear_embeddings:
            await self._clear_all_embeddings()

    async def _clear_all_embeddings(self) -> None:
        """Clear all embeddings from the vector store.

        This ensures orphaned embeddings are removed during force rebuilds.
        Failures are logged but don't prevent the graph rebuild from completing.
        """
        if self._is_staging:
            raise RuntimeError("Staged graphs cannot mutate external embeddings")
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

            def _work() -> Dict[str, Any]:
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

            return await self._run_db(_work)

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

            def _work() -> List[GraphNode]:
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

            return await self._run_db(_work)

    async def get_node_by_id(self, node_id: str) -> Optional[GraphNode]:
        """Get a single node by its ID."""
        async with self._lock:
            conn = self._connect()

            def _work() -> Optional[GraphNode]:
                cur = conn.execute(
                    f"SELECT {self._NODE_COLS} FROM {_NODE_TABLE} WHERE node_id = ?",
                    (node_id,),
                )
                row = cur.fetchone()
                return self._row_to_node(row) if row else None

            return await self._run_db(_work)

    async def get_all_nodes(self) -> List[GraphNode]:
        """Get all nodes in the graph."""
        async with self._lock:
            conn = self._connect()

            def _work() -> List[GraphNode]:
                cur = conn.execute(
                    f"SELECT {self._NODE_COLS} FROM {_NODE_TABLE} ORDER BY file, line, name"
                )
                return [self._row_to_node(row) for row in cur.fetchall()]

            return await self._run_db(_work)

    async def get_nodes_by_file(self, file: str) -> List[GraphNode]:
        """Get file symbols, sharing the connection when called by a batch owner."""
        file_variants = self._file_path_variants(file)
        placeholders = ",".join("?" for _ in file_variants)

        def _work(conn: sqlite3.Connection) -> List[GraphNode]:
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

        batch_conn = self._get_active_write_batch_connection()
        if batch_conn is not None:
            return await self._run_db(lambda: _work(batch_conn))
        async with self._lock:
            conn = self._connect()
            return await self._run_db(lambda: _work(conn))

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
            await self._run_db(lambda: _apply(batch_conn))
            return
        async with self.write_batch():
            conn = self._connect()
            await self._run_db(lambda: _apply(conn))

    async def update_nodes_metadata(self, pairs: List[Tuple[str, Dict[str, Any]]]) -> None:
        """Batched ``update_node_metadata``: one SELECT + one executemany UPDATE
        instead of N round trips (co-design review item 20a). Unknown node ids
        are skipped, same as the single-node path. A repeated node id compounds
        its patches in call order (matching sequential update_node_metadata
        calls) rather than the last one silently winning; an ``embedding_ref``
        omitted from a later patch for the same id leaves the earlier one in
        place, same as the single-node path only ever touching that column
        when a patch explicitly supplies it.
        """
        if not pairs:
            return

        def _apply(conn: sqlite3.Connection) -> None:
            combined_metadata: Dict[str, Dict[str, Any]] = {}
            combined_embedding_ref: Dict[str, str] = {}
            order: List[str] = []
            for node_id, metadata in pairs:
                if node_id not in combined_metadata:
                    combined_metadata[node_id] = {}
                    order.append(node_id)
                patch = dict(metadata or {})
                embedding_ref = patch.pop("embedding_ref", None)
                combined_metadata[node_id].update(patch)
                if embedding_ref is not None:
                    combined_embedding_ref[node_id] = embedding_ref

            placeholders = ",".join("?" for _ in order)
            cur = conn.execute(
                f"SELECT node_id, metadata FROM {_NODE_TABLE} WHERE node_id IN ({placeholders})",
                order,
            )
            existing = {str(row[0]): row[1] for row in cur.fetchall()}
            embed_rows = []
            plain_rows = []
            for node_id in order:
                if node_id not in existing:
                    continue
                merged = json.loads(existing[node_id]) if existing[node_id] else {}
                merged.update(combined_metadata[node_id])
                if node_id in combined_embedding_ref:
                    embed_rows.append(
                        (json.dumps(merged), combined_embedding_ref[node_id], node_id)
                    )
                else:
                    plain_rows.append((json.dumps(merged), node_id))
            if embed_rows:
                conn.executemany(
                    f"UPDATE {_NODE_TABLE} SET metadata = ?, embedding_ref = ? WHERE node_id = ?",
                    embed_rows,
                )
            if plain_rows:
                conn.executemany(
                    f"UPDATE {_NODE_TABLE} SET metadata = ? WHERE node_id = ?",
                    plain_rows,
                )

        batch_conn = self._get_active_write_batch_connection()
        if batch_conn is not None:
            await self._run_db(lambda: _apply(batch_conn))
            return
        async with self.write_batch():
            conn = self._connect()
            await self._run_db(lambda: _apply(conn))

    async def update_file_mtime(
        self, file: str, mtime: float, content_hash: str | None = None
    ) -> None:
        """Record file modification time (and content hash) for staleness tracking."""
        file = self._canonical_file_path(file)
        batch_conn = self._get_active_write_batch_connection()
        if batch_conn is not None:
            await self._run_db(
                lambda: self._update_file_mtime_conn(batch_conn, file, mtime, content_hash)
            )
            return
        async with self._lock:
            conn = self._connect()

            def _work() -> None:
                self._update_file_mtime_conn(conn, file, mtime, content_hash)
                conn.commit()

            await self._run_db(_work)

    async def get_file_hashes(self, files: List[str]) -> Dict[str, str]:
        """Stored content hashes for the given files (files without a hash omitted).

        Loads the mtime table once instead of one SELECT per file (co-design
        review item 20a) — `file` is the table's primary key, so the load is
        a plain dict; variant matching happens in Python afterward.
        """
        async with self._lock:
            conn = self._connect()

            def _work() -> Dict[str, str]:
                cur = conn.execute(f"SELECT file, content_hash FROM {_MTIME_TABLE}")
                return {str(row[0]): row[1] for row in cur.fetchall()}

            stored_hashes = await self._run_db(_work)
        hashes: Dict[str, str] = {}
        for file in files:
            for variant in self._file_path_variants(file):
                value = stored_hashes.get(variant)
                if value:
                    hashes[file] = str(value)
                    break
        return hashes

    async def get_stale_files(self, file_mtimes: Dict[str, float]) -> List[str]:
        """Get files that have changed since last index.

        Loads the mtime table once instead of one SELECT per file (co-design
        review item 20a).
        """
        async with self._lock:
            conn = self._connect()

            def _work() -> Dict[str, Any]:
                cur = conn.execute(f"SELECT file, mtime FROM {_MTIME_TABLE}")
                return {str(row[0]): row[1] for row in cur.fetchall()}

            stored_mtimes = await self._run_db(_work)
        stale = []
        for file, current_mtime in file_mtimes.items():
            recorded_mtime = None
            for variant in self._file_path_variants(file):
                candidate = stored_mtimes.get(variant)
                if candidate is not None and (recorded_mtime is None or candidate > recorded_mtime):
                    recorded_mtime = candidate
            if recorded_mtime is None or recorded_mtime < current_mtime:
                stale.append(file)
        return stale

    async def get_indexed_files(self) -> List[str]:
        """Get the set of files currently tracked for graph staleness."""
        async with self._lock:
            conn = self._connect()

            def _work() -> List[str]:
                cur = conn.execute(f"SELECT file FROM {_MTIME_TABLE} ORDER BY file")
                return [str(row[0]) for row in cur.fetchall()]

            return await self._run_db(_work)

    async def delete_by_file(self, file: str) -> None:
        """Delete all nodes, edges, and embeddings for a specific file (for incremental reindex)."""
        if self._is_staging:
            raise RuntimeError("Staged graphs cannot mutate external embeddings")
        # Get nodes before deletion for embedding cleanup
        nodes = await self.get_nodes_by_file(file)
        node_ids = [n.node_id for n in nodes]

        # Delete graph nodes and edges
        batch_conn = self._get_active_write_batch_connection()
        if batch_conn is not None:
            await self._run_db(lambda: self._delete_by_file_conn(batch_conn, file))
        else:
            async with self._lock:
                conn = self._connect()

                def _work() -> None:
                    self._delete_by_file_conn(conn, file)
                    conn.commit()

                await self._run_db(_work)

        # Clean up embeddings from vector store
        await self._delete_embeddings_for_file(file, node_ids)

    async def get_all_edges(self) -> List[GraphEdge]:
        """Get all edges in the graph (bulk retrieval for loading into memory)."""
        async with self._lock:
            conn = self._connect()

            def _work() -> List[GraphEdge]:
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

            return await self._run_db(_work)

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
            cur = await self._run_db(lambda: conn.execute(query, params))

            while True:
                rows = await self._run_db(lambda: cur.fetchmany(batch_size))
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
            cur = await self._run_db(lambda: conn.execute(query, params))

            while True:
                rows = await self._run_db(lambda: cur.fetchmany(batch_size))
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
            cur = await self._run_db(lambda: conn.execute(query, params))

            while True:
                rows = await self._run_db(lambda: cur.fetchmany(batch_size))
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
