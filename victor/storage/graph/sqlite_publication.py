"""Private, store-owned staging for a SQLite graph replacement.

Only graph rows, their FTS index and a publication receipt commit together.
The indexing pipeline does not use this capability yet. Parser completeness,
external vectors, cache invalidation and whole-request reader snapshots are
separate contracts. A receipt describes a past publication, not current readiness.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
import tempfile
import uuid
from contextlib import asynccontextmanager, closing
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, AsyncIterator

if TYPE_CHECKING:
    from victor.storage.graph.sqlite_store import SqliteGraphStore

_RECEIPT_KEY = "graph_last_publication_v1"
# Explicit stored columns, never FTS shadow tables or authoritative project state.
_COLUMNS = {
    "graph_node": (
        "node_id,type,name,file,line,end_line,lang,signature,docstring,parent_id,"
        "embedding_ref,metadata,ast_kind,scope_id,statement_type,requirement_id,visibility"
    ),
    "graph_edge": "src,dst,type,weight,file,metadata",
    "graph_file_mtime": "file,mtime,indexed_at,content_hash",
}
_ORDER = {"graph_node": "node_id", "graph_edge": "src,dst,type", "graph_file_mtime": "file"}
_DEPENDENTS = (
    "graph_requirement",
    "graph_subgraph_node",
    "graph_subgraph",
    "graph_module_metric",
    "cpg_fragment_node",
    "cpg_fragment_edge",
    "graph_query_cache",
)


@dataclass(frozen=True)
class GraphPublicationReceipt:
    publication_id: str
    content_digest: str
    repository_root: str
    schema_version: int = 1


def _fingerprint(conn: sqlite3.Connection, *, receipt: bool) -> str:
    digest = hashlib.sha256()
    for table, columns in _COLUMNS.items():
        actual = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
        if actual != set(columns.split(",")):
            raise ValueError(f"Unsupported graph schema: {table}")
        digest.update(table.encode())
        for row in conn.execute(f"SELECT {columns} FROM {table} ORDER BY {_ORDER[table]}"):
            encoded = json.dumps(tuple(row), ensure_ascii=True, allow_nan=False).encode()
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
    if receipt:
        row = conn.execute(
            "SELECT value FROM _project_metadata WHERE key=?", (_RECEIPT_KEY,)
        ).fetchone()
        digest.update(json.dumps(tuple(row) if row else None).encode())
    return digest.hexdigest()


def _baseline(conn: sqlite3.Connection) -> str:
    conn.execute("BEGIN")
    try:
        return _fingerprint(conn, receipt=True)
    finally:
        conn.rollback()


def _validate_graph(conn: sqlite3.Connection) -> None:
    if conn.execute("PRAGMA quick_check").fetchall() != [("ok",)]:
        raise ValueError("Staged graph integrity check failed")
    checks = (
        "SELECT 1 FROM graph_edge e LEFT JOIN graph_node s ON s.node_id=e.src "
        "LEFT JOIN graph_node d ON d.node_id=e.dst WHERE s.node_id IS NULL OR d.node_id IS NULL LIMIT 1",
        "SELECT 1 FROM graph_node n LEFT JOIN graph_node p ON p.node_id=n.parent_id "
        "WHERE n.parent_id IS NOT NULL AND p.node_id IS NULL LIMIT 1",
        "SELECT 1 FROM graph_node WHERE embedding_ref IS NOT NULL "
        "OR json_extract(metadata,'$.has_embedding')=1 LIMIT 1",
    )
    if any(conn.execute(sql).fetchone() is not None for sql in checks):
        raise ValueError("Staged graph has dangling references or unsupported embeddings")


def _check_dependents(conn: sqlite3.Connection) -> None:
    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    for table in _DEPENDENTS:
        if table in tables and conn.execute(f"SELECT 1 FROM {table} LIMIT 1").fetchone():
            raise ValueError(f"Graph publication cannot reconcile dependent state: {table}")
    if conn.execute(
        "SELECT 1 FROM graph_node WHERE embedding_ref IS NOT NULL "
        "OR json_extract(metadata,'$.has_embedding')=1 LIMIT 1"
    ).fetchone():
        raise ValueError("Graph publication cannot reconcile existing embeddings")


def _copy_rows(source: sqlite3.Connection, target: sqlite3.Connection, table: str) -> None:
    columns = _COLUMNS[table]
    cursor = source.execute(f"SELECT {columns} FROM {table}")
    placeholders = ",".join("?" for _ in columns.split(","))
    while rows := cursor.fetchmany(256):
        target.executemany(f"INSERT INTO {table} ({columns}) VALUES ({placeholders})", rows)


def _write_receipt(conn: sqlite3.Connection, receipt: GraphPublicationReceipt) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO _project_metadata (key,value,updated_at) "
        "VALUES (?,?,datetime('now'))",
        (_RECEIPT_KEY, json.dumps(receipt.__dict__, sort_keys=True)),
    )


class GraphReplacement:
    """Owned one-attempt replacement handle; created only by its store context."""

    def __init__(self, owner: SqliteGraphStore, stage: SqliteGraphStore, baseline: str) -> None:
        self.store = stage
        self.publication_id = uuid.uuid4().hex
        self._owner = owner
        self._task = asyncio.current_task()
        self._baseline = baseline
        self._attempted = False
        self._active = True

    async def publish(self) -> GraphPublicationReceipt:
        if not self._active or self._attempted:
            raise RuntimeError("Graph publication is already attempted or its context has closed")
        if asyncio.current_task() is not self._task:
            raise RuntimeError("Only the staging context owner may publish its replacement")
        self._attempted = True
        self._owner._reject_batch_lifecycle("publish replacement")
        if self.store.repo_root != self._owner.repo_root:
            raise ValueError("Staging and publication repository roots differ")
        await self.store.close()  # settle writes, then forbid automatic reopening
        async with self._owner.write_batch():
            conn = self._owner._get_active_write_batch_connection()
            assert conn is not None
            return await self._owner._run_db(lambda: self._publish_rows(conn))

    def _publish_rows(self, conn: sqlite3.Connection) -> GraphPublicationReceipt:
        if _fingerprint(conn, receipt=True) != self._baseline:
            raise RuntimeError("Graph changed while replacement was being prepared")
        _check_dependents(conn)
        with closing(
            sqlite3.connect(self.store.db_path.resolve().as_uri() + "?mode=ro", uri=True)
        ) as source:
            source.execute("BEGIN")
            content_digest = _fingerprint(source, receipt=False)
            _validate_graph(source)
            # Parent nodes may be inserted after their children in the copied rows.
            conn.execute("PRAGMA defer_foreign_keys=ON")
            for table in ("graph_edge", "graph_node", "graph_file_mtime"):
                conn.execute(f"DELETE FROM {table}")
            for table in _COLUMNS:
                _copy_rows(source, conn, table)
            if _fingerprint(conn, receipt=False) != content_digest:
                raise ValueError("Graph replacement changed stored values during publication")
            conn.execute(
                "INSERT INTO graph_node_fts(graph_node_fts,rank) VALUES ('integrity-check',1)"
            )
            receipt = GraphPublicationReceipt(
                self.publication_id, content_digest, str(self._owner.repo_root)
            )
            _write_receipt(conn, receipt)
            return receipt


async def _settle_close(store: SqliteGraphStore) -> None:
    cleanup = asyncio.create_task(store.close())
    cancelled = False
    while not cleanup.done():
        try:
            await asyncio.shield(cleanup)
        except asyncio.CancelledError:
            cancelled = True
    cleanup.result()
    if cancelled:
        raise asyncio.CancelledError


@asynccontextmanager
async def stage_replacement(owner: SqliteGraphStore) -> AsyncIterator[GraphReplacement]:
    from victor.storage.graph.sqlite_store import SqliteGraphStore

    owner._reject_batch_lifecycle("stage replacement")
    if owner._is_staging:
        raise RuntimeError("Nested staging is unsupported")
    async with owner._lock:
        conn = owner._connect()
        baseline = await owner._run_db(lambda: _baseline(conn))
    with tempfile.TemporaryDirectory(prefix="graph-stage-", dir=owner.db_path.parent) as directory:
        stage = SqliteGraphStore(owner.repo_root, _staging_path=Path(directory) / "graph.db")
        replacement = GraphReplacement(owner, stage, baseline)
        try:
            yield replacement
        finally:
            replacement._active = False
            await _settle_close(stage)
