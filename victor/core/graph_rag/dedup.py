"""Conservative repair of duplicate SQLite graph definitions.

Connectivity and matching leaf names are never evidence of duplication. Only
identical, fully located definitions with compatible source metadata can merge.
External vectors cannot join a SQLite transaction; a durable pending marker
keeps failed embedding refreshes retryable without repeating graph mutations.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from contextlib import closing
from pathlib import Path
from typing import Any, Callable, Iterable, NoReturn, TypeVar

_T = TypeVar("_T")


async def settled_thread(operation: Callable[[], _T]) -> _T:
    """Settle SQLite writes before cancellation releases the index lock."""
    task = asyncio.create_task(asyncio.to_thread(operation))
    cancelled = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled = True
    result = task.result()
    if cancelled:
        raise asyncio.CancelledError
    return result


@dataclass(frozen=True)
class DuplicateGroup:
    """Reviewed definition IDs and a fingerprint for stale-plan detection."""

    node_ids: tuple[str, ...]
    fingerprint: str
    name: str
    file: str
    line: int


@dataclass(frozen=True)
class MergeResult:
    """Committed canonical identity and removed identities."""

    merged_id: str
    removed_ids: tuple[str, ...]


def _reject_json_constant(value: str) -> NoReturn:
    raise ValueError(f"Non-finite JSON constant in graph metadata: {value}")


def _metadata(value: str | None) -> dict[str, Any]:
    data = json.loads(value or "{}", parse_constant=_reject_json_constant)
    if not isinstance(data, dict):
        raise ValueError("Graph metadata must be a JSON object")
    return data


def _fingerprint(rows: list[dict[str, Any]]) -> str:
    return hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()


def _identity(row: dict[str, Any]) -> str:
    metadata = _metadata(row["metadata"])
    # Embedding evidence can differ across indexing runs. Source hashes,
    # compiler identities, qualified names and all other metadata must agree.
    metadata = {
        k: v
        for k, v in metadata.items()
        if (not k.startswith("embedding_") or k == "embedding_excluded")
        and k not in {"has_embedding", "content_version", "deduplication_sources"}
    }
    definition = {k: v for k, v in row.items() if k not in {"node_id", "embedding_ref", "metadata"}}
    return json.dumps([definition, metadata], sort_keys=True)


class SymbolDeduplicator:
    """Repair graph rows on a caller-owned, idle SQLite connection.

    The connection's row factory is left untouched. Discovery performs reads
    only. Apply owns one BEGIN IMMEDIATE transaction and rolls back every
    group on failure; it never commits an unrelated caller transaction.
    """

    def __init__(self, db: sqlite3.Connection) -> None:
        self.db = db

    def _rows(self, sql: str, params: Iterable[Any] = ()) -> list[dict[str, Any]]:
        cursor = self.db.execute(sql, tuple(params))
        columns = [col[0] for col in cursor.description]
        return [dict(zip(columns, row)) for row in cursor]

    def find_duplicates(self) -> list[DuplicateGroup]:
        """Group exact source definitions, excluding incomplete/ambiguous rows.

        SQL prunes singletons before Python inspects JSON, keeping statements
        and the rest of the corpus out of the allocation-heavy repair path.
        Parent and scope identity are intentionally conservative: duplicates
        below duplicated parents can be discovered on a subsequent pass.
        """
        candidates = self._rows(
            "WITH candidates AS (SELECT file,type,name,line,end_line,signature "
            "FROM graph_node WHERE type IN ('function','method','class','constructor') "
            "AND file<>'' AND line>0 AND end_line>=line "
            "AND signature IS NOT NULL AND trim(signature)<>'' "
            "GROUP BY file,type,name,line,end_line,signature HAVING count(*)>1) "
            "SELECT n.* FROM graph_node n JOIN candidates c "
            "ON n.file=c.file AND n.type=c.type AND n.name=c.name AND n.line=c.line "
            "AND n.end_line=c.end_line AND n.signature=c.signature ORDER BY n.node_id"
        )
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in candidates:
            try:
                if _metadata(row["metadata"]).get("embedding_excluded"):
                    continue
                grouped[_identity(row)].append(row)
            except (ValueError, TypeError):
                continue  # malformed evidence is never an automatic merge
        return [
            DuplicateGroup(
                tuple(row["node_id"] for row in rows),
                _fingerprint(rows),
                rows[0]["name"],
                rows[0]["file"],
                rows[0]["line"],
            )
            for rows in grouped.values()
            if len(rows) > 1
        ]

    def merge_nodes(self, group: DuplicateGroup) -> MergeResult:
        """Merge one reviewed group, preserving inbound/outbound relationships."""
        return self.apply([group])[0]

    def apply(self, groups: Iterable[DuplicateGroup]) -> list[MergeResult]:
        """Commit all groups atomically, rejecting stale or fabricated plans."""
        if self.db.in_transaction:
            raise ValueError("Deduplication requires an idle connection, not a caller transaction")
        groups = list(groups)
        if not groups:
            return []
        if not self.db.execute("PRAGMA foreign_keys").fetchone()[0]:
            raise ValueError("Deduplication requires foreign key enforcement")
        self.db.execute("BEGIN IMMEDIATE")
        try:
            edge_columns = {r[1] for r in self.db.execute("PRAGMA table_info(graph_edge)")}
            if edge_columns != {"src", "dst", "type", "weight", "file", "metadata"}:
                raise ValueError("Unsupported graph edge schema; refusing to discard columns")
            # Fragment stores use a separate identity model; refuse to silently
            # leave stored references behind until a reconciliation exists.
            tables = {
                r[0] for r in self.db.execute("SELECT name FROM sqlite_master WHERE type='table'")
            }
            for table in ("cpg_fragment_node", "cpg_fragment_edge"):
                if table in tables and self.db.execute(f"SELECT 1 FROM {table} LIMIT 1").fetchone():
                    raise ValueError(f"Cannot reconcile dependent state: {table}")
            results = []
            # Validate every plan before mutating parents/scopes used by another.
            validated = []
            seen: set[str] = set()
            current_groups = set(self.find_duplicates())
            for group in groups:
                ids = group.node_ids
                if len(ids) < 2 or tuple(sorted(set(ids))) != ids or seen.intersection(ids):
                    raise ValueError("Invalid or overlapping duplicate group")
                seen.update(ids)
                rows = self._rows(
                    f"SELECT * FROM graph_node WHERE node_id IN ({','.join('?' for _ in ids)}) ORDER BY node_id",
                    ids,
                )
                if len(rows) != len(ids) or _fingerprint(rows) != group.fingerprint:
                    raise ValueError("Duplicate group changed since discovery")
                if len({_identity(row) for row in rows}) != 1:
                    raise ValueError("Group contains distinct definitions")
                if group not in current_groups:
                    raise ValueError("Group is not a complete duplicate definition")
                validated.append((group, rows))
            # Cached subgraphs contain serialized edges; invalidate the complete
            # derived cache rather than rewriting only the junction rows.
            for table in (
                "graph_subgraph_node",
                "graph_subgraph",
                "graph_query_cache",
                "graph_module_metric",
            ):
                if table in tables:
                    self.db.execute(f"DELETE FROM {table}")
            for group, rows in validated:
                results.append(self._merge(group, rows))
            self.db.commit()
            return results
        except BaseException:
            self.db.rollback()
            raise

    def _merge(self, group: DuplicateGroup, rows: list[dict[str, Any]]) -> MergeResult:
        canonical, *removed = group.node_ids
        identities = set(group.node_ids)
        placeholders = ",".join("?" for _ in group.node_ids)
        edges = self._rows(
            f"SELECT * FROM graph_edge WHERE src IN ({placeholders}) OR dst IN ({placeholders}) ORDER BY src,dst,type",
            (*group.node_ids, *group.node_ids),
        )
        combined: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
        for edge in edges:
            src = canonical if edge["src"] in identities else edge["src"]
            dst = canonical if edge["dst"] in identities else edge["dst"]
            combined[(src, dst, edge["type"])].append(edge)
        self.db.execute(
            f"DELETE FROM graph_edge WHERE src IN ({placeholders}) OR dst IN ({placeholders})",
            (*group.node_ids, *group.node_ids),
        )
        for (src, dst, kind), originals in combined.items():
            edge = originals[0]
            metadata = _metadata(edge["metadata"])
            if len(originals) > 1:
                metadata["deduplication_edges"] = [
                    {**e, "metadata": _metadata(e["metadata"])} for e in originals
                ]
            weights = [e["weight"] for e in originals if e["weight"] is not None]
            self.db.execute(
                "INSERT INTO graph_edge (src,dst,type,weight,file,metadata) VALUES (?,?,?,?,?,?)",
                (
                    src,
                    dst,
                    kind,
                    max(weights) if weights else None,
                    edge.get("file"),
                    json.dumps(metadata),
                ),
            )
        for old in removed:
            for column in ("parent_id", "scope_id"):
                self.db.execute(
                    f"UPDATE graph_node SET {column}=? WHERE {column}=?", (canonical, old)
                )
        metadata = _metadata(rows[0]["metadata"])
        sources = []
        for row in rows:
            evidence = _metadata(row["metadata"])
            sources.extend(evidence.pop("deduplication_sources", []))
            sources.append(
                {
                    "node_id": row["node_id"],
                    "embedding_ref": row["embedding_ref"],
                    "metadata": evidence,
                }
            )
        metadata = {
            k: v
            for k, v in metadata.items()
            if not k.startswith("embedding_") and k != "content_version"
        }
        metadata["has_embedding"] = False
        metadata["deduplication_sources"] = sources
        vector_ids = set(group.node_ids)
        vector_ids.update(row["embedding_ref"] for row in rows if row["embedding_ref"])
        metadata["deduplication_pending"] = {"vector_ids": sorted(vector_ids)}
        self.db.execute(
            "UPDATE graph_node SET metadata=?,embedding_ref=NULL WHERE node_id=?",
            (json.dumps(metadata), canonical),
        )
        for old in removed:
            self.db.execute("DELETE FROM graph_node WHERE node_id=?", (old,))
        return MergeResult(canonical, tuple(removed))


def pending_embeddings(db: sqlite3.Connection) -> dict[str, dict[str, Any]]:
    """Load durable repair work left by committed graph merges."""
    return {
        row[0]: _metadata(row[1])["deduplication_pending"]
        for row in db.execute(
            "SELECT node_id,metadata FROM graph_node "
            "WHERE CASE WHEN json_valid(metadata) "
            "THEN json_type(metadata,'$.deduplication_pending') END='object'"
        )
    }


async def update_embeddings(project_root: Path) -> int:
    """Remove obsolete vectors and refresh pending canonical symbols.

    Caller holds the repository index lock. Failure leaves the committed graph
    and pending marker intact; re-running apply resumes only derived work.
    Model/profile validation and persistence use the existing indexing pipeline.
    """
    from victor.core.graph_rag.config import GraphIndexConfig
    from victor.core.graph_rag.indexing import GraphIndexingPipeline
    from victor.storage.graph.sqlite_store import SqliteGraphStore

    db_path = project_root / ".victor" / "project.db"

    def load() -> dict[str, dict[str, Any]]:
        with closing(sqlite3.connect(db_path.resolve().as_uri() + "?mode=ro", uri=True)) as db:
            return pending_embeddings(db)

    pending = await asyncio.to_thread(load)
    if not pending:
        return 0
    store = SqliteGraphStore(project_root)
    pipeline = GraphIndexingPipeline(
        store, GraphIndexConfig(root_path=project_root, enable_ccg=False)
    )
    provider = pipeline._get_vector_provider()
    try:
        if provider is None:
            raise RuntimeError("Embedding vector store is unavailable; repair remains pending")
        obsolete: set[str] = set()
        for item in pending.values():
            for vector_id in item["vector_ids"]:
                obsolete.add(vector_id)
                if vector_id.startswith(("emb:", "symbol:")):
                    obsolete.add(vector_id.split(":", 1)[1])

        def protected_vectors() -> set[str]:
            # Older indexes may share a vector reference across graph rows.
            # A removed identity does not make its vector obsolete while a
            # surviving row still references it.
            protected = set(pending)
            with closing(sqlite3.connect(db_path.resolve().as_uri() + "?mode=ro", uri=True)) as db:
                for node_id, ref in db.execute("SELECT node_id,embedding_ref FROM graph_node"):
                    if node_id in obsolete:
                        protected.add(node_id)
                    if ref:
                        if ref in obsolete:
                            protected.add(ref)
                        if ref.startswith(("emb:", "symbol:")):
                            suffix = ref.split(":", 1)[1]
                            if suffix in obsolete:
                                protected.add(suffix)
            return protected

        protected = await asyncio.to_thread(protected_vectors)
        for vector_id in sorted(obsolete - protected):
            await provider.delete_document(vector_id)
        stats = await pipeline.refresh_embeddings(set(pending))
        if stats.error_count:
            raise RuntimeError("; ".join(stats.errors))

        def finish() -> None:
            with closing(sqlite3.connect(db_path.resolve().as_uri() + "?mode=rw", uri=True)) as db:
                db.execute("BEGIN IMMEDIATE")
                try:
                    for node_id, expected in pending.items():
                        row = db.execute(
                            "SELECT embedding_ref,metadata FROM graph_node WHERE node_id=?",
                            (node_id,),
                        ).fetchone()
                        metadata = _metadata(row[1]) if row else {}
                        if not row or not row[0] or not metadata.get("has_embedding"):
                            raise RuntimeError(f"Embedding refresh incomplete for {node_id}")
                        if metadata.get("deduplication_pending") != expected:
                            raise RuntimeError(f"Pending repair changed for {node_id}")
                        metadata.pop("deduplication_pending")
                        db.execute(
                            "UPDATE graph_node SET metadata=? WHERE node_id=?",
                            (json.dumps(metadata), node_id),
                        )
                    db.commit()
                except BaseException:
                    db.rollback()
                    raise

        await settled_thread(finish)
        return len(pending)
    finally:
        try:
            if provider is not None:
                await provider.close()
        finally:
            await store.close()
