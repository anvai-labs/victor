"""Real SQLite/FTS regression coverage for conservative graph deduplication."""

import json
import sqlite3

import pytest

from victor.storage.graph.sqlite_store import _TABLE_SCHEMA, FTS_SCHEMA


@pytest.fixture
def db(tmp_path):
    conn = sqlite3.connect(tmp_path / "project.db")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(_TABLE_SCHEMA + FTS_SCHEMA)
    yield conn
    conn.close()


def node(db, node_id, **overrides):
    values = {
        "node_id": node_id,
        "type": "function",
        "name": "execute",
        "file": "src/a.py",
        "line": 10,
        "end_line": 12,
        "signature": "def execute():",
        "metadata": '{"source_hash":"same"}',
    }
    values.update(overrides)
    db.execute(
        f"INSERT INTO graph_node ({','.join(values)}) VALUES ({','.join('?' for _ in values)})",
        tuple(values.values()),
    )
    db.commit()


def test_only_same_definition_is_duplicate(db):
    from victor.core.graph_rag.dedup import SymbolDeduplicator

    node(db, "a")
    node(db, "b", embedding_ref="emb:b")
    node(db, "other-file", file="src/b.py")
    node(db, "other-line", line=20, end_line=22)
    node(db, "overload", signature="def execute(x):")
    node(db, "unknown", signature=None)
    node(db, "stale", metadata='{"source_hash":"changed"}')
    node(db, "scope", scope_id="another")
    node(db, "bad-a", metadata='{"score":NaN}')
    node(db, "bad-b", metadata='{"score":NaN}')
    groups = SymbolDeduplicator(db).find_duplicates()
    assert [g.node_ids for g in groups] == [("a", "b")]


def test_merge_preserves_relationships_children_provenance_and_fts(db):
    from victor.core.graph_rag.dedup import SymbolDeduplicator

    node(db, "a", embedding_ref="emb:a")
    node(db, "b", embedding_ref="emb:b")
    node(db, "child", line=11, parent_id="b", scope_id="b")
    db.executemany(
        "INSERT INTO graph_edge VALUES (?,?,?,?,?,?)",
        [
            ("a", "child", "CALLS", 0.4, "src/a.py", '{"site":1}'),
            ("b", "child", "CALLS", 0.9, "src/a.py", '{"site":2}'),
            ("child", "b", "REFERENCES", 1.0, "src/a.py", "{}"),
            ("b", "a", "CALLS", 1.0, "src/a.py", "{}"),
        ],
    )
    db.commit()
    dedup = SymbolDeduplicator(db)
    result = dedup.merge_nodes(dedup.find_duplicates()[0])
    assert result.removed_ids == ("b",)
    assert db.execute("SELECT parent_id,scope_id FROM graph_node WHERE node_id='child'").fetchone()[
        :
    ] == ("a", "a")
    assert set(tuple(r) for r in db.execute("SELECT src,dst,type FROM graph_edge")) == {
        ("a", "child", "CALLS"),
        ("child", "a", "REFERENCES"),
        ("a", "a", "CALLS"),
    }
    edge = db.execute(
        "SELECT weight,metadata FROM graph_edge WHERE src='a' AND dst='child'"
    ).fetchone()
    assert edge[0] == 0.9
    assert {e["metadata"]["site"] for e in json.loads(edge[1])["deduplication_edges"]} == {1, 2}
    canonical = db.execute(
        "SELECT embedding_ref,metadata FROM graph_node WHERE node_id='a'"
    ).fetchone()
    assert canonical[0] is None
    assert json.loads(canonical[1])["deduplication_pending"]["vector_ids"] == [
        "a",
        "b",
        "emb:a",
        "emb:b",
    ]
    assert dedup.find_duplicates() == []
    assert {
        r[0]
        for r in db.execute(
            "SELECT node_id FROM graph_node_fts WHERE graph_node_fts MATCH 'execute'"
        )
    } == {"a", "child"}
    assert db.execute("PRAGMA foreign_key_check").fetchall() == []


def test_stale_plan_and_transaction_failure_leave_graph_unchanged(db):
    from victor.core.graph_rag.dedup import SymbolDeduplicator

    node(db, "a")
    node(db, "b")
    dedup = SymbolDeduplicator(db)
    group = dedup.find_duplicates()[0]
    db.execute("UPDATE graph_node SET signature='changed' WHERE node_id='b'")
    db.commit()
    with pytest.raises(ValueError, match="changed"):
        dedup.merge_nodes(group)
    db.execute("UPDATE graph_node SET signature='def execute():' WHERE node_id='b'")
    db.commit()
    db.executescript(
        "CREATE TRIGGER fail_delete BEFORE DELETE ON graph_node BEGIN SELECT RAISE(ABORT,'injected'); END;"
    )
    before = list(db.iterdump())
    with pytest.raises(sqlite3.IntegrityError, match="injected"):
        dedup.merge_nodes(dedup.find_duplicates()[0])
    assert list(db.iterdump()) == before


def test_caller_transaction_is_not_committed(db):
    from victor.core.graph_rag.dedup import SymbolDeduplicator

    node(db, "a")
    node(db, "b")
    dedup = SymbolDeduplicator(db)
    group = dedup.find_duplicates()[0]
    db.execute("BEGIN")
    with pytest.raises(ValueError, match="transaction"):
        dedup.merge_nodes(group)
    assert db.in_transaction
    db.rollback()


@pytest.mark.parametrize("failure", [False, True])
async def test_embedding_refresh_uses_existing_pipeline_and_remains_retryable(
    db, tmp_path, monkeypatch, failure
):
    from unittest.mock import AsyncMock, MagicMock
    import numpy as np
    from victor_codegraph.embeddings import EmbeddingProfile
    from victor.core.graph_rag.dedup import (
        SymbolDeduplicator,
        update_embeddings,
        pending_embeddings,
    )
    from victor.core.graph_rag.indexing import GraphIndexingPipeline
    from victor.storage.graph.protocol import GraphNode

    node(db, "a", embedding_ref="emb:a")
    node(db, "b", embedding_ref="symbol:shared")
    node(db, "unrelated", line=20, end_line=22, embedding_ref="symbol:shared")
    dedup = SymbolDeduplicator(db)
    dedup.merge_nodes(dedup.find_duplicates()[0])

    from victor.storage.graph.sqlite_store import SqliteGraphStore

    root = tmp_path / "embedding-project"
    store = SqliteGraphStore(root)
    await store.upsert_nodes(
        [
            GraphNode(**{**dict(row), "metadata": json.loads(row["metadata"] or "{}")})
            for row in db.execute("SELECT * FROM graph_node")
        ]
    )
    await store.close()
    project_db = root / ".victor" / "project.db"

    provider = MagicMock()
    provider.delete_document = AsyncMock()
    provider.close = AsyncMock()
    provider.index_embedded_documents = AsyncMock(
        side_effect=RuntimeError("offline") if failure else None
    )
    service = MagicMock()
    service.profile = EmbeddingProfile("test", "a" * 40, 4)
    service.embed_batch = AsyncMock(side_effect=lambda texts: np.ones((len(texts), 4)))
    monkeypatch.setattr(GraphIndexingPipeline, "_get_vector_provider", lambda self: provider)
    monkeypatch.setattr("victor.storage.embeddings.service.get_embedding_service", lambda: service)

    if failure:
        with pytest.raises(RuntimeError, match="offline"):
            await update_embeddings(root)
        with sqlite3.connect(project_db) as project:
            assert set(pending_embeddings(project)) == {"a"}
            assert (
                project.execute(
                    "SELECT embedding_ref FROM graph_node WHERE node_id='a'"
                ).fetchone()[0]
                is None
            )
    else:
        assert await update_embeddings(root) == 1
        with sqlite3.connect(project_db) as project:
            assert pending_embeddings(project) == {}
        documents = provider.index_embedded_documents.await_args.args[0]
        assert [doc["id"] for doc in documents] == ["a"]
        assert documents[0]["metadata"]["embedding_profile_id"] == service.profile.identity
        assert documents[0]["vector"] == [1.0, 1.0, 1.0, 1.0]
    assert "a" not in [call.args[0] for call in provider.delete_document.await_args_list]
    deleted = {call.args[0] for call in provider.delete_document.await_args_list}
    assert "b" in deleted
    assert not {"symbol:shared", "shared"}.intersection(deleted)


def test_multi_group_apply_is_atomic_and_rejects_unsafe_groups(db):
    from dataclasses import replace
    from victor.core.graph_rag.dedup import SymbolDeduplicator

    for node_id in ("a", "b"):
        node(db, node_id)
    for node_id in ("c", "d"):
        node(db, node_id, line=20, end_line=22)
    dedup = SymbolDeduplicator(db)
    groups = dedup.find_duplicates()
    with pytest.raises(ValueError, match="overlapping"):
        dedup.apply([groups[0], groups[0]])
    with pytest.raises(ValueError, match="changed"):
        dedup.apply([groups[0], replace(groups[1], fingerprint="stale")])
    assert db.execute("SELECT count(*) FROM graph_node").fetchone()[0] == 4
    assert len(dedup.apply(groups)) == 2


def test_conflicting_metadata_and_populated_fragment_state_are_not_lost(db):
    from victor.core.graph_rag.dedup import SymbolDeduplicator

    node(db, "a")
    node(db, "b")
    node(db, "invalid", metadata="[]")
    dedup = SymbolDeduplicator(db)
    groups = dedup.find_duplicates()
    db.executescript(
        "CREATE TABLE cpg_fragment_node (node_id TEXT); INSERT INTO cpg_fragment_node VALUES ('b');"
    )
    with pytest.raises(ValueError, match="dependent state"):
        dedup.apply(groups)
    assert db.execute("SELECT count(*) FROM graph_node").fetchone()[0] == 3


async def test_cancelled_sqlite_work_settles_before_returning():
    import asyncio
    import threading
    from victor.core.graph_rag.dedup import settled_thread

    started, finish, committed = threading.Event(), threading.Event(), threading.Event()

    def operation():
        started.set()
        assert finish.wait(timeout=5)
        committed.set()

    task = asyncio.create_task(settled_thread(operation))
    try:
        assert await asyncio.to_thread(started.wait, 5)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
    finally:
        finish.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert committed.is_set()


def test_apply_invalidates_derived_caches_and_requires_foreign_keys(db):
    from victor.core.graph_rag.dedup import SymbolDeduplicator

    node(db, "a")
    node(db, "b")
    db.executescript(
        "CREATE TABLE graph_query_cache (result TEXT);"
        "INSERT INTO graph_query_cache VALUES ('b');"
        "CREATE TABLE graph_subgraph (anchor_node_id TEXT);"
        "INSERT INTO graph_subgraph VALUES ('b');"
    )
    dedup = SymbolDeduplicator(db)
    groups = dedup.find_duplicates()
    db.execute("PRAGMA foreign_keys=OFF")
    with pytest.raises(ValueError, match="foreign key"):
        dedup.apply(groups)
    assert db.execute("SELECT count(*) FROM graph_query_cache").fetchone()[0] == 1
    db.execute("PRAGMA foreign_keys=ON")
    dedup.apply(groups)
    assert db.execute("SELECT count(*) FROM graph_query_cache").fetchone()[0] == 0
    assert db.execute("SELECT count(*) FROM graph_subgraph").fetchone()[0] == 0


def test_unknown_edge_schema_is_rejected_before_mutation(db):
    from victor.core.graph_rag.dedup import SymbolDeduplicator

    node(db, "a")
    node(db, "b")
    db.execute("ALTER TABLE graph_edge ADD COLUMN evidence TEXT")
    db.commit()
    dedup = SymbolDeduplicator(db)
    with pytest.raises(ValueError, match="edge schema"):
        dedup.apply(dedup.find_duplicates())
    assert db.execute("SELECT count(*) FROM graph_node").fetchone()[0] == 2
