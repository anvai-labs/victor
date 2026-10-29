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

"""Tests for SQLite graph store cleanup operations.

These tests verify that delete_by_repo() and related cleanup operations
work correctly for force mode rebuilds.
"""

import asyncio
import sqlite3
from contextlib import closing
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from victor.core.database import reset_project_database
from victor.core.schema import Schema
from victor.storage.graph.protocol import GraphNode, GraphEdge
from victor.storage.graph.sqlite_store import SqliteGraphStore


@pytest.fixture
async def publication_store(tmp_path):
    store = SqliteGraphStore(tmp_path)
    await store.upsert_nodes([GraphNode("old", "function", "previous", "old.py")])
    await store.update_file_mtime("old.py", 1.0, "old-hash")
    try:
        yield store
    finally:
        await store.close()
        reset_project_database(tmp_path)


async def test_staged_replacement_preserves_sessions_and_reader_snapshot(publication_store):
    store = publication_store
    reader = sqlite3.connect(store.db_path)
    reader.execute("BEGIN")
    assert reader.execute("SELECT name FROM graph_node").fetchall() == [("previous",)]
    try:
        async with store._stage_replacement() as replacement:
            assert replacement.store.repo_root == store.repo_root
            assert replacement.store.db_path != store.db_path
            with closing(sqlite3.connect(replacement.store.db_path)) as stage_conn, stage_conn:
                tables = {r[0] for r in stage_conn.execute("SELECT name FROM sqlite_master")}
                assert "sessions" not in tables and "messages" not in tables
            await replacement.store.upsert_nodes(
                [GraphNode("new", "function", "replacement", "new.py")]
            )
            await replacement.store.update_file_mtime("new.py", 2.0, "new-hash")
            with closing(sqlite3.connect(store.db_path)) as session, session:
                session.execute(Schema.CONV_SESSION)
                session.execute(Schema.CONV_MESSAGE)
                session.execute(
                    "INSERT INTO sessions (session_id,created_at,last_activity) "
                    "VALUES ('concurrent',datetime('now'),datetime('now'))"
                )
                session.execute(
                    "INSERT INTO messages "
                    "(id,session_id,role,content,timestamp,token_count,priority) "
                    "VALUES ('message','concurrent','user','preserve me',"
                    "datetime('now'),2,1)"
                )
            receipt = await replacement.publish()
            assert receipt.publication_id == replacement.publication_id
            assert [n.node_id for n in await store.get_all_nodes()] == ["new"]
            assert [n.node_id for n in await store.search_symbols("replacement")] == ["new"]
            assert await store.search_symbols("previous") == []
            with pytest.raises(RuntimeError, match="sealed"):
                await replacement.store.upsert_nodes([GraphNode("late", "file", "late", "x")])
            with pytest.raises(RuntimeError, match="already"):
                await replacement.publish()
            assert reader.execute("SELECT name FROM graph_node").fetchall() == [("previous",)]
            reader.rollback()
            assert reader.execute("SELECT name FROM graph_node").fetchall() == [("replacement",)]
            assert reader.execute("SELECT session_id,content FROM messages").fetchall() == [
                ("concurrent", "preserve me")
            ]
    finally:
        reader.close()


@pytest.mark.parametrize("same_content", [False, True])
async def test_staged_replacement_rejects_competing_graph_write(publication_store, same_content):
    store = publication_store
    async with store._stage_replacement() as first, store._stage_replacement() as second:
        if same_content:
            await first.store.upsert_nodes(await store.get_all_nodes())
            with closing(sqlite3.connect(store.db_path)) as source:
                row = source.execute("SELECT * FROM graph_file_mtime").fetchone()
            with closing(sqlite3.connect(first.store.db_path)) as target, target:
                target.execute("INSERT INTO graph_file_mtime VALUES (?,?,?,?)", row)
        else:
            await first.store.upsert_nodes([GraphNode("winner", "file", "winner", "a.py")])
        await first.publish()
        with pytest.raises(RuntimeError, match="changed"):
            await second.publish()
    assert [n.node_id for n in await store.get_all_nodes()] == (
        ["old"] if same_content else ["winner"]
    )


@pytest.mark.parametrize("invalid", ["dangling_edge", "dangling_parent", "embedding"])
async def test_staged_replacement_rejects_invalid_graph(publication_store, invalid):
    store = publication_store
    async with store._stage_replacement() as replacement:
        node = GraphNode("new", "file", "new", "new.py")
        if invalid == "dangling_parent":
            node.parent_id = "missing"
        elif invalid == "embedding":
            node.embedding_ref = "external-vector"
        # Inject invalid persisted input directly: publication must validate storage,
        # independently of which insertion API produced it.
        with closing(sqlite3.connect(replacement.store.db_path)) as conn, conn:
            conn.execute(
                "INSERT INTO graph_node (node_id,type,name,file,parent_id,embedding_ref) "
                "VALUES (?,?,?,?,?,?)",
                (node.node_id, node.type, node.name, node.file, node.parent_id, node.embedding_ref),
            )
            if invalid == "dangling_edge":
                conn.execute(
                    "INSERT INTO graph_edge (src,dst,type) VALUES ('new','absent','CALLS')"
                )
        with pytest.raises(ValueError):
            await replacement.publish()
    assert [n.node_id for n in await store.get_all_nodes()] == ["old"]


@pytest.mark.parametrize(
    "failure", ["graph_node", "graph_edge", "graph_file_mtime", "fts", "sql_constraint", "receipt"]
)
async def test_staged_replacement_copy_failure_rolls_back(publication_store, monkeypatch, failure):
    from victor.storage.graph import sqlite_publication as publication

    store = publication_store
    original_copy = publication._copy_rows
    original_receipt = publication._write_receipt

    def fail_copy(source, target, table):
        original_copy(source, target, table)
        if table == failure:
            raise RuntimeError("injected copy failure")

    def fail_receipt(conn, receipt):
        original_receipt(conn, receipt)
        raise RuntimeError("injected receipt failure")

    monkeypatch.setattr(publication, "_copy_rows", fail_copy)
    if failure == "receipt":
        monkeypatch.setattr(publication, "_write_receipt", fail_receipt)
    async with store._stage_replacement() as replacement:
        await replacement.store.upsert_nodes([GraphNode("new", "file", "fresh", "new.py")])
        await replacement.store.upsert_edges([GraphEdge("new", "new", "CALLS")])
        await replacement.store.update_file_mtime("new.py", 2.0, "new-hash")
        if failure == "fts":
            # Simulate a broken trigger without replacing the publication validator.
            with closing(sqlite3.connect(store.db_path)) as conn, conn:
                conn.execute("DROP TRIGGER graph_node_ai")
        elif failure == "sql_constraint":
            with closing(sqlite3.connect(store.db_path)) as conn, conn:
                conn.execute(
                    "CREATE TRIGGER reject_new BEFORE INSERT ON graph_node "
                    "BEGIN SELECT RAISE(ABORT, 'reject new node'); END"
                )
        with pytest.raises((RuntimeError, sqlite3.DatabaseError)):
            await replacement.publish()
    assert [n.node_id for n in await store.get_all_nodes()] == ["old"]
    assert await store.get_indexed_files() == ["old.py"]
    assert [n.node_id for n in await store.search_symbols("previous")] == ["old"]
    assert await store.search_symbols("fresh") == []
    with closing(sqlite3.connect(store.db_path)) as conn, conn:
        assert (
            conn.execute(
                "SELECT value FROM _project_metadata WHERE key='graph_last_publication_v1'"
            ).fetchone()
            is None
        )


@pytest.mark.parametrize(
    "mutation", ["empty", "schema", "root", "dependent", "query_cache", "external_write"]
)
async def test_staged_replacement_admission(publication_store, mutation):
    store = publication_store
    async with store._stage_replacement() as replacement:
        if mutation == "schema":
            with closing(sqlite3.connect(replacement.store.db_path)) as conn, conn:
                conn.execute("ALTER TABLE graph_node ADD COLUMN future_column TEXT")
        elif mutation == "root":
            replacement.store._project_root = store.repo_root / "different"
        elif mutation == "dependent":
            from victor.core.schema import Schema

            with closing(sqlite3.connect(store.db_path)) as conn, conn:
                conn.execute(Schema.GRAPH_REQUIREMENT)
                conn.execute(
                    "INSERT INTO graph_requirement (requirement_id,type,title) "
                    "VALUES ('old','requirement','preserve me')"
                )
        elif mutation == "query_cache":
            from victor.core.graph_rag.query_cache import _SqlitePersistence

            cache = _SqlitePersistence(str(store.repo_root))
            cache.put("stale", {"nodes": ["old"]}, 10)
        elif mutation == "external_write":
            with closing(sqlite3.connect(store.db_path)) as conn, conn:
                conn.execute("UPDATE graph_node SET name='concurrent' WHERE node_id='old'")
        if mutation == "empty":
            await replacement.publish()
            assert await store.get_all_nodes() == []
        else:
            with pytest.raises((RuntimeError, ValueError)):
                await replacement.publish()
            assert [n.node_id for n in await store.get_all_nodes()] == ["old"]


async def test_staged_replacement_prohibits_external_effects_and_context_escape(publication_store):
    store = publication_store
    async with store._stage_replacement() as replacement:
        path = replacement.store.db_path
        await replacement.store.upsert_nodes([GraphNode("new", "file", "new", "new.py")])
        with pytest.raises(RuntimeError, match="external embeddings"):
            await replacement.store.delete_by_file("new.py")
        with pytest.raises(RuntimeError, match="external embeddings"):
            await replacement.store.delete_by_repo(clear_embeddings=True)
        assert [n.node_id for n in await replacement.store.get_all_nodes()] == ["new"]
    assert not path.exists()
    with pytest.raises(RuntimeError, match="closed"):
        await replacement.publish()
    assert [n.node_id for n in await store.get_all_nodes()] == ["old"]


async def test_staged_replacement_rejects_child_publication(publication_store):
    store = publication_store
    async with store._stage_replacement() as replacement:
        await replacement.store.upsert_nodes([GraphNode("new", "file", "new", "new.py")])
        with pytest.raises(RuntimeError, match="context owner"):
            await asyncio.create_task(replacement.publish())
        assert [n.node_id for n in await store.get_all_nodes()] == ["old"]
        # A rejected child neither consumes the attempt nor seals the stage.
        await replacement.publish()
    assert [n.node_id for n in await store.get_all_nodes()] == ["new"]


async def test_staged_replacement_process_death_before_commit(publication_store):
    import subprocess
    import sys
    import textwrap

    store = publication_store
    script = textwrap.dedent("""
        import asyncio
        import os
        import sys
        from pathlib import Path
        from victor.storage.graph.sqlite_store import SqliteGraphStore
        from victor.storage.graph.protocol import GraphNode
        from victor.storage.graph import sqlite_publication as publication

        async def main():
            store = SqliteGraphStore(Path(sys.argv[1]))
            original = publication._write_receipt
            def die_after_receipt(conn, receipt):
                original(conn, receipt)
                os._exit(73)  # No Python cleanup: copy and receipt are uncommitted.
            publication._write_receipt = die_after_receipt
            async with store._stage_replacement() as stage:
                await stage.store.upsert_nodes([GraphNode("new", "file", "new", "new.py")])
                await stage.publish()
        asyncio.run(main())
    """)
    result = await asyncio.to_thread(
        subprocess.run,
        [sys.executable, "-c", script, str(store.repo_root)],
        cwd=Path(__file__).resolve().parents[4],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 73, result.stderr
    with closing(sqlite3.connect(store.db_path)) as observer:
        assert observer.execute("PRAGMA integrity_check").fetchone() == ("ok",)
        assert observer.execute("SELECT node_id FROM graph_node").fetchall() == [("old",)]
        assert observer.execute("SELECT file FROM graph_file_mtime").fetchall() == [("old.py",)]
        assert (
            observer.execute(
                "SELECT value FROM _project_metadata WHERE key='graph_last_publication_v1'"
            ).fetchone()
            is None
        )
    assert [n.node_id for n in await store.search_symbols("previous")] == ["old"]
    # Abrupt death can leave a private stage; it is never treated as published.
    assert list(store.db_path.parent.glob("graph-stage-*/graph.db"))


@pytest.mark.asyncio
class TestDeleteByRepo:
    """Test delete_by_repo() method for full repository cleanup."""

    @pytest.fixture
    async def graph_store(self, tmp_path: Path) -> SqliteGraphStore:
        """Create a graph store for testing."""
        store = SqliteGraphStore(project_path=tmp_path)
        await store.initialize()
        yield store
        await store.close()
        reset_project_database(tmp_path)

    async def test_delete_by_repo_removes_all_nodes(self, graph_store: SqliteGraphStore) -> None:
        """Verify all nodes are deleted by delete_by_repo()."""
        # Insert test nodes
        nodes = [
            GraphNode(
                node_id=f"node:{i}",
                type="FUNCTION",
                name=f"func_{i}",
                file=f"/path/to/file{i}.py",
                line=10 + i,
                end_line=20 + i,
                lang="python",
            )
            for i in range(5)
        ]
        await graph_store.upsert_nodes(nodes)

        # Verify nodes exist
        stats_before = await graph_store.stats()
        assert stats_before["nodes"] == 5

        # Delete by repo
        await graph_store.delete_by_repo()

        # Verify all nodes are deleted
        stats_after = await graph_store.stats()
        assert stats_after["nodes"] == 0

    async def test_delete_by_repo_removes_all_edges(self, graph_store: SqliteGraphStore) -> None:
        """Verify all edges are deleted by delete_by_repo()."""
        # First add nodes (edges require nodes to exist)
        nodes = [
            GraphNode(
                node_id=f"node:{i}",
                type="FUNCTION",
                name=f"func_{i}",
                file="/path/to/file.py",
                line=10 + i,
                end_line=20 + i,
                lang="python",
            )
            for i in range(3)
        ]
        await graph_store.upsert_nodes(nodes)

        # Insert test edges
        edges = [
            GraphEdge(
                src="node:0",
                dst="node:1",
                type="CALLS",
                weight=1.0,
                metadata={"file": "/path/to/file.py"},
            ),
            GraphEdge(
                src="node:1",
                dst="node:2",
                type="CALLS",
                weight=1.0,
                metadata={"file": "/path/to/file.py"},
            ),
        ]
        await graph_store.upsert_edges(edges)

        # Verify edges exist
        stats_before = await graph_store.stats()
        assert stats_before["edges"] == 2

        # Delete by repo
        await graph_store.delete_by_repo()

        # Verify all edges are deleted
        stats_after = await graph_store.stats()
        assert stats_after["edges"] == 0

    async def test_delete_by_repo_removes_file_mtime_entries(
        self, graph_store: SqliteGraphStore
    ) -> None:
        """Verify file mtime tracking is cleared."""
        # Update some file mtimes
        await graph_store.update_file_mtime("/path/to/file1.py", 123456.0)
        await graph_store.update_file_mtime("/path/to/file2.py", 123457.0)

        # Verify mtimes exist via stats
        stats_before = await graph_store.stats()
        assert stats_before["indexed_files"] == 2

        # Delete by repo
        await graph_store.delete_by_repo()

        # Verify mtimes are cleared
        stats_after = await graph_store.stats()
        assert stats_after["indexed_files"] == 0

    async def test_delete_by_repo_handles_empty_database(
        self, graph_store: SqliteGraphStore
    ) -> None:
        """Verify graceful handling of empty database."""
        # Database is already empty from fixture
        stats_before = await graph_store.stats()
        assert stats_before["nodes"] == 0
        assert stats_before["edges"] == 0

        # Should not raise
        await graph_store.delete_by_repo()

        # Still empty
        stats_after = await graph_store.stats()
        assert stats_after["nodes"] == 0
        assert stats_after["edges"] == 0

    async def test_delete_by_repo_within_write_batch(self, graph_store: SqliteGraphStore) -> None:
        """Verify delete_by_repo works correctly within a write_batch context."""
        # Insert test data
        nodes = [
            GraphNode(
                node_id=f"node:{i}",
                type="FUNCTION",
                name=f"func_{i}",
                file="/path/to/file.py",
                line=10 + i,
                end_line=20 + i,
                lang="python",
            )
            for i in range(3)
        ]
        await graph_store.upsert_nodes(nodes)

        # Verify data exists
        stats_before = await graph_store.stats()
        assert stats_before["nodes"] == 3

        # Delete within batch context
        async with graph_store.write_batch():
            await graph_store.delete_by_repo()

        # Verify data is deleted
        stats_after = await graph_store.stats()
        assert stats_after["nodes"] == 0


@pytest.mark.asyncio
class TestDeleteByFileIntegrity:
    """Test delete_by_file() for atomic cleanup of all file-related data."""

    @pytest.fixture
    async def graph_store(self, tmp_path: Path, monkeypatch) -> SqliteGraphStore:
        """Create a graph store for testing."""
        store = SqliteGraphStore(project_path=tmp_path)
        await store.initialize()
        monkeypatch.setattr(store, "_delete_embeddings_for_file", AsyncMock())
        yield store
        await store.close()
        reset_project_database(tmp_path)

    async def test_delete_by_file_removes_nodes(self, graph_store: SqliteGraphStore) -> None:
        """Verify delete_by_file removes all nodes for a file."""
        # Insert nodes from different files
        nodes = [
            GraphNode(
                node_id="node:file1_func1",
                type="FUNCTION",
                name="func1",
                file="/path/to/file1.py",
                line=10,
                end_line=20,
                lang="python",
            ),
            GraphNode(
                node_id="node:file1_func2",
                type="FUNCTION",
                name="func2",
                file="/path/to/file1.py",
                line=30,
                end_line=40,
                lang="python",
            ),
            GraphNode(
                node_id="node:file2_func1",
                type="FUNCTION",
                name="func3",
                file="/path/to/file2.py",  # Different file
                line=10,
                end_line=20,
                lang="python",
            ),
        ]
        await graph_store.upsert_nodes(nodes)

        # Delete file1.py
        await graph_store.delete_by_file("/path/to/file1.py")

        # Verify file1 nodes are gone, file2 nodes remain
        stats = await graph_store.stats()
        assert stats["nodes"] == 1, "Only file2 nodes should remain"

    async def test_delete_by_file_removes_edges(self, graph_store: SqliteGraphStore) -> None:
        """Verify delete_by_file removes all edges for a file."""
        # Insert nodes
        nodes = [
            GraphNode(
                node_id=f"node:{i}",
                type="FUNCTION",
                name=f"func{i}",
                file="/path/to/file1.py" if i < 2 else "/path/to/file2.py",
                line=10 + i,
                end_line=20 + i,
                lang="python",
            )
            for i in range(3)
        ]
        await graph_store.upsert_nodes(nodes)

        # Insert edges (some from file1, some cross-file)
        edges = [
            GraphEdge(
                src="node:0",
                dst="node:1",
                type="CALLS",
                weight=1.0,
                metadata={"file": "/path/to/file1.py"},
            ),
            GraphEdge(
                src="node:0",
                dst="node:2",
                type="CALLS",
                weight=1.0,
                metadata={"file": "/path/to/file1.py"},
            ),
            GraphEdge(
                src="node:1",
                dst="node:2",
                type="CALLS",
                weight=1.0,
                metadata={"file": "/path/to/file1.py"},
            ),
        ]
        await graph_store.upsert_edges(edges)

        stats_before = await graph_store.stats()
        edges_before = stats_before["edges"]

        # Delete file1.py - should remove edges that have file1 nodes
        await graph_store.delete_by_file("/path/to/file1.py")

        stats_after = await graph_store.stats()
        assert stats_after["edges"] < edges_before, "Edges should be removed"

    async def test_delete_by_file_removes_mtime(self, graph_store: SqliteGraphStore) -> None:
        """Verify delete_by_file removes file mtime entry."""
        # Update file mtimes
        await graph_store.update_file_mtime("/path/to/file1.py", 123456.0)
        await graph_store.update_file_mtime("/path/to/file2.py", 123457.0)

        stats_before = await graph_store.stats()
        assert stats_before["indexed_files"] == 2

        # Delete file1.py
        await graph_store.delete_by_file("/path/to/file1.py")

        stats_after = await graph_store.stats()
        assert stats_after["indexed_files"] == 1, "Only file2 mtime should remain"


@pytest.mark.asyncio
@pytest.mark.parametrize("synchronous", [1, 2])
@pytest.mark.parametrize("fail", [False, True])
async def test_delete_preserves_policy_and_sessions(tmp_path, monkeypatch, synchronous, fail):
    from victor.core.database import reset_project_database
    from victor.core.schema import Schema

    store = SqliteGraphStore(project_path=tmp_path)
    try:
        await store.initialize()
        await store.upsert_nodes([GraphNode(node_id="old", type="file", name="old", file="old.py")])
        conn = store._connect()
        conn.execute(Schema.CONV_SESSION)
        conn.execute(
            "INSERT INTO sessions (session_id, created_at, last_activity) VALUES ('chat', 'now', 'now')"
        )
        conn.commit()
        conn.execute(f"PRAGMA synchronous={synchronous}")
        pragmas = ("synchronous", "journal_mode", "cache_size", "mmap_size", "temp_store")
        before = {key: conn.execute(f"PRAGMA {key}").fetchone()[0] for key in pragmas}
        original = store._delete_by_repo_conn
        observed = []

        def delete(connection):
            observed.append(connection.execute("PRAGMA synchronous").fetchone()[0])
            original(connection)
            if fail:
                raise RuntimeError("injected deletion failure")

        monkeypatch.setattr(store, "_delete_by_repo_conn", delete)
        if fail:
            with pytest.raises(RuntimeError, match="injected deletion failure"):
                await store.delete_by_repo()
        else:
            await store.delete_by_repo()
        assert observed == [synchronous]
        assert {key: conn.execute(f"PRAGMA {key}").fetchone()[0] for key in pragmas} == before
        assert not conn.in_transaction
        await store.upsert_nodes(
            [GraphNode(node_id="later", type="file", name="later", file="later.py")]
        )
        with closing(sqlite3.connect(store.db_path)) as observer, observer:
            assert {r[0] for r in observer.execute("SELECT node_id FROM graph_node")} == (
                {"old", "later"} if fail else {"later"}
            )
            assert observer.execute("SELECT session_id FROM sessions").fetchall() == [("chat",)]
    finally:
        await store.close()
        reset_project_database(tmp_path)


@pytest.mark.asyncio
async def test_nested_repository_delete_does_not_commit_outer_transaction(tmp_path):
    from victor.core.database import reset_project_database

    store = SqliteGraphStore(project_path=tmp_path)
    try:
        await store.upsert_nodes([GraphNode(node_id="old", type="file", name="old", file="old.py")])
        with pytest.raises(RuntimeError, match="abort outer batch"):
            async with store.write_batch():
                await store.delete_by_repo()
                await store.upsert_nodes(
                    [GraphNode(node_id="new", type="file", name="new", file="new.py")]
                )
                raise RuntimeError("abort outer batch")
        with closing(sqlite3.connect(store.db_path)) as observer, observer:
            assert observer.execute("SELECT node_id FROM graph_node").fetchall() == [("old",)]
            assert observer.execute(
                "SELECT node_id FROM graph_node_fts WHERE graph_node_fts MATCH 'old'"
            ).fetchall() == [("old",)]
            assert (
                observer.execute(
                    "SELECT node_id FROM graph_node_fts WHERE graph_node_fts MATCH 'new'"
                ).fetchall()
                == []
            )
        await store.upsert_nodes(
            [GraphNode(node_id="later", type="file", name="later", file="later.py")]
        )
        with closing(sqlite3.connect(store.db_path)) as observer, observer:
            assert observer.execute(
                "SELECT node_id FROM graph_node_fts WHERE graph_node_fts MATCH 'later'"
            ).fetchall() == [("later",)]
    finally:
        await store.close()
        reset_project_database(tmp_path)
