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
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
import sqlite3

from victor.core.database import reset_project_database
from victor.storage.graph.protocol import GraphNode, GraphEdge
from victor.storage.graph.sqlite_store import SqliteGraphStore


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
        with sqlite3.connect(store.db_path) as observer:
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
        with sqlite3.connect(store.db_path) as observer:
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
        with sqlite3.connect(store.db_path) as observer:
            assert observer.execute(
                "SELECT node_id FROM graph_node_fts WHERE graph_node_fts MATCH 'later'"
            ).fetchall() == [("later",)]
    finally:
        await store.close()
        reset_project_database(tmp_path)
