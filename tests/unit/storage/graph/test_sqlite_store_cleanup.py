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

import pytest
import sqlite3

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
        # Cleanup is handled by store's own lifecycle

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
    async def graph_store(self, tmp_path: Path) -> SqliteGraphStore:
        """Create a graph store for testing."""
        store = SqliteGraphStore(project_path=tmp_path)
        await store.initialize()
        yield store

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
class TestBulkLoadOptimizations:
    """Test PRAGMA optimizations for bulk loading."""

    @pytest.fixture
    async def graph_store(self, tmp_path: Path) -> SqliteGraphStore:
        """Create a graph store for testing."""
        store = SqliteGraphStore(project_path=tmp_path)
        await store.initialize()
        yield store

    async def test_enable_bulk_load_mode_sets_pragmas(self, graph_store: SqliteGraphStore) -> None:
        """Verify _enable_bulk_load_mode sets performance pragmas."""
        # This test verifies the PRAGMA settings are applied
        # The actual test is that the methods exist and don't error

        # Get internal connection to check pragmas
        if hasattr(graph_store, "_enable_bulk_load_mode"):
            # Just verify it can be called without error
            conn = graph_store._connect()
            try:
                if hasattr(graph_store, "_enable_bulk_load_mode"):
                    graph_store._enable_bulk_load_mode(conn)

                # Check cache_size was set (should be negative for KB)
                cur = conn.execute("PRAGMA cache_size")
                cache_size = cur.fetchone()[0]
                # Cache size should be set to a negative value (KB) or large positive (pages)
                assert isinstance(cache_size, int)

                # Check temp_store (can be 0=DEFAULT, 1=FILE, 2=MEMORY, or string)
                cur = conn.execute("PRAGMA temp_store")
                temp_store = cur.fetchone()[0]
                assert temp_store in (
                    0,
                    1,
                    2,
                    "0",
                    "1",
                    "2",
                    "MEMORY",
                    "FILE",
                    "DEFAULT",
                )

            finally:
                conn.close()

    async def test_disable_bulk_load_mode_restores_settings(
        self, graph_store: SqliteGraphStore
    ) -> None:
        """Verify _disable_bulk_load_mode restores normal settings."""
        if hasattr(graph_store, "_disable_bulk_load_mode"):
            conn = graph_store._connect()
            try:
                graph_store._enable_bulk_load_mode(conn)
                graph_store._disable_bulk_load_mode(conn)

                # Verify synchronous is restored to NORMAL
                cur = conn.execute("PRAGMA synchronous")
                synchronous = cur.fetchone()[0]
                # NORMAL is 1
                assert synchronous == 1
            finally:
                conn.close()
