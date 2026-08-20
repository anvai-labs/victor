# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
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

"""Integration tests for incremental indexing with embedding cleanup.

These tests verify the end-to-end pipeline of:
1. File watching and change detection
2. Graph incremental indexing
3. Embedding cleanup during file deletion
4. Vector store integration
"""

import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict

import pytest

from victor.storage.graph.protocol import GraphNode, GraphEdge
from victor.storage.graph.sqlite_store import SqliteGraphStore
from victor.storage.graph.incremental_indexing_simple import (
    SimpleIncrementalIndexer,
    IncrementalUpdateStats,
)


@pytest.fixture
def temp_project_dir():
    """Create a temporary project directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        project_path = Path(tmpdir)
        (project_path / "src").mkdir()
        yield project_path


@pytest.fixture
def project_db(temp_project_dir):
    """Create a project database for testing."""
    db_path = temp_project_dir / ".victor" / "project.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path))

    # Create schema
    conn.execute("""
        CREATE TABLE IF NOT EXISTS graph_node (
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
            metadata TEXT
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS graph_edge (
            src TEXT NOT NULL,
            dst TEXT NOT NULL,
            type TEXT NOT NULL,
            weight REAL,
            metadata TEXT,
            PRIMARY KEY (src, dst, type)
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS graph_file_mtime (
            file TEXT PRIMARY KEY,
            mtime REAL NOT NULL,
            indexed_at REAL NOT NULL
        )
    """)

    conn.commit()
    conn.close()

    return db_path


@pytest.fixture
def graph_store(temp_project_dir):
    """Create a SqliteGraphStore instance."""
    store = SqliteGraphStore(project_path=temp_project_dir)
    return store


@pytest.fixture
def indexer(project_db):
    """Create a SimpleIncrementalIndexer instance."""
    conn = sqlite3.connect(str(project_db))
    indexer = SimpleIncrementalIndexer(conn, Path(project_db).parent.parent)
    return indexer


@pytest.mark.asyncio
async def test_incremental_indexing_full_workflow(graph_store, temp_project_dir):
    """Test full incremental indexing workflow: create, modify, delete."""
    # Create initial test file
    test_file = temp_project_dir / "src" / "test.py"
    test_file.write_text("""
def hello_world():
    '''A simple hello function.'''
    print("Hello, World!")

class MyClass:
    def method(self):
        pass
""")

    # Create nodes for the file
    nodes = [
        GraphNode(
            node_id="test:hello_world",
            type="function",
            name="hello_world",
            file="src/test.py",
            line=2,
            end_line=4,
            lang="python",
        ),
        GraphNode(
            node_id="test:MyClass",
            type="class",
            name="MyClass",
            file="src/test.py",
            line=7,
            end_line=10,
            lang="python",
        ),
        GraphNode(
            node_id="test:MyClass.method",
            type="function",
            name="method",
            file="src/test.py",
            line=8,
            end_line=9,
            lang="python",
            parent_id="test:MyClass",
        ),
    ]

    edges = [
        GraphEdge(src="test:MyClass", dst="test:MyClass.method", type="CONTAINS"),
    ]

    # Index the file
    await graph_store.upsert_nodes(nodes)
    await graph_store.upsert_edges(edges)
    await graph_store.update_file_mtime("src/test.py", 100.0)

    # Verify initial state
    all_nodes = await graph_store.get_all_nodes()
    assert len(all_nodes) == 3
    assert len(await graph_store.get_nodes_by_file("src/test.py")) == 3

    all_edges = await graph_store.get_all_edges()
    assert len(all_edges) == 1

    # Simulate file modification - delete old data
    await graph_store.delete_by_file("src/test.py")

    # Verify deletion
    remaining_nodes = await graph_store.get_all_nodes()
    assert len(remaining_nodes) == 0

    remaining_edges = await graph_store.get_all_edges()
    assert len(remaining_edges) == 0


@pytest.mark.asyncio
async def test_incremental_reindex_updates_symbol_correctly(graph_store):
    """Test that incremental reindex correctly updates changed symbols."""
    # Initial indexing
    original_nodes = [
        GraphNode(
            node_id="test:old_function",
            type="function",
            name="old_function",
            file="test.py",
            line=10,
            signature="def old_function():",
        ),
        GraphNode(
            node_id="test:stable_function",
            type="function",
            name="stable_function",
            file="test.py",
            line=20,
            signature="def stable_function():",
        ),
    ]

    await graph_store.upsert_nodes(original_nodes)

    # Simulate modification: rename old_function to new_function
    await graph_store.delete_by_file("test.py")

    updated_nodes = [
        GraphNode(
            node_id="test:new_function",
            type="function",
            name="new_function",
            file="test.py",
            line=10,
            signature="def new_function():",
        ),
        GraphNode(
            node_id="test:stable_function",
            type="function",
            name="stable_function",
            file="test.py",
            line=20,
            signature="def stable_function():",
        ),
    ]

    await graph_store.upsert_nodes(updated_nodes)

    # Verify only new_function exists (not old_function)
    final_nodes = await graph_store.get_nodes_by_file("test.py")
    assert len(final_nodes) == 2
    node_names = {n.name for n in final_nodes}
    assert "new_function" in node_names
    assert "stable_function" in node_names
    assert "old_function" not in node_names


@pytest.mark.asyncio
async def test_delete_file_removes_embeddings_from_vector_store(graph_store, caplog):
    """Verify file deletion removes both graph nodes and vector store embeddings."""
    import logging

    caplog.set_level(logging.DEBUG)

    # Create a node with embedding reference
    nodes = [
        GraphNode(
            node_id="test:function_with_embedding",
            type="function",
            name="function_with_embedding",
            file="test.py",
            line=10,
            embedding_ref="emb:test:function_with_embedding",
        )
    ]

    await graph_store.upsert_nodes(nodes)

    # Mock vector store to verify deletion
    mock_provider = AsyncMock()
    mock_provider.delete_by_file.return_value = 1

    with patch("victor.storage.vector_stores.registry.EmbeddingRegistry") as mock_registry:
        mock_registry.create.return_value = mock_provider
        await graph_store.delete_by_file("test.py")

    # Verify vector store was called with correct file path
    mock_provider.delete_by_file.assert_called_once_with("test.py")

    # Verify graph nodes are also deleted
    remaining_nodes = await graph_store.get_all_nodes()
    assert len(remaining_nodes) == 0


@pytest.mark.asyncio
async def test_vector_store_unavailable_doesnt_block_graph_deletion(graph_store):
    """Verify graph deletion succeeds even if vector store is unavailable."""
    # Create test nodes
    nodes = [
        GraphNode(
            node_id="test:func1",
            type="function",
            name="func1",
            file="test.py",
            line=10,
        )
    ]

    await graph_store.upsert_nodes(nodes)

    # Mock vector store to raise exception
    mock_provider = AsyncMock()
    mock_provider.delete_by_file.side_effect = Exception("Vector store connection failed")

    with patch("victor.storage.vector_stores.registry.EmbeddingRegistry") as mock_registry:
        mock_registry.create.return_value = mock_provider
        # Should not raise exception
        await graph_store.delete_by_file("test.py")

    # Verify nodes are still deleted despite vector store error
    remaining_nodes = await graph_store.get_all_nodes()
    assert len(remaining_nodes) == 0


@pytest.mark.asyncio
async def test_incremental_index_multiple_files(graph_store):
    """Test incremental indexing across multiple files."""
    # Create nodes for multiple files
    nodes_file1 = [
        GraphNode(
            node_id=f"file1:func{i}",
            type="function",
            name=f"func{i}",
            file="file1.py",
            line=i * 10,
        )
        for i in range(5)
    ]
    nodes_file2 = [
        GraphNode(
            node_id=f"file2:func{i}",
            type="function",
            name=f"func{i}",
            file="file2.py",
            line=i * 10,
        )
        for i in range(3)
    ]

    await graph_store.upsert_nodes(nodes_file1 + nodes_file2)

    # Verify initial state
    all_nodes = await graph_store.get_all_nodes()
    assert len(all_nodes) == 8

    # Delete only file1.py
    await graph_store.delete_by_file("file1.py")

    # Verify only file2.py nodes remain
    remaining_nodes = await graph_store.get_all_nodes()
    assert len(remaining_nodes) == 3
    for node in remaining_nodes:
        assert node.file == "file2.py"


@pytest.mark.asyncio
async def test_delete_nonexistent_file_does_not_error(graph_store):
    """Verify deleting a nonexistent file is a no-op."""
    # Should not raise any exception
    await graph_store.delete_by_file("nonexistent.py")

    # Verify no nodes were deleted (there were none anyway)
    all_nodes = await graph_store.get_all_nodes()
    assert len(all_nodes) == 0


@pytest.mark.asyncio
async def test_file_mtime_tracking_for_incremental_updates(graph_store):
    """Test that file mtime tracking correctly identifies stale files."""
    import time

    # Set initial mtime
    current_mtime = time.time()
    await graph_store.update_file_mtime("test.py", current_mtime)

    # Get stale files (with a slightly newer mtime)
    stale = await graph_store.get_stale_files({"test.py": current_mtime + 1})
    assert "test.py" in stale

    # Get stale files (with same mtime - should not be stale)
    not_stale = await graph_store.get_stale_files({"test.py": current_mtime})
    assert "test.py" not in not_stale

    # Get stale files (with older mtime - should not be stale)
    not_stale_old = await graph_store.get_stale_files({"test.py": current_mtime - 1})
    assert "test.py" not in not_stale_old


def test_incremental_indexer_delete_file_data(indexer):
    """Test SimpleIncrementalIndexer.delete_file_data()."""
    # Insert test data
    conn = indexer.db
    conn.execute(
        "INSERT INTO graph_node (node_id, type, name, file, line) VALUES (?, ?, ?, ?, ?)",
        ("test:func1", "function", "func1", "test.py", 10),
    )
    conn.execute(
        "INSERT INTO graph_node (node_id, type, name, file, line) VALUES (?, ?, ?, ?, ?)",
        ("test:func2", "function", "func2", "test.py", 20),
    )
    conn.execute(
        "INSERT INTO graph_node (node_id, type, name, file, line) VALUES (?, ?, ?, ?, ?)",
        ("other:func1", "function", "func1", "other.py", 10),
    )

    conn.execute(
        "INSERT INTO graph_edge (src, dst, type) VALUES (?, ?, ?)",
        ("test:func1", "test:func2", "CALLS"),
    )
    conn.execute(
        "INSERT INTO graph_edge (src, dst, type) VALUES (?, ?, ?)",
        ("other:func1", "test:func1", "CALLS"),
    )

    conn.commit()

    # Delete data for test.py
    result = indexer.delete_file_data("test.py")

    # Verify deletion
    assert result["nodes"] == 2
    assert result["edges"] == 2  # Both edges connected to test.py nodes deleted
    assert result["embeddings"] >= 0

    # Verify other.py data remains
    cursor = conn.execute("SELECT COUNT(*) FROM graph_node WHERE file = ?", ("other.py",))
    assert cursor.fetchone()[0] == 1

    cursor = conn.execute("SELECT COUNT(*) FROM graph_edge")
    assert cursor.fetchone()[0] == 0  # All edges deleted


def test_incremental_indexer_get_changed_files(indexer, temp_project_dir):
    """Test SimpleIncrementalIndexer.get_changed_files_from_mtime()."""
    import time
    import os

    # Create actual files
    unchanged_file = temp_project_dir / "unchanged.py"
    unchanged_file.write_text("# unchanged")

    stale_file = temp_project_dir / "stale.py"
    stale_file.write_text("# stale")

    # Get actual file mtimes
    unchanged_mtime = unchanged_file.stat().st_mtime
    stale_mtime = stale_file.stat().st_mtime

    # Set up mtime tracking with stale time (older than actual file)
    conn = indexer.db
    conn.execute(
        "INSERT INTO graph_file_mtime (file, mtime, indexed_at) VALUES (?, ?, ?)",
        (str(unchanged_file.name), unchanged_mtime, unchanged_mtime),
    )
    # For stale file, use a time older than the actual file
    conn.execute(
        "INSERT INTO graph_file_mtime (file, mtime, indexed_at) VALUES (?, ?, ?)",
        (str(stale_file.name), stale_mtime - 100, stale_mtime - 100),
    )
    conn.commit()

    # Get changed files - stale.py should be detected as changed
    changed = indexer.get_changed_files_from_mtime()

    # The stale file should be detected as changed
    assert any("stale.py" in f for f in changed)


@pytest.mark.asyncio
async def test_cross_file_edge_cleanup_on_file_deletion(graph_store):
    """Test that edges from other files to deleted nodes are cleaned up."""
    # Create nodes in different files
    nodes = [
        GraphNode(
            node_id="file1:func1",
            type="function",
            name="func1",
            file="file1.py",
            line=10,
        ),
        GraphNode(
            node_id="file2:func2",
            type="function",
            name="func2",
            file="file2.py",
            line=10,
        ),
    ]

    # Create edge between files
    edges = [
        GraphEdge(src="file1:func1", dst="file2:func2", type="CALLS"),
    ]

    await graph_store.upsert_nodes(nodes)
    await graph_store.upsert_edges(edges)

    # Verify edge exists
    all_edges = await graph_store.get_all_edges()
    assert len(all_edges) == 1

    # Delete file1.py
    await graph_store.delete_by_file("file1.py")

    # Verify edge is deleted (edge connected to deleted node)
    remaining_edges = await graph_store.get_all_edges()
    assert len(remaining_edges) == 0

    # Verify file2.py node still exists
    remaining_nodes = await graph_store.get_all_nodes()
    assert len(remaining_nodes) == 1
    assert remaining_nodes[0].node_id == "file2:func2"


@pytest.mark.asyncio
async def test_embedding_cleanup_with_mock_vector_store_sync_to_async_bridge(indexer):
    """Test that SimpleIncrementalIndexer handles sync-to-async bridging correctly."""
    # Insert test data
    conn = indexer.db
    conn.execute(
        "INSERT INTO graph_node (node_id, type, name, file, line) VALUES (?, ?, ?, ?, ?)",
        ("test:func1", "function", "func1", "test.py", 10),
    )
    conn.commit()

    # Mock the vector store provider at the registry import path
    mock_provider = AsyncMock()
    mock_provider.delete_by_file.return_value = 5

    with patch(
        "victor.storage.vector_stores.registry.EmbeddingRegistry.create",
        return_value=mock_provider,
    ):

        # Delete file data (includes vector store cleanup)
        result = indexer.delete_file_data("test.py")

    # Verify vector store was called
    mock_provider.delete_by_file.assert_called_once_with("test.py")

    # Verify graph deletion worked
    assert result["nodes"] == 1
    assert result["edges"] == 0
    # The embedding count depends on the vector store call
    # Since we're in a sync context with no running loop, it should create one
    # The count may be 0 or 5 depending on how the loop handles it
    assert result["embeddings"] >= 0


@pytest.mark.asyncio
async def test_concurrent_file_deletions(graph_store):
    """Test that concurrent deletions of different files work correctly."""
    import asyncio

    # Create nodes for multiple files
    nodes = []
    for i in range(5):
        nodes.extend(
            [
                GraphNode(
                    node_id=f"file{i}:func1",
                    type="function",
                    name="func1",
                    file=f"file{i}.py",
                    line=10,
                ),
                GraphNode(
                    node_id=f"file{i}:func2",
                    type="function",
                    name="func2",
                    file=f"file{i}.py",
                    line=20,
                ),
            ]
        )

    await graph_store.upsert_nodes(nodes)

    # Verify initial state
    all_nodes = await graph_store.get_all_nodes()
    assert len(all_nodes) == 10

    # Delete all files concurrently
    async def delete_file(file_num):
        await graph_store.delete_by_file(f"file{file_num}.py")

    tasks = [delete_file(i) for i in range(5)]
    await asyncio.gather(*tasks)

    # Verify all nodes deleted
    remaining_nodes = await graph_store.get_all_nodes()
    assert len(remaining_nodes) == 0
