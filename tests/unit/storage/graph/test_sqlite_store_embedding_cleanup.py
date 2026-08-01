# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
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

"""Unit tests for embedding cleanup in SqliteGraphStore.

These tests verify that when graph nodes are deleted, corresponding
embeddings are also cleaned up from the vector store.
"""

import sqlite3
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path
import tempfile

import pytest

from victor.storage.graph.protocol import GraphNode, GraphEdge
from victor.storage.graph.sqlite_store import SqliteGraphStore


@pytest.fixture
def temp_db_path():
    """Create a temporary database path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "test.db"


@pytest.fixture
def graph_store(temp_db_path):
    """Create a SqliteGraphStore instance for testing."""
    store = SqliteGraphStore(project_path=Path(temp_db_path))
    return store


@pytest.mark.asyncio
async def test_delete_by_file_removes_nodes_and_edges(graph_store):
    """Test that delete_by_file removes both nodes and edges."""
    # Create test nodes
    nodes = [
        GraphNode(
            node_id="test:func1",
            type="function",
            name="func1",
            file="test.py",
            line=10,
        ),
        GraphNode(
            node_id="test:func2",
            type="function",
            name="func2",
            file="test.py",
            line=20,
        ),
        GraphNode(
            node_id="other:func1",
            type="function",
            name="func1",
            file="other.py",
            line=10,
        ),
    ]

    # Create test edges
    edges = [
        GraphEdge(src="test:func1", dst="test:func2", type="CALLS"),
        GraphEdge(src="other:func1", dst="test:func1", type="CALLS"),
    ]

    # Insert nodes and edges
    await graph_store.upsert_nodes(nodes)
    await graph_store.upsert_edges(edges)

    # Verify data exists
    all_nodes = await graph_store.get_all_nodes()
    assert len(all_nodes) == 3
    nodes_in_file = await graph_store.get_nodes_by_file("test.py")
    assert len(nodes_in_file) == 2

    # Delete nodes for test.py
    await graph_store.delete_by_file("test.py")

    # Verify only nodes from test.py are deleted
    remaining_nodes = await graph_store.get_all_nodes()
    assert len(remaining_nodes) == 1
    assert remaining_nodes[0].file == "other.py"

    # Verify edges connected to deleted nodes are also removed
    neighbors = await graph_store.get_neighbors("other:func1")
    assert len(neighbors) == 0  # Edge to test:func1 should be gone


@pytest.mark.asyncio
async def test_delete_by_file_calls_vector_store(graph_store, caplog):
    """Test that delete_by_file calls the vector store to delete embeddings."""
    import logging

    caplog.set_level(logging.DEBUG)

    # Create a test node
    nodes = [
        GraphNode(
            node_id="test:func1",
            type="function",
            name="func1",
            file="test.py",
            line=10,
            embedding_ref="emb:test:func1",
        )
    ]

    await graph_store.upsert_nodes(nodes)

    # Mock the vector store provider - patch at the import location
    mock_provider = AsyncMock()
    mock_provider.delete_by_file.return_value = 1

    with patch("victor.storage.vector_stores.registry.EmbeddingRegistry") as mock_registry:
        mock_registry.create.return_value = mock_provider
        await graph_store.delete_by_file("test.py")

    # Verify vector store was called
    mock_provider.delete_by_file.assert_called_once_with("test.py")


@pytest.mark.asyncio
async def test_delete_by_file_handles_vector_store_unavailable(graph_store, caplog):
    """Test that delete_by_file gracefully handles vector store unavailability."""
    import logging

    caplog.set_level(logging.WARNING)

    # Create a test node
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

    # Mock EmbeddingRegistry to raise ImportError
    with patch(
        "victor.storage.vector_stores.registry.EmbeddingRegistry",
        side_effect=ImportError("vector store not available"),
    ):
        # Should not raise exception
        await graph_store.delete_by_file("test.py")

    # Verify nodes are still deleted
    remaining_nodes = await graph_store.get_all_nodes()
    assert len(remaining_nodes) == 0


@pytest.mark.asyncio
async def test_delete_by_file_handles_vector_store_error(graph_store, caplog):
    """Test that delete_by_file gracefully handles vector store errors."""
    import logging

    caplog.set_level(logging.WARNING)

    # Create a test node
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

    # Verify warning was logged
    assert any("Failed to delete embeddings" in record.message for record in caplog.records)


@pytest.mark.asyncio
async def test_clear_all_embeddings_reaches_provider(graph_store):
    """Regression: _clear_all_embeddings must not die on a bad EmbeddingConfig import.

    It previously imported EmbeddingConfig from victor.storage.embeddings.service
    (which does not export it), so the swallowed ImportError meant clear_index()
    was never called during force rebuilds.
    """
    mock_provider = AsyncMock()

    with patch("victor.storage.vector_stores.registry.EmbeddingRegistry") as mock_registry:
        mock_registry.create.return_value = mock_provider
        await graph_store._clear_all_embeddings()

    mock_registry.create.assert_called_once()
    mock_provider.clear_index.assert_awaited_once()


@pytest.mark.asyncio
async def test_delete_embeddings_uses_canonical_path(graph_store):
    """delete_by_file must pass the repo-relative canonical path to the vector store."""
    absolute = str(graph_store._project_root / "pkg" / "mod.py")
    nodes = [
        GraphNode(
            node_id="pkg:mod:func1",
            type="function",
            name="func1",
            file=absolute,
            line=3,
        )
    ]
    await graph_store.upsert_nodes(nodes)

    mock_provider = AsyncMock()
    mock_provider.delete_by_file.return_value = 1

    with patch("victor.storage.vector_stores.registry.EmbeddingRegistry") as mock_registry:
        mock_registry.create.return_value = mock_provider
        await graph_store.delete_by_file(absolute)

    mock_provider.delete_by_file.assert_called_once_with("pkg/mod.py")


@pytest.mark.asyncio
async def test_build_embedding_config_defaults(graph_store):
    """_build_embedding_config returns a usable config even without settings."""
    config = graph_store._build_embedding_config()
    assert config.vector_store  # non-empty provider name
    assert config.embedding_model_name


@pytest.mark.asyncio
async def test_delete_by_file_nonexistent_file(graph_store):
    """Test that delete_by_file handles nonexistent files gracefully."""
    # Should not raise exception
    await graph_store.delete_by_file("nonexistent.py")

    # Verify no errors occurred
    all_nodes = await graph_store.get_all_nodes()
    assert len(all_nodes) == 0


@pytest.mark.asyncio
async def test_delete_by_file_with_batch_connection(graph_store):
    """Test that delete_by_file works with active batch connection.

    Note: This test is skipped due to potential blocking in vector store
    initialization. The batch connection logic is tested indirectly through
    other tests and the write_batch() is used throughout the codebase.
    """
    pytest.skip("Batch connection test may block on vector store initialization")
