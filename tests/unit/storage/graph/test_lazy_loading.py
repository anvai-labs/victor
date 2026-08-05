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

"""Tests for lazy loading in graph store (PH4-006)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from victor.storage.graph.protocol import GraphNode, GraphEdge
from victor.storage.graph.sqlite_store import SqliteGraphStore


@pytest.fixture
def temp_db_path(tmp_path: Path) -> Path:
    """Create a temporary database path."""
    return tmp_path / "test.db"


@pytest.fixture
def graph_store(temp_db_path: Path) -> SqliteGraphStore:
    """Create a graph store for testing."""
    return SqliteGraphStore(project_path=temp_db_path)


@pytest.fixture
async def populated_graph_store(graph_store: SqliteGraphStore) -> SqliteGraphStore:
    """Create a graph store populated with test data."""
    await graph_store.initialize()

    # Create test nodes
    nodes = [
        GraphNode(
            node_id=f"n{i}",
            type="function",
            name=f"func_{i}",
            file=f"file_{i // 10}.py",
            line=i,
            lang="python",
        )
        for i in range(100)
    ]

    # Create test edges
    edges = []
    for i in range(100):
        if i < 99:
            edges.append(GraphEdge(src=f"n{i}", dst=f"n{i+1}", type="CALLS"))
        if i < 90:
            edges.append(GraphEdge(src=f"n{i}", dst=f"n{i+10}", type="REFERENCES"))

    await graph_store.upsert_nodes(nodes)
    await graph_store.upsert_edges(edges)

    return graph_store


class TestLazyNodeIteration:
    """Tests for lazy node iteration."""

    @pytest.mark.asyncio
    async def test_iter_nodes_all(self, populated_graph_store: SqliteGraphStore):
        """Test iterating over all nodes in batches."""
        batches = []
        async for batch in populated_graph_store.iter_nodes(batch_size=20):
            batches.append(batch)

        # Should have 5 batches of 20 nodes each
        assert len(batches) == 5

        # Each batch should have 20 nodes except possibly the last
        for i, batch in enumerate(batches):
            if i < 4:
                assert len(batch) == 20
            else:
                assert len(batch) == 20

        # Total nodes should be 100
        total_nodes = sum(len(batch) for batch in batches)
        assert total_nodes == 100

    @pytest.mark.asyncio
    async def test_iter_nodes_with_filter(self, populated_graph_store: SqliteGraphStore):
        """Test iterating nodes with filters."""
        # Filter by file
        batches = []
        async for batch in populated_graph_store.iter_nodes(batch_size=5, file="file_0.py"):
            batches.append(batch)

        # file_0.py should have nodes 0-9 (10 nodes)
        total_nodes = sum(len(batch) for batch in batches)
        assert total_nodes == 10

        # All nodes should be from file_0.py
        for batch in batches:
            for node in batch:
                assert node.file == "file_0.py"

    @pytest.mark.asyncio
    async def test_iter_nodes_by_type(self, populated_graph_store: SqliteGraphStore):
        """Test iterating nodes filtered by type."""
        batches = []
        async for batch in populated_graph_store.iter_nodes(batch_size=10, type="function"):
            batches.append(batch)

        # All nodes should be functions
        for batch in batches:
            for node in batch:
                assert node.type == "function"

    @pytest.mark.asyncio
    async def test_iter_nodes_single_batch(self, populated_graph_store: SqliteGraphStore):
        """Test iteration with batch size larger than result set."""
        batches = []
        async for batch in populated_graph_store.iter_nodes(batch_size=200):
            batches.append(batch)

        # Should have only one batch
        assert len(batches) == 1
        assert len(batches[0]) == 100

    @pytest.mark.asyncio
    async def test_iter_nodes_empty_result(self, graph_store: SqliteGraphStore):
        """Test iteration with no matching nodes."""
        await graph_store.initialize()

        batches = []
        async for batch in graph_store.iter_nodes(name="nonexistent"):
            batches.append(batch)

        # Should have no batches
        assert len(batches) == 0


class TestLazyEdgeIteration:
    """Tests for lazy edge iteration."""

    @pytest.mark.asyncio
    async def test_iter_edges_all(self, populated_graph_store: SqliteGraphStore):
        """Test iterating over all edges in batches."""
        batches = []
        async for batch in populated_graph_store.iter_edges(batch_size=50):
            batches.append(batch)

        # Should have multiple batches
        assert len(batches) > 0

        # Total edges should be 199 (99 CALLS + 90 REFERENCES = 189)
        total_edges = sum(len(batch) for batch in batches)
        assert total_edges == 189

    @pytest.mark.asyncio
    async def test_iter_edges_filtered(self, populated_graph_store: SqliteGraphStore):
        """Test iterating edges filtered by type."""
        batches = []
        async for batch in populated_graph_store.iter_edges(batch_size=20, edge_types=["CALLS"]):
            batches.append(batch)

        # Should only have CALLS edges (99)
        total_edges = sum(len(batch) for batch in batches)
        assert total_edges == 99

        # All edges should be CALLS
        for batch in batches:
            for edge in batch:
                assert edge.type == "CALLS"

    @pytest.mark.asyncio
    async def test_iter_edges_multiple_types(self, populated_graph_store: SqliteGraphStore):
        """Test iterating edges with multiple type filters."""
        batches = []
        async for batch in populated_graph_store.iter_edges(
            batch_size=30, edge_types=["CALLS", "REFERENCES"]
        ):
            batches.append(batch)

        # Should have all edges
        total_edges = sum(len(batch) for batch in batches)
        assert total_edges == 189

        # All edges should be either CALLS or REFERENCES
        for batch in batches:
            for edge in batch:
                assert edge.type in {"CALLS", "REFERENCES"}

    @pytest.mark.asyncio
    async def test_iter_edges_empty_result(self, graph_store: SqliteGraphStore):
        """Test iteration with no matching edges."""
        await graph_store.initialize()

        batches = []
        async for batch in graph_store.iter_edges(edge_types=["NONEXISTENT"]):
            batches.append(batch)

        # Should have no batches
        assert len(batches) == 0


class TestLazyNeighborIteration:
    """Tests for lazy neighbor iteration."""

    @pytest.mark.asyncio
    async def test_iter_neighbors_out(self, populated_graph_store: SqliteGraphStore):
        """Test iterating outgoing neighbors."""
        batches = []
        async for batch in populated_graph_store.iter_neighbors(
            node_id="n0", batch_size=10, direction="out"
        ):
            batches.append(batch)

        # n0 should have 2 outgoing edges: n0->n1 (CALLS), n0->n10 (REFERENCES)
        total_edges = sum(len(batch) for batch in batches)
        assert total_edges == 2

        # Check destinations
        all_edges = [edge for batch in batches for edge in batch]
        destinations = {edge.dst for edge in all_edges}
        assert destinations == {"n1", "n10"}

    @pytest.mark.asyncio
    async def test_iter_neighbors_in(self, populated_graph_store: SqliteGraphStore):
        """Test iterating incoming neighbors."""
        batches = []
        async for batch in populated_graph_store.iter_neighbors(
            node_id="n10", batch_size=10, direction="in"
        ):
            batches.append(batch)

        # n10 should have 2 incoming edges: n9->n10 (CALLS), n0->n10 (REFERENCES)
        total_edges = sum(len(batch) for batch in batches)
        assert total_edges == 2

    @pytest.mark.asyncio
    async def test_iter_neighbors_both(self, populated_graph_store: SqliteGraphStore):
        """Test iterating neighbors in both directions."""
        batches = []
        async for batch in populated_graph_store.iter_neighbors(
            node_id="n10", batch_size=10, direction="both"
        ):
            batches.append(batch)

        # n10 should have 4 edges total (2 in, 2 out)
        total_edges = sum(len(batch) for batch in batches)
        assert total_edges == 4

    @pytest.mark.asyncio
    async def test_iter_neighbors_filtered(self, populated_graph_store: SqliteGraphStore):
        """Test iterating neighbors with edge type filter."""
        batches = []
        async for batch in populated_graph_store.iter_neighbors(
            node_id="n0", batch_size=10, edge_types=["CALLS"], direction="out"
        ):
            batches.append(batch)

        # Should only have CALLS edge (n0->n1)
        total_edges = sum(len(batch) for batch in batches)
        assert total_edges == 1

        all_edges = [edge for batch in batches for edge in batch]
        assert all_edges[0].type == "CALLS"
        assert all_edges[0].dst == "n1"

    @pytest.mark.asyncio
    async def test_iter_neighbors_nonexistent(self, graph_store: SqliteGraphStore):
        """Test iterating neighbors for nonexistent node."""
        await graph_store.initialize()

        batches = []
        async for batch in graph_store.iter_neighbors(node_id="nonexistent", batch_size=10):
            batches.append(batch)

        # Should have no batches
        assert len(batches) == 0

    @pytest.mark.asyncio
    async def test_iter_neighbors_invalid_direction(self, populated_graph_store: SqliteGraphStore):
        """Test that invalid direction raises error."""
        with pytest.raises(ValueError, match="Unsupported graph traversal direction"):
            async for _ in populated_graph_store.iter_neighbors(
                node_id="n0", direction="invalid"  # type: ignore
            ):
                pass


class TestLazyLoadingMemoryEfficiency:
    """Tests for memory efficiency of lazy loading."""

    @pytest.mark.asyncio
    async def test_iter_nodes_memory_efficient(self, populated_graph_store: SqliteGraphStore):
        """Test that lazy iteration doesn't load all nodes at once."""
        batch_count = 0
        max_batch_size = 0

        async for batch in populated_graph_store.iter_nodes(batch_size=10):
            batch_count += 1
            max_batch_size = max(max_batch_size, len(batch))

            # Each batch should be at most 10 nodes
            assert len(batch) <= 10

        # Should have processed all nodes in batches
        assert batch_count == 10  # 100 nodes / 10 per batch
        assert max_batch_size == 10

    @pytest.mark.asyncio
    async def test_iter_edges_memory_efficient(self, populated_graph_store: SqliteGraphStore):
        """Test that lazy iteration doesn't load all edges at once."""
        max_batch_size = 0

        async for batch in populated_graph_store.iter_edges(batch_size=25):
            max_batch_size = max(max_batch_size, len(batch))

            # Each batch should be at most 25 edges
            assert len(batch) <= 25

        # Max batch size should not exceed the limit
        assert max_batch_size == 25

    @pytest.mark.asyncio
    async def test_early_termination(self, populated_graph_store: SqliteGraphStore):
        """Test that early termination works correctly."""
        batch_count = 0
        max_batches = 2

        async for batch in populated_graph_store.iter_nodes(batch_size=10):
            batch_count += 1
            if batch_count >= max_batches:
                break

        # Should only have processed 2 batches
        assert batch_count == 2


class TestLazyLoadingIntegration:
    """Integration tests for lazy loading with MultiHopRetriever."""

    @pytest.mark.asyncio
    async def test_retriever_uses_lazy_loading(self, populated_graph_store: SqliteGraphStore):
        """Test that MultiHopRetriever can use lazy loading."""
        from victor.core.graph_rag.retrieval import MultiHopRetriever

        # Create a mock config with lazy loading enabled
        config = MagicMock()
        config.seed_count = 5
        config.max_hops = 2
        config.max_nodes = 20
        config.use_lazy_loading = True
        config.lazy_load_batch_size = 10
        config.edge_types = None
        config.centrality_weight = 0
        config.size_penalty_weight = 0

        retriever = MultiHopRetriever(populated_graph_store, config)

        # Check that lazy loading is detected
        assert retriever._use_lazy_loading(config) is True

    @pytest.mark.asyncio
    async def test_retriever_lazy_neighbor_loading(self, populated_graph_store: SqliteGraphStore):
        """Test lazy neighbor loading in retriever."""
        from victor.core.graph_rag.retrieval import MultiHopRetriever

        config = MagicMock()
        config.seed_count = 3
        config.max_hops = 2
        config.max_nodes = 15
        config.use_lazy_loading = True
        config.lazy_load_batch_size = 5
        config.max_neighbors_per_node = 10
        config.edge_types = None
        config.centrality_weight = 0
        config.size_penalty_weight = 0

        retriever = MultiHopRetriever(populated_graph_store, config)

        # Get neighbors using lazy loading
        neighbors = await retriever._get_neighbors_lazy("n0", None, config)

        # Should have neighbors
        assert len(neighbors) > 0
        assert all(isinstance(n, GraphEdge) for n in neighbors)

    @pytest.mark.asyncio
    async def test_retriever_lazy_fallback(self, populated_graph_store: SqliteGraphStore):
        """Test fallback to regular get_neighbors when lazy loading fails."""
        from victor.core.graph_rag.retrieval import MultiHopRetriever

        config = MagicMock()
        config.seed_count = 3
        config.max_hops = 2
        config.max_nodes = 15
        config.use_lazy_loading = True
        config.lazy_load_batch_size = 5
        config.edge_types = None
        config.centrality_weight = 0
        config.size_penalty_weight = 0

        # Create retriever with mocked store that raises error on iter_neighbors
        mock_store = MagicMock(spec=populated_graph_store)
        mock_store.get_neighbors = AsyncMock(return_value=[])
        mock_store.iter_neighbors = MagicMock(side_effect=Exception("Lazy loading failed"))

        retriever = MultiHopRetriever(mock_store, config)

        # Should fall back to get_neighbors
        neighbors = await retriever._get_neighbors_lazy("n0", None, config)

        # Should have called get_neighbors as fallback
        mock_store.get_neighbors.assert_called_once()
        assert neighbors == []

    @pytest.mark.asyncio
    async def test_retriever_lazy_based_on_size(self, populated_graph_store: SqliteGraphStore):
        """Test that lazy loading is used for large max_nodes."""
        from victor.core.graph_rag.retrieval import MultiHopRetriever

        # Create config with max_nodes and top_k set
        class MockConfig:
            def __init__(self, max_nodes_val, top_k_val=100):
                self.max_nodes = max_nodes_val
                self.top_k = top_k_val

        config_large = MockConfig(150, top_k_val=150)
        config_small = MockConfig(50, top_k_val=50)

        retriever = MultiHopRetriever(populated_graph_store, config_large)

        # Should use lazy loading for large max_nodes
        assert retriever._use_lazy_loading(config_large) is True

        # Small max_nodes should not trigger lazy loading
        assert retriever._use_lazy_loading(config_small) is False
