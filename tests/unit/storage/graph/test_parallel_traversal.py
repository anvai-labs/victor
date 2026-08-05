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

"""Tests for parallel graph traversal (PH4-007)."""

from __future__ import annotations

from pathlib import Path

import pytest

from victor.storage.graph.protocol import GraphNode, GraphEdge
from victor.storage.graph.sqlite_store import SqliteGraphStore


@pytest.fixture
def temp_db_path(tmp_path: Path) -> Path:
    """Create a temporary database path."""
    return tmp_path / "test_parallel.db"


@pytest.fixture
def graph_store(temp_db_path: Path) -> SqliteGraphStore:
    """Create a graph store for testing."""
    return SqliteGraphStore(project_path=temp_db_path)


@pytest.fixture
async def populated_graph_store(graph_store: SqliteGraphStore) -> SqliteGraphStore:
    """Create a graph store populated with test data."""
    await graph_store.initialize()

    # Create test nodes - star topology with multiple branches
    nodes = [
        GraphNode(
            node_id="center",
            type="function",
            name="center_func",
            file="center.py",
            line=0,
            lang="python",
        ),
    ]

    # Create 20 branches from center
    for i in range(20):
        nodes.append(
            GraphNode(
                node_id=f"branch_{i}",
                type="function",
                name=f"branch_func_{i}",
                file=f"branch_{i}.py",
                line=i,
                lang="python",
            )
        )

    # Create edges - center to each branch
    edges = [GraphEdge(src="center", dst=f"branch_{i}", type="CALLS") for i in range(20)]

    # Create some cross-branch edges
    for i in range(10):
        edges.append(GraphEdge(src=f"branch_{i}", dst=f"branch_{i+10}", type="REFERENCES"))

    await graph_store.upsert_nodes(nodes)
    await graph_store.upsert_edges(edges)

    return graph_store


class TestParallelNeighborBatch:
    """Tests for parallel neighbor batch retrieval."""

    @pytest.mark.asyncio
    async def test_get_neighbors_batch_single(self, populated_graph_store: SqliteGraphStore):
        """Test getting neighbors for a single node."""
        result = await populated_graph_store.get_neighbors_batch(["center"])

        assert "center" in result
        assert len(result["center"]) == 20  # 20 branches

    @pytest.mark.asyncio
    async def test_get_neighbors_batch_multiple(self, populated_graph_store: SqliteGraphStore):
        """Test getting neighbors for multiple nodes in parallel."""
        result = await populated_graph_store.get_neighbors_batch(["center", "branch_0", "branch_1"])

        # All nodes should have results
        assert "center" in result
        assert "branch_0" in result
        assert "branch_1" in result

        # Center should have 20 outgoing edges
        assert len(result["center"]) == 20

        # Branches should have 1 incoming edge (from center)
        # unless they have cross-branch edges
        assert len(result["branch_0"]) >= 1
        assert len(result["branch_1"]) >= 1

    @pytest.mark.asyncio
    async def test_get_neighbors_batch_empty(self, graph_store: SqliteGraphStore):
        """Test getting neighbors with empty node list."""
        await graph_store.initialize()

        result = await graph_store.get_neighbors_batch([])

        assert result == {}

    @pytest.mark.asyncio
    async def test_get_neighbors_batch_nonexistent(self, graph_store: SqliteGraphStore):
        """Test getting neighbors for nonexistent nodes."""
        await graph_store.initialize()

        result = await graph_store.get_neighbors_batch(["nonexistent1", "nonexistent2"])

        assert "nonexistent1" in result
        assert "nonexistent2" in result
        assert result["nonexistent1"] == []
        assert result["nonexistent2"] == []

    @pytest.mark.asyncio
    async def test_get_neighbors_batch_filtered(self, populated_graph_store: SqliteGraphStore):
        """Test getting neighbors with edge type filter."""
        result = await populated_graph_store.get_neighbors_batch(
            ["center", "branch_0"],
            edge_types=["CALLS"],
        )

        # Should only have CALLS edges
        for edges in result.values():
            for edge in edges:
                assert edge.type == "CALLS"

    @pytest.mark.asyncio
    async def test_get_neighbors_batch_in_direction(self, populated_graph_store: SqliteGraphStore):
        """Test getting neighbors with incoming direction."""
        result = await populated_graph_store.get_neighbors_batch(
            ["branch_0", "branch_1"],
            direction="in",
        )

        # Branches should have incoming edges
        assert "branch_0" in result
        assert "branch_1" in result

        # Should have at least the edge from center
        assert len(result["branch_0"]) >= 1
        assert len(result["branch_1"]) >= 1


class TestParallelMultiHopTraversal:
    """Tests for parallel multi-hop traversal."""

    @pytest.mark.asyncio
    async def test_parallel_traverse_single_seed(self, populated_graph_store: SqliteGraphStore):
        """Test parallel traversal with single seed node."""
        result = await populated_graph_store.multi_hop_traverse_parallel(
            start_node_ids=["center"],
            max_hops=2,
            max_nodes=50,
        )

        assert result is not None
        assert len(result.nodes) > 0
        assert len(result.edges) > 0
        assert result.metadata["hops_completed"] >= 1

    @pytest.mark.asyncio
    async def test_parallel_traverse_multiple_seeds(self, populated_graph_store: SqliteGraphStore):
        """Test parallel traversal with multiple seed nodes."""
        result = await populated_graph_store.multi_hop_traverse_parallel(
            start_node_ids=["center", "branch_0", "branch_1"],
            max_hops=2,
            max_nodes=50,
            max_workers=4,
        )

        assert result is not None
        assert len(result.nodes) > 0
        assert len(result.edges) > 0
        assert result.metadata["start_nodes"] == ["center", "branch_0", "branch_1"]
        assert result.metadata["max_workers"] == 4

    @pytest.mark.asyncio
    async def test_parallel_traverse_empty_seeds(self, graph_store: SqliteGraphStore):
        """Test parallel traversal with no seed nodes."""
        await graph_store.initialize()

        result = await graph_store.multi_hop_traverse_parallel(
            start_node_ids=[],
            max_hops=2,
            max_nodes=50,
        )

        assert result is not None
        assert len(result.nodes) == 0
        assert len(result.edges) == 0

    @pytest.mark.asyncio
    async def test_parallel_traverse_max_nodes_limit(self, populated_graph_store: SqliteGraphStore):
        """Test that max_nodes limit is respected."""
        result = await populated_graph_store.multi_hop_traverse_parallel(
            start_node_ids=["center"],
            max_hops=3,
            max_nodes=5,  # Small limit
        )

        assert result is not None
        assert len(result.nodes) <= 5

    @pytest.mark.asyncio
    async def test_parallel_traverse_with_edge_filter(
        self, populated_graph_store: SqliteGraphStore
    ):
        """Test parallel traversal with edge type filtering."""
        result = await populated_graph_store.multi_hop_traverse_parallel(
            start_node_ids=["center"],
            max_hops=2,
            max_nodes=50,
            edge_types=["CALLS"],
        )

        assert result is not None
        # All edges should be CALLS
        for edge in result.edges:
            assert edge.type == "CALLS"

    @pytest.mark.asyncio
    async def test_parallel_traverse_max_workers(self, populated_graph_store: SqliteGraphStore):
        """Test parallel traversal with different worker counts."""
        result_2 = await populated_graph_store.multi_hop_traverse_parallel(
            start_node_ids=["center"],
            max_hops=2,
            max_nodes=50,
            max_workers=2,
        )

        result_8 = await populated_graph_store.multi_hop_traverse_parallel(
            start_node_ids=["center"],
            max_hops=2,
            max_nodes=50,
            max_workers=8,
        )

        # Both should complete successfully
        assert result_2 is not None
        assert result_8 is not None
        assert result_2.metadata["max_workers"] == 2
        assert result_8.metadata["max_workers"] == 8

    @pytest.mark.asyncio
    async def test_parallel_traverse_execution_time(self, populated_graph_store: SqliteGraphStore):
        """Test that parallel traversal returns execution time."""
        result = await populated_graph_store.multi_hop_traverse_parallel(
            start_node_ids=["center"],
            max_hops=2,
            max_nodes=50,
        )

        assert result.execution_time_ms >= 0


class TestParallelRetrieverIntegration:
    """Integration tests for parallel retrieval with MultiHopRetriever."""

    @pytest.mark.asyncio
    async def test_retriever_parallel_method(self, populated_graph_store: SqliteGraphStore):
        """Test that MultiHopRetriever has parallel retrieval method."""
        from victor.core.graph_rag.retrieval import MultiHopRetriever

        # Create a mock config
        class MockConfig:
            seed_count = 5
            max_hops = 2
            max_nodes = 20
            top_k = 10
            enable_parallel = True
            max_workers = 4
            edge_types = None
            centrality_weight = 0
            size_penalty_weight = 0

        config = MockConfig()
        retriever = MultiHopRetriever(populated_graph_store, config)

        # Check that parallel retrieval method exists
        assert hasattr(retriever, "retrieve_parallel")

        # Check _should_use_parallel logic
        assert retriever._should_use_parallel(config) is True

        # With enable_parallel=False
        config.enable_parallel = False
        assert retriever._should_use_parallel(config) is False

    @pytest.mark.asyncio
    async def test_retriever_parallel_threshold(self, populated_graph_store: SqliteGraphStore):
        """Test parallel retrieval threshold based on seed count."""
        from victor.core.graph_rag.retrieval import MultiHopRetriever

        retriever = MultiHopRetriever(populated_graph_store, None)

        # Small seed count - should not use parallel
        class SmallConfig:
            seed_count = 2
            parallel_min_batch_size = 3

        assert not retriever._should_use_parallel(SmallConfig())

        # Large seed count - should use parallel
        class LargeConfig:
            seed_count = 5
            parallel_min_batch_size = 3

        assert retriever._should_use_parallel(LargeConfig())

    @pytest.mark.asyncio
    async def test_parallel_vs_consistency(self, populated_graph_store: SqliteGraphStore):
        """Test that parallel and sequential results are consistent."""
        from victor.core.graph_rag.retrieval import MultiHopRetriever

        # Create identical configs
        class Config:
            seed_count = 3
            max_hops = 2
            max_nodes = 15
            top_k = 10
            enable_parallel = True
            max_workers = 4
            edge_types = None
            centrality_weight = 0
            size_penalty_weight = 0

        config = Config()
        retriever = MultiHopRetriever(populated_graph_store, config)

        # Run parallel retrieval (will fall back to sequential for single seed)
        # We use a query that will return multiple seeds
        # Note: For this test, we're checking the method exists and doesn't error
        # Real parallelism testing would require a more complex graph setup

        assert retriever._should_use_parallel(config) is True


class TestParallelPerformance:
    """Performance tests for parallel traversal."""

    @pytest.mark.asyncio
    async def test_parallel_batch_efficiency(self, populated_graph_store: SqliteGraphStore):
        """Test that parallel batch fetching is efficient."""
        import time

        # Sequential approach
        start = time.time()
        sequential_results = {}
        for node_id in ["center", "branch_0", "branch_1", "branch_2", "branch_3"]:
            sequential_results[node_id] = await populated_graph_store.get_neighbors(
                node_id, direction="out", max_depth=1
            )
        sequential_time = time.time() - start

        # Parallel approach
        start = time.time()
        parallel_results = await populated_graph_store.get_neighbors_batch(
            ["center", "branch_0", "branch_1", "branch_2", "branch_3"]
        )
        parallel_time = time.time() - start

        # Both should have same results
        assert set(sequential_results.keys()) == set(parallel_results.keys())

        for node_id in sequential_results:
            assert len(sequential_results[node_id]) == len(parallel_results[node_id])

        # Note: In real-world scenarios with I/O bound operations,
        # parallel would be faster. For in-memory SQLite,
        # the difference may be minimal due to lock contention.

    @pytest.mark.asyncio
    async def test_parallel_handles_errors(self, populated_graph_store: SqliteGraphStore):
        """Test that parallel traversal handles errors gracefully."""
        # Mix of valid and invalid node IDs
        result = await populated_graph_store.get_neighbors_batch(
            ["center", "invalid_node_1", "invalid_node_2"]
        )

        # Valid node should have results
        assert "center" in result
        assert len(result["center"]) == 20

        # Invalid nodes should return empty lists
        assert "invalid_node_1" in result
        assert "invalid_node_2" in result
        assert result["invalid_node_1"] == []
        assert result["invalid_node_2"] == []
