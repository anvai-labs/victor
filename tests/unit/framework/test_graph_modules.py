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

"""Tests for codebase graph modules - achieving 70%+ coverage."""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

from victor.storage.graph.protocol import GraphNode, GraphEdge
from victor.storage.graph.registry import create_graph_store
from victor.storage.graph.memory_store import MemoryGraphStore


class TestGraphNode:
    """Tests for GraphNode dataclass."""

    def test_basic_creation(self):
        """Test basic GraphNode creation."""
        node = GraphNode(
            node_id="test_id",
            type="function",
            name="test_func",
            file="test.py",
        )
        assert node.node_id == "test_id"
        assert node.type == "function"
        assert node.name == "test_func"
        assert node.file == "test.py"

    def test_optional_fields_default(self):
        """Test optional fields default to None."""
        node = GraphNode(
            node_id="id",
            type="class",
            name="MyClass",
            file="module.py",
        )
        assert node.line is None
        assert node.end_line is None
        assert node.lang is None
        assert node.signature is None
        assert node.docstring is None
        assert node.parent_id is None
        assert node.embedding_ref is None

    def test_metadata_default(self):
        """Test metadata defaults to empty dict."""
        node = GraphNode(
            node_id="id",
            type="class",
            name="MyClass",
            file="module.py",
        )
        assert node.metadata == {}

    def test_full_node_creation(self):
        """Test node creation with all fields."""
        node = GraphNode(
            node_id="full_id",
            type="method",
            name="my_method",
            file="module.py",
            line=10,
            end_line=25,
            lang="python",
            signature="def my_method(self, arg1: str) -> bool",
            docstring="A test method.",
            parent_id="parent_class_id",
            embedding_ref="emb_123",
            metadata={"complexity": 5, "is_async": True},
        )
        assert node.line == 10
        assert node.end_line == 25
        assert node.lang == "python"
        assert node.signature == "def my_method(self, arg1: str) -> bool"
        assert node.docstring == "A test method."
        assert node.parent_id == "parent_class_id"
        assert node.embedding_ref == "emb_123"
        assert node.metadata["complexity"] == 5


class TestGraphEdge:
    """Tests for GraphEdge dataclass."""

    def test_basic_creation(self):
        """Test basic GraphEdge creation."""
        edge = GraphEdge(
            src="node1",
            dst="node2",
            type="CALLS",
        )
        assert edge.src == "node1"
        assert edge.dst == "node2"
        assert edge.type == "CALLS"

    def test_optional_fields(self):
        """Test optional fields default values."""
        edge = GraphEdge(
            src="a",
            dst="b",
            type="REFERENCES",
        )
        assert edge.weight is None
        assert edge.metadata == {}

    def test_full_edge_creation(self):
        """Test edge creation with all fields."""
        edge = GraphEdge(
            src="caller",
            dst="callee",
            type="CALLS",
            weight=0.8,
            metadata={"line": 42, "count": 3},
        )
        assert edge.weight == 0.8
        assert edge.metadata["line"] == 42

    def test_edge_types(self):
        """Test different edge types."""
        edge_types = [
            "CALLS",
            "REFERENCES",
            "CONTAINS",
            "INHERITS",
            "IMPLEMENTS",
            "COMPOSED_OF",
            "IMPORTS",
        ]
        for edge_type in edge_types:
            edge = GraphEdge(src="a", dst="b", type=edge_type)
            assert edge.type == edge_type


class TestCreateGraphStore:
    """Tests for graph store factory."""

    def test_create_sqlite_store(self):
        """Test creating SQLite store."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.db"
            store = create_graph_store("sqlite", path)
            assert store is not None

    def test_create_sqlite_default(self):
        """Test default store is SQLite."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.db"
            store = create_graph_store("", path)
            assert store is not None

    def test_create_memory_store(self):
        """Test creating memory store."""
        store = create_graph_store("memory", Path("."))
        assert isinstance(store, MemoryGraphStore)

    def test_create_sqlite_case_insensitive(self):
        """Test SQLite name is case insensitive."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.db"
            store = create_graph_store("SQLITE", path)
            assert store is not None

    def test_invalid_backend_raises(self):
        """Test invalid backend raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported graph store backend"):
            create_graph_store("invalid_backend", Path("."))

    @patch("victor.storage.graph.registry.DuckDBGraphStore", None)
    def test_duckdb_not_installed(self):
        """Test DuckDB raises when not installed."""
        with pytest.raises(ValueError, match="DuckDB.*not installed"):
            create_graph_store("duckdb", Path("."))


class TestMemoryGraphStore:
    """Tests for MemoryGraphStore."""

    @pytest.fixture
    def store(self):
        """Create a MemoryGraphStore instance."""
        return MemoryGraphStore()

    @pytest.fixture
    def sample_nodes(self):
        """Create sample nodes for testing."""
        return [
            GraphNode(node_id="n1", type="function", name="func1", file="a.py", line=10),
            GraphNode(node_id="n2", type="function", name="func2", file="a.py", line=20),
            GraphNode(node_id="n3", type="class", name="MyClass", file="b.py", line=5),
        ]

    @pytest.fixture
    def sample_edges(self):
        """Create sample edges for testing."""
        return [
            GraphEdge(src="n1", dst="n2", type="CALLS", weight=1.0),
            GraphEdge(src="n2", dst="n3", type="REFERENCES", weight=0.5),
        ]

    @pytest.mark.asyncio
    async def test_upsert_nodes(self, store, sample_nodes):
        """Test upserting nodes."""
        await store.upsert_nodes(sample_nodes)
        # Verify nodes were added using find_nodes
        nodes = await store.find_nodes(name="func1")
        assert len(nodes) >= 1
        assert nodes[0].name == "func1"

    @pytest.mark.asyncio
    async def test_upsert_edges(self, store, sample_nodes, sample_edges):
        """Test upserting edges."""
        await store.upsert_nodes(sample_nodes)
        await store.upsert_edges(sample_edges)
        # Verify edges were added using get_neighbors
        edges = await store.get_neighbors("n1")
        assert len(edges) >= 1

    @pytest.mark.asyncio
    async def test_find_nodes_by_name(self, store, sample_nodes):
        """Test finding nodes by name."""
        await store.upsert_nodes(sample_nodes)
        nodes = await store.find_nodes(name="func1")
        assert len(nodes) >= 1
        assert any(n.name == "func1" for n in nodes)

    @pytest.mark.asyncio
    async def test_find_nodes_by_type(self, store, sample_nodes):
        """Test finding nodes by type."""
        await store.upsert_nodes(sample_nodes)
        nodes = await store.find_nodes(type="class")
        assert len(nodes) >= 1
        assert all(n.type == "class" for n in nodes)

    @pytest.mark.asyncio
    async def test_find_nodes_by_file(self, store, sample_nodes):
        """Test finding nodes by file."""
        await store.upsert_nodes(sample_nodes)
        nodes = await store.find_nodes(file="a.py")
        assert len(nodes) >= 2

    @pytest.mark.asyncio
    async def test_find_node_by_name(self, store, sample_nodes):
        """Test finding node by name (alternative to get_node_by_id)."""
        await store.upsert_nodes(sample_nodes)
        nodes = await store.find_nodes(name="func2")
        assert len(nodes) >= 1
        assert nodes[0].name == "func2"

    @pytest.mark.asyncio
    async def test_find_node_not_found(self, store):
        """Test finding non-existent node."""
        nodes = await store.find_nodes(name="nonexistent")
        assert len(nodes) == 0

    @pytest.mark.asyncio
    async def test_find_nodes_by_file_filter(self, store, sample_nodes):
        """Test finding all nodes in a file using find_nodes."""
        await store.upsert_nodes(sample_nodes)
        nodes = await store.find_nodes(file="a.py")
        assert len(nodes) == 2

    @pytest.mark.asyncio
    async def test_get_neighbors(self, store, sample_nodes, sample_edges):
        """Test getting neighbors."""
        await store.upsert_nodes(sample_nodes)
        await store.upsert_edges(sample_edges)
        edges = await store.get_neighbors("n1")
        assert len(edges) >= 1

    @pytest.mark.asyncio
    async def test_get_neighbors_returns_incoming_and_outgoing_edges(self, store, sample_nodes):
        """Test default neighbor traversal includes both edge directions."""
        await store.upsert_nodes(sample_nodes)
        await store.upsert_edges(
            [
                GraphEdge(src="n1", dst="n2", type="CALLS", weight=1.0),
                GraphEdge(src="n2", dst="n3", type="REFERENCES", weight=0.5),
            ]
        )
        edges = await store.get_neighbors("n2")
        assert {(edge.src, edge.dst, edge.type) for edge in edges} == {
            ("n1", "n2", "CALLS"),
            ("n2", "n3", "REFERENCES"),
        }

    @pytest.mark.asyncio
    async def test_get_neighbors_respects_direction_and_depth(self, store, sample_nodes):
        """Test directional traversal can walk multiple hops."""
        await store.upsert_nodes(sample_nodes)
        await store.upsert_edges(
            [
                GraphEdge(src="n1", dst="n2", type="CALLS", weight=1.0),
                GraphEdge(src="n2", dst="n3", type="CALLS", weight=1.0),
            ]
        )
        outgoing = await store.get_neighbors("n1", direction="out", max_depth=2)
        incoming = await store.get_neighbors("n3", direction="in", max_depth=2)

        assert {(edge.src, edge.dst) for edge in outgoing} == {
            ("n1", "n2"),
            ("n2", "n3"),
        }
        assert {(edge.src, edge.dst) for edge in incoming} == {
            ("n1", "n2"),
            ("n2", "n3"),
        }

    @pytest.mark.asyncio
    async def test_get_neighbors_with_edge_types(self, store, sample_nodes, sample_edges):
        """Test getting neighbors filtered by edge type."""
        await store.upsert_nodes(sample_nodes)
        await store.upsert_edges(sample_edges)
        edges = await store.get_neighbors("n2", edge_types=["REFERENCES"])
        assert all(e.type == "REFERENCES" for e in edges)

    @pytest.mark.asyncio
    async def test_delete_by_repo(self, store, sample_nodes):
        """Test clearing all data."""
        await store.upsert_nodes(sample_nodes)
        await store.delete_by_repo()
        stats = await store.stats()
        # MemoryGraphStore returns "nodes" key
        assert stats.get("nodes", 0) == 0

    @pytest.mark.asyncio
    async def test_stats(self, store, sample_nodes, sample_edges):
        """Test getting stats."""
        await store.upsert_nodes(sample_nodes)
        await store.upsert_edges(sample_edges)
        stats = await store.stats()
        # MemoryGraphStore returns dict with "nodes", "edges", "path" keys
        assert isinstance(stats, dict)
        assert "nodes" in stats
        assert "edges" in stats


class TestGraphAnalyzer:
    """Tests for GraphAnalyzer class from graph_tool."""

    @pytest.fixture
    def analyzer(self):
        """Create a GraphAnalyzer instance."""
        graph_tool = pytest.importorskip("victor.tools.graph_tool")
        GraphAnalyzer = graph_tool.GraphAnalyzer

        return GraphAnalyzer()

    @pytest.fixture
    def sample_nodes(self):
        """Create sample nodes."""
        return [
            GraphNode(node_id="a", type="function", name="func_a", file="test.py"),
            GraphNode(node_id="b", type="function", name="func_b", file="test.py"),
            GraphNode(node_id="c", type="function", name="func_c", file="test.py"),
            GraphNode(node_id="d", type="class", name="MyClass", file="test.py"),
        ]

    @pytest.fixture
    def sample_edges(self):
        """Create sample edges."""
        return [
            GraphEdge(src="a", dst="b", type="CALLS", weight=1.0),
            GraphEdge(src="b", dst="c", type="CALLS", weight=1.0),
            GraphEdge(src="a", dst="c", type="REFERENCES", weight=0.5),
            GraphEdge(src="d", dst="a", type="CONTAINS", weight=1.0),
        ]

    def test_add_node(self, analyzer):
        """Test adding nodes."""
        node = GraphNode(node_id="test", type="function", name="test_func", file="test.py")
        analyzer.add_node(node)
        assert "test" in analyzer.nodes

    def test_add_edge(self, analyzer, sample_nodes):
        """Test adding edges."""
        for node in sample_nodes:
            analyzer.add_node(node)
        edge = GraphEdge(src="a", dst="b", type="CALLS", weight=1.0)
        analyzer.add_edge(edge)
        assert len(analyzer.outgoing["a"]) == 1
        assert len(analyzer.incoming["b"]) == 1

    def test_get_neighbors_outgoing(self, analyzer, sample_nodes, sample_edges):
        """Test getting outgoing neighbors."""
        for node in sample_nodes:
            analyzer.add_node(node)
        for edge in sample_edges:
            analyzer.add_edge(edge)

        result = analyzer.get_neighbors("a", direction="out", max_depth=1)
        assert result["source"] == "a"
        assert result["total_neighbors"] >= 1

    def test_get_neighbors_incoming(self, analyzer, sample_nodes, sample_edges):
        """Test getting incoming neighbors."""
        for node in sample_nodes:
            analyzer.add_node(node)
        for edge in sample_edges:
            analyzer.add_edge(edge)

        result = analyzer.get_neighbors("b", direction="in", max_depth=1)
        assert result["source"] == "b"
        # b has incoming edge from a
        assert result["total_neighbors"] >= 1

    def test_get_neighbors_both(self, analyzer, sample_nodes, sample_edges):
        """Test getting both incoming and outgoing neighbors."""
        for node in sample_nodes:
            analyzer.add_node(node)
        for edge in sample_edges:
            analyzer.add_edge(edge)

        result = analyzer.get_neighbors("b", direction="both", max_depth=1)
        assert result["source"] == "b"
        # b has incoming from a and outgoing to c
        assert result["total_neighbors"] >= 2

    def test_get_neighbors_with_edge_filter(self, analyzer, sample_nodes, sample_edges):
        """Test filtering neighbors by edge type."""
        for node in sample_nodes:
            analyzer.add_node(node)
        for edge in sample_edges:
            analyzer.add_edge(edge)

        result = analyzer.get_neighbors("a", direction="out", edge_types=["CALLS"], max_depth=1)
        for depth_neighbors in result["neighbors_by_depth"].values():
            for neighbor in depth_neighbors:
                assert neighbor["edge_type"] == "CALLS"

    def test_get_neighbors_max_depth(self, analyzer, sample_nodes, sample_edges):
        """Test neighbor traversal depth."""
        for node in sample_nodes:
            analyzer.add_node(node)
        for edge in sample_edges:
            analyzer.add_edge(edge)

        # With max_depth=2, should reach c from a via b
        result = analyzer.get_neighbors("a", direction="out", max_depth=2)
        assert 1 in result["neighbors_by_depth"] or 2 in result["neighbors_by_depth"]

    def test_pagerank_empty_graph(self, analyzer):
        """Test PageRank on empty graph."""
        result = analyzer.pagerank()
        assert result == []

    def test_pagerank_basic(self, analyzer, sample_nodes, sample_edges):
        """Test basic PageRank."""
        for node in sample_nodes:
            analyzer.add_node(node)
        for edge in sample_edges:
            analyzer.add_edge(edge)

        result = analyzer.pagerank(iterations=20, top_k=10)
        assert len(result) > 0
        assert "rank" in result[0]
        assert "node_id" in result[0]
        assert "score" in result[0]

    def test_pagerank_with_edge_filter(self, analyzer, sample_nodes, sample_edges):
        """Test PageRank with edge type filter."""
        for node in sample_nodes:
            analyzer.add_node(node)
        for edge in sample_edges:
            analyzer.add_edge(edge)

        result = analyzer.pagerank(edge_types=["CALLS"], iterations=10, top_k=10)
        assert isinstance(result, list)

    def test_pagerank_top_k(self, analyzer, sample_nodes, sample_edges):
        """Test PageRank returns top_k results."""
        for node in sample_nodes:
            analyzer.add_node(node)
        for edge in sample_edges:
            analyzer.add_edge(edge)

        result = analyzer.pagerank(top_k=2)
        assert len(result) <= 2


class TestGraphAnalyzerCentrality:
    """Tests for GraphAnalyzer centrality methods."""

    @pytest.fixture
    def analyzer_with_data(self):
        """Create analyzer with test data."""
        graph_tool = pytest.importorskip("victor.tools.graph_tool")
        GraphAnalyzer = graph_tool.GraphAnalyzer

        analyzer = GraphAnalyzer()

        nodes = [
            GraphNode(node_id="hub", type="function", name="hub_func", file="test.py"),
            GraphNode(node_id="spoke1", type="function", name="spoke1", file="test.py"),
            GraphNode(node_id="spoke2", type="function", name="spoke2", file="test.py"),
            GraphNode(node_id="spoke3", type="function", name="spoke3", file="test.py"),
        ]
        for node in nodes:
            analyzer.add_node(node)

        # Hub connected to all spokes
        edges = [
            GraphEdge(src="hub", dst="spoke1", type="CALLS"),
            GraphEdge(src="hub", dst="spoke2", type="CALLS"),
            GraphEdge(src="hub", dst="spoke3", type="CALLS"),
            GraphEdge(src="spoke1", dst="hub", type="CALLS"),
        ]
        for edge in edges:
            analyzer.add_edge(edge)

        return analyzer

    def test_degree_centrality_calculation(self, analyzer_with_data):
        """Test degree centrality calculation."""
        result = analyzer_with_data.degree_centrality(top_k=4)
        assert len(result) > 0
        # Hub should have high centrality
        hub_entry = next((r for r in result if r["node_id"] == "hub"), None)
        assert hub_entry is not None
        assert "degree" in hub_entry


class TestGraphAnalyzerPath:
    """Tests for GraphAnalyzer shortest path methods."""

    @pytest.fixture
    def analyzer_with_path(self):
        """Create analyzer with path test data."""
        graph_tool = pytest.importorskip("victor.tools.graph_tool")
        GraphAnalyzer = graph_tool.GraphAnalyzer

        analyzer = GraphAnalyzer()

        nodes = [
            GraphNode(node_id="start", type="function", name="start", file="test.py"),
            GraphNode(node_id="mid", type="function", name="mid", file="test.py"),
            GraphNode(node_id="end", type="function", name="end", file="test.py"),
            GraphNode(node_id="isolated", type="function", name="isolated", file="test.py"),
        ]
        for node in nodes:
            analyzer.add_node(node)

        edges = [
            GraphEdge(src="start", dst="mid", type="CALLS"),
            GraphEdge(src="mid", dst="end", type="CALLS"),
        ]
        for edge in edges:
            analyzer.add_edge(edge)

        return analyzer

    def test_shortest_path_exists(self, analyzer_with_path):
        """Test finding shortest path."""
        result = analyzer_with_path.shortest_path("start", "end")
        assert result is not None
        assert result.get("found") is True
        assert "path" in result
        assert len(result["path"]) >= 2

    def test_shortest_path_no_path(self, analyzer_with_path):
        """Test when no path exists."""
        result = analyzer_with_path.shortest_path("start", "isolated")
        # Returns {"found": False, "source": ..., "target": ..., "message": ...}
        assert result.get("found") is False
        assert "message" in result


class TestGraphStoreProtocol:
    """Tests for GraphStoreProtocol compliance."""

    def test_protocol_methods_exist(self):
        """Test that protocol defines all required methods."""
        from victor.storage.graph.protocol import GraphStoreProtocol
        import inspect

        # Get all abstract methods from protocol
        methods = [
            name for name, _ in inspect.getmembers(GraphStoreProtocol, predicate=inspect.isfunction)
        ]

        expected_methods = [
            "initialize",
            "close",
            "upsert_nodes",
            "upsert_edges",
            "get_neighbors",
            "find_nodes",
            "search_symbols",
            "get_node_by_id",
            "get_all_nodes",
            "get_nodes_by_file",
            "update_file_mtime",
            "get_stale_files",
            "delete_by_file",
            "delete_by_repo",
            "stats",
            "get_all_edges",
        ]

        for method in expected_methods:
            assert method in methods or hasattr(GraphStoreProtocol, method)


class TestGraphModuleExports:
    def test_graph_package_exports_traversal_direction_type(self):
        from victor.storage.graph import GraphTraversalDirection

        assert GraphTraversalDirection is not None


class TestGraphEdgeTypes:
    """Tests for graph edge type constants."""

    def test_all_edge_types_constant(self):
        """Test ALL_EDGE_TYPES structural invariants.

        The EdgeType enum grows as new relationship types are added, so we
        assert invariants rather than a hardcoded list:
          - always a sorted list (so membership lookups are deterministic)
          - no duplicates
          - always contains the core relationship types
        """
        graph_tool = pytest.importorskip(
            "victor.tools.graph_tool",
            reason="victor.tools.graph_tool module not available",
        )
        ALL_EDGE_TYPES = graph_tool.ALL_EDGE_TYPES

        # Structural invariants
        assert isinstance(ALL_EDGE_TYPES, list)
        assert ALL_EDGE_TYPES == sorted(ALL_EDGE_TYPES), "ALL_EDGE_TYPES must be sorted"
        assert len(ALL_EDGE_TYPES) == len(
            set(ALL_EDGE_TYPES)
        ), "ALL_EDGE_TYPES must have no duplicates"

        # Core types that must always be present
        core_types = {
            "CALLS",
            "REFERENCES",
            "CONTAINS",
            "INHERITS",
            "IMPLEMENTS",
            "COMPOSED_OF",
            "IMPORTS",
        }
        assert core_types.issubset(
            set(ALL_EDGE_TYPES)
        ), f"Missing core types: {core_types - set(ALL_EDGE_TYPES)}"


class TestGraphModes:
    """Tests for graph mode types."""

    def test_graph_mode_types(self):
        """Test GraphMode literal types."""
        graph_tool = pytest.importorskip(
            "victor.tools.graph_tool",
            reason="victor.tools.graph_tool module not available",
        )
        GraphMode = graph_tool.GraphMode

        # These should all be valid modes
        modes = [
            "find",
            "neighbors",
            "pagerank",
            "centrality",
            "path",
            "impact",
            "clusters",
            "stats",
            "subgraph",
            "file_deps",
            "patterns",
            "module_pagerank",
            "module_centrality",
            "call_flow",
        ]
        # Just verify we can reference GraphMode
        assert GraphMode is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
