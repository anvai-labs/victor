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

"""Unit tests for graph protocol dataclasses."""

from __future__ import annotations

import pytest

from victor.storage.graph.protocol import (
    GraphNode,
    GraphEdge,
    RequirementNode,
    Subgraph,
    GraphQueryResult,
    GraphTraversalDirection,
)


class TestGraphNode:
    """Tests for GraphNode dataclass."""

    def test_create_minimal_node(self) -> None:
        """Test creating a minimal GraphNode."""
        node = GraphNode(
            node_id="test_id",
            type="function",
            name="test_func",
            file="/path/to/file.py",
        )
        assert node.node_id == "test_id"
        assert node.type == "function"
        assert node.name == "test_func"
        assert node.file == "/path/to/file.py"
        assert node.line is None
        assert node.end_line is None

    def test_create_full_node(self) -> None:
        """Test creating a GraphNode with all fields."""
        node = GraphNode(
            node_id="test_id",
            type="function",
            name="test_func",
            file="/path/to/file.py",
            line=10,
            end_line=20,
            lang="python",
            signature="def test_func():",
            docstring="Test function",
            parent_id="parent_id",
            embedding_ref="emb_ref",
            metadata={"key": "value"},
        )
        assert node.line == 10
        assert node.end_line == 20
        assert node.lang == "python"
        assert node.signature == "def test_func():"
        assert node.docstring == "Test function"
        assert node.parent_id == "parent_id"
        assert node.embedding_ref == "emb_ref"
        assert node.metadata == {"key": "value"}

    def test_create_node_with_v5_fields(self) -> None:
        """Test creating a GraphNode with v5 CCG fields."""
        node = GraphNode(
            node_id="test_id",
            type="statement",
            name="test_stmt",
            file="/path/to/file.py",
            line=15,
            ast_kind="if_statement",
            scope_id="scope_123",
            statement_type="condition",
            requirement_id="req_456",
            visibility="public",
        )
        assert node.ast_kind == "if_statement"
        assert node.scope_id == "scope_123"
        assert node.statement_type == "condition"
        assert node.requirement_id == "req_456"
        assert node.visibility == "public"

    def test_node_metadata_default_factory(self) -> None:
        """Test that metadata has a default factory."""
        node1 = GraphNode(
            node_id="id1",
            type="function",
            name="func1",
            file="file1.py",
        )
        node2 = GraphNode(
            node_id="id2",
            type="function",
            name="func2",
            file="file2.py",
        )
        # Each node should have its own metadata dict
        node1.metadata["key"] = "value1"
        node2.metadata["key"] = "value2"
        assert node1.metadata["key"] == "value1"
        assert node2.metadata["key"] == "value2"


class TestGraphEdge:
    """Tests for GraphEdge dataclass."""

    def test_create_minimal_edge(self) -> None:
        """Test creating a minimal GraphEdge."""
        edge = GraphEdge(
            src="node_a",
            dst="node_b",
            type="CALLS",
        )
        assert edge.src == "node_a"
        assert edge.dst == "node_b"
        assert edge.type == "CALLS"
        assert edge.weight is None
        assert edge.metadata == {}

    def test_create_weighted_edge(self) -> None:
        """Test creating a weighted GraphEdge."""
        edge = GraphEdge(
            src="node_a",
            dst="node_b",
            type="CALLS",
            weight=0.8,
        )
        assert edge.weight == 0.8

    def test_create_edge_with_metadata(self) -> None:
        """Test creating an edge with metadata."""
        edge = GraphEdge(
            src="node_a",
            dst="node_b",
            type="CALLS",
            metadata={"line": 42, "context": "test"},
        )
        assert edge.metadata == {"line": 42, "context": "test"}

    def test_edge_metadata_default_factory(self) -> None:
        """Test that metadata has a default factory."""
        edge1 = GraphEdge(src="a", dst="b", type="CALLS")
        edge2 = GraphEdge(src="c", dst="d", type="REFERENCES")
        edge1.metadata["key"] = "value1"
        edge2.metadata["key"] = "value2"
        assert edge1.metadata["key"] == "value1"
        assert edge2.metadata["key"] == "value2"


class TestRequirementNode:
    """Tests for RequirementNode dataclass."""

    def test_create_requirement_node(self) -> None:
        """Test creating a RequirementNode."""
        req = RequirementNode(
            requirement_id="req_123",
            type="feature",
            source="github_issue",
            title="Add user authentication",
            description="Implement OAuth2 login",
            priority=0.8,
            status="open",
        )
        assert req.requirement_id == "req_123"
        assert req.type == "feature"
        assert req.source == "github_issue"
        assert req.title == "Add user authentication"
        assert req.description == "Implement OAuth2 login"
        assert req.priority == 0.8
        assert req.status == "open"

    def test_requirement_node_defaults(self) -> None:
        """Test RequirementNode default values."""
        req = RequirementNode(
            requirement_id="req_456",
            type="bug",
            title="Fix crash",
        )
        assert req.source is None
        assert req.description is None
        assert req.priority == 0.5
        assert req.status == "open"
        assert req.metadata == {}


class TestSubgraph:
    """Tests for Subgraph dataclass."""

    def test_create_subgraph(self) -> None:
        """Test creating a Subgraph."""
        subgraph = Subgraph(
            subgraph_id="sub_123",
            anchor_node_id="node_a",
            radius=2,
            edge_types=["CALLS", "REFERENCES"],
            node_ids=["node_a", "node_b", "node_c"],
            edges=[],
            node_count=3,
        )
        assert subgraph.subgraph_id == "sub_123"
        assert subgraph.anchor_node_id == "node_a"
        assert subgraph.radius == 2
        assert subgraph.edge_types == ["CALLS", "REFERENCES"]
        assert subgraph.node_ids == ["node_a", "node_b", "node_c"]
        assert subgraph.node_count == 3

    def test_subgraph_defaults(self) -> None:
        """Test Subgraph default values."""
        subgraph = Subgraph(
            subgraph_id="sub_456",
            anchor_node_id="node_x",
            radius=1,
            edge_types=[],
            node_ids=[],
            edges=[],
        )
        assert subgraph.node_count == 0
        assert subgraph.computed_at is None


class TestGraphQueryResult:
    """Tests for GraphQueryResult dataclass."""

    def test_create_query_result(self) -> None:
        """Test creating a GraphQueryResult."""
        result = GraphQueryResult(
            nodes=[],
            edges=[],
            subgraphs=[],
            query="test query",
            execution_time_ms=150.5,
        )
        assert result.nodes == []
        assert result.edges == []
        assert result.subgraphs == []
        assert result.query == "test query"
        assert result.execution_time_ms == 150.5
        assert result.metadata == {}

    def test_query_result_with_metadata(self) -> None:
        """Test GraphQueryResult with metadata."""
        result = GraphQueryResult(
            nodes=[],
            edges=[],
            subgraphs=[],
            query="test",
            metadata={"hop_count": 2, "node_count": 10},
        )
        assert result.metadata == {"hop_count": 2, "node_count": 10}

    def test_query_result_to_dict(self) -> None:
        """Test GraphQueryResult to_dict method."""
        node = GraphNode(
            node_id="n1",
            type="function",
            name="func",
            file="file.py",
        )
        result = GraphQueryResult(
            nodes=[node],
            edges=[],
            subgraphs=[],
            query="test",
            execution_time_ms=100.0,
        )
        d = result.to_dict()
        assert d["node_count"] == 1
        assert d["edge_count"] == 0
        assert d["subgraph_count"] == 0
        assert d["query"] == "test"
        assert d["execution_time_ms"] == 100.0


class TestGraphTraversalDirection:
    """Tests for GraphTraversalDirection literal."""

    def test_valid_directions(self) -> None:
        """Test valid direction values."""
        assert "out" in GraphTraversalDirection.__args__
        assert "in" in GraphTraversalDirection.__args__
        assert "both" in GraphTraversalDirection.__args__

    def test_invalid_direction_raises(self) -> None:
        """Test that invalid direction raises error."""
        from victor.storage.graph.sqlite_store import SqliteGraphStore

        store = SqliteGraphStore()
        with pytest.raises(ValueError, match="Unsupported graph traversal direction"):
            # Use sync wrapper to test the validation
            import asyncio

            async def test_invalid():
                await store.get_neighbors("test_id", direction="invalid")

            asyncio.run(test_invalid())
