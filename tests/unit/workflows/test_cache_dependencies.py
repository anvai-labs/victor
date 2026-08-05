# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
# SPDX-License-Identifier: Apache-2.0
"""Tests for DependencyGraph and CascadingInvalidator."""

from dataclasses import dataclass
from typing import Dict, List, Optional
from unittest.mock import MagicMock

import pytest

from victor.workflows.cache import (
    DependencyGraph,
    CascadingInvalidator,
    WorkflowCache,
    WorkflowCacheConfig,
)


@dataclass
class MockNode:
    """Mock workflow node for testing."""

    id: str
    next_nodes: Optional[List[str]] = None
    branches: Optional[Dict[str, str]] = None


@dataclass
class MockWorkflow:
    """Mock workflow definition for testing."""

    nodes: Dict[str, MockNode]


class TestDependencyGraph:
    """Test DependencyGraph class."""

    def test_init_empty(self):
        """Should initialize with empty collections."""
        graph = DependencyGraph()
        assert len(graph.dependents) == 0
        assert len(graph.dependencies) == 0

    def test_add_dependency(self):
        """Should track dependencies bidirectionally."""
        graph = DependencyGraph()
        graph.add_dependency("B", "A")  # B depends on A

        assert "A" in graph.get_dependencies("B")
        assert "B" in graph.get_dependents("A")

    def test_add_multiple_dependencies(self):
        """Should handle multiple dependencies."""
        graph = DependencyGraph()
        graph.add_dependency("C", "A")
        graph.add_dependency("C", "B")

        deps = graph.get_dependencies("C")
        assert "A" in deps
        assert "B" in deps

    def test_add_multiple_dependents(self):
        """Should handle multiple dependents."""
        graph = DependencyGraph()
        graph.add_dependency("B", "A")
        graph.add_dependency("C", "A")

        dependents = graph.get_dependents("A")
        assert "B" in dependents
        assert "C" in dependents

    def test_get_dependencies_empty(self):
        """Should return empty set for unknown node."""
        graph = DependencyGraph()
        deps = graph.get_dependencies("unknown")
        assert len(deps) == 0

    def test_get_dependents_empty(self):
        """Should return empty set for unknown node."""
        graph = DependencyGraph()
        dependents = graph.get_dependents("unknown")
        assert len(dependents) == 0

    def test_get_cascade_set_direct(self):
        """Should find direct dependents."""
        graph = DependencyGraph()
        graph.add_dependency("B", "A")

        cascade = graph.get_cascade_set("A")
        assert "B" in cascade

    def test_get_cascade_set_transitive(self):
        """Should find transitive dependents."""
        graph = DependencyGraph()
        graph.add_dependency("B", "A")
        graph.add_dependency("C", "B")
        graph.add_dependency("D", "C")

        cascade = graph.get_cascade_set("A")
        assert "B" in cascade
        assert "C" in cascade
        assert "D" in cascade

    def test_get_cascade_set_branching(self):
        """Should handle branching dependencies."""
        #     A
        #    / \
        #   B   C
        #    \ /
        #     D
        graph = DependencyGraph()
        graph.add_dependency("B", "A")
        graph.add_dependency("C", "A")
        graph.add_dependency("D", "B")
        graph.add_dependency("D", "C")

        cascade = graph.get_cascade_set("A")
        assert "B" in cascade
        assert "C" in cascade
        assert "D" in cascade

    def test_get_cascade_set_empty(self):
        """Should return empty set for leaf node."""
        graph = DependencyGraph()
        graph.add_dependency("B", "A")

        cascade = graph.get_cascade_set("B")
        assert len(cascade) == 0

    def test_get_all_upstream_direct(self):
        """Should find direct dependencies."""
        graph = DependencyGraph()
        graph.add_dependency("B", "A")

        upstream = graph.get_all_upstream("B")
        assert "A" in upstream

    def test_get_all_upstream_transitive(self):
        """Should find transitive dependencies."""
        graph = DependencyGraph()
        graph.add_dependency("B", "A")
        graph.add_dependency("C", "B")

        upstream = graph.get_all_upstream("C")
        assert "B" in upstream
        assert "A" in upstream

    def test_from_workflow_next_nodes(self):
        """Should build graph from next_nodes."""
        workflow = MockWorkflow(
            nodes={
                "A": MockNode(id="A", next_nodes=["B"]),
                "B": MockNode(id="B", next_nodes=["C"]),
                "C": MockNode(id="C"),
            }
        )

        graph = DependencyGraph.from_workflow(workflow)

        # B depends on A, C depends on B
        assert "A" in graph.get_dependencies("B")
        assert "B" in graph.get_dependencies("C")

    def test_from_workflow_branches(self):
        """Should build graph from branches."""
        workflow = MockWorkflow(
            nodes={
                "condition": MockNode(
                    id="condition",
                    branches={"true": "branch_a", "false": "branch_b"},
                ),
                "branch_a": MockNode(id="branch_a"),
                "branch_b": MockNode(id="branch_b"),
            }
        )

        graph = DependencyGraph.from_workflow(workflow)

        # Both branches depend on condition
        assert "condition" in graph.get_dependencies("branch_a")
        assert "condition" in graph.get_dependencies("branch_b")

    def test_to_dict(self):
        """Should serialize to dictionary."""
        graph = DependencyGraph()
        graph.add_dependency("B", "A")

        d = graph.to_dict()
        assert "dependents" in d
        assert "dependencies" in d
        assert "B" in d["dependents"]["A"]


class TestCascadingInvalidator:
    """Test CascadingInvalidator class."""

    def test_invalidate_with_cascade_single(self):
        """Should invalidate single node."""
        cache = MagicMock(spec=WorkflowCache)
        cache.invalidate.return_value = 1

        graph = DependencyGraph()
        invalidator = CascadingInvalidator(cache, graph)

        count = invalidator.invalidate_with_cascade("A")

        cache.invalidate.assert_called_once_with("A")
        assert count == 1

    def test_invalidate_with_cascade_multiple(self):
        """Should invalidate node and all dependents."""
        cache = MagicMock(spec=WorkflowCache)
        cache.invalidate.return_value = 1

        graph = DependencyGraph()
        graph.add_dependency("B", "A")
        graph.add_dependency("C", "B")

        invalidator = CascadingInvalidator(cache, graph)
        count = invalidator.invalidate_with_cascade("A")

        # Should invalidate A, B, C
        assert cache.invalidate.call_count == 3
        assert count == 3

    def test_invalidate_with_cascade_tracks_count(self):
        """Should track total invalidated entries."""
        cache = MagicMock(spec=WorkflowCache)
        # Simulate different entry counts per node
        cache.invalidate.side_effect = [2, 3, 1]

        graph = DependencyGraph()
        graph.add_dependency("B", "A")
        graph.add_dependency("C", "A")

        invalidator = CascadingInvalidator(cache, graph)
        count = invalidator.invalidate_with_cascade("A")

        assert count == 6  # 2 + 3 + 1

    def test_invalidate_upstream(self):
        """Should invalidate all upstream nodes."""
        cache = MagicMock(spec=WorkflowCache)
        cache.invalidate.return_value = 1

        graph = DependencyGraph()
        graph.add_dependency("B", "A")
        graph.add_dependency("C", "B")

        invalidator = CascadingInvalidator(cache, graph)
        count = invalidator.invalidate_upstream("C")

        # Should invalidate A, B and their cascades
        assert cache.invalidate.call_count >= 2

    def test_invalidate_no_dependents(self):
        """Should only invalidate the node if no dependents."""
        cache = MagicMock(spec=WorkflowCache)
        cache.invalidate.return_value = 1

        graph = DependencyGraph()
        invalidator = CascadingInvalidator(cache, graph)

        count = invalidator.invalidate_with_cascade("leaf")

        cache.invalidate.assert_called_once_with("leaf")
        assert count == 1


class TestWorkflowCacheDependencyMethods:
    """Test WorkflowCache.set_dependency_graph and invalidate_cascade."""

    def test_set_dependency_graph(self):
        """Should set dependency graph and create invalidator."""
        cache = WorkflowCache(WorkflowCacheConfig(enabled=True))
        graph = DependencyGraph()
        graph.add_dependency("B", "A")

        cache.set_dependency_graph(graph)

        assert hasattr(cache, "_dependency_graph")
        assert cache._dependency_graph is graph
        assert hasattr(cache, "_invalidator")
        assert isinstance(cache._invalidator, CascadingInvalidator)

    def test_invalidate_cascade_with_graph(self):
        """Should cascade invalidation when graph is set."""
        cache = WorkflowCache(WorkflowCacheConfig(enabled=True))
        graph = DependencyGraph()
        graph.add_dependency("B", "A")
        graph.add_dependency("C", "B")

        cache.set_dependency_graph(graph)

        # invalidate_cascade should use the cascading invalidator
        count = cache.invalidate_cascade("A")
        # Even with empty cache, the method should work
        assert count >= 0

    def test_invalidate_cascade_without_graph(self):
        """Should fall back to simple invalidation without graph."""
        cache = WorkflowCache(WorkflowCacheConfig(enabled=True))

        # Without set_dependency_graph, should use simple invalidation
        count = cache.invalidate_cascade("A")
        assert count == 0  # No entries to invalidate

    def test_invalidate_cascade_disabled_cache(self):
        """Should handle disabled cache gracefully."""
        cache = WorkflowCache(WorkflowCacheConfig(enabled=False))
        graph = DependencyGraph()
        graph.add_dependency("B", "A")

        cache.set_dependency_graph(graph)
        count = cache.invalidate_cascade("A")

        assert count == 0  # Cache is disabled
