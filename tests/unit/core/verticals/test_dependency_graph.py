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

"""Unit tests for ExtensionDependencyGraph module."""

from __future__ import annotations

import pytest

from victor.core.verticals.dependency_graph import (
    DependencyCycleError,
    DependencyNode,
    LoadOrder,
    ExtensionDependencyGraph,
)
from victor_contracts.verticals.manifest import ExtensionManifest, ExtensionDependency


class TestDependencyNode:
    """Test suite for DependencyNode dataclass."""

    def test_create_node_minimal(self):
        """Test creating a node with minimal fields."""
        node = DependencyNode(vertical_name="test", version="1.0.0")

        assert node.vertical_name == "test"
        assert node.version == "1.0.0"
        assert node.manifest is None
        assert node.dependencies == set()
        assert node.dependents == set()
        assert node.load_priority == 0
        assert node.loaded is False

    def test_create_node_full(self):
        """Test creating a node with all fields."""
        manifest = ExtensionManifest(name="test", version="1.0.0")
        node = DependencyNode(
            vertical_name="test",
            version="1.0.0",
            manifest=manifest,
            dependencies={"dep1", "dep2"},
            dependents={"dependent1"},
            load_priority=100,
            loaded=True,
        )

        assert node.vertical_name == "test"
        assert node.version == "1.0.0"
        assert node.manifest == manifest
        assert node.dependencies == {"dep1", "dep2"}
        assert node.dependents == {"dependent1"}
        assert node.load_priority == 100
        assert node.loaded is True

    def test_node_equality(self):
        """Test node equality based on vertical_name."""
        node1 = DependencyNode(vertical_name="test", version="1.0.0")
        node2 = DependencyNode(vertical_name="test", version="2.0.0")
        node3 = DependencyNode(vertical_name="other", version="1.0.0")

        assert node1 == node2
        assert node1 != node3

    def test_node_hash(self):
        """Test node hashing based on vertical_name."""
        node1 = DependencyNode(vertical_name="test", version="1.0.0")
        node2 = DependencyNode(vertical_name="test", version="2.0.0")

        assert hash(node1) == hash(node2)


class TestLoadOrder:
    """Test suite for LoadOrder dataclass."""

    def test_create_load_order(self):
        """Test creating a load order."""
        order = LoadOrder(order=["dep1", "dep2", "main"])

        assert order.order == ["dep1", "dep2", "main"]
        assert order.missing_dependencies == set()
        assert order.missing_optional == set()
        assert order.cycles == []

    def test_can_load_property(self):
        """Test can_load property."""
        # No missing deps or cycles
        order1 = LoadOrder(order=["dep1", "main"])
        assert order1.can_load is True

        # Has missing dependencies
        order2 = LoadOrder(order=["main"], missing_dependencies={"dep1"})
        assert order2.can_load is False

        # Has cycles
        order3 = LoadOrder(order=[], cycles=[["a", "b", "a"]])
        assert order3.can_load is False

    def test_has_cycles_property(self):
        """Test has_cycles property."""
        order1 = LoadOrder(order=["dep1", "main"])
        assert order1.has_cycles is False

        order2 = LoadOrder(order=[], cycles=[["a", "b", "a"]])
        assert order2.has_cycles is True


class TestExtensionDependencyGraph:
    """Test suite for ExtensionDependencyGraph class."""

    def setup_method(self):
        """Create fresh graph for each test."""
        self.graph = ExtensionDependencyGraph()

    def test_add_vertical(self):
        """Test adding a vertical to the graph."""
        self.graph.add_vertical("test", "1.0.0")

        assert self.graph.has_vertical("test")
        assert "test" in self.graph.list_verticals()

    def test_add_vertical_with_priority(self):
        """Test adding a vertical with load priority."""
        self.graph.add_vertical("test", "1.0.0", load_priority=100)

        assert self.graph.has_vertical("test")
        # Priority is stored but affects order resolution
        assert self.graph._nodes["test"].load_priority == 100

    def test_remove_vertical(self):
        """Test removing a vertical from the graph."""
        self.graph.add_vertical("test", "1.0.0")
        self.graph.add_vertical("dep", "1.0.0")
        self.graph.add_dependency("test", "dep")

        assert self.graph.has_vertical("test")
        assert self.graph.has_vertical("dep")

        self.graph.remove_vertical("test")

        assert not self.graph.has_vertical("test")
        assert self.graph.has_vertical("dep")
        # Dependency should be removed from dep's dependents
        assert "test" not in self.graph.get_dependents("dep")

    def test_add_dependency(self):
        """Test adding a dependency relationship."""
        self.graph.add_vertical("main", "1.0.0")
        self.graph.add_vertical("dep", "1.0.0")

        self.graph.add_dependency("main", "dep")

        assert "dep" in self.graph.get_dependencies("main")
        assert "main" in self.graph.get_dependents("dep")

    def test_add_dependency_vertical_not_found(self):
        """Test adding dependency for non-existent vertical."""
        self.graph.add_vertical("main", "1.0.0")

        with pytest.raises(ValueError, match="not in graph"):
            self.graph.add_dependency("main", "nonexistent")

    def test_add_optional_dependency_missing(self):
        """Test optional dependency with missing target."""
        self.graph.add_vertical("main", "1.0.0")

        # Should not raise for optional dependency
        self.graph.add_dependency("main", "missing", required=False)

        # Missing dependency not in graph
        assert not self.graph.has_vertical("missing")

    def test_get_dependencies(self):
        """Test getting dependencies for a vertical."""
        self.graph.add_vertical("main", "1.0.0")
        self.graph.add_vertical("dep1", "1.0.0")
        self.graph.add_vertical("dep2", "1.0.0")

        self.graph.add_dependency("main", "dep1")
        self.graph.add_dependency("main", "dep2")

        deps = self.graph.get_dependencies("main")
        assert deps == {"dep1", "dep2"}

    def test_get_dependencies_not_found(self):
        """Test getting dependencies for non-existent vertical."""
        with pytest.raises(ValueError, match="not in graph"):
            self.graph.get_dependencies("nonexistent")

    def test_get_dependents(self):
        """Test getting dependents of a vertical."""
        self.graph.add_vertical("dep", "1.0.0")
        self.graph.add_vertical("main1", "1.0.0")
        self.graph.add_vertical("main2", "1.0.0")

        self.graph.add_dependency("main1", "dep")
        self.graph.add_dependency("main2", "dep")

        dependents = self.graph.get_dependents("dep")
        assert dependents == {"main1", "main2"}

    def test_resolve_load_order_simple(self):
        """Test resolving load order with simple chain."""
        self.graph.add_vertical("dep", "1.0.0")
        self.graph.add_vertical("main", "1.0.0")

        self.graph.add_dependency("main", "dep")

        order = self.graph.resolve_load_order()

        assert order.can_load
        assert order.order == ["dep", "main"]

    def test_resolve_load_order_complex(self):
        """Test resolving load order with complex dependencies."""
        # A depends on B and C
        # B depends on D
        # C depends on D
        # Expected order: D -> B/C -> A (B and C can be in any order relative to each other)
        self.graph.add_vertical("a", "1.0.0")
        self.graph.add_vertical("b", "1.0.0")
        self.graph.add_vertical("c", "1.0.0")
        self.graph.add_vertical("d", "1.0.0")

        self.graph.add_dependency("a", "b")
        self.graph.add_dependency("a", "c")
        self.graph.add_dependency("b", "d")
        self.graph.add_dependency("c", "d")

        order = self.graph.resolve_load_order()

        assert order.can_load
        assert order.order[0] == "d"  # D must be first
        assert order.order[-1] == "a"  # A must be last
        assert set(order.order) == {"a", "b", "c", "d"}

    def test_resolve_load_order_priority(self):
        """Test that load priority affects order."""
        self.graph.add_vertical("low", "1.0.0", load_priority=10)
        self.graph.add_vertical("high", "1.0.0", load_priority=100)
        self.graph.add_vertical("medium", "1.0.0", load_priority=50)

        # No dependencies - order by priority only
        order = self.graph.resolve_load_order()

        assert order.can_load
        assert order.order == ["high", "medium", "low"]

    def test_resolve_load_order_priority_with_dependencies(self):
        """Test that dependencies take priority over load priority."""
        # Even though high has higher priority, dep must load first
        self.graph.add_vertical("dep", "1.0.0", load_priority=10)
        self.graph.add_vertical("high", "1.0.0", load_priority=100)
        self.graph.add_vertical("low", "1.0.0", load_priority=50)

        self.graph.add_dependency("high", "dep")
        self.graph.add_dependency("low", "dep")

        order = self.graph.resolve_load_order()

        assert order.can_load
        assert order.order[0] == "dep"  # Dep must be first
        # high and low both depend on dep, high loads first due to priority
        assert order.order[1] == "high"
        assert order.order[2] == "low"

    def test_detect_cycle_simple(self):
        """Test detecting a simple circular dependency."""
        self.graph.add_vertical("a", "1.0.0")
        self.graph.add_vertical("b", "1.0.0")

        self.graph.add_dependency("a", "b")
        self.graph.add_dependency("b", "a")

        order = self.graph.resolve_load_order()

        assert not order.can_load
        assert len(order.cycles) == 1
        assert order.cycles[0] in ([["a", "b", "a"], ["b", "a", "b"]])

    def test_detect_cycle_complex(self):
        """Test detecting a complex circular dependency."""
        self.graph.add_vertical("a", "1.0.0")
        self.graph.add_vertical("b", "1.0.0")
        self.graph.add_vertical("c", "1.0.0")

        self.graph.add_dependency("a", "b")
        self.graph.add_dependency("b", "c")
        self.graph.add_dependency("c", "a")

        order = self.graph.resolve_load_order()

        assert not order.can_load
        assert len(order.cycles) >= 1

    def test_get_load_sequence(self):
        """Test getting load sequence for a specific vertical."""
        self.graph.add_vertical("dep1", "1.0.0")
        self.graph.add_vertical("dep2", "1.0.0")
        self.graph.add_vertical("main", "1.0.0")

        self.graph.add_dependency("main", "dep1")
        self.graph.add_dependency("main", "dep2")
        self.graph.add_dependency("dep1", "dep2")

        sequence = self.graph.get_load_sequence("main")

        assert sequence == ["dep2", "dep1", "main"]

    def test_get_load_sequence_not_found(self):
        """Test getting load sequence for non-existent vertical."""
        with pytest.raises(ValueError, match="not in graph"):
            self.graph.get_load_sequence("nonexistent")

    def test_get_load_sequence_with_cycle(self):
        """Test getting load sequence with circular dependency."""
        self.graph.add_vertical("a", "1.0.0")
        self.graph.add_vertical("b", "1.0.0")

        self.graph.add_dependency("a", "b")
        self.graph.add_dependency("b", "a")

        with pytest.raises(DependencyCycleError):
            self.graph.get_load_sequence("a")

    def test_build_from_manifests(self):
        """Test building graph from manifests dict."""
        manifest1 = ExtensionManifest(
            name="main",
            version="1.0.0",
            extension_dependencies=[
                ExtensionDependency(extension_name="dep1", optional=False),
                ExtensionDependency(extension_name="dep2", optional=True),
            ],
        )
        manifest2 = ExtensionManifest(name="dep1", version="1.0.0")
        manifest3 = ExtensionManifest(name="dep2", version="1.0.0")

        manifests = {
            "main": manifest1,
            "dep1": manifest2,
            "dep2": manifest3,
        }

        self.graph.build_from_manifests(manifests)

        assert self.graph.has_vertical("main")
        assert self.graph.has_vertical("dep1")
        assert self.graph.has_vertical("dep2")

        # Check dependencies
        deps = self.graph.get_dependencies("main")
        assert "dep1" in deps

    def test_clear(self):
        """Test clearing the graph."""
        self.graph.add_vertical("test", "1.0.0")
        assert self.graph.has_vertical("test")

        self.graph.clear()

        assert not self.graph.has_vertical("test")
        assert len(self.graph.list_verticals()) == 0

    def test_list_verticals(self):
        """Test listing all verticals."""
        self.graph.add_vertical("test1", "1.0.0")
        self.graph.add_vertical("test2", "1.0.0")
        self.graph.add_vertical("test3", "1.0.0")

        verticals = self.graph.list_verticals()

        assert set(verticals) == {"test1", "test2", "test3"}

    def test_has_vertical(self):
        """Test checking if vertical exists."""
        assert not self.graph.has_vertical("test")

        self.graph.add_vertical("test", "1.0.0")

        assert self.graph.has_vertical("test")

    def test_get_graph_depth_empty(self):
        """Test graph depth with no nodes."""
        assert self.graph.get_graph_depth() == 0

    def test_get_graph_depth_no_deps(self):
        """Test graph depth with no dependencies."""
        self.graph.add_vertical("test", "1.0.0")
        assert self.graph.get_graph_depth() == 0

    def test_get_graph_depth_chain(self):
        """Test graph depth with dependency chain."""
        self.graph.add_vertical("a", "1.0.0")
        self.graph.add_vertical("b", "1.0.0")
        self.graph.add_vertical("c", "1.0.0")

        self.graph.add_dependency("c", "b")
        self.graph.add_dependency("b", "a")

        # Depth should be 2: a (0) -> b (1) -> c (2)
        assert self.graph.get_graph_depth() == 2

    def test_get_graph_depth_complex(self):
        """Test graph depth with complex dependencies."""
        self.graph.add_vertical("a", "1.0.0")
        self.graph.add_vertical("b", "1.0.0")
        self.graph.add_vertical("c", "1.0.0")
        self.graph.add_vertical("d", "1.0.0")

        # a depends on b and c
        # b depends on d
        # c depends on d
        self.graph.add_dependency("a", "b")
        self.graph.add_dependency("a", "c")
        self.graph.add_dependency("b", "d")
        self.graph.add_dependency("c", "d")

        # Depth should be 2: d (0) -> b/c (1) -> a (2)
        assert self.graph.get_graph_depth() == 2


class TestDependencyCycleError:
    """Test suite for DependencyCycleError exception."""

    def test_create_error(self):
        """Test creating cycle error."""
        cycle = ["a", "b", "c", "a"]
        error = DependencyCycleError(cycle)

        assert error.cycle == cycle
        assert "Circular dependency detected" in str(error)
        assert "a -> b -> c -> a" in str(error)

    def test_create_error_with_custom_message(self):
        """Test creating cycle error with custom message."""
        cycle = ["x", "y", "x"]
        error = DependencyCycleError(cycle, "Custom message")

        assert error.cycle == cycle
        assert str(error) == "Custom message"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
