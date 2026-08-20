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

"""Performance benchmarks for dependency resolution.

These benchmarks validate that the ExtensionDependencyGraph meets
the performance target of < 10ms for dependency resolution.

Target Metrics:
    - Dependency resolution: < 10ms
    - Topological sort performance
    - Circular dependency detection speed
"""

from __future__ import annotations

import time
import pytest

from victor.core.verticals.dependency_graph import (
    ExtensionDependencyGraph,
    DependencyNode,
    DependencyCycleError,
)


class TestDependencyResolutionPerformance:
    """Performance benchmarks for dependency resolution."""

    def test_simple_chain_resolution_performance(self):
        """Test dependency resolution for simple chain is fast."""
        graph = ExtensionDependencyGraph()

        # Create a simple dependency chain: A -> B -> C -> D
        graph.add_vertical("A", version="1.0.0", load_priority=0)
        graph.add_vertical("B", version="1.0.0", load_priority=0)
        graph.add_vertical("C", version="1.0.0", load_priority=0)
        graph.add_vertical("D", version="1.0.0", load_priority=0)

        graph.add_dependency("A", "B")
        graph.add_dependency("B", "C")
        graph.add_dependency("C", "D")

        # Time the resolution
        start = time.perf_counter()
        load_order = graph.resolve_load_order()
        duration_ms = (time.perf_counter() - start) * 1000

        # Assert: resolution should be very fast (< 5ms for simple chain)
        assert duration_ms < 5, f"Simple chain resolution took {duration_ms:.2f}ms, target < 5ms"

        # Verify order is correct
        order = load_order.order
        assert order == ["D", "C", "B", "A"] or order == ["A", "B", "C", "D"]

    def test_complex_graph_resolution_performance(self):
        """Test dependency resolution for complex graph is fast."""
        graph = ExtensionDependencyGraph()

        # Create a complex dependency graph
        # Base layer (no dependencies)
        for i in range(10):
            graph.add_vertical(f"base_{i}", version="1.0.0", load_priority=0)

        # Middle layer (depends on base)
        for i in range(10):
            graph.add_vertical(f"mid_{i}", version="1.0.0", load_priority=0)
            graph.add_dependency(f"mid_{i}", f"base_{i % 10}")

        # Top layer (depends on middle)
        for i in range(10):
            graph.add_vertical(f"top_{i}", version="1.0.0", load_priority=0)
            graph.add_dependency(f"top_{i}", f"mid_{i % 10}")

        # Time the resolution
        start = time.perf_counter()
        load_order = graph.resolve_load_order()
        duration_ms = (time.perf_counter() - start) * 1000

        # Assert: resolution should be fast (< 10ms for complex graph)
        assert duration_ms < 10, f"Complex graph resolution took {duration_ms:.2f}ms, target < 10ms"

        # Verify all nodes are included
        assert len(load_order.order) == 30

    def test_topological_sort_performance(self):
        """Test topological sort performance."""
        graph = ExtensionDependencyGraph()

        # Create a graph with 50 nodes
        for i in range(50):
            graph.add_vertical(f"vertical_{i}", version="1.0.0", load_priority=i)

        # Create dependencies (each node depends on previous 3)
        for i in range(3, 50):
            graph.add_dependency(f"vertical_{i}", f"vertical_{i-1}")
            graph.add_dependency(f"vertical_{i}", f"vertical_{i-2}")
            graph.add_dependency(f"vertical_{i}", f"vertical_{i-3}")

        # Time the resolution
        start = time.perf_counter()
        load_order = graph.resolve_load_order()
        duration_ms = (time.perf_counter() - start) * 1000

        # Assert: should be fast even with 50 nodes
        assert (
            duration_ms < 15
        ), f"Topological sort with 50 nodes took {duration_ms:.2f}ms, target < 15ms"

        # Verify all nodes are included
        assert len(load_order.order) == 50

    def test_priority_sorting_performance(self):
        """Test priority-based sorting performance."""
        graph = ExtensionDependencyGraph()

        # Create nodes with different priorities
        for i in range(50):
            graph.add_vertical(f"vertical_{i}", version="1.0.0", load_priority=i % 10)

        # Time the resolution
        start = time.perf_counter()
        load_order = graph.resolve_load_order()
        duration_ms = (time.perf_counter() - start) * 1000

        # Assert: priority sorting should be fast
        assert (
            duration_ms < 5
        ), f"Priority sorting with 50 nodes took {duration_ms:.2f}ms, target < 5ms"

    def test_cycle_detection_performance(self):
        """Test circular dependency detection performance."""
        graph = ExtensionDependencyGraph()

        # Create a graph with a cycle
        graph.add_vertical("A", version="1.0.0", load_priority=0)
        graph.add_vertical("B", version="1.0.0", load_priority=0)
        graph.add_vertical("C", version="1.0.0", load_priority=0)

        graph.add_dependency("A", "B")
        graph.add_dependency("B", "C")
        graph.add_dependency("C", "A")  # Creates cycle

        # Time the cycle detection
        start = time.perf_counter()
        load_order = graph.resolve_load_order()
        duration_ms = (time.perf_counter() - start) * 1000

        # Assert: cycle detection should be fast
        assert duration_ms < 5, f"Cycle detection took {duration_ms:.2f}ms, target < 5ms"

        # Verify cycle was detected
        assert len(load_order.cycles) > 0, "Should have detected cycle"

    def test_removal_performance(self):
        """Test node removal performance."""
        graph = ExtensionDependencyGraph()

        # Add many nodes
        for i in range(50):
            graph.add_vertical(f"vertical_{i}", version="1.0.0", load_priority=0)

        # Add dependencies
        for i in range(1, 50):
            graph.add_dependency(f"vertical_{i}", f"vertical_{i-1}")

        # Time removal of multiple nodes
        start = time.perf_counter()
        for i in range(10):
            graph.remove_vertical(f"vertical_{i}")
        duration_ms = (time.perf_counter() - start) * 1000

        # Assert: removal should be fast
        assert duration_ms < 10, f"Removing 10 nodes took {duration_ms:.2f}ms, target < 10ms"

    def test_missing_dependency_check_performance(self):
        """Test missing dependency check performance."""
        graph = ExtensionDependencyGraph()

        # Add nodes with dependencies
        for i in range(50):
            graph.add_vertical(f"vertical_{i}", version="1.0.0", load_priority=0)
            if i > 0:
                graph.add_dependency(f"vertical_{i}", f"vertical_{i-1}")

        # Time the resolution (which validates dependencies)
        start = time.perf_counter()
        load_order = graph.resolve_load_order()
        duration_ms = (time.perf_counter() - start) * 1000

        # Assert: validation should be fast
        assert duration_ms < 5, f"Dependency validation took {duration_ms:.2f}ms, target < 5ms"

        # Verify no missing dependencies for valid graph
        assert len(load_order.missing_dependencies) == 0
        assert load_order.can_load


class TestDependencyGraphScalability:
    """Test scalability of dependency graph operations."""

    def test_large_graph_performance(self):
        """Test performance with large graph (100 nodes)."""
        graph = ExtensionDependencyGraph()

        # Create a large graph
        for i in range(100):
            graph.add_vertical(f"vertical_{i}", version="1.0.0", load_priority=i % 10)

        # Create dependencies
        for i in range(10, 100):
            graph.add_dependency(f"vertical_{i}", f"vertical_{i-10}")

        # Time the resolution
        start = time.perf_counter()
        load_order = graph.resolve_load_order()
        duration_ms = (time.perf_counter() - start) * 1000

        # Should still be reasonably fast
        assert duration_ms < 50, f"Large graph (100 nodes) took {duration_ms:.2f}ms, target < 50ms"
        assert len(load_order.order) == 100

    def test_memory_efficiency(self):
        """Test that graph operations are memory efficient."""
        import gc

        graph = ExtensionDependencyGraph()

        # Get baseline
        gc.collect()
        baseline_objects = len(gc.get_objects())

        # Add many nodes and dependencies
        for i in range(100):
            graph.add_vertical(f"vertical_{i}", version="1.0.0", load_priority=0)
            if i > 0:
                graph.add_dependency(f"vertical_{i}", f"vertical_{i-1}")

        # Force collection
        gc.collect()
        growth = len(gc.get_objects()) - baseline_objects

        # Memory growth should be reasonable (< 5000 objects)
        assert growth < 5000, f"Memory grew by {growth} objects for 100 nodes"

    def test_idempotent_operations_performance(self):
        """Test that repeated operations are fast (idempotent)."""
        graph = ExtensionDependencyGraph()

        # Add some nodes
        for i in range(10):
            graph.add_vertical(f"vertical_{i}", version="1.0.0", load_priority=0)

        # Add same dependency multiple times (should be idempotent)
        start = time.perf_counter()
        for _ in range(100):
            graph.add_dependency("vertical_0", "vertical_1")
        duration_ms = (time.perf_counter() - start) * 1000

        # Should be very fast (idempotent operations)
        assert (
            duration_ms < 10
        ), f"100 idempotent operations took {duration_ms:.2f}ms, target < 10ms"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
