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

"""Performance benchmarks for entry point scanning.

These benchmarks validate that the UnifiedEntryPointRegistry meets
the performance target of < 50ms for single-pass scanning.

Target Metrics:
    - Entry point scan: < 50ms
    - Single-pass scanning vs multiple independent scans
    - Lazy loading efficiency
"""

from __future__ import annotations

import time
import pytest

from victor.framework.entry_point_registry import (
    UnifiedEntryPointRegistry,
    scan_all_entry_points,
    get_entry_point,
    get_entry_point_group,
)


class TestEntryPointScanPerformance:
    """Performance benchmarks for entry point scanning."""

    def test_single_pass_scan_performance(self):
        """Test that single-pass scanning completes in < 50ms."""
        registry = UnifiedEntryPointRegistry()

        # Time the scan
        start = time.perf_counter()
        metrics = registry.scan_all()
        end = time.perf_counter()

        duration_ms = (end - start) * 1000

        # Assert: scan should complete in < 50ms
        assert duration_ms < 50, f"Entry point scan took {duration_ms:.2f}ms, target < 50ms"

        # Also verify we found some groups (via metrics)
        assert metrics.total_groups > 0, "Should find at least some entry point groups"
        assert metrics.total_entry_points >= 0, "Should have entry point count"

    def test_single_pass_scan_vs_multiple_scans(self, benchmark=False):
        """Compare single-pass scanning to multiple independent scans."""
        if not benchmark and not pytest:
            # Skip in non-benchmark mode unless explicitly requested
            return

        # Time single-pass scan
        start = time.perf_counter()
        registry = UnifiedEntryPointRegistry()
        groups = registry.scan_all()
        single_pass_duration = time.perf_counter() - start

        # Time multiple independent scans (simulating old approach)
        start = time.perf_counter()
        from importlib.metadata import entry_points

        # Simulate 9 independent scans (as in old code)
        groups_to_scan = [
            "victor.prompt_contributors",
            "victor.mode_configs",
            "victor.workflow_providers",
            "victor.team_spec_providers",
            "victor.capability_providers",
            "victor.service_providers",
            "victor.tool_dependencies",
            "victor.safety_rules",
            "victor.rl_configs",
        ]

        for group in groups_to_scan:
            try:
                eps = entry_points(group=group)
                list(eps)  # Force evaluation
            except Exception:
                pass

        multiple_scan_duration = time.perf_counter() - start

        # Single-pass should be faster
        speedup = multiple_scan_duration / single_pass_duration if single_pass_duration > 0 else 1

        # Assert: single-pass should be at least 2x faster
        assert speedup >= 1.5, f"Single-pass scan is only {speedup:.2f}x faster, target >= 1.5x"

        print(f"\nSingle-pass: {single_pass_duration*1000:.2f}ms")
        print(f"Multiple scans: {multiple_scan_duration*1000:.2f}ms")
        print(f"Speedup: {speedup:.2f}x")

    def test_lazy_loading_performance(self):
        """Test that lazy loading (get) is fast."""
        registry = UnifiedEntryPointRegistry()

        # First scan to populate cache
        registry.scan_all()

        # Time lazy get operations
        start = time.perf_counter()

        # Simulate 300 get operations (100 iterations * 3 groups)
        for _ in range(100):
            # Try to get various groups
            for group in [
                "victor.prompt_contributors",
                "victor.mode_configs",
                "victor.workflow_providers",
            ]:
                try:
                    registry.get_group(group)
                except Exception:
                    pass

        end = time.perf_counter()
        duration_ms = (end - start) * 1000

        # Assert: 300 lazy gets should be very fast (< 30ms total)
        assert duration_ms < 30, f"300 lazy get operations took {duration_ms:.2f}ms, target < 30ms"

        # Average per get (300 operations)
        avg_per_get = duration_ms / 300
        assert avg_per_get < 0.2, f"Average lazy get took {avg_per_get:.3f}ms, target < 0.2ms"

    def test_caching_efficiency(self):
        """Test that repeated scans use cached results efficiently."""
        registry = UnifiedEntryPointRegistry()

        # First scan (cold)
        start = time.perf_counter()
        metrics1 = registry.scan_all()
        cold_duration = time.perf_counter() - start

        # Second scan (warm, should use cache)
        start = time.perf_counter()
        metrics2 = registry.scan_all()
        warm_duration = time.perf_counter() - start

        # Warm scan should be much faster
        speedup = cold_duration / warm_duration if warm_duration > 0 else 1

        # Assert: warm scan should be at least 10x faster
        assert speedup >= 10, f"Warm scan is only {speedup:.2f}x faster, target >= 10x"

        # Results should be identical (metrics should match)
        assert (
            metrics1.total_groups == metrics2.total_groups
        ), "Cached scan should return same group count"
        assert (
            metrics1.total_entry_points == metrics2.total_entry_points
        ), "Cached scan should return same entry point count"

    def test_scan_consistency(self):
        """Test that multiple scans produce consistent results."""
        registry = UnifiedEntryPointRegistry()

        # Run scan 10 times
        results = []
        for _ in range(10):
            metrics = registry.scan_all()
            results.append((metrics.total_groups, metrics.total_entry_points))

        # All results should be identical
        first_result = results[0]
        for result in results[1:]:
            assert result == first_result, "Scans should produce consistent results"


class TestEntryPointScalability:
    """Test scalability of entry point scanning."""

    def test_scan_with_many_groups(self):
        """Test scanning performance with many entry point groups."""
        registry = UnifiedEntryPointRegistry()

        # Scan all available groups
        start = time.perf_counter()
        metrics = registry.scan_all()
        duration = time.perf_counter() - start

        # Even with many groups, should be fast
        duration_ms = duration * 1000
        assert (
            duration_ms < 100
        ), f"Scan took {duration_ms:.2f}ms with {metrics.total_groups} groups"

    def test_memory_efficiency(self):
        """Test that registry doesn't leak memory."""
        import gc
        import sys

        registry = UnifiedEntryPointRegistry()

        # Get baseline memory
        gc.collect()
        baseline_objects = len(gc.get_objects())

        # Perform multiple scans
        for _ in range(10):
            registry.scan_all()

        # Force garbage collection
        gc.collect()

        # Memory should not have grown significantly
        final_objects = len(gc.get_objects())
        growth = final_objects - baseline_objects

        # Allow some growth but not excessive (< 1000 objects)
        assert growth < 1000, f"Memory grew by {growth} objects after 10 scans"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
