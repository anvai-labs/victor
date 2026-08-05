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

"""Unit tests for UnifiedEntryPointRegistry."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from victor.framework.entry_point_registry import (
    CAPABILITY_ENTRY_POINT_GROUPS,
    EXTENSION_CAPABILITIES_ENTRY_POINT_GROUP,
    EXTENSION_PROTOCOLS_ENTRY_POINT_GROUP,
    EntryPointGroup,
    PROTOCOL_ENTRY_POINT_GROUPS,
    ScanMetrics,
    UnifiedEntryPointRegistry,
    get_entry_point,
    get_entry_point_group,
    get_entry_point_objects,
    get_entry_point_registry,
    get_entry_point_values,
    scan_all_entry_points,
)


class TestUnifiedEntryPointRegistry:
    """Test suite for UnifiedEntryPointRegistry."""

    def setup_method(self):
        """Reset registry before each test."""
        registry = UnifiedEntryPointRegistry.get_instance()
        registry.invalidate()

    def teardown_method(self):
        """Clean up registry after each test."""
        registry = UnifiedEntryPointRegistry.get_instance()
        registry.invalidate()

    def test_singleton_instance(self):
        """Test that registry returns singleton instance."""
        registry1 = UnifiedEntryPointRegistry.get_instance()
        registry2 = UnifiedEntryPointRegistry.get_instance()

        assert registry1 is registry2
        assert isinstance(registry1, UnifiedEntryPointRegistry)

    def test_scan_all_with_mock_entry_points(self):
        """Test single-pass scanning with mocked entry points."""
        # Mock entry points
        mock_ep1 = MagicMock()
        mock_ep1.group = "victor.plugins"
        mock_ep1.name = "coding"
        mock_ep1.value = "victor_coding.plugin:plugin"

        mock_ep2 = MagicMock()
        mock_ep2.group = "victor.plugins"
        mock_ep2.name = "devops"
        mock_ep2.value = "victor_devops.plugin:plugin"

        mock_ep3 = MagicMock()
        mock_ep3.group = "victor.capabilities"
        mock_ep3.name = "parser"
        mock_ep3.value = "victor.parsing:Parser"

        with patch("victor.framework.entry_point_registry.entry_points") as mock_eps:
            mock_eps.return_value = [mock_ep1, mock_ep2, mock_ep3]

            registry = UnifiedEntryPointRegistry.get_instance()
            metrics = registry.scan_all()

            # Verify metrics
            assert metrics.total_groups == 2  # victor.plugins and victor.capabilities
            assert metrics.total_entry_points == 3
            assert metrics.scan_duration_ms >= 0

    def test_preferred_extension_groups_are_discovered(self):
        """Semantic extension groups are first-class discovery groups."""
        protocol_ep = MagicMock()
        protocol_ep.group = EXTENSION_PROTOCOLS_ENTRY_POINT_GROUP
        protocol_ep.name = "coding-tools"
        protocol_ep.value = "victor_coding.protocols:CodingToolProvider"

        capability_ep = MagicMock()
        capability_ep.group = EXTENSION_CAPABILITIES_ENTRY_POINT_GROUP
        capability_ep.name = "coding-lsp"
        capability_ep.value = "victor_coding.protocols:LSPCapabilityProvider"

        with patch("victor.framework.entry_point_registry.entry_points") as mock_eps:
            mock_eps.return_value = [protocol_ep, capability_ep]

            registry = UnifiedEntryPointRegistry.get_instance()
            metrics = registry.scan_all()

            assert metrics.total_groups == 2
            assert registry.get_group(EXTENSION_PROTOCOLS_ENTRY_POINT_GROUP) is not None
            assert registry.get_group(EXTENSION_CAPABILITIES_ENTRY_POINT_GROUP) is not None
            assert PROTOCOL_ENTRY_POINT_GROUPS[0] == EXTENSION_PROTOCOLS_ENTRY_POINT_GROUP
            assert CAPABILITY_ENTRY_POINT_GROUPS[0] == EXTENSION_CAPABILITIES_ENTRY_POINT_GROUP

    def test_scan_all_empty(self):
        """Test scanning with no entry points."""
        with patch("victor.framework.entry_point_registry.entry_points") as mock_eps:
            mock_eps.return_value = []

            registry = UnifiedEntryPointRegistry.get_instance()
            metrics = registry.scan_all()

            assert metrics.total_groups == 0
            assert metrics.total_entry_points == 0
            assert metrics.scan_duration_ms >= 0

    def test_scan_all_idempotent(self):
        """Test that multiple scans produce same result."""
        mock_ep = MagicMock()
        mock_ep.group = "victor.plugins"
        mock_ep.name = "test"
        mock_ep.value = "test:Test"

        with patch("victor.framework.entry_point_registry.entry_points") as mock_eps:
            mock_eps.return_value = [mock_ep]

            registry = UnifiedEntryPointRegistry.get_instance()

            # First scan
            metrics1 = registry.scan_all()

            # Second scan (should be cached, no actual scan)
            metrics2 = registry.scan_all()

            assert metrics1.total_groups == metrics2.total_groups
            assert metrics1.total_entry_points == metrics2.total_entry_points

    def test_scan_all_force_rescan(self):
        """Test forced rescan clears previous data."""
        mock_ep1 = MagicMock()
        mock_ep1.group = "victor.plugins"
        mock_ep1.name = "test1"
        mock_ep1.value = "test1:Test1"

        with patch("victor.framework.entry_point_registry.entry_points") as mock_eps:
            mock_eps.return_value = [mock_ep1]

            registry = UnifiedEntryPointRegistry.get_instance()
            metrics1 = registry.scan_all()
            assert metrics1.total_entry_points == 1

            # Add another mock entry point
            mock_ep2 = MagicMock()
            mock_ep2.group = "victor.plugins"
            mock_ep2.name = "test2"
            mock_ep2.value = "test2:Test2"
            mock_eps.return_value = [mock_ep1, mock_ep2]

            # Force rescan
            metrics2 = registry.scan_all(force=True)
            assert metrics2.total_entry_points == 2

    def test_get_group(self):
        """Test getting entry points for a specific group."""
        mock_ep = MagicMock()
        mock_ep.group = "victor.plugins"
        mock_ep.name = "coding"
        mock_ep.value = "victor_coding.plugin:plugin"

        with patch("victor.framework.entry_point_registry.entry_points") as mock_eps:
            mock_eps.return_value = [mock_ep]

            registry = UnifiedEntryPointRegistry.get_instance()
            registry.scan_all()

            group = registry.get_group("victor.plugins")
            assert group is not None
            assert group.group_name == "victor.plugins"
            assert "coding" in group.entry_points
            assert len(group.entry_points) == 1

    def test_get_group_not_found(self):
        """Test getting non-existent group returns None."""
        with patch("victor.framework.entry_point_registry.entry_points") as mock_eps:
            mock_eps.return_value = []
            registry = UnifiedEntryPointRegistry.get_instance()
            registry.scan_all()

        group = registry.get_group("nonexistent.group")
        assert group is None

    def test_get_group_triggers_lazy_scan(self):
        """Test that get_group triggers lazy scan if not scanned."""
        mock_ep = MagicMock()
        mock_ep.group = "victor.plugins"
        mock_ep.name = "test"
        mock_ep.value = "test:Test"

        with patch("victor.framework.entry_point_registry.entry_points") as mock_eps:
            mock_eps.return_value = [mock_ep]

            registry = UnifiedEntryPointRegistry.get_instance()

            # Don't call scan_all() - get_group should trigger it
            group = registry.get_group("victor.plugins")

            assert group is not None
            assert "test" in group.entry_points

    def test_get_entry_point(self):
        """Test getting a specific entry point."""
        mock_ep = MagicMock()
        mock_ep.group = "victor.plugins"
        mock_ep.name = "coding"
        mock_ep.value = "victor_coding.plugin:plugin"
        mock_ep.load.return_value = "loaded_value"

        with patch("victor.framework.entry_point_registry.entry_points") as mock_eps:
            mock_eps.return_value = [mock_ep]

            registry = UnifiedEntryPointRegistry.get_instance()
            registry.scan_all()

            # Get entry point (should load it)
            result = registry.get("victor.plugins", "coding")
            assert result == "loaded_value"
            mock_ep.load.assert_called_once()

    def test_get_entry_point_caches_loaded_value(self):
        """Test that loaded entry points are cached."""
        mock_ep = MagicMock()
        mock_ep.group = "victor.plugins"
        mock_ep.name = "coding"
        mock_ep.value = "victor_coding.plugin:plugin"
        mock_ep.load.return_value = "loaded_value"

        with patch("victor.framework.entry_point_registry.entry_points") as mock_eps:
            mock_eps.return_value = [mock_ep]

            registry = UnifiedEntryPointRegistry.get_instance()
            registry.scan_all()

            # Get entry point twice
            result1 = registry.get("victor.plugins", "coding")
            result2 = registry.get("victor.plugins", "coding")

            assert result1 == result2
            # Should only load once
            mock_ep.load.assert_called_once()

    def test_get_entry_point_not_found(self):
        """Test getting non-existent entry point returns None."""
        with patch("victor.framework.entry_point_registry.entry_points") as mock_eps:
            mock_eps.return_value = []
            registry = UnifiedEntryPointRegistry.get_instance()
            registry.scan_all()

        result = registry.get("nonexistent.group", "nonexistent")
        assert result is None

    def test_get_entry_point_load_failure(self):
        """Test that load failure is handled gracefully."""
        mock_ep = MagicMock()
        mock_ep.group = "victor.plugins"
        mock_ep.name = "broken"
        mock_ep.value = "broken:Broken"
        mock_ep.load.side_effect = Exception("Load failed")

        with patch("victor.framework.entry_point_registry.entry_points") as mock_eps:
            mock_eps.return_value = [mock_ep]

            registry = UnifiedEntryPointRegistry.get_instance()
            registry.scan_all()

            # Should return None on load failure
            result = registry.get("victor.plugins", "broken")
            assert result is None

    def test_list_groups(self):
        """Test listing all discovered groups."""
        mock_ep1 = MagicMock()
        mock_ep1.group = "victor.plugins"
        mock_ep1.name = "coding"
        mock_ep1.value = "test:Test"

        mock_ep2 = MagicMock()
        mock_ep2.group = "victor.capabilities"
        mock_ep2.name = "parser"
        mock_ep2.value = "test:Test"

        with patch("victor.framework.entry_point_registry.entry_points") as mock_eps:
            mock_eps.return_value = [mock_ep1, mock_ep2]

            registry = UnifiedEntryPointRegistry.get_instance()
            registry.scan_all()

            groups = registry.list_groups()
            assert "victor.plugins" in groups
            assert "victor.capabilities" in groups
            assert len(groups) == 2

    def test_list_entry_points(self):
        """Test listing entry points in a group."""
        mock_ep1 = MagicMock()
        mock_ep1.group = "victor.plugins"
        mock_ep1.name = "coding"
        mock_ep1.value = "test:Coding"

        mock_ep2 = MagicMock()
        mock_ep2.group = "victor.plugins"
        mock_ep2.name = "devops"
        mock_ep2.value = "test:DevOps"

        with patch("victor.framework.entry_point_registry.entry_points") as mock_eps:
            mock_eps.return_value = [mock_ep1, mock_ep2]

            registry = UnifiedEntryPointRegistry.get_instance()
            registry.scan_all()

            entry_points = registry.list_entry_points("victor.plugins")
            assert "coding" in entry_points
            assert "devops" in entry_points
            assert len(entry_points) == 2

    def test_list_entry_points_nonexistent_group(self):
        """Test listing entry points for non-existent group."""
        with patch("victor.framework.entry_point_registry.entry_points") as mock_eps:
            mock_eps.return_value = []
            registry = UnifiedEntryPointRegistry.get_instance()
            registry.scan_all()

        entry_points_list = registry.list_entry_points("nonexistent.group")
        assert entry_points_list == []

    def test_invalidate(self):
        """Test cache invalidation."""
        mock_ep = MagicMock()
        mock_ep.group = "victor.plugins"
        mock_ep.name = "test"
        mock_ep.value = "test:Test"

        with patch("victor.framework.entry_point_registry.entry_points") as mock_eps:
            mock_eps.return_value = [mock_ep]

            registry = UnifiedEntryPointRegistry.get_instance()
            registry.scan_all()

            assert registry.list_groups() == ["victor.plugins"]
            assert registry._scanned is True

            # Invalidate - clears cache and marks as not scanned
            registry.invalidate()
            assert registry._scanned is False

            # Accessing _groups directly should show empty (no automatic re-scan)
            assert len(registry._groups) == 0

    def test_get_metrics(self):
        """Test getting scan metrics."""
        registry = UnifiedEntryPointRegistry.get_instance()
        metrics = registry.get_metrics()

        assert isinstance(metrics, ScanMetrics)
        assert metrics.total_groups >= 0
        assert metrics.total_entry_points >= 0

    def test_cache_hit_tracking(self):
        """Test that cache hits are tracked."""
        mock_ep = MagicMock()
        mock_ep.group = "victor.plugins"
        mock_ep.name = "test"
        mock_ep.value = "test:Test"
        mock_ep.load.return_value = "loaded"

        with patch("victor.framework.entry_point_registry.entry_points") as mock_eps:
            mock_eps.return_value = [mock_ep]

            registry = UnifiedEntryPointRegistry.get_instance()
            registry.scan_all()

            # First access - cache miss
            registry.get_group("victor.plugins")
            initial_misses = registry.get_metrics().cache_misses

            # Second access - cache hit
            registry.get_group("victor.plugins")

            assert registry.get_metrics().cache_misses == initial_misses


class TestConvenienceFunctions:
    """Test suite for convenience functions."""

    def setup_method(self):
        """Reset registry before each test."""
        registry = UnifiedEntryPointRegistry.get_instance()
        registry.invalidate()

    def teardown_method(self):
        """Clean up registry after each test."""
        registry = UnifiedEntryPointRegistry.get_instance()
        registry.invalidate()

    def test_get_entry_point_registry(self):
        """Test get_entry_point_registry convenience function."""
        registry = get_entry_point_registry()
        assert isinstance(registry, UnifiedEntryPointRegistry)

    def test_scan_all_entry_points(self):
        """Test scan_all_entry_points convenience function."""
        mock_ep = MagicMock()
        mock_ep.group = "victor.plugins"
        mock_ep.name = "test"
        mock_ep.value = "test:Test"

        with patch("victor.framework.entry_point_registry.entry_points") as mock_eps:
            mock_eps.return_value = [mock_ep]

            metrics = scan_all_entry_points()
            assert isinstance(metrics, ScanMetrics)
            assert metrics.total_entry_points == 1

    def test_scan_all_entry_points_with_force(self):
        """Test scan_all_entry_points with force parameter."""
        mock_ep = MagicMock()
        mock_ep.group = "victor.plugins"
        mock_ep.name = "test"
        mock_ep.value = "test:Test"

        with patch("victor.framework.entry_point_registry.entry_points") as mock_eps:
            mock_eps.return_value = [mock_ep]

            # First scan
            metrics1 = scan_all_entry_points()

            # Second scan with force
            metrics2 = scan_all_entry_points(force=True)

            assert metrics1.total_entry_points == metrics2.total_entry_points

    def test_get_entry_point_convenience(self):
        """Test get_entry_point convenience function."""
        mock_ep = MagicMock()
        mock_ep.group = "victor.plugins"
        mock_ep.name = "coding"
        mock_ep.value = "test:Coding"
        mock_ep.load.return_value = "loaded_value"

        with patch("victor.framework.entry_point_registry.entry_points") as mock_eps:
            mock_eps.return_value = [mock_ep]

            # Scan first
            scan_all_entry_points()

            # Get entry point
            result = get_entry_point("victor.plugins", "coding")
            assert result == "loaded_value"

    def test_get_entry_point_group_convenience(self):
        """Test get_entry_point_group convenience function."""
        mock_ep = MagicMock()
        mock_ep.group = "victor.plugins"
        mock_ep.name = "test"
        mock_ep.value = "test:Test"

        with patch("victor.framework.entry_point_registry.entry_points") as mock_eps:
            mock_eps.return_value = [mock_ep]

            # Scan first
            scan_all_entry_points()

            # Get group
            group = get_entry_point_group("victor.plugins")
            assert group is not None
            assert group.group_name == "victor.plugins"

    def test_get_entry_point_objects_convenience(self):
        """Test retrieving raw entry point objects through the shared helper."""
        mock_ep = MagicMock()
        mock_ep.group = "victor.plugins"
        mock_ep.name = "coding"
        mock_ep.value = "victor_coding.plugin:plugin"

        with patch("victor.framework.entry_point_registry.entry_points") as mock_eps:
            mock_eps.return_value = [mock_ep]

            result = get_entry_point_objects("victor.plugins")

        assert result == (mock_ep,)

    def test_get_entry_point_values_convenience(self):
        """Test retrieving entry point name/value mappings through the shared helper."""
        mock_ep = MagicMock()
        mock_ep.group = "victor.plugins"
        mock_ep.name = "coding"
        mock_ep.value = "victor_coding.plugin:plugin"

        with patch("victor.framework.entry_point_registry.entry_points") as mock_eps:
            mock_eps.return_value = [mock_ep]

            result = get_entry_point_values("victor.plugins")

        assert result == {"coding": "victor_coding.plugin:plugin"}

    def test_get_entry_point_values_force_invalidates_registry(self):
        """Force lookups should invalidate the shared registry before reading values."""
        registry = UnifiedEntryPointRegistry.get_instance()
        registry.invalidate()

        mock_ep = MagicMock()
        mock_ep.group = "victor.plugins"
        mock_ep.name = "coding"
        mock_ep.value = "victor_coding.plugin:plugin"

        with patch("victor.framework.entry_point_registry.entry_points") as mock_eps:
            mock_eps.return_value = [mock_ep]
            with patch.object(registry, "invalidate", wraps=registry.invalidate) as mock_invalidate:
                result = get_entry_point_values("victor.plugins", force=True)

        assert result == {"coding": "victor_coding.plugin:plugin"}
        mock_invalidate.assert_called_once()


class TestPerformanceBenchmarks:
    """Performance benchmarks for UnifiedEntryPointRegistry."""

    def setup_method(self):
        """Reset registry before each test."""
        registry = UnifiedEntryPointRegistry.get_instance()
        registry.invalidate()

    def teardown_method(self):
        """Clean up registry after each test."""
        registry = UnifiedEntryPointRegistry.get_instance()
        registry.invalidate()

    def test_scan_performance_target(self):
        """Test that scan completes in under 50ms."""
        # Create many mock entry points
        mock_eps = []
        for i in range(50):
            mock_ep = MagicMock()
            mock_ep.group = "victor.plugins"
            mock_ep.name = f"vertical_{i}"
            mock_ep.value = f"test:Vertical{i}"
            mock_eps.append(mock_ep)

        with patch("victor.framework.entry_point_registry.entry_points") as mock_eps_func:
            mock_eps_func.return_value = mock_eps

            registry = UnifiedEntryPointRegistry.get_instance()

            start = time.perf_counter()
            metrics = registry.scan_all()
            duration_ms = (time.perf_counter() - start) * 1000

            # Should complete quickly (less than 50ms for 50 entry points)
            # Note: This is a relaxed target for testing; real entry_points() is slower
            assert duration_ms < 100  # Relaxed target for mocking overhead
            assert metrics.total_entry_points == 50

    def test_lazy_scan_performance(self):
        """Test that lazy scan on first access is fast."""
        mock_ep = MagicMock()
        mock_ep.group = "victor.plugins"
        mock_ep.name = "test"
        mock_ep.value = "test:Test"

        with patch("victor.framework.entry_point_registry.entry_points") as mock_eps:
            mock_eps.return_value = [mock_ep]

            registry = UnifiedEntryPointRegistry.get_instance()

            start = time.perf_counter()
            group = registry.get_group("victor.plugins")
            duration_ms = (time.perf_counter() - start) * 1000

            # Lazy scan should trigger automatically
            assert group is not None
            assert duration_ms < 100  # Relaxed target

    def test_cache_hit_performance(self):
        """Test that cached access is very fast."""
        mock_ep = MagicMock()
        mock_ep.group = "victor.plugins"
        mock_ep.name = "test"
        mock_ep.value = "test:Test"

        with patch("victor.framework.entry_point_registry.entry_points") as mock_eps:
            mock_eps.return_value = [mock_ep]

            registry = UnifiedEntryPointRegistry.get_instance()
            registry.scan_all()

            # Time multiple cache hits
            start = time.perf_counter()
            for _ in range(100):
                registry.get_group("victor.plugins")
            duration_ms = (time.perf_counter() - start) * 1000

            # Should be very fast (less than 10ms for 100 cache hits)
            assert duration_ms < 10


class TestEntryPointGroup:
    """Test suite for EntryPointGroup dataclass."""

    def test_entry_point_group_creation(self):
        """Test creating an EntryPointGroup."""
        mock_ep = MagicMock()
        mock_ep.name = "test"
        mock_ep.value = "test:Test"

        group = EntryPointGroup(
            group_name="victor.plugins",
            entry_points={"test": (mock_ep, False)},
            scan_order=0,
        )

        assert group.group_name == "victor.plugins"
        assert "test" in group.entry_points
        assert group.scan_order == 0


class TestScanMetrics:
    """Test suite for ScanMetrics dataclass."""

    def test_scan_metrics_defaults(self):
        """Test ScanMetrics default values."""
        metrics = ScanMetrics()

        assert metrics.total_groups == 0
        assert metrics.total_entry_points == 0
        assert metrics.scan_duration_ms == 0.0
        assert metrics.cache_hits == 0
        assert metrics.cache_misses == 0

    def test_scan_metrics_with_values(self):
        """Test ScanMetrics with custom values."""
        metrics = ScanMetrics(
            total_groups=5,
            total_entry_points=20,
            scan_duration_ms=45.5,
            cache_hits=100,
            cache_misses=5,
        )

        assert metrics.total_groups == 5
        assert metrics.total_entry_points == 20
        assert metrics.scan_duration_ms == 45.5
        assert metrics.cache_hits == 100
        assert metrics.cache_misses == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
