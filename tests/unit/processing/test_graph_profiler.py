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

"""Tests for graph profiling utilities (PH4-008)."""

from __future__ import annotations

import asyncio
import time

import pytest

from victor.processing.graph_profiler import (
    GraphProfiler,
    OperationMetrics,
    ProfileReport,
    ProfilingConfig,
    configure_graph_profiler,
    get_graph_profiler,
    profile_graph_operation,
    reset_graph_profiler,
)


class TestGraphProfiler:
    """Tests for GraphProfiler class."""

    def test_profiler_initialization(self):
        """Test profiler initialization with defaults."""
        profiler = GraphProfiler()

        assert profiler.is_enabled() is True
        assert profiler._track_memory is False

    def test_profiler_disabled(self):
        """Test profiler when disabled."""
        profiler = GraphProfiler(enabled=False)

        assert profiler.is_enabled() is False

    def test_profiler_with_memory_tracking(self):
        """Test profiler with memory tracking enabled."""
        profiler = GraphProfiler(track_memory=True)

        assert profiler._track_memory is True

    def test_profile_operation_context_manager(self):
        """Test profiling an operation with context manager."""
        profiler = GraphProfiler()

        with profiler.profile_operation("test_operation"):
            time.sleep(0.01)  # Simulate some work

        metrics = profiler.get_metrics("test_operation")
        assert metrics is not None
        assert metrics.name == "test_operation"
        assert metrics.call_count == 1
        assert metrics.total_time_ms > 0
        assert metrics.avg_time_ms > 0

    def test_profile_operation_multiple_calls(self):
        """Test profiling multiple calls to the same operation."""
        profiler = GraphProfiler()

        for _ in range(5):
            with profiler.profile_operation("repeat_operation"):
                time.sleep(0.001)

        metrics = profiler.get_metrics("repeat_operation")
        assert metrics.call_count == 5
        assert metrics.total_time_ms > 0
        assert metrics.avg_time_ms == metrics.total_time_ms / 5

    def test_profile_operation_when_disabled(self):
        """Test that operations are not tracked when profiler is disabled."""
        profiler = GraphProfiler(enabled=False)

        with profiler.profile_operation("disabled_operation"):
            time.sleep(0.01)

        metrics = profiler.get_metrics("disabled_operation")
        assert metrics is None

    def test_record_call_manual(self):
        """Test manual recording of operation calls."""
        profiler = GraphProfiler()

        profiler.record_call("manual_operation", 50.0)
        profiler.record_call("manual_operation", 100.0)

        metrics = profiler.get_metrics("manual_operation")
        assert metrics.call_count == 2
        assert metrics.total_time_ms == 150.0
        assert metrics.avg_time_ms == 75.0

    def test_record_call_with_error(self):
        """Test recording calls with errors."""
        profiler = GraphProfiler()

        profiler.record_call("error_operation", 50.0, error=False)
        profiler.record_call("error_operation", 50.0, error=True)
        profiler.record_call("error_operation", 50.0, error=True)

        metrics = profiler.get_metrics("error_operation")
        assert metrics.call_count == 3
        assert metrics.error_count == 2

    def test_get_metrics_nonexistent(self):
        """Test getting metrics for non-existent operation."""
        profiler = GraphProfiler()

        metrics = profiler.get_metrics("nonexistent")
        assert metrics is None

    def test_get_report(self):
        """Test generating a profile report."""
        profiler = GraphProfiler()

        with profiler.profile_operation("op1"):
            time.sleep(0.01)

        with profiler.profile_operation("op2"):
            time.sleep(0.005)

        report = profiler.get_report()

        assert isinstance(report, ProfileReport)
        assert len(report.operations) == 2
        assert "op1" in report.operations
        assert "op2" in report.operations
        assert report.total_time_ms > 0

    def test_report_hot_paths(self):
        """Test hot path identification in report."""
        profiler = GraphProfiler()

        # Create operations with different total times
        for _ in range(10):
            with profiler.profile_operation("hot_op"):
                time.sleep(0.01)

        for _ in range(2):
            with profiler.profile_operation("cold_op"):
                time.sleep(0.001)

        report = profiler.get_report()

        hot_paths = report.get_hot_paths(top_n=5)
        assert len(hot_paths) > 0
        # hot_op should be first (highest total time)
        assert hot_paths[0][0] == "hot_op"

    def test_report_slowest_operations(self):
        """Test slowest operations identification."""
        profiler = GraphProfiler()

        with profiler.profile_operation("fast_op"):
            time.sleep(0.001)

        with profiler.profile_operation("slow_op"):
            time.sleep(0.01)

        report = profiler.get_report()

        slowest = report.get_slowest_operations(top_n=5)
        assert len(slowest) > 0
        # slow_op should have higher average time
        assert slowest[0][0] == "slow_op"

    def test_report_most_frequent(self):
        """Test most frequent operations identification."""
        profiler = GraphProfiler()

        for _ in range(10):
            with profiler.profile_operation("frequent_op"):
                pass

        for _ in range(2):
            with profiler.profile_operation("rare_op"):
                pass

        report = profiler.get_report()

        most_frequent = report.get_most_frequent(top_n=5)
        assert len(most_frequent) > 0
        assert most_frequent[0][0] == "frequent_op"
        assert most_frequent[0][1] == 10

    def test_report_recommendations(self):
        """Test optimization recommendations generation."""
        profiler = GraphProfiler()

        # Create a slow, frequently called operation
        for _ in range(15):
            with profiler.profile_operation("slow_frequent"):
                time.sleep(0.01)

        # Create high variance operation
        with profiler.profile_operation("variance_op"):
            time.sleep(0.001)
        with profiler.profile_operation("variance_op"):
            time.sleep(0.1)

        # Create error-prone operation
        for i in range(10):
            profiler.record_call("error_op", 10.0, error=(i % 2 == 0))

        report = profiler.get_report()

        assert len(report.recommendations) > 0
        # Check for slow operation recommendation
        assert any("slow_frequent" in r for r in report.recommendations) or any(
            "error" in r.lower() for r in report.recommendations
        )

    def test_reset(self):
        """Test resetting profiler data."""
        profiler = GraphProfiler()

        with profiler.profile_operation("test_op"):
            pass

        assert profiler.get_metrics("test_op") is not None

        profiler.reset()

        assert profiler.get_metrics("test_op") is None

    def test_get_stats(self):
        """Test getting profiler statistics."""
        profiler = GraphProfiler()

        with profiler.profile_operation("op1"):
            time.sleep(0.01)
        with profiler.profile_operation("op2"):
            time.sleep(0.01)

        stats = profiler.get_stats()

        assert stats["enabled"] is True
        assert stats["track_memory"] is False
        assert stats["operation_count"] == 2
        assert stats["total_calls"] == 2
        assert stats["total_time_ms"] > 0
        assert stats["avg_time_ms"] > 0


class TestGlobalProfiler:
    """Tests for global profiler singleton."""

    def test_get_graph_profiler_singleton(self):
        """Test that get_graph_profiler returns same instance."""
        reset_graph_profiler()

        profiler1 = get_graph_profiler()
        profiler2 = get_graph_profiler()

        assert profiler1 is profiler2

    def test_configure_graph_profiler(self):
        """Test configuring the global profiler."""
        reset_graph_profiler()

        profiler = configure_graph_profiler(
            enabled=True,
            track_memory=True,
        )

        assert profiler.is_enabled() is True
        assert profiler._track_memory is True

        # Verify it's the same instance
        assert get_graph_profiler() is profiler

    def test_reset_graph_profiler(self):
        """Test resetting the global profiler."""
        reset_graph_profiler()

        profiler = get_graph_profiler()
        profiler.record_call("test", 10.0)

        reset_graph_profiler()

        new_profiler = get_graph_profiler()
        assert new_profiler.get_metrics("test") is None


class TestProfilingDecorator:
    """Tests for profile_graph_operation decorator."""

    def test_decorator_sync_function(self):
        """Test decorator on synchronous function."""
        reset_graph_profiler()
        profiler = get_graph_profiler()

        @profile_graph_operation(profiler, "sync_test")
        def sync_function():
            time.sleep(0.01)
            return "result"

        result = sync_function()

        assert result == "result"
        metrics = profiler.get_metrics("sync_test")
        assert metrics is not None
        assert metrics.call_count == 1

    def test_decorator_async_function(self):
        """Test decorator on async function."""
        reset_graph_profiler()
        profiler = get_graph_profiler()

        @profile_graph_operation(profiler, "async_test")
        async def async_function():
            await asyncio.sleep(0.01)
            return "async_result"

        async def run_test():
            result = await async_function()
            assert result == "async_result"

            metrics = profiler.get_metrics("async_test")
            assert metrics is not None
            assert metrics.call_count == 1

        asyncio.run(run_test())

    def test_decorator_with_default_profiler(self):
        """Test decorator using global profiler."""
        reset_graph_profiler()

        @profile_graph_operation()
        def default_profiler_function():
            time.sleep(0.01)
            return "result"

        default_profiler_function()

        profiler = get_graph_profiler()
        metrics = profiler.get_metrics("default_profiler_function")
        assert metrics is not None
        assert metrics.call_count == 1

    def test_decorator_with_operation_name(self):
        """Test decorator with custom operation name."""
        reset_graph_profiler()

        @profile_graph_operation(operation_name="custom_name")
        def some_function():
            time.sleep(0.01)
            return "result"

        some_function()

        profiler = get_graph_profiler()
        metrics = profiler.get_metrics("custom_name")
        assert metrics is not None


class TestProfilingConfig:
    """Tests for ProfilingConfig dataclass."""

    def test_profiling_config_defaults(self):
        """Test default configuration values."""
        config = ProfilingConfig()

        assert config.enabled is False
        assert config.track_memory is False
        assert config.report_threshold_ms == 10.0
        assert config.sample_rate == 1.0
        assert config.max_tracked_operations == 100

    def test_profiling_config_custom(self):
        """Test custom configuration values."""
        config = ProfilingConfig(
            enabled=True,
            track_memory=True,
            report_threshold_ms=50.0,
            sample_rate=0.5,
            max_tracked_operations=200,
        )

        assert config.enabled is True
        assert config.track_memory is True
        assert config.report_threshold_ms == 50.0
        assert config.sample_rate == 0.5
        assert config.max_tracked_operations == 200


class TestOperationMetrics:
    """Tests for OperationMetrics dataclass."""

    def test_operation_metrics_defaults(self):
        """Test default metric values."""
        metrics = OperationMetrics(name="test")

        assert metrics.name == "test"
        assert metrics.call_count == 0
        assert metrics.total_time_ms == 0.0
        assert metrics.min_time_ms == float("inf")
        assert metrics.max_time_ms == 0.0
        assert metrics.last_time_ms == 0.0
        assert metrics.error_count == 0
        assert metrics.memory_bytes == 0

    def test_operation_metrics_avg_time_calculation(self):
        """Test average time calculation."""
        metrics = OperationMetrics(name="test")

        metrics.call_count = 3
        metrics.total_time_ms = 150.0

        assert metrics.avg_time_ms == 50.0

    def test_operation_metrics_avg_time_zero_calls(self):
        """Test average time when no calls recorded."""
        metrics = OperationMetrics(name="test")

        assert metrics.avg_time_ms == 0.0


class TestProfileReport:
    """Tests for ProfileReport dataclass."""

    def test_profile_report_defaults(self):
        """Test default report values."""
        report = ProfileReport()

        assert report.operations == {}
        assert report.hot_paths == []
        assert report.recommendations == []
        assert report.total_time_ms == 0.0


class TestProfilingIntegration:
    """Integration tests for profiling with graph operations."""

    def test_profiling_multiple_operations(self):
        """Test profiling multiple related operations."""
        profiler = GraphProfiler()

        # Simulate a multi-stage operation
        with profiler.profile_operation("stage1_load"):
            time.sleep(0.01)

        with profiler.profile_operation("stage2_process"):
            time.sleep(0.02)

        with profiler.profile_operation("stage3_save"):
            time.sleep(0.005)

        report = profiler.get_report()

        assert len(report.operations) == 3
        assert report.total_time_ms > 35  # 10 + 20 + 5 ms minimum

        # Verify operations are ordered by total time in hot paths
        hot_paths = report.get_hot_paths()
        assert hot_paths[0][0] == "stage2_process"  # Should be highest

    def test_profiling_nested_operations(self):
        """Test profiling nested operations."""
        profiler = GraphProfiler()

        with profiler.profile_operation("outer_operation"):
            time.sleep(0.01)
            with profiler.profile_operation("inner_operation"):
                time.sleep(0.005)

        # Both operations should be tracked
        outer_metrics = profiler.get_metrics("outer_operation")
        inner_metrics = profiler.get_metrics("inner_operation")

        assert outer_metrics is not None
        assert inner_metrics is not None
        assert outer_metrics.call_count == 1
        assert inner_metrics.call_count == 1

    def test_profiling_error_handling(self):
        """Test that profiler handles exceptions in operations."""
        profiler = GraphProfiler()

        try:
            with profiler.profile_operation("failing_operation"):
                time.sleep(0.01)
                raise ValueError("Test error")
        except ValueError:
            pass

        # Operation should still be tracked despite error
        metrics = profiler.get_metrics("failing_operation")
        assert metrics is not None
        assert metrics.call_count == 1
        assert metrics.total_time_ms > 0
