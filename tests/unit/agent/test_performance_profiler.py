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

"""Tests for the PerformanceProfiler module."""

import asyncio
import time

import pytest

from victor.agent.performance_profiler import (
    PerformanceProfiler,
    ProfileReport,
    Span,
    SpanStatus,
    get_profiler,
    reset_profiler,
)

# =============================================================================
# Span Tests
# =============================================================================


class TestSpan:
    """Tests for the Span dataclass."""

    def test_span_creation(self):
        """Test basic span creation."""
        span = Span(name="test_op", category="test")

        assert span.name == "test_op"
        assert span.category == "test"
        assert span.status == SpanStatus.RUNNING
        assert span.parent_id is None
        assert span.end_time is None
        assert span.duration_ms is None
        assert span.is_complete is False

    def test_span_completion(self):
        """Test span completion."""
        span = Span(name="test_op", category="test")
        time.sleep(0.01)  # Small delay to ensure measurable duration
        span.complete()

        assert span.status == SpanStatus.COMPLETED
        assert span.end_time is not None
        assert span.duration_ms is not None
        assert span.duration_ms > 0
        assert span.is_complete is True
        assert span.error is None

    def test_span_completion_with_error(self):
        """Test span completion with error."""
        span = Span(name="test_op", category="test")
        span.complete(error="Something went wrong")

        assert span.status == SpanStatus.FAILED
        assert span.is_complete is True
        assert span.error == "Something went wrong"

    def test_span_metadata(self):
        """Test span metadata operations."""
        span = Span(name="test_op", category="test")
        span.add_metadata("key1", "value1")
        span.add_metadata("key2", 42)

        assert span.metadata["key1"] == "value1"
        assert span.metadata["key2"] == 42

    def test_span_to_dict(self):
        """Test span serialization."""
        span = Span(name="test_op", category="test", metadata={"key": "value"})
        span.complete()
        data = span.to_dict()

        assert data["name"] == "test_op"
        assert data["category"] == "test"
        assert data["status"] == "completed"
        assert data["metadata"] == {"key": "value"}
        assert "duration_ms" in data
        assert data["duration_ms"] is not None


# =============================================================================
# PerformanceProfiler Tests
# =============================================================================


class TestPerformanceProfiler:
    """Tests for the PerformanceProfiler class."""

    def test_profiler_creation(self):
        """Test profiler creation."""
        profiler = PerformanceProfiler()
        assert profiler.enabled is True

        profiler_disabled = PerformanceProfiler(enabled=False)
        assert profiler_disabled.enabled is False

    def test_profiler_enable_disable(self):
        """Test enabling/disabling profiler."""
        profiler = PerformanceProfiler()
        assert profiler.enabled is True

        profiler.disable()
        assert profiler.enabled is False

        profiler.enable()
        assert profiler.enabled is True

    def test_span_context_manager(self):
        """Test span context manager."""
        profiler = PerformanceProfiler()

        with profiler.span("test_operation", category="test") as span:
            assert span.name == "test_operation"
            assert span.category == "test"
            assert span.status == SpanStatus.RUNNING
            time.sleep(0.01)

        assert span.status == SpanStatus.COMPLETED
        assert span.duration_ms is not None
        assert span.duration_ms >= 10  # At least 10ms

    def test_span_context_manager_with_metadata(self):
        """Test span context manager with metadata."""
        profiler = PerformanceProfiler()

        with profiler.span("test_op", category="test", key1="value1") as span:
            span.add_metadata("key2", "value2")

        assert span.metadata["key1"] == "value1"
        assert span.metadata["key2"] == "value2"

    def test_span_context_manager_with_exception(self):
        """Test span context manager handles exceptions."""
        profiler = PerformanceProfiler()

        with pytest.raises(ValueError):
            with profiler.span("failing_op", category="test") as span:
                raise ValueError("Test error")

        assert span.status == SpanStatus.FAILED
        assert span.error == "Test error"

    def test_nested_spans(self):
        """Test nested span hierarchy."""
        profiler = PerformanceProfiler()

        with profiler.span("parent", category="test") as parent:
            with profiler.span("child", category="test") as child:
                time.sleep(0.01)

        assert child.parent_id == parent.span_id
        assert child.span_id in parent.children

    def test_disabled_profiler_no_overhead(self):
        """Test disabled profiler has minimal overhead."""
        profiler = PerformanceProfiler(enabled=False)

        with profiler.span("test_op", category="test"):
            pass

        # Span should still work but not be tracked
        report = profiler.get_report()
        assert report.span_count == 0

    def test_profiler_reset(self):
        """Test profiler reset."""
        profiler = PerformanceProfiler()

        with profiler.span("test_op", category="test"):
            pass

        assert len(profiler._spans) == 1

        profiler.reset()
        assert len(profiler._spans) == 0

    def test_get_spans_by_category(self):
        """Test getting spans by category."""
        profiler = PerformanceProfiler()

        with profiler.span("op1", category="cat1"):
            pass
        with profiler.span("op2", category="cat2"):
            pass
        with profiler.span("op3", category="cat1"):
            pass

        cat1_spans = profiler.get_spans_by_category("cat1")
        assert len(cat1_spans) == 2

        cat2_spans = profiler.get_spans_by_category("cat2")
        assert len(cat2_spans) == 1

    def test_get_active_spans(self):
        """Test getting active spans."""
        profiler = PerformanceProfiler()

        # No active spans initially
        assert len(profiler.get_active_spans()) == 0

        # During span execution, there should be an active span
        # (Can't easily test this without threading)


# =============================================================================
# Profile Decorator Tests
# =============================================================================


class TestProfileDecorator:
    """Tests for the @profile decorator."""

    def test_sync_function_profiling(self):
        """Test profiling sync functions."""
        profiler = PerformanceProfiler()

        @profiler.profile(category="test")
        def sync_func():
            time.sleep(0.01)
            return 42

        result = sync_func()

        assert result == 42

        spans = profiler.get_spans_by_category("test")
        assert len(spans) == 1
        assert spans[0].name == "sync_func"
        assert spans[0].duration_ms >= 10

    def test_async_function_profiling(self):
        """Test profiling async functions."""
        profiler = PerformanceProfiler()

        @profiler.profile(category="async_test")
        async def async_func():
            await asyncio.sleep(0.01)
            return "async_result"

        result = asyncio.run(async_func())

        assert result == "async_result"

        spans = profiler.get_spans_by_category("async_test")
        assert len(spans) == 1
        assert spans[0].name == "async_func"

    def test_custom_span_name(self):
        """Test custom span name in decorator."""
        profiler = PerformanceProfiler()

        @profiler.profile(name="custom_name", category="test")
        def func():
            pass

        func()

        spans = profiler.get_spans_by_category("test")
        assert len(spans) == 1
        assert spans[0].name == "custom_name"


# =============================================================================
# ProfileReport Tests
# =============================================================================


class TestProfileReport:
    """Tests for the ProfileReport class."""

    def test_report_generation(self):
        """Test report generation."""
        profiler = PerformanceProfiler()

        with profiler.span("op1", category="cat1"):
            time.sleep(0.01)
        with profiler.span("op2", category="cat2"):
            time.sleep(0.02)

        report = profiler.get_report()

        assert report.span_count == 2
        assert report.total_duration_ms > 0
        assert "cat1" in report.spans_by_category
        assert "cat2" in report.spans_by_category

    def test_category_stats(self):
        """Test category statistics."""
        profiler = PerformanceProfiler()

        with profiler.span("op1", category="test"):
            time.sleep(0.01)
        with profiler.span("op2", category="test"):
            time.sleep(0.02)
        with profiler.span("op3", category="test"):
            time.sleep(0.03)

        report = profiler.get_report()
        stats = report.get_category_stats("test")

        assert stats["count"] == 3
        assert stats["total_ms"] > 0
        assert stats["avg_ms"] > 0
        assert stats["min_ms"] <= stats["avg_ms"] <= stats["max_ms"]

    def test_empty_category_stats(self):
        """Test stats for non-existent category."""
        profiler = PerformanceProfiler()
        report = profiler.get_report()

        stats = report.get_category_stats("nonexistent")

        assert stats["count"] == 0
        assert stats["total_ms"] == 0
        assert stats["avg_ms"] == 0

    def test_slowest_spans(self):
        """Test getting slowest spans."""
        profiler = PerformanceProfiler()

        with profiler.span("fast", category="test"):
            time.sleep(0.01)
        with profiler.span("slow", category="test"):
            time.sleep(0.05)
        with profiler.span("medium", category="test"):
            time.sleep(0.02)

        report = profiler.get_report()
        slowest = report.get_slowest_spans(2)

        assert len(slowest) == 2
        assert slowest[0].name == "slow"
        # Second should be either medium or fast, depending on exact timing

    def test_to_markdown(self):
        """Test markdown report generation."""
        profiler = PerformanceProfiler()

        with profiler.span("test_op", category="test"):
            time.sleep(0.01)

        report = profiler.get_report()
        markdown = report.to_markdown()

        assert "# Performance Profile Report" in markdown
        assert "test" in markdown
        assert "test_op" in markdown

    def test_to_dict(self):
        """Test dictionary report generation."""
        profiler = PerformanceProfiler()

        with profiler.span("test_op", category="test"):
            pass

        report = profiler.get_report()
        data = report.to_dict()

        assert "total_duration_ms" in data
        assert "span_count" in data
        assert "categories" in data
        assert "test" in data["categories"]


# =============================================================================
# Global Profiler Tests
# =============================================================================


class TestGlobalProfiler:
    """Tests for global profiler functions."""

    def test_get_profiler(self):
        """Test getting global profiler."""
        reset_profiler()  # Start fresh
        profiler = get_profiler()

        assert profiler is not None
        assert isinstance(profiler, PerformanceProfiler)

    def test_get_profiler_singleton(self):
        """Test global profiler is a singleton."""
        reset_profiler()
        profiler1 = get_profiler()
        profiler2 = get_profiler()

        assert profiler1 is profiler2

    def test_reset_profiler(self):
        """Test resetting global profiler."""
        reset_profiler()
        profiler = get_profiler()

        with profiler.span("test", category="test"):
            pass

        assert len(profiler._spans) == 1

        reset_profiler()
        assert len(profiler._spans) == 0


# =============================================================================
# Edge Cases and Integration Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_deeply_nested_spans(self):
        """Test deeply nested span hierarchy."""
        profiler = PerformanceProfiler()

        with profiler.span("level1", category="test"):
            with profiler.span("level2", category="test"):
                with profiler.span("level3", category="test"):
                    with profiler.span("level4", category="test"):
                        pass

        assert len(profiler._spans) == 4

    def test_concurrent_spans(self):
        """Test spans don't interfere across concurrent calls."""
        profiler = PerformanceProfiler()

        async def task(name: str):
            with profiler.span(name, category="concurrent"):
                await asyncio.sleep(0.01)

        async def run_concurrent():
            await asyncio.gather(
                task("task1"),
                task("task2"),
                task("task3"),
            )

        asyncio.run(run_concurrent())

        spans = profiler.get_spans_by_category("concurrent")
        assert len(spans) == 3

    def test_zero_duration_span(self):
        """Test span with near-zero duration."""
        profiler = PerformanceProfiler()

        with profiler.span("instant", category="test"):
            pass  # No delay

        spans = profiler.get_spans_by_category("test")
        assert len(spans) == 1
        assert spans[0].duration_ms is not None
        assert spans[0].duration_ms >= 0


# =============================================================================
# Marker Tests
# =============================================================================


@pytest.mark.unit
class TestProfilerMarker:
    """Unit tests marked for CI."""

    def test_profiler_basic(self):
        """Basic profiler functionality."""
        profiler = PerformanceProfiler()

        with profiler.span("test", category="unit"):
            pass

        assert profiler.get_report().span_count == 1
