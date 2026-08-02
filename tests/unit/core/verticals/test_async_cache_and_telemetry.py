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

"""Unit tests for AsyncSafeCacheManager and telemetry modules."""

from __future__ import annotations

import asyncio
import time

import pytest

from victor.core.verticals.async_cache_manager import AsyncSafeCacheManager
from victor.core.verticals.telemetry import (
    VerticalLoadSpan,
    vertical_load_span,
    VerticalLoadTelemetry,
    get_telemetry,
)


class TestAsyncSafeCacheManager:
    """Test suite for AsyncSafeCacheManager."""

    def setup_method(self):
        """Create fresh cache for each test."""
        self.cache = AsyncSafeCacheManager()

    def test_get_or_create_simple(self):
        """Test basic get_or_create functionality."""
        call_count = 0

        def factory():
            nonlocal call_count
            call_count += 1
            return "created_value"

        # First call - should invoke factory
        result1 = self.cache.get_or_create("test", "key1", factory)
        assert result1 == "created_value"
        assert call_count == 1

        # Second call - should use cache
        result2 = self.cache.get_or_create("test", "key1", factory)
        assert result2 == "created_value"
        assert call_count == 1  # Factory not called again

    def test_get_or_create_double_lock(self):
        """Test double-check locking pattern."""
        call_count = 0

        def factory():
            nonlocal call_count
            call_count += 1
            time.sleep(0.01)  # Simulate slow creation
            return "value"

        import threading

        results = []
        errors = []

        def worker():
            try:
                result = self.cache.get_or_create("test", "key", factory)
                results.append(result)
            except Exception as e:
                errors.append(e)

        # Create multiple threads that try to create the same value
        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Only one thread should have called the factory
        assert call_count == 1
        assert len(results) == 10
        assert all(r == "value" for r in results)
        assert len(errors) == 0

    def test_get_or_create_async_factory(self):
        """Test get_or_create_async with async factory."""

        async def async_factory():
            await asyncio.sleep(0.01)
            return "async_value"

        async def test():
            result = await self.cache.get_or_create_async("test", "key1", async_factory)
            assert result == "async_value"

        asyncio.run(test())

    def test_get_or_create_sync_factory_in_async_context(self):
        """Test get_or_create_async with sync factory."""

        def sync_factory():
            return "sync_value"

        async def test():
            result = await self.cache.get_or_create_async("test", "key1", sync_factory)
            assert result == "sync_value"

        asyncio.run(test())

    def test_get_if_cached_hit(self):
        """Test get_if_cached when value is cached."""
        self.cache._cache[self.cache._make_key("test", "key1")] = "cached_value"

        found, value = self.cache.get_if_cached("test", "key1")

        assert found is True
        assert value == "cached_value"

    def test_get_if_cached_miss(self):
        """Test get_if_cached when value is not cached."""
        found, value = self.cache.get_if_cached("test", "key1")

        assert found is False
        assert value is None

    def test_load_optional_cached(self):
        """Test load_optional when value is cached."""
        self.cache._cache[self.cache._make_key("test", "key1")] = "cached_value"

        result = self.cache.load_optional("test", "key1", lambda: "loaded")

        assert result == "cached_value"

    def test_load_optional_not_cached(self):
        """Test load_optional when value is not cached."""
        result = self.cache.load_optional("test", "key1", lambda: "loaded")

        assert result == "loaded"

    def test_load_optional_returns_none(self):
        """Test load_optional when loader returns None."""
        result = self.cache.load_optional("test", "key1", lambda: None)

        assert result is None
        # Should not cache None
        cache_key = self.cache._make_key("test", "key1")
        assert cache_key not in self.cache._cache

    def test_invalidate_all(self):
        """Test invalidating all cache entries."""
        self.cache._cache["test:key1"] = "value1"
        self.cache._cache["test:key2"] = "value2"
        self.cache._cache["other:key1"] = "value3"

        count = self.cache.invalidate()

        assert count == 3
        assert len(self.cache._cache) == 0
        assert len(self.cache._locks) == 0

    def test_invalidate_namespace(self):
        """Test invalidating all entries in a namespace."""
        self.cache._cache["test:key1"] = "value1"
        self.cache._cache["test:key2"] = "value2"
        self.cache._cache["other:key1"] = "value3"

        count = self.cache.invalidate(namespace="test")

        assert count == 2
        assert "test:key1" not in self.cache._cache
        assert "test:key2" not in self.cache._cache
        assert "other:key1" in self.cache._cache

    def test_invalidate_specific_key(self):
        """Test invalidating a specific key."""
        self.cache._cache["test:key1"] = "value1"
        self.cache._cache["test:key2"] = "value2"

        count = self.cache.invalidate(namespace="test", key="key1")

        assert count == 1
        assert "test:key1" not in self.cache._cache
        assert "test:key2" in self.cache._cache

    def test_get_stats(self):
        """Test getting cache statistics."""
        # Add some cached values by using the cache
        self.cache.get_or_create("test", "key1", lambda: "value1")
        self.cache.get_or_create("test", "key2", lambda: "value2")

        # Use cache to increment hit count
        self.cache.get_or_create("test", "key1", lambda: "value1")
        self.cache.get_or_create("test", "key2", lambda: "value2")

        stats = self.cache.get_stats()

        assert stats["cache_size"] == 2
        assert stats["lock_count"] == 2  # Locks created during get_or_create
        assert stats["hit_count"] == 2  # Two cache hits from second calls
        assert stats["miss_count"] == 2  # Two misses from first calls
        assert stats["hit_rate"] == 0.5  # 2/4

    def test_concurrent_access_different_keys(self):
        """Test concurrent access to different keys doesn't block."""
        import threading

        results = []
        errors = []

        def worker(key):
            try:

                def factory():
                    time.sleep(0.01)
                    return f"value_{key}"

                result = self.cache.get_or_create("test", key, factory)
                results.append((key, result))
            except Exception as e:
                errors.append(e)

        # Create threads for different keys
        threads = [threading.Thread(target=worker, args=(f"key{i}",)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 5
        assert len(errors) == 0
        # Each key should have been created
        assert set(k for k, v in results) == {f"key{i}" for i in range(5)}


class TestVerticalLoadSpan:
    """Test suite for VerticalLoadSpan dataclass."""

    def test_create_span(self):
        """Test creating a load span."""
        span = VerticalLoadSpan(
            vertical_name="test",
            operation="load",
            start_time_ns=time.time_ns(),
        )

        assert span.vertical_name == "test"
        assert span.operation == "load"
        assert span.status == "pending"
        assert span.end_time_ns is None

    def test_span_duration(self):
        """Test span duration calculation."""
        start = time.time_ns()
        span = VerticalLoadSpan(
            vertical_name="test",
            operation="load",
            start_time_ns=start,
            end_time_ns=start + 100_000_000,  # 100ms
        )

        assert span.duration_ms == 100.0

    def test_span_duration_not_ended(self):
        """Test span duration when not ended."""
        span = VerticalLoadSpan(
            vertical_name="test",
            operation="load",
            start_time_ns=time.time_ns(),
        )

        assert span.duration_ms is None

    def test_span_is_success(self):
        """Test is_success property."""
        span = VerticalLoadSpan(
            vertical_name="test",
            operation="load",
            start_time_ns=time.time_ns(),
            status="success",
        )

        assert span.is_success is True
        assert span.is_error is False

    def test_span_is_error(self):
        """Test is_error property."""
        span = VerticalLoadSpan(
            vertical_name="test",
            operation="load",
            start_time_ns=time.time_ns(),
            status="error",
            error="Failed",
        )

        assert span.is_error is True
        assert span.is_success is False


class TestVerticalLoadSpanContextManager:
    """Test suite for vertical_load_span context manager."""

    def test_context_manager_basic(self):
        """Test basic context manager usage."""
        emitted_spans = []

        def emit_callback(span):
            emitted_spans.append(span)

        with vertical_load_span("test", "load", emit_callback) as span:
            span.status = "success"

        assert len(emitted_spans) == 1
        assert emitted_spans[0].vertical_name == "test"
        assert emitted_spans[0].operation == "load"
        assert emitted_spans[0].status == "success"
        assert emitted_spans[0].duration_ms is not None

    def test_context_manager_with_error(self):
        """Test context manager with exception."""
        emitted_spans = []

        def emit_callback(span):
            emitted_spans.append(span)

        try:
            with vertical_load_span("test", "load", emit_callback) as span:
                raise ValueError("Test error")
        except ValueError:
            pass

        assert len(emitted_spans) == 1
        # Span should record error status
        assert emitted_spans[0].status == "pending"  # Not automatically set on error
        assert emitted_spans[0].duration_ms is not None


class TestVerticalLoadTelemetry:
    """Test suite for VerticalLoadTelemetry class."""

    def setup_method(self):
        """Create fresh telemetry for each test."""
        self.telemetry = VerticalLoadTelemetry()

    def test_track_load(self):
        """Test tracking a load operation."""
        with self.telemetry.track_load("coding", "load") as span:
            span.status = "success"

        spans = self.telemetry._spans
        assert len(spans) == 1
        assert spans[0].vertical_name == "coding"
        assert spans[0].operation == "load"
        assert spans[0].is_success

    def test_track_multiple_loads(self):
        """Test tracking multiple operations."""
        with self.telemetry.track_load("coding", "discover"):
            pass

        with self.telemetry.track_load("devops", "load") as span:
            span.status = "success"

        with self.telemetry.track_load("coding", "activate") as span:
            span.status = "success"

        assert len(self.telemetry._spans) == 3

    def test_track_extension_load(self):
        """Test tracking extension load."""
        self.telemetry.track_extension_load("coding", "middleware", True, 150.0)

        spans = self.telemetry._spans
        assert len(spans) == 1
        assert spans[0].vertical_name == "coding"
        assert spans[0].operation == "load_extension"
        assert spans[0].metadata["extension_type"] == "middleware"

    def test_get_metrics_empty(self):
        """Test getting metrics when no spans recorded."""
        metrics = self.telemetry.get_metrics()
        assert metrics == {}

    def test_get_metrics_with_spans(self):
        """Test getting metrics with recorded spans."""
        with self.telemetry.track_load("coding", "load") as span:
            # Simulate a 50ms operation
            time.sleep(0.05)
            span.status = "success"

        with self.telemetry.track_load("coding", "load") as span:
            # Simulate a 100ms operation
            time.sleep(0.1)
            span.status = "error"

        metrics = self.telemetry.get_metrics()

        assert metrics["total_operations"] == 2
        assert metrics["successful"] == 1
        assert metrics["errors"] == 1
        assert metrics["success_rate"] == 0.5
        assert metrics["avg_duration_ms"] > 50  # Should be around 75ms
        assert metrics["avg_duration_ms"] < 200

    def test_get_slow_operations(self):
        """Test getting slow operations."""
        with self.telemetry.track_load("coding", "load") as span:
            # Simulate a 600ms operation
            time.sleep(0.6)
            span.status = "success"

        slow_ops = self.telemetry.get_slow_operations(threshold_ms=500.0)

        assert len(slow_ops) == 1
        assert slow_ops[0]["vertical"] == "coding"
        assert slow_ops[0]["duration_ms"] > 500.0

    def test_clear(self):
        """Test clearing telemetry."""
        with self.telemetry.track_load("test", "load"):
            pass

        assert len(self.telemetry._spans) == 1

        self.telemetry.clear()

        assert len(self.telemetry._spans) == 0


class TestGlobalTelemetry:
    """Test suite for global telemetry instance."""

    def test_get_telemetry_singleton(self):
        """Test that get_telemetry returns singleton."""
        telemetry1 = get_telemetry()
        telemetry2 = get_telemetry()

        assert telemetry1 is telemetry2
        assert isinstance(telemetry1, VerticalLoadTelemetry)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
