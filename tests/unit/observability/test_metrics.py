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

"""Tests for metrics collection module."""

import pytest
import time

from victor.observability.metrics import (
    Counter,
    Gauge,
    Histogram,
    MetricsCollector,
    MetricsRegistry,
    Timer,
)

# =============================================================================
# Counter Tests
# =============================================================================


class TestCounter:
    """Tests for Counter metric."""

    def test_initial_value_zero(self):
        """Counter should start at zero."""
        counter = Counter("test_counter", "Test counter")
        assert counter.value == 0

    def test_increment(self):
        """Counter should increment."""
        counter = Counter("test_counter", "Test counter")
        counter.increment()
        assert counter.value == 1

    def test_increment_by_amount(self):
        """Counter should increment by specified amount."""
        counter = Counter("test_counter", "Test counter")
        counter.increment(5)
        assert counter.value == 5

    def test_increment_negative_raises(self):
        """Counter should reject negative increment."""
        counter = Counter("test_counter", "Test counter")
        with pytest.raises(ValueError):
            counter.increment(-1)

    def test_reset(self):
        """Counter should reset to zero."""
        counter = Counter("test_counter", "Test counter")
        counter.increment(10)
        counter.reset()
        assert counter.value == 0

    def test_collect(self):
        """Counter should return correct dict."""
        counter = Counter("test_counter", "Test description")
        counter.increment(5)

        data = counter.collect()

        assert data["type"] == "counter"
        assert data["name"] == "test_counter"
        assert data["description"] == "Test description"
        assert data["value"] == 5


# =============================================================================
# Gauge Tests
# =============================================================================


class TestGauge:
    """Tests for Gauge metric."""

    def test_set_value(self):
        """Gauge should set value."""
        gauge = Gauge("test_gauge", "Test gauge")
        gauge.set(42)
        assert gauge.value == 42

    def test_increment(self):
        """Gauge should increment."""
        gauge = Gauge("test_gauge", "Test gauge")
        gauge.set(10)
        gauge.increment(5)
        assert gauge.value == 15

    def test_decrement(self):
        """Gauge should decrement."""
        gauge = Gauge("test_gauge", "Test gauge")
        gauge.set(10)
        gauge.decrement(3)
        assert gauge.value == 7

    def test_reset(self):
        """Gauge should reset to zero."""
        gauge = Gauge("test_gauge", "Test gauge")
        gauge.set(100)
        gauge.reset()
        assert gauge.value == 0

    def test_collect(self):
        """Gauge should return correct dict."""
        gauge = Gauge("test_gauge", "Test description")
        gauge.set(42)

        data = gauge.collect()

        assert data["type"] == "gauge"
        assert data["name"] == "test_gauge"
        assert data["value"] == 42


# =============================================================================
# Histogram Tests
# =============================================================================


class TestHistogram:
    """Tests for Histogram metric."""

    def test_observe(self):
        """Histogram should record observations."""
        histogram = Histogram("test_histogram", "Test histogram")
        histogram.observe(10)
        histogram.observe(20)
        histogram.observe(30)

        assert histogram.count == 3
        assert histogram.sum == 60

    def test_mean(self):
        """Histogram should calculate mean."""
        histogram = Histogram("test_histogram", "Test histogram")
        histogram.observe(10)
        histogram.observe(20)
        histogram.observe(30)

        assert histogram.mean == 20

    def test_mean_empty(self):
        """Histogram mean should be None when empty."""
        histogram = Histogram("test_histogram", "Test histogram")
        assert histogram.mean is None

    def test_percentiles(self):
        """Histogram should calculate percentiles."""
        histogram = Histogram("test_histogram", "Test histogram")
        for i in range(100):
            histogram.observe(i)

        p50 = histogram.percentile(50)
        assert p50 is not None
        assert 45 <= p50 <= 55  # Approximately median

    def test_custom_buckets(self):
        """Histogram should use custom buckets."""
        histogram = Histogram("test_histogram", "Test histogram", buckets=(10, 50, 100))
        histogram.observe(25)
        histogram.observe(75)

        data = histogram.collect()
        assert 10 in data["buckets"]
        assert 50 in data["buckets"]
        assert 100 in data["buckets"]

    def test_reset(self):
        """Histogram should reset all values."""
        histogram = Histogram("test_histogram", "Test histogram")
        histogram.observe(10)
        histogram.observe(20)
        histogram.reset()

        assert histogram.count == 0
        assert histogram.sum == 0

    def test_collect(self):
        """Histogram should return complete data."""
        histogram = Histogram("test_histogram", "Test description")
        histogram.observe(10)
        histogram.observe(90)

        data = histogram.collect()

        assert data["type"] == "histogram"
        assert data["name"] == "test_histogram"
        assert data["count"] == 2
        assert data["sum"] == 100
        assert "buckets" in data
        assert "p50" in data
        assert "p95" in data


# =============================================================================
# Timer Tests
# =============================================================================


class TestTimer:
    """Tests for Timer metric."""

    def test_context_manager(self):
        """Timer context manager should record duration."""
        timer = Timer("test_timer", "Test timer")

        with timer.time():
            time.sleep(0.01)

        assert timer.count == 1
        assert timer.sum > 0  # Should have recorded something

    def test_decorator(self):
        """Timer decorator should record duration."""
        timer = Timer("test_timer", "Test timer")

        @timer.timed
        def slow_func():
            time.sleep(0.01)
            return "done"

        result = slow_func()

        assert result == "done"
        assert timer.count == 1


# =============================================================================
# MetricsRegistry Tests
# =============================================================================


class TestMetricsRegistry:
    """Tests for MetricsRegistry."""

    def test_singleton(self):
        """Registry should be singleton."""
        r1 = MetricsRegistry.get_instance()
        r2 = MetricsRegistry.get_instance()

        assert r1 is r2

    def test_create_counter(self):
        """Registry should create counter."""
        registry = MetricsRegistry()
        counter = registry.counter("test_counter", "Test")

        assert isinstance(counter, Counter)
        assert counter.name == "test_counter"

    def test_create_gauge(self):
        """Registry should create gauge."""
        registry = MetricsRegistry()
        gauge = registry.gauge("test_gauge", "Test")

        assert isinstance(gauge, Gauge)

    def test_create_histogram(self):
        """Registry should create histogram."""
        registry = MetricsRegistry()
        histogram = registry.histogram("test_histogram", "Test")

        assert isinstance(histogram, Histogram)

    def test_create_timer(self):
        """Registry should create timer."""
        registry = MetricsRegistry()
        timer = registry.timer("test_timer", "Test")

        assert isinstance(timer, Timer)

    def test_get_existing_metric(self):
        """Registry should return existing metric."""
        registry = MetricsRegistry()
        counter1 = registry.counter("same_name", "Test")
        counter2 = registry.counter("same_name", "Test")

        assert counter1 is counter2

    def test_get_metric(self):
        """Registry should get metric by name."""
        registry = MetricsRegistry()
        counter = registry.counter("lookup_test", "Test")

        found = registry.get("lookup_test")

        assert found is counter

    def test_get_nonexistent(self):
        """Registry should return None for nonexistent."""
        registry = MetricsRegistry()
        found = registry.get("does_not_exist")

        assert found is None

    def test_unregister(self):
        """Registry should unregister metric."""
        registry = MetricsRegistry()
        registry.counter("to_remove", "Test")

        assert registry.unregister("to_remove")
        assert registry.get("to_remove") is None

    def test_collect_all(self):
        """Registry should collect all metrics."""
        registry = MetricsRegistry()
        registry.clear()  # Start fresh

        registry.counter("c1", "Counter 1").increment()
        registry.gauge("g1", "Gauge 1").set(10)

        data = registry.collect()

        assert len(data) == 2
        assert any(d["name"] == "c1" for d in data)
        assert any(d["name"] == "g1" for d in data)

    def test_reset_all(self):
        """Registry should reset all metrics."""
        registry = MetricsRegistry()
        registry.clear()

        counter = registry.counter("c1", "Counter 1")
        counter.increment(10)
        gauge = registry.gauge("g1", "Gauge 1")
        gauge.set(20)

        registry.reset_all()

        assert counter.value == 0
        assert gauge.value == 0

    def test_clear(self):
        """Registry should clear all metrics."""
        registry = MetricsRegistry()
        registry.counter("c1", "Counter 1")
        registry.counter("c2", "Counter 2")

        registry.clear()

        assert registry.metric_count == 0

    def test_metrics_with_labels(self):
        """Registry should handle metrics with labels."""
        registry = MetricsRegistry()
        registry.clear()

        c1 = registry.counter("http_requests", "Requests", {"method": "GET"})
        c2 = registry.counter("http_requests", "Requests", {"method": "POST"})

        c1.increment()
        c2.increment(2)

        assert c1.value == 1
        assert c2.value == 2
        assert c1 is not c2


# =============================================================================
# MetricsCollector Tests
# =============================================================================


class TestMetricsCollector:
    """Tests for MetricsCollector."""

    def test_creates_standard_metrics(self):
        """Collector should create standard metrics."""
        registry = MetricsRegistry()
        registry.clear()

        collector = MetricsCollector(registry=registry, prefix="test")

        assert collector.tool_calls is not None
        assert collector.model_requests is not None
        assert collector.errors is not None

    def test_get_summary(self):
        """Collector should return summary."""
        registry = MetricsRegistry()
        registry.clear()

        collector = MetricsCollector(registry=registry, prefix="test")
        collector.tool_calls.increment(5)
        collector.errors.increment(1)

        summary = collector.get_summary()

        assert summary["tool_calls"] == 5
        assert summary["errors"] == 1
