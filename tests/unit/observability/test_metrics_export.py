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

"""Tests for Resilience Metrics Dashboard Export.

Tests cover:
- Metric collection from circuit breakers
- Metric collection from health checkers
- Metric collection from resilient providers
- Export formats (JSON, Prometheus, Summary)
- Report generation
"""

import json
import pytest
from unittest.mock import MagicMock
from datetime import datetime

from victor.providers.metrics_export import (
    ResilienceMetricsExporter,
    CircuitBreakerMetrics,
    HealthMetrics,
    ResilienceMetrics,
    DashboardReport,
    get_metrics_exporter,
    reset_metrics_exporter,
)


class TestCircuitBreakerMetrics:
    """Tests for CircuitBreakerMetrics dataclass."""

    def test_default_values(self):
        """Test default values for circuit breaker metrics."""
        metrics = CircuitBreakerMetrics(name="test", state="closed")

        assert metrics.failure_count == 0
        assert metrics.error_rate == 0.0
        assert metrics.availability == 100.0

    def test_custom_values(self):
        """Test custom values."""
        metrics = CircuitBreakerMetrics(
            name="test",
            state="open",
            failure_count=5,
            total_calls=100,
            error_rate=5.0,
        )

        assert metrics.name == "test"
        assert metrics.state == "open"
        assert metrics.failure_count == 5


class TestHealthMetrics:
    """Tests for HealthMetrics dataclass."""

    def test_default_values(self):
        """Test default values for health metrics."""
        metrics = HealthMetrics(provider_name="test", status="healthy")

        assert metrics.latency_ms == 0.0
        assert metrics.uptime_percent == 100.0

    def test_custom_values(self):
        """Test custom values."""
        metrics = HealthMetrics(
            provider_name="anthropic",
            status="degraded",
            latency_ms=5500.0,
            uptime_percent=95.5,
        )

        assert metrics.provider_name == "anthropic"
        assert metrics.status == "degraded"
        assert metrics.latency_ms == 5500.0


class TestResilienceMetrics:
    """Tests for ResilienceMetrics dataclass."""

    def test_default_values(self):
        """Test default values for resilience metrics."""
        metrics = ResilienceMetrics(provider_name="test")

        assert metrics.total_requests == 0
        assert metrics.success_rate == 0.0

    def test_custom_values(self):
        """Test custom values."""
        metrics = ResilienceMetrics(
            provider_name="main",
            total_requests=1000,
            primary_successes=950,
            fallback_successes=40,
            total_failures=10,
            success_rate=99.0,
        )

        assert metrics.total_requests == 1000
        assert metrics.success_rate == 99.0


class TestDashboardReport:
    """Tests for DashboardReport dataclass."""

    def test_empty_report(self):
        """Test creating an empty report."""
        report = DashboardReport()

        assert len(report.circuit_breakers) == 0
        assert len(report.health) == 0
        assert len(report.resilience) == 0
        assert "timestamp" in report.to_dict()

    def test_to_dict(self):
        """Test converting report to dictionary."""
        report = DashboardReport(
            circuit_breakers={"test": CircuitBreakerMetrics(name="test", state="closed")}
        )

        d = report.to_dict()

        assert "circuit_breakers" in d
        assert "test" in d["circuit_breakers"]
        assert d["circuit_breakers"]["test"]["state"] == "closed"

    def test_to_json(self):
        """Test exporting report as JSON."""
        report = DashboardReport()

        json_str = report.to_json()
        parsed = json.loads(json_str)

        assert "timestamp" in parsed
        assert "circuit_breakers" in parsed


class TestResilienceMetricsExporter:
    """Tests for ResilienceMetricsExporter class."""

    @pytest.fixture
    def exporter(self):
        """Create a fresh exporter."""
        reset_metrics_exporter()
        return ResilienceMetricsExporter()

    @pytest.fixture
    def mock_circuit_breaker(self):
        """Create a mock circuit breaker."""
        cb = MagicMock()
        cb.get_stats.return_value = {
            "state": "closed",
            "failure_count": 2,
            "success_count": 98,
            "total_calls": 100,
            "total_failures": 2,
            "total_rejected": 0,
            "last_failure_time": None,
            "state_changes": 0,
        }
        return cb

    @pytest.fixture
    def mock_health_checker(self):
        """Create a mock health checker."""
        checker = MagicMock()

        # Mock health check result
        result = MagicMock()
        result.status.value = "healthy"
        result.latency_ms = 150.0
        result.timestamp = datetime.now()

        checker._latest = {"anthropic": result}
        checker.get_stats.return_value = {"total_checks": 10}
        checker.get_provider_history.return_value = [result] * 5
        checker.calculate_uptime.return_value = 99.5

        return checker

    @pytest.fixture
    def mock_resilient_provider(self):
        """Create a mock resilient provider."""
        provider = MagicMock()
        provider.get_resilience_stats.return_value = {
            "total_requests": 500,
            "primary_successes": 480,
            "fallback_successes": 15,
            "total_failures": 5,
            "retry_attempts": 10,
        }
        return provider

    def test_register_circuit_breaker(self, exporter, mock_circuit_breaker):
        """Test registering a circuit breaker."""
        exporter.register_circuit_breaker("test", mock_circuit_breaker)

        assert "test" in exporter._circuit_breakers

    def test_register_health_checker(self, exporter, mock_health_checker):
        """Test registering a health checker."""
        exporter.register_health_checker(mock_health_checker)

        assert exporter._health_checker is mock_health_checker

    def test_register_resilient_provider(self, exporter, mock_resilient_provider):
        """Test registering a resilient provider."""
        exporter.register_resilient_provider("main", mock_resilient_provider)

        assert "main" in exporter._resilient_providers

    def test_add_custom_metric(self, exporter):
        """Test adding custom metrics."""
        exporter.add_custom_metric("custom_value", 42)

        assert exporter._custom_metrics["custom_value"] == 42

    def test_collect_circuit_breaker_metrics(self, exporter, mock_circuit_breaker):
        """Test collecting circuit breaker metrics."""
        exporter.register_circuit_breaker("test", mock_circuit_breaker)

        metrics = exporter._collect_circuit_breaker_metrics()

        assert "test" in metrics
        assert metrics["test"].state == "closed"
        assert metrics["test"].total_calls == 100

    def test_collect_circuit_breaker_error_rate(self, exporter, mock_circuit_breaker):
        """Test that error rate is calculated correctly."""
        exporter.register_circuit_breaker("test", mock_circuit_breaker)

        metrics = exporter._collect_circuit_breaker_metrics()

        # 2 failures out of 100 calls = 2%
        assert metrics["test"].error_rate == 2.0

    def test_collect_health_metrics(self, exporter, mock_health_checker):
        """Test collecting health metrics."""
        exporter.register_health_checker(mock_health_checker)

        metrics = exporter._collect_health_metrics()

        assert "anthropic" in metrics
        assert metrics["anthropic"].status == "healthy"
        assert metrics["anthropic"].latency_ms == 150.0

    def test_collect_health_metrics_empty(self, exporter):
        """Test collecting health metrics without checker."""
        metrics = exporter._collect_health_metrics()

        assert len(metrics) == 0

    def test_collect_resilience_metrics(self, exporter, mock_resilient_provider):
        """Test collecting resilience metrics."""
        exporter.register_resilient_provider("main", mock_resilient_provider)

        metrics = exporter._collect_resilience_metrics()

        assert "main" in metrics
        assert metrics["main"].total_requests == 500
        assert metrics["main"].primary_successes == 480

    def test_generate_report(
        self,
        exporter,
        mock_circuit_breaker,
        mock_health_checker,
        mock_resilient_provider,
    ):
        """Test generating complete report."""
        exporter.register_circuit_breaker("cb_test", mock_circuit_breaker)
        exporter.register_health_checker(mock_health_checker)
        exporter.register_resilient_provider("main", mock_resilient_provider)

        report = exporter.generate_report()

        assert "cb_test" in report.circuit_breakers
        assert "anthropic" in report.health
        assert "main" in report.resilience
        assert "timestamp" in report.to_dict()

    def test_generate_summary(
        self,
        exporter,
        mock_circuit_breaker,
        mock_health_checker,
        mock_resilient_provider,
    ):
        """Test summary generation."""
        exporter.register_circuit_breaker("cb_test", mock_circuit_breaker)
        exporter.register_health_checker(mock_health_checker)
        exporter.register_resilient_provider("main", mock_resilient_provider)

        report = exporter.generate_report()

        assert "circuit_breakers" in report.summary
        assert "health" in report.summary
        assert "resilience" in report.summary

    def test_export_json(self, exporter, mock_circuit_breaker):
        """Test JSON export."""
        exporter.register_circuit_breaker("test", mock_circuit_breaker)

        json_str = exporter.export_json()
        parsed = json.loads(json_str)

        assert "circuit_breakers" in parsed
        assert "test" in parsed["circuit_breakers"]

    def test_export_prometheus(self, exporter, mock_circuit_breaker):
        """Test Prometheus format export."""
        exporter.register_circuit_breaker("test", mock_circuit_breaker)

        prometheus = exporter.export_prometheus()

        assert "victor_circuit_breaker_test_state" in prometheus
        assert "victor_circuit_breaker_test_total_calls" in prometheus

    def test_export_prometheus_with_health(self, exporter, mock_health_checker):
        """Test Prometheus export with health metrics."""
        exporter.register_health_checker(mock_health_checker)

        prometheus = exporter.export_prometheus()

        assert "victor_health_anthropic_status" in prometheus
        assert "victor_health_anthropic_latency_ms" in prometheus

    def test_get_summary(self, exporter, mock_circuit_breaker):
        """Test human-readable summary."""
        exporter.register_circuit_breaker("test", mock_circuit_breaker)

        summary = exporter.get_summary()

        assert "Resilience Metrics Summary" in summary
        assert "Circuit Breakers:" in summary

    def test_summary_shows_unhealthy(self, exporter):
        """Test that summary shows unhealthy providers."""
        # Create unhealthy health checker
        checker = MagicMock()
        result = MagicMock()
        result.status.value = "unhealthy"
        result.latency_ms = 20000.0
        result.timestamp = datetime.now()

        checker._latest = {"bad_provider": result}
        checker.get_stats.return_value = {}
        checker.get_provider_history.return_value = []
        checker.calculate_uptime.return_value = 50.0

        exporter.register_health_checker(checker)

        summary = exporter.get_summary()

        assert "Unhealthy Providers:" in summary
        assert "bad_provider" in summary


class TestGlobalMetricsExporter:
    """Tests for global metrics exporter singleton."""

    def test_get_metrics_exporter_singleton(self):
        """Test that get_metrics_exporter returns singleton."""
        reset_metrics_exporter()

        exporter1 = get_metrics_exporter()
        exporter2 = get_metrics_exporter()

        assert exporter1 is exporter2

    def test_reset_metrics_exporter(self):
        """Test resetting the global exporter."""
        exporter1 = get_metrics_exporter()
        reset_metrics_exporter()
        exporter2 = get_metrics_exporter()

        assert exporter1 is not exporter2


class TestEdgeCases:
    """Tests for edge cases."""

    @pytest.fixture
    def exporter(self):
        reset_metrics_exporter()
        return ResilienceMetricsExporter()

    def test_empty_report_generates_valid_json(self, exporter):
        """Test that empty report generates valid JSON."""
        json_str = exporter.export_json()
        parsed = json.loads(json_str)

        assert parsed is not None
        assert "summary" in parsed

    def test_empty_report_generates_valid_prometheus(self, exporter):
        """Test that empty report generates valid Prometheus output."""
        prometheus = exporter.export_prometheus()

        # Should be empty string for no metrics
        assert isinstance(prometheus, str)

    def test_circuit_breaker_without_stats(self, exporter):
        """Test handling circuit breaker without get_stats."""
        cb = MagicMock()
        del cb.get_stats  # Remove get_stats method

        exporter.register_circuit_breaker("test", cb)
        metrics = exporter._collect_circuit_breaker_metrics()

        # Should handle gracefully
        assert "test" in metrics

    def test_zero_division_in_error_rate(self, exporter):
        """Test handling zero total calls in error rate calculation."""
        cb = MagicMock()
        cb.get_stats.return_value = {
            "state": "closed",
            "total_calls": 0,
            "total_failures": 0,
            "total_rejected": 0,
        }

        exporter.register_circuit_breaker("test", cb)
        metrics = exporter._collect_circuit_breaker_metrics()

        assert metrics["test"].error_rate == 0.0
        assert metrics["test"].availability == 100.0

    def test_resilient_provider_fallback_to_stats(self, exporter):
        """Test resilient provider metrics with _stats attribute."""
        provider = MagicMock(spec=[])  # Empty spec to avoid auto-creating methods
        provider._stats = {
            "total_requests": 100,
            "primary_successes": 90,
            "fallback_successes": 5,
            "total_failures": 5,
        }

        exporter.register_resilient_provider("test", provider)
        metrics = exporter._collect_resilience_metrics()

        assert "test" in metrics
        assert metrics["test"].total_requests == 100
