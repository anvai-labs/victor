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

"""Resilience Metrics Dashboard Export.

Aggregates and exports metrics from:
- Circuit breakers (state, trip counts, recovery time)
- Health checks (uptime, latency, status history)
- Retry statistics (attempts, success rates)
- Provider performance (request counts, error rates)

Supports multiple export formats:
- JSON (for web dashboards)
- Prometheus (for monitoring systems)
- Summary (for logs/CLI)

Usage:
    from victor.providers.metrics_export import ResilienceMetricsExporter

    # Create exporter
    exporter = ResilienceMetricsExporter()

    # Register components
    exporter.register_circuit_breaker("anthropic", cb)
    exporter.register_health_checker(health_checker)
    exporter.register_resilient_provider("main", provider)

    # Export to JSON
    report = exporter.export_json()

    # Export for Prometheus
    metrics = exporter.export_prometheus()

    # Get summary
    summary = exporter.get_summary()
"""

from victor.core.json_utils import json_dumps
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional

from victor.core.circuit_breaker import CircuitState
from victor.core.health import HealthStatus

logger = logging.getLogger(__name__)


class ExportFormat(Enum):
    """Supported export formats."""

    JSON = "json"
    PROMETHEUS = "prometheus"
    SUMMARY = "summary"


@dataclass
class CircuitBreakerMetrics:
    """Aggregated metrics for a circuit breaker."""

    name: str
    state: str
    failure_count: int = 0
    success_count: int = 0
    total_calls: int = 0
    total_failures: int = 0
    total_rejected: int = 0
    last_failure_time: Optional[str] = None
    state_changes: int = 0
    error_rate: float = 0.0
    availability: float = 100.0


@dataclass
class HealthMetrics:
    """Aggregated health metrics for a provider."""

    provider_name: str
    status: str
    latency_ms: float = 0.0
    uptime_percent: float = 100.0
    check_count: int = 0
    healthy_count: int = 0
    degraded_count: int = 0
    unhealthy_count: int = 0
    last_check_time: Optional[str] = None


@dataclass
class ResilienceMetrics:
    """Aggregated metrics for a resilient provider."""

    provider_name: str
    total_requests: int = 0
    primary_successes: int = 0
    fallback_successes: int = 0
    total_failures: int = 0
    retry_attempts: int = 0
    success_rate: float = 0.0
    fallback_rate: float = 0.0


@dataclass
class DashboardReport:
    """Complete dashboard report with all metrics.

    Attributes:
        timestamp: Report generation time
        circuit_breakers: Circuit breaker metrics by name
        health: Health metrics by provider
        resilience: Resilience metrics by provider
        summary: High-level summary stats
    """

    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    circuit_breakers: Dict[str, CircuitBreakerMetrics] = field(default_factory=dict)
    health: Dict[str, HealthMetrics] = field(default_factory=dict)
    resilience: Dict[str, ResilienceMetrics] = field(default_factory=dict)
    summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp,
            "circuit_breakers": {name: vars(cb) for name, cb in self.circuit_breakers.items()},
            "health": {name: vars(h) for name, h in self.health.items()},
            "resilience": {name: vars(r) for name, r in self.resilience.items()},
            "summary": self.summary,
        }

    def to_json(self, indent: int = 2) -> str:
        """Export as JSON string."""
        return json_dumps(self.to_dict(), indent=indent)


class ResilienceMetricsExporter:
    """Exports resilience metrics for dashboards.

    Aggregates data from circuit breakers, health checkers, and
    resilient providers into a unified format suitable for monitoring
    dashboards.

    Usage:
        exporter = ResilienceMetricsExporter()

        # Register components
        exporter.register_circuit_breaker("anthropic", anthropic_cb)
        exporter.register_health_checker(health_checker)
        exporter.register_resilient_provider("main", resilient_provider)

        # Generate report
        report = exporter.generate_report()

        # Export in different formats
        json_data = report.to_json()
        prometheus = exporter.export_prometheus()
    """

    def __init__(self):
        """Initialize metrics exporter."""
        self._circuit_breakers: Dict[str, Any] = {}
        self._health_checker: Optional[Any] = None
        self._resilient_providers: Dict[str, Any] = {}
        self._custom_metrics: Dict[str, Any] = {}

        logger.debug("ResilienceMetricsExporter initialized")

    def register_circuit_breaker(self, name: str, circuit_breaker: Any) -> None:
        """Register a circuit breaker for metrics export.

        Args:
            name: Identifier for this circuit breaker
            circuit_breaker: CircuitBreaker instance
        """
        self._circuit_breakers[name] = circuit_breaker
        logger.debug(f"Registered circuit breaker: {name}")

    def register_health_checker(self, health_checker: Any) -> None:
        """Register a health checker for metrics export.

        Args:
            health_checker: ProviderHealthChecker instance
        """
        self._health_checker = health_checker
        logger.debug("Registered health checker")

    def register_resilient_provider(self, name: str, provider: Any) -> None:
        """Register a resilient provider for metrics export.

        Args:
            name: Identifier for this provider
            provider: ResilientProvider or ManagedProvider instance
        """
        self._resilient_providers[name] = provider
        logger.debug(f"Registered resilient provider: {name}")

    def add_custom_metric(self, name: str, value: Any) -> None:
        """Add a custom metric to the export.

        Args:
            name: Metric name
            value: Metric value (should be JSON-serializable)
        """
        self._custom_metrics[name] = value

    def _collect_circuit_breaker_metrics(self) -> Dict[str, CircuitBreakerMetrics]:
        """Collect metrics from all registered circuit breakers."""
        metrics = {}

        for name, cb in self._circuit_breakers.items():
            try:
                stats = cb.get_stats() if hasattr(cb, "get_stats") else {}

                total_calls = stats.get("total_calls", 0)
                total_failures = stats.get("total_failures", 0)
                total_rejected = stats.get("total_rejected", 0)

                # Calculate error rate
                error_rate = (total_failures / total_calls * 100) if total_calls > 0 else 0.0

                # Calculate availability (excluding rejected requests)
                successful = total_calls - total_failures - total_rejected
                availability = (successful / total_calls * 100) if total_calls > 0 else 100.0

                metrics[name] = CircuitBreakerMetrics(
                    name=name,
                    state=stats.get("state", "unknown"),
                    failure_count=stats.get("failure_count", 0),
                    success_count=stats.get("success_count", 0),
                    total_calls=total_calls,
                    total_failures=total_failures,
                    total_rejected=total_rejected,
                    last_failure_time=stats.get("last_failure_time"),
                    state_changes=stats.get("state_changes", 0),
                    error_rate=round(error_rate, 2),
                    availability=round(availability, 2),
                )
            except Exception as e:
                logger.warning(f"Failed to collect metrics for circuit breaker {name}: {e}")

        return metrics

    def _collect_health_metrics(self) -> Dict[str, HealthMetrics]:
        """Collect metrics from health checker."""
        metrics = {}

        if not self._health_checker:
            return metrics

        try:
            _stats = (
                self._health_checker.get_stats()
                if hasattr(self._health_checker, "get_stats")
                else {}
            )

            # Get latest results for each provider
            latest_results = getattr(self._health_checker, "_latest", {})

            for provider_name, result in latest_results.items():
                # Get history counts
                history = self._health_checker.get_provider_history(provider_name)
                healthy_count = sum(1 for h in history if h.status.value == "healthy")
                degraded_count = sum(1 for h in history if h.status.value == "degraded")
                unhealthy_count = sum(1 for h in history if h.status.value == "unhealthy")

                # Get uptime
                uptime = self._health_checker.calculate_uptime(provider_name)

                metrics[provider_name] = HealthMetrics(
                    provider_name=provider_name,
                    status=result.status.value,
                    latency_ms=round(result.latency_ms, 2),
                    uptime_percent=round(uptime, 2),
                    check_count=len(history),
                    healthy_count=healthy_count,
                    degraded_count=degraded_count,
                    unhealthy_count=unhealthy_count,
                    last_check_time=result.timestamp.isoformat(),
                )
        except Exception as e:
            logger.warning(f"Failed to collect health metrics: {e}")

        return metrics

    def _collect_resilience_metrics(self) -> Dict[str, ResilienceMetrics]:
        """Collect metrics from resilient providers."""
        metrics = {}

        for name, provider in self._resilient_providers.items():
            try:
                # Try different methods to get stats
                if hasattr(provider, "get_resilience_stats"):
                    stats = provider.get_resilience_stats() or {}
                elif hasattr(provider, "get_stats"):
                    stats = provider.get_stats() or {}
                elif hasattr(provider, "_stats"):
                    stats = provider._stats or {}
                else:
                    stats = {}

                total_requests = stats.get("total_requests", 0)
                primary_successes = stats.get("primary_successes", 0)
                fallback_successes = stats.get("fallback_successes", 0)
                total_failures = stats.get("total_failures", 0)

                # Calculate rates
                total_successes = primary_successes + fallback_successes
                success_rate = (
                    (total_successes / total_requests * 100) if total_requests > 0 else 0.0
                )
                fallback_rate = (
                    (fallback_successes / total_successes * 100) if total_successes > 0 else 0.0
                )

                metrics[name] = ResilienceMetrics(
                    provider_name=name,
                    total_requests=total_requests,
                    primary_successes=primary_successes,
                    fallback_successes=fallback_successes,
                    total_failures=total_failures,
                    retry_attempts=stats.get("retry_attempts", 0),
                    success_rate=round(success_rate, 2),
                    fallback_rate=round(fallback_rate, 2),
                )
            except Exception as e:
                logger.warning(f"Failed to collect resilience metrics for {name}: {e}")

        return metrics

    def _generate_summary(
        self,
        cb_metrics: Dict[str, CircuitBreakerMetrics],
        health_metrics: Dict[str, HealthMetrics],
        resilience_metrics: Dict[str, ResilienceMetrics],
    ) -> Dict[str, Any]:
        """Generate summary statistics."""
        summary = {
            "total_providers": len(health_metrics) or len(resilience_metrics),
            "circuit_breakers": {
                "total": len(cb_metrics),
                "open": sum(1 for cb in cb_metrics.values() if cb.state == CircuitState.OPEN.value),
                "half_open": sum(
                    1 for cb in cb_metrics.values() if cb.state == CircuitState.HALF_OPEN.value
                ),
                "closed": sum(
                    1 for cb in cb_metrics.values() if cb.state == CircuitState.CLOSED.value
                ),
            },
            "health": {
                "healthy": sum(
                    1 for h in health_metrics.values() if h.status == HealthStatus.HEALTHY.value
                ),
                "degraded": sum(
                    1 for h in health_metrics.values() if h.status == HealthStatus.DEGRADED.value
                ),
                "unhealthy": sum(
                    1 for h in health_metrics.values() if h.status == HealthStatus.UNHEALTHY.value
                ),
                "average_latency_ms": round(
                    (
                        sum(h.latency_ms for h in health_metrics.values()) / len(health_metrics)
                        if health_metrics
                        else 0.0
                    ),
                    2,
                ),
            },
            "resilience": {
                "total_requests": sum(r.total_requests for r in resilience_metrics.values()),
                "total_failures": sum(r.total_failures for r in resilience_metrics.values()),
                "average_success_rate": round(
                    (
                        sum(r.success_rate for r in resilience_metrics.values())
                        / len(resilience_metrics)
                        if resilience_metrics
                        else 0.0
                    ),
                    2,
                ),
            },
            "custom_metrics": self._custom_metrics,
        }
        return summary

    def generate_report(self) -> DashboardReport:
        """Generate complete dashboard report.

        Returns:
            DashboardReport with all aggregated metrics
        """
        cb_metrics = self._collect_circuit_breaker_metrics()
        health_metrics = self._collect_health_metrics()
        resilience_metrics = self._collect_resilience_metrics()
        summary = self._generate_summary(cb_metrics, health_metrics, resilience_metrics)

        return DashboardReport(
            circuit_breakers=cb_metrics,
            health=health_metrics,
            resilience=resilience_metrics,
            summary=summary,
        )

    def export_json(self, indent: int = 2) -> str:
        """Export metrics as JSON string.

        Args:
            indent: JSON indentation level

        Returns:
            JSON string with all metrics
        """
        report = self.generate_report()
        return report.to_json(indent=indent)

    def export_prometheus(self) -> str:
        """Export metrics in Prometheus format.

        Returns:
            Prometheus-compatible metrics text
        """
        lines = []
        report = self.generate_report()

        # Circuit breaker metrics
        for name, cb in report.circuit_breakers.items():
            prefix = f"victor_circuit_breaker_{name.replace('-', '_')}"
            lines.append(
                f"# HELP {prefix}_state Circuit breaker state (0=closed, 1=half_open, 2=open)"
            )
            state_value = {"closed": 0, "half_open": 1, "open": 2}.get(cb.state, -1)
            lines.append(f"{prefix}_state {state_value}")
            lines.append(f"{prefix}_total_calls {cb.total_calls}")
            lines.append(f"{prefix}_total_failures {cb.total_failures}")
            lines.append(f"{prefix}_error_rate {cb.error_rate}")
            lines.append(f"{prefix}_availability {cb.availability}")

        # Health metrics
        for name, h in report.health.items():
            prefix = f"victor_health_{name.replace('-', '_')}"
            status_value = {"healthy": 0, "degraded": 1, "unhealthy": 2}.get(h.status, -1)
            lines.append(f"{prefix}_status {status_value}")
            lines.append(f"{prefix}_latency_ms {h.latency_ms}")
            lines.append(f"{prefix}_uptime_percent {h.uptime_percent}")

        # Resilience metrics
        for name, r in report.resilience.items():
            prefix = f"victor_resilience_{name.replace('-', '_')}"
            lines.append(f"{prefix}_total_requests {r.total_requests}")
            lines.append(f"{prefix}_primary_successes {r.primary_successes}")
            lines.append(f"{prefix}_fallback_successes {r.fallback_successes}")
            lines.append(f"{prefix}_success_rate {r.success_rate}")

        return "\n".join(lines)

    def get_summary(self) -> str:
        """Get human-readable summary.

        Returns:
            Summary text suitable for logging or CLI output
        """
        report = self.generate_report()
        summary = report.summary

        lines = [
            "=== Resilience Metrics Summary ===",
            f"Timestamp: {report.timestamp}",
            "",
            "Circuit Breakers:",
            f"  Total: {summary['circuit_breakers']['total']}",
            f"  Open: {summary['circuit_breakers']['open']}",
            f"  Half-Open: {summary['circuit_breakers']['half_open']}",
            f"  Closed: {summary['circuit_breakers']['closed']}",
            "",
            "Health Status:",
            f"  Healthy: {summary['health']['healthy']}",
            f"  Degraded: {summary['health']['degraded']}",
            f"  Unhealthy: {summary['health']['unhealthy']}",
            f"  Avg Latency: {summary['health']['average_latency_ms']}ms",
            "",
            "Resilience:",
            f"  Total Requests: {summary['resilience']['total_requests']}",
            f"  Total Failures: {summary['resilience']['total_failures']}",
            f"  Avg Success Rate: {summary['resilience']['average_success_rate']}%",
        ]

        # Add circuit breaker details if any are unhealthy
        open_breakers = [
            name
            for name, cb in report.circuit_breakers.items()
            if cb.state in ("open", "half_open")
        ]
        if open_breakers:
            lines.append("")
            lines.append("Tripped Circuit Breakers:")
            for name in open_breakers:
                cb = report.circuit_breakers[name]
                lines.append(f"  - {name}: {cb.state} (failures: {cb.failure_count})")

        # Add unhealthy providers
        unhealthy = [name for name, h in report.health.items() if h.status == "unhealthy"]
        if unhealthy:
            lines.append("")
            lines.append("Unhealthy Providers:")
            for name in unhealthy:
                h = report.health[name]
                lines.append(f"  - {name}: {h.latency_ms}ms latency")

        return "\n".join(lines)


# Singleton instance
_metrics_exporter: Optional[ResilienceMetricsExporter] = None


def get_metrics_exporter() -> ResilienceMetricsExporter:
    """Get the global metrics exporter instance.

    Returns:
        Global ResilienceMetricsExporter singleton
    """
    global _metrics_exporter
    if _metrics_exporter is None:
        _metrics_exporter = ResilienceMetricsExporter()
    return _metrics_exporter


def reset_metrics_exporter() -> None:
    """Reset the global metrics exporter (mainly for testing)."""
    global _metrics_exporter
    _metrics_exporter = None
