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

"""Tests for Phase 9 Framework Facades.

These tests verify that the facade modules correctly re-export components
from the underlying modules and that the exports are identical objects
(not copies), ensuring the Facade Pattern is correctly implemented.

The facade pattern provides:
1. Single import point for framework users
2. No code duplication - pure re-exports
3. Backward compatibility with original modules
"""

import pytest


class TestResilienceFacade:
    """Tests for resilience facade re-exports."""

    def test_circuit_breaker_reexport_identity(self):
        """Verify CircuitBreaker is same object from both modules."""
        from victor.framework.resilience import CircuitBreaker as FacadeCircuitBreaker
        from victor.providers.circuit_breaker import (
            CircuitBreaker as OriginalCircuitBreaker,
        )

        assert FacadeCircuitBreaker is OriginalCircuitBreaker

    def test_circuit_breaker_error_reexport_identity(self):
        """Verify CircuitBreakerError is same object from both modules."""
        from victor.framework.resilience import CircuitBreakerError as FacadeError
        from victor.core.circuit_breaker import CircuitBreakerError as CoreError
        from victor.providers.circuit_breaker import (
            CircuitBreakerError as OriginalError,
        )

        assert FacadeError is OriginalError
        assert OriginalError is CoreError

    def test_circuit_breaker_registry_reexport_identity(self):
        """Verify CircuitBreakerRegistry is same object from both modules."""
        from victor.framework.resilience import CircuitBreakerRegistry as FacadeRegistry
        from victor.providers.circuit_breaker import (
            CircuitBreakerRegistry as OriginalRegistry,
        )

        assert FacadeRegistry is OriginalRegistry

    def test_circuit_state_reexport_identity(self):
        """Verify CircuitState is same enum from both modules."""
        from victor.framework.resilience import CircuitState as FacadeState
        from victor.core.circuit_breaker import CircuitState as CoreState
        from victor.providers.circuit_breaker import CircuitState as OriginalState

        assert FacadeState is OriginalState
        assert OriginalState is CoreState
        assert FacadeState.CLOSED is OriginalState.CLOSED
        assert FacadeState.OPEN is OriginalState.OPEN
        assert FacadeState.HALF_OPEN is OriginalState.HALF_OPEN

    def test_resilient_provider_reexport_identity(self):
        """Verify ResilientProvider is same object from both modules."""
        from victor.framework.resilience import ResilientProvider as FacadeProvider
        from victor.providers.resilience import ResilientProvider as OriginalProvider

        assert FacadeProvider is OriginalProvider

    def test_circuit_breaker_config_reexport_identity(self):
        """Verify CircuitBreakerConfig is same object from both modules."""
        from victor.framework.resilience import CircuitBreakerConfig as FacadeConfig
        from victor.core.circuit_breaker import CircuitBreakerConfig as CoreConfig
        from victor.providers.resilience import CircuitBreakerConfig as OriginalConfig

        assert FacadeConfig is OriginalConfig
        assert OriginalConfig is CoreConfig

    def test_retry_config_reexport_identity(self):
        """Verify ProviderRetryConfig is same object from both modules."""
        from victor.framework.resilience import ProviderRetryConfig as FacadeConfig
        from victor.providers.resilience import ProviderRetryConfig as OriginalConfig

        assert FacadeConfig is OriginalConfig

    def test_exponential_backoff_reexport_identity(self):
        """Verify ExponentialBackoffStrategy is same object from both modules."""
        from victor.framework.resilience import (
            ExponentialBackoffStrategy as FacadeStrategy,
        )
        from victor.core.retry import ExponentialBackoffStrategy as OriginalStrategy

        assert FacadeStrategy is OriginalStrategy

    def test_linear_backoff_reexport_identity(self):
        """Verify LinearBackoffStrategy is same object from both modules."""
        from victor.framework.resilience import LinearBackoffStrategy as FacadeStrategy
        from victor.core.retry import LinearBackoffStrategy as OriginalStrategy

        assert FacadeStrategy is OriginalStrategy

    def test_fixed_delay_reexport_identity(self):
        """Verify FixedDelayStrategy is same object from both modules."""
        from victor.framework.resilience import FixedDelayStrategy as FacadeStrategy
        from victor.core.retry import FixedDelayStrategy as OriginalStrategy

        assert FacadeStrategy is OriginalStrategy

    def test_no_retry_reexport_identity(self):
        """Verify NoRetryStrategy is same object from both modules."""
        from victor.framework.resilience import NoRetryStrategy as FacadeStrategy
        from victor.core.retry import NoRetryStrategy as OriginalStrategy

        assert FacadeStrategy is OriginalStrategy

    def test_retry_executor_reexport_identity(self):
        """Verify RetryExecutor is same object from both modules."""
        from victor.framework.resilience import RetryExecutor as FacadeExecutor
        from victor.core.retry import RetryExecutor as OriginalExecutor

        assert FacadeExecutor is OriginalExecutor

    def test_retry_context_reexport_identity(self):
        """Verify RetryContext is same object from both modules."""
        from victor.framework.resilience import RetryContext as FacadeContext
        from victor.core.retry import RetryContext as OriginalContext

        assert FacadeContext is OriginalContext

    def test_with_retry_decorator_reexport_identity(self):
        """Verify with_retry decorator is same function from both modules."""
        from victor.framework.resilience import with_retry as facade_with_retry
        from victor.core.retry import with_retry as original_with_retry

        assert facade_with_retry is original_with_retry

    def test_with_retry_sync_reexport_identity(self):
        """Verify with_retry_sync is same function from both modules."""
        from victor.framework.resilience import with_retry_sync as facade_func
        from victor.core.retry import with_retry_sync as original_func

        assert facade_func is original_func

    def test_tool_retry_strategy_reexport_identity(self):
        """Verify tool_retry_strategy is same function from both modules."""
        from victor.framework.resilience import tool_retry_strategy as facade_func
        from victor.core.retry import tool_retry_strategy as original_func

        assert facade_func is original_func

    def test_provider_retry_strategy_reexport_identity(self):
        """Verify provider_retry_strategy is same function from both modules."""
        from victor.framework.resilience import provider_retry_strategy as facade_func
        from victor.core.retry import provider_retry_strategy as original_func

        assert facade_func is original_func

    def test_connection_retry_strategy_reexport_identity(self):
        """Verify connection_retry_strategy is same function from both modules."""
        from victor.framework.resilience import connection_retry_strategy as facade_func
        from victor.core.retry import connection_retry_strategy as original_func

        assert facade_func is original_func

    def test_all_exports_present(self):
        """Verify all declared exports are present in the module."""
        from victor.framework import resilience

        expected_exports = [
            # Circuit Breaker (Standalone)
            "CircuitBreaker",
            "CircuitBreakerError",
            "CircuitBreakerRegistry",
            "CircuitState",
            # Resilient Provider
            "CircuitBreakerConfig",
            "CircuitBreakerState",
            "CircuitOpenError",
            "ProviderUnavailableError",
            "ResilientProvider",
            "ProviderRetryConfig",
            "RetryExhaustedError",
            "ProviderRetryStrategy",
            # Unified Retry Strategies
            "ExponentialBackoffStrategy",
            "FixedDelayStrategy",
            "LinearBackoffStrategy",
            "NoRetryStrategy",
            "RetryContext",
            "RetryExecutor",
            "RetryOutcome",
            "RetryResult",
            "BaseRetryStrategy",
            "connection_retry_strategy",
            "provider_retry_strategy",
            "tool_retry_strategy",
            "with_retry",
            "with_retry_sync",
        ]

        for name in expected_exports:
            assert hasattr(resilience, name), f"Missing export: {name}"

    def test_framework_import_works(self):
        """Test that imports from victor.framework work."""
        from victor.framework import (
            CircuitBreaker,
            CircuitBreakerRegistry,
            CircuitState,
            ExponentialBackoffStrategy,
            ResilientProvider,
            BaseRetryStrategy,
            with_retry,
        )

        # Basic validation that classes are importable and usable
        assert CircuitBreaker is not None
        assert CircuitState.CLOSED.value == "closed"
        assert BaseRetryStrategy is not None


class TestHealthFacade:
    """Tests for health facade re-exports."""

    def test_health_checker_reexport_identity(self):
        """Verify HealthChecker is same object from both modules."""
        from victor.framework.health import HealthChecker as FacadeChecker
        from victor.core.health import HealthChecker as OriginalChecker

        assert FacadeChecker is OriginalChecker

    def test_health_status_reexport_identity(self):
        """Verify HealthStatus is same enum from both modules."""
        from victor.framework.health import HealthStatus as FacadeStatus
        from victor.core.health import HealthStatus as OriginalStatus

        assert FacadeStatus is OriginalStatus
        assert FacadeStatus.HEALTHY is OriginalStatus.HEALTHY
        assert FacadeStatus.DEGRADED is OriginalStatus.DEGRADED
        assert FacadeStatus.UNHEALTHY is OriginalStatus.UNHEALTHY

    def test_component_health_reexport_identity(self):
        """Verify ComponentHealth is same object from both modules."""
        from victor.framework.health import ComponentHealth as FacadeHealth
        from victor.core.health import ComponentHealth as OriginalHealth

        assert FacadeHealth is OriginalHealth

    def test_health_report_reexport_identity(self):
        """Verify HealthReport is same object from both modules."""
        from victor.framework.health import HealthReport as FacadeReport
        from victor.core.health import HealthReport as OriginalReport

        assert FacadeReport is OriginalReport

    def test_base_health_check_reexport_identity(self):
        """Verify BaseHealthCheck is same object from both modules."""
        from victor.framework.health import BaseHealthCheck as FacadeCheck
        from victor.core.health import BaseHealthCheck as OriginalCheck

        assert FacadeCheck is OriginalCheck

    def test_provider_health_check_reexport_identity(self):
        """Verify ProviderHealthCheck is same object from both modules."""
        from victor.framework.health import ProviderHealthCheck as FacadeCheck
        from victor.core.health import ProviderHealthCheck as OriginalCheck

        assert FacadeCheck is OriginalCheck

    def test_tool_health_check_reexport_identity(self):
        """Verify ToolHealthCheck is same object from both modules."""
        from victor.framework.health import ToolHealthCheck as FacadeCheck
        from victor.core.health import ToolHealthCheck as OriginalCheck

        assert FacadeCheck is OriginalCheck

    def test_cache_health_check_reexport_identity(self):
        """Verify CacheHealthCheck is same object from both modules."""
        from victor.framework.health import CacheHealthCheck as FacadeCheck
        from victor.core.health import CacheHealthCheck as OriginalCheck

        assert FacadeCheck is OriginalCheck

    def test_memory_health_check_reexport_identity(self):
        """Verify MemoryHealthCheck is same object from both modules."""
        from victor.framework.health import MemoryHealthCheck as FacadeCheck
        from victor.core.health import MemoryHealthCheck as OriginalCheck

        assert FacadeCheck is OriginalCheck

    def test_callable_health_check_reexport_identity(self):
        """Verify CallableHealthCheck is same object from both modules."""
        from victor.framework.health import CallableHealthCheck as FacadeCheck
        from victor.core.health import CallableHealthCheck as OriginalCheck

        assert FacadeCheck is OriginalCheck

    def test_create_default_health_checker_reexport_identity(self):
        """Verify create_default_health_checker is same function from both modules."""
        from victor.framework.health import (
            create_default_health_checker as facade_func,
        )
        from victor.core.health import create_default_health_checker as original_func

        assert facade_func is original_func

    def test_provider_health_checker_reexport_identity(self):
        """Verify ProviderHealthChecker is same object from both modules."""
        from victor.framework.health import ProviderHealthChecker as FacadeChecker
        from victor.providers.health import ProviderHealthChecker as OriginalChecker

        assert FacadeChecker is OriginalChecker

    def test_provider_health_result_reexport_identity(self):
        """Verify ProviderHealthResult is same object from both modules."""
        from victor.framework.health import ProviderHealthResult as FacadeResult
        from victor.providers.health import ProviderHealthResult as OriginalResult

        assert FacadeResult is OriginalResult

    def test_health_check_result_alias(self):
        """Verify HealthCheckResult is an alias for ProviderHealthResult."""
        from victor.framework.health import HealthCheckResult, ProviderHealthResult

        assert HealthCheckResult is ProviderHealthResult

    def test_check_provider_health_reexport_identity(self):
        """Verify check_provider_health is same function from both modules."""
        from victor.framework.health import check_provider_health as facade_func
        from victor.providers.health import check_provider_health as original_func

        assert facade_func is original_func

    def test_all_exports_present(self):
        """Verify all declared exports are present in the module."""
        from victor.framework import health

        expected_exports = [
            # Core Health Check System
            "BaseHealthCheck",
            "CacheHealthCheck",
            "CallableHealthCheck",
            "ComponentHealth",
            "HealthCheckProtocol",
            "HealthChecker",
            "HealthReport",
            "HealthStatus",
            "MemoryHealthCheck",
            "ProviderHealthCheck",
            "ToolHealthCheck",
            "create_default_health_checker",
            # Provider-Specific Health
            "HealthCheckResult",
            "ProviderHealthChecker",
            "ProviderHealthResult",
            "check_provider_health",
        ]

        for name in expected_exports:
            assert hasattr(health, name), f"Missing export: {name}"

    def test_framework_import_works(self):
        """Test that imports from victor.framework work."""
        from victor.framework import (
            ComponentHealth,
            HealthChecker,
            HealthStatus,
            create_default_health_checker,
        )

        # Basic validation that classes are importable and usable
        assert HealthChecker is not None
        assert HealthStatus.HEALTHY.value == "healthy"


class TestMetricsFacade:
    """Tests for metrics facade re-exports."""

    def test_metrics_registry_reexport_identity(self):
        """Verify MetricsRegistry is same object from both modules."""
        from victor.framework.metrics import MetricsRegistry as FacadeRegistry
        from victor.observability.metrics import MetricsRegistry as OriginalRegistry

        assert FacadeRegistry is OriginalRegistry

    def test_counter_reexport_identity(self):
        """Verify Counter is same object from both modules."""
        from victor.framework.metrics import Counter as FacadeCounter
        from victor.observability.metrics import Counter as OriginalCounter

        assert FacadeCounter is OriginalCounter

    def test_gauge_reexport_identity(self):
        """Verify Gauge is same object from both modules."""
        from victor.framework.metrics import Gauge as FacadeGauge
        from victor.observability.metrics import Gauge as OriginalGauge

        assert FacadeGauge is OriginalGauge

    def test_histogram_reexport_identity(self):
        """Verify Histogram is same object from both modules."""
        from victor.framework.metrics import Histogram as FacadeHistogram
        from victor.observability.metrics import Histogram as OriginalHistogram

        assert FacadeHistogram is OriginalHistogram

    def test_timer_reexport_identity(self):
        """Verify Timer is same object from both modules."""
        from victor.framework.metrics import Timer as FacadeTimer
        from victor.observability.metrics import Timer as OriginalTimer

        assert FacadeTimer is OriginalTimer

    def test_metric_reexport_identity(self):
        """Verify Metric is same object from both modules."""
        from victor.framework.metrics import Metric as FacadeMetric
        from victor.observability.metrics import Metric as OriginalMetric

        assert FacadeMetric is OriginalMetric

    def test_metric_labels_reexport_identity(self):
        """Verify MetricLabels is same object from both modules."""
        from victor.framework.metrics import MetricLabels as FacadeLabels
        from victor.observability.metrics import MetricLabels as OriginalLabels

        assert FacadeLabels is OriginalLabels

    def test_metrics_collector_reexport_identity(self):
        """Verify MetricsCollector is same object from both modules."""
        from victor.framework.metrics import MetricsCollector as FacadeCollector
        from victor.observability.metrics import MetricsCollector as OriginalCollector

        assert FacadeCollector is OriginalCollector

    def test_timer_context_reexport_identity(self):
        """Verify TimerContext is same object from both modules."""
        from victor.framework.metrics import TimerContext as FacadeContext
        from victor.observability.metrics import TimerContext as OriginalContext

        assert FacadeContext is OriginalContext

    def test_setup_opentelemetry_reexport_identity(self):
        """Verify setup_opentelemetry is same function from both modules."""
        from victor.framework.metrics import setup_opentelemetry as facade_func
        from victor.observability.telemetry import setup_opentelemetry as original_func

        assert facade_func is original_func

    def test_get_tracer_reexport_identity(self):
        """Verify get_tracer is same function from both modules."""
        from victor.framework.metrics import get_tracer as facade_func
        from victor.observability.telemetry import get_tracer as original_func

        assert facade_func is original_func

    def test_get_meter_reexport_identity(self):
        """Verify get_meter is same function from both modules."""
        from victor.framework.metrics import get_meter as facade_func
        from victor.observability.telemetry import get_meter as original_func

        assert facade_func is original_func

    def test_is_telemetry_enabled_reexport_identity(self):
        """Verify is_telemetry_enabled is same function from both modules."""
        from victor.framework.metrics import is_telemetry_enabled as facade_func
        from victor.observability.telemetry import is_telemetry_enabled as original_func

        assert facade_func is original_func

    def test_all_exports_present(self):
        """Verify all declared exports are present in the module."""
        from victor.framework import metrics

        expected_exports = [
            # Metrics System
            "Counter",
            "Gauge",
            "Histogram",
            "Metric",
            "MetricLabels",
            "MetricsCollector",
            "MetricsRegistry",
            "Timer",
            "TimerContext",
            # Telemetry
            "get_meter",
            "get_tracer",
            "is_telemetry_enabled",
            "setup_opentelemetry",
        ]

        for name in expected_exports:
            assert hasattr(metrics, name), f"Missing export: {name}"

    def test_framework_import_works(self):
        """Test that imports from victor.framework work."""
        from victor.framework import (
            Counter,
            Gauge,
            Histogram,
            MetricsRegistry,
            Timer,
            get_tracer,
            setup_opentelemetry,
        )

        # Basic validation that classes are importable and usable
        assert MetricsRegistry is not None
        assert Counter is not None


class TestFrameworkMainExports:
    """Tests for Phase 9 exports from main framework module."""

    def test_resilience_exports_in_main_module(self):
        """Test that resilience exports are available from victor.framework."""
        from victor.framework import (
            CircuitBreaker,
            CircuitBreakerConfig,
            CircuitState,
            ExponentialBackoffStrategy,
            ResilientProvider,
            BaseRetryStrategy,
            with_retry,
        )

        assert CircuitBreaker is not None
        assert CircuitState.CLOSED.value == "closed"
        assert BaseRetryStrategy is not None

    def test_health_exports_in_main_module(self):
        """Test that health exports are available from victor.framework."""
        from victor.framework import (
            ComponentHealth,
            HealthChecker,
            HealthStatus,
            ProviderHealthChecker,
            ProviderHealthResult,
            create_default_health_checker,
        )

        assert HealthChecker is not None
        assert HealthStatus.HEALTHY.value == "healthy"

    def test_metrics_exports_in_main_module(self):
        """Test that metrics exports are available from victor.framework."""
        from victor.framework import (
            Counter,
            Gauge,
            Histogram,
            MetricsRegistry,
            Timer,
            get_tracer,
            setup_opentelemetry,
        )

        assert MetricsRegistry is not None
        assert Counter is not None

    def test_phase9_export_counts(self):
        """Verify Phase 9 added expected number of exports."""
        from victor.framework import (
            _HEALTH_EXPORTS,
            _METRICS_EXPORTS,
            _RESILIENCE_EXPORTS,
        )

        assert (
            len(_RESILIENCE_EXPORTS) == 26
        ), f"Expected 26 resilience exports, got {len(_RESILIENCE_EXPORTS)}"
        assert len(_HEALTH_EXPORTS) == 19, f"Expected 19 health exports, got {len(_HEALTH_EXPORTS)}"
        assert (
            len(_METRICS_EXPORTS) == 13
        ), f"Expected 13 metrics exports, got {len(_METRICS_EXPORTS)}"

        # Total new exports from Phase 9
        total = len(_RESILIENCE_EXPORTS) + len(_HEALTH_EXPORTS) + len(_METRICS_EXPORTS)
        assert total == 58, f"Expected 58 total Phase 9 exports, got {total}"
