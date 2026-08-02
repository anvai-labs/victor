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

"""Health check system for production monitoring.

This module provides a comprehensive health check framework that:
- Supports multiple health check types (liveness, readiness, startup)
- Aggregates health from multiple components
- Provides detailed diagnostics for debugging
- Integrates with Kubernetes health probes

Design Patterns:
- Composite Pattern: HealthChecker aggregates multiple checks
- Strategy Pattern: Different health check implementations
- Observer Pattern: Health status change notifications

Example:
    from victor.core.health import (
        HealthChecker,
        HealthStatus,
        ComponentHealth,
        ProviderHealthCheck,
        ToolHealthCheck,
    )

    # Create health checker
    checker = HealthChecker()

    # Add component health checks
    checker.add_check(ProviderHealthCheck("anthropic", provider))
    checker.add_check(ToolHealthCheck("filesystem", filesystem_tool))
    checker.add_check(CacheHealthCheck("redis", cache))

    # Get overall health
    health = await checker.check_health()
    print(f"Status: {health.status}")
    print(f"Components: {health.components}")

    # For Kubernetes probes
    is_ready = await checker.is_ready()
    is_alive = await checker.is_alive()
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


class HealthStatus(str, Enum):
    """Health status enumeration."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ComponentHealth:
    """Health status for a single component."""

    name: str
    status: HealthStatus
    message: Optional[str] = None
    latency_ms: Optional[float] = None
    details: Dict[str, Any] = field(default_factory=dict)
    last_check: Optional[datetime] = None
    consecutive_failures: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "latency_ms": self.latency_ms,
            "details": self.details,
            "last_check": self.last_check.isoformat() if self.last_check else None,
            "consecutive_failures": self.consecutive_failures,
        }


@dataclass
class HealthReport:
    """Aggregated health report for the system."""

    status: HealthStatus
    components: Dict[str, ComponentHealth]
    timestamp: datetime
    version: str = "1.0.0"
    uptime_seconds: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "status": self.status.value,
            "timestamp": self.timestamp.isoformat(),
            "version": self.version,
            "uptime_seconds": self.uptime_seconds,
            "components": {name: comp.to_dict() for name, comp in self.components.items()},
        }

    @property
    def is_healthy(self) -> bool:
        """Check if overall status is healthy."""
        return self.status == HealthStatus.HEALTHY

    @property
    def is_degraded(self) -> bool:
        """Check if status is degraded."""
        return self.status == HealthStatus.DEGRADED

    @property
    def unhealthy_components(self) -> List[str]:
        """Get list of unhealthy component names."""
        return [
            name for name, comp in self.components.items() if comp.status == HealthStatus.UNHEALTHY
        ]


@runtime_checkable
class HealthCheckProtocol(Protocol):
    """Protocol for health check implementations."""

    @property
    def name(self) -> str:
        """Get the component name."""
        ...

    async def check(self) -> ComponentHealth:
        """Perform health check and return status."""
        ...


class BaseHealthCheck(ABC):
    """Abstract base class for health checks."""

    def __init__(
        self,
        name: str,
        timeout: float = 5.0,
        critical: bool = True,
    ) -> None:
        """Initialize health check.

        Args:
            name: Component name.
            timeout: Check timeout in seconds.
            critical: Whether failure makes system unhealthy (vs degraded).
        """
        self._name = name
        self._timeout = timeout
        self._critical = critical
        self._consecutive_failures = 0
        self._last_health: Optional[ComponentHealth] = None

    @property
    def name(self) -> str:
        """Get component name."""
        return self._name

    @property
    def is_critical(self) -> bool:
        """Check if this is a critical component."""
        return self._critical

    @abstractmethod
    async def _do_check(self) -> ComponentHealth:
        """Perform the actual health check.

        Returns:
            Component health status.
        """
        pass

    async def check(self) -> ComponentHealth:
        """Perform health check with timeout and error handling.

        Returns:
            Component health status.
        """
        start = time.perf_counter()

        try:
            health = await asyncio.wait_for(
                self._do_check(),
                timeout=self._timeout,
            )
            health.latency_ms = (time.perf_counter() - start) * 1000
            health.last_check = datetime.now(timezone.utc)

            if health.status == HealthStatus.HEALTHY:
                self._consecutive_failures = 0
            else:
                self._consecutive_failures += 1

            health.consecutive_failures = self._consecutive_failures
            self._last_health = health
            return health

        except asyncio.TimeoutError:
            self._consecutive_failures += 1
            health = ComponentHealth(
                name=self._name,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check timed out after {self._timeout}s",
                latency_ms=(time.perf_counter() - start) * 1000,
                last_check=datetime.now(timezone.utc),
                consecutive_failures=self._consecutive_failures,
            )
            self._last_health = health
            return health

        except Exception as e:
            self._consecutive_failures += 1
            health = ComponentHealth(
                name=self._name,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check failed: {str(e)}",
                latency_ms=(time.perf_counter() - start) * 1000,
                last_check=datetime.now(timezone.utc),
                consecutive_failures=self._consecutive_failures,
            )
            self._last_health = health
            return health


class CallableHealthCheck(BaseHealthCheck):
    """Health check using a callable function.

    Example:
        async def check_db():
            await db.ping()
            return ComponentHealth(name="db", status=HealthStatus.HEALTHY)

        checker.add_check(CallableHealthCheck("database", check_db))
    """

    def __init__(
        self,
        name: str,
        check_fn: Callable[[], ComponentHealth],
        timeout: float = 5.0,
        critical: bool = True,
    ) -> None:
        """Initialize callable health check.

        Args:
            name: Component name.
            check_fn: Function that performs the check.
            timeout: Check timeout.
            critical: Whether critical component.
        """
        super().__init__(name, timeout, critical)
        self._check_fn = check_fn

    async def _do_check(self) -> ComponentHealth:
        """Execute the callable."""
        result = self._check_fn()
        if asyncio.iscoroutine(result):
            return await result
        return result


class ProviderHealthCheck(BaseHealthCheck):
    """Health check for LLM providers.

    Verifies provider connectivity and basic functionality.
    """

    def __init__(
        self,
        name: str,
        provider: Any,
        timeout: float = 10.0,
    ) -> None:
        """Initialize provider health check.

        Args:
            name: Provider name.
            provider: Provider instance.
            timeout: Check timeout.
        """
        super().__init__(f"provider.{name}", timeout, critical=True)
        self._provider = provider

    async def _do_check(self) -> ComponentHealth:
        """Check provider health."""
        details: Dict[str, Any] = {}

        # Check if provider has health check method
        if hasattr(self._provider, "health_check"):
            try:
                result = await self._provider.health_check()
                details["provider_response"] = result
                return ComponentHealth(
                    name=self._name,
                    status=HealthStatus.HEALTHY,
                    message="Provider is responsive",
                    details=details,
                )
            except Exception as e:
                return ComponentHealth(
                    name=self._name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Provider health check failed: {e}",
                    details=details,
                )

        # Fallback: check if provider has required attributes
        has_name = hasattr(self._provider, "name")
        has_chat = hasattr(self._provider, "chat") or hasattr(self._provider, "stream_chat")

        details["has_name"] = has_name
        details["has_chat"] = has_chat

        if has_name and has_chat:
            return ComponentHealth(
                name=self._name,
                status=HealthStatus.HEALTHY,
                message="Provider interface available",
                details=details,
            )

        return ComponentHealth(
            name=self._name,
            status=HealthStatus.DEGRADED,
            message="Provider missing required interface",
            details=details,
        )


class ToolHealthCheck(BaseHealthCheck):
    """Health check for tools.

    Verifies tool is properly configured and can execute.
    """

    def __init__(
        self,
        name: str,
        tool: Any,
        timeout: float = 5.0,
        critical: bool = False,
    ) -> None:
        """Initialize tool health check.

        Args:
            name: Tool name.
            tool: Tool instance.
            timeout: Check timeout.
            critical: Whether tool is critical.
        """
        super().__init__(f"tool.{name}", timeout, critical)
        self._tool = tool

    async def _do_check(self) -> ComponentHealth:
        """Check tool health."""
        details: Dict[str, Any] = {}

        # Check required attributes
        has_name = hasattr(self._tool, "name")
        has_execute = hasattr(self._tool, "execute")
        has_params = hasattr(self._tool, "parameters")

        details["has_name"] = has_name
        details["has_execute"] = has_execute
        details["has_parameters"] = has_params

        if has_name:
            details["tool_name"] = getattr(self._tool, "name", None)

        if has_name and has_execute and has_params:
            return ComponentHealth(
                name=self._name,
                status=HealthStatus.HEALTHY,
                message="Tool is properly configured",
                details=details,
            )

        missing = []
        if not has_name:
            missing.append("name")
        if not has_execute:
            missing.append("execute")
        if not has_params:
            missing.append("parameters")

        return ComponentHealth(
            name=self._name,
            status=HealthStatus.DEGRADED,
            message=f"Tool missing: {', '.join(missing)}",
            details=details,
        )


class CacheHealthCheck(BaseHealthCheck):
    """Health check for cache systems."""

    def __init__(
        self,
        name: str,
        cache: Any,
        timeout: float = 3.0,
        critical: bool = False,
    ) -> None:
        """Initialize cache health check.

        Args:
            name: Cache name.
            cache: Cache instance.
            timeout: Check timeout.
            critical: Whether cache is critical.
        """
        super().__init__(f"cache.{name}", timeout, critical)
        self._cache = cache

    async def _do_check(self) -> ComponentHealth:
        """Check cache health by doing a round-trip."""
        test_key = "__health_check__"
        test_value = "healthy"

        try:
            # Try to set and get a value
            if hasattr(self._cache, "set"):
                self._cache.set(test_key, test_value, ttl=60)

            if hasattr(self._cache, "get"):
                result = self._cache.get(test_key)
                if result == test_value:
                    return ComponentHealth(
                        name=self._name,
                        status=HealthStatus.HEALTHY,
                        message="Cache is operational",
                    )
                return ComponentHealth(
                    name=self._name,
                    status=HealthStatus.DEGRADED,
                    message="Cache read/write mismatch",
                )

            # Can't verify, assume ok if methods exist
            return ComponentHealth(
                name=self._name,
                status=HealthStatus.HEALTHY,
                message="Cache interface available",
            )

        except Exception as e:
            return ComponentHealth(
                name=self._name,
                status=HealthStatus.UNHEALTHY,
                message=f"Cache error: {e}",
            )


class MemoryHealthCheck(BaseHealthCheck):
    """Health check for memory usage."""

    def __init__(
        self,
        warning_threshold_mb: float = 1000,
        critical_threshold_mb: float = 2000,
    ) -> None:
        """Initialize memory health check.

        Args:
            warning_threshold_mb: Memory threshold for degraded status.
            critical_threshold_mb: Memory threshold for unhealthy status.
        """
        super().__init__("system.memory", timeout=1.0, critical=False)
        self._warning_mb = warning_threshold_mb
        self._critical_mb = critical_threshold_mb

    async def _do_check(self) -> ComponentHealth:
        """Check memory usage."""
        try:
            import psutil

            process = psutil.Process()
            memory_info = process.memory_info()
            rss_mb = memory_info.rss / (1024 * 1024)

            details = {
                "rss_mb": round(rss_mb, 2),
                "warning_threshold_mb": self._warning_mb,
                "critical_threshold_mb": self._critical_mb,
            }

            if rss_mb >= self._critical_mb:
                return ComponentHealth(
                    name=self._name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Memory usage critical: {rss_mb:.0f}MB",
                    details=details,
                )

            if rss_mb >= self._warning_mb:
                return ComponentHealth(
                    name=self._name,
                    status=HealthStatus.DEGRADED,
                    message=f"Memory usage high: {rss_mb:.0f}MB",
                    details=details,
                )

            return ComponentHealth(
                name=self._name,
                status=HealthStatus.HEALTHY,
                message=f"Memory usage: {rss_mb:.0f}MB",
                details=details,
            )

        except ImportError:
            return ComponentHealth(
                name=self._name,
                status=HealthStatus.UNKNOWN,
                message="psutil not installed",
            )


# =============================================================================
# Health Checker (Composite Pattern)
# =============================================================================


class HealthChecker:
    """Aggregates multiple health checks into a single report.

    Implements the Composite pattern to combine multiple health checks.
    Also implements caching to avoid excessive health checking.

    Example:
        checker = HealthChecker()
        checker.add_check(ProviderHealthCheck("anthropic", provider))
        checker.add_check(ToolHealthCheck("read", read_tool))

        report = await checker.check_health()
        print(report.to_dict())
    """

    def __init__(
        self,
        cache_ttl: float = 5.0,
        version: str = "1.0.0",
    ) -> None:
        """Initialize health checker.

        Args:
            cache_ttl: How long to cache health results.
            version: Application version for reports.
        """
        self._checks: Dict[str, BaseHealthCheck] = {}
        self._cache_ttl = cache_ttl
        self._version = version
        self._start_time = time.time()
        self._cached_report: Optional[HealthReport] = None
        self._cache_time: Optional[float] = None
        self._on_status_change: List[Callable[[HealthStatus, HealthStatus], None]] = []

    def add_check(self, check: BaseHealthCheck) -> "HealthChecker":
        """Add a health check.

        Args:
            check: Health check to add.

        Returns:
            Self for chaining.
        """
        self._checks[check.name] = check
        return self

    def remove_check(self, name: str) -> "HealthChecker":
        """Remove a health check.

        Args:
            name: Check name to remove.

        Returns:
            Self for chaining.
        """
        if name in self._checks:
            del self._checks[name]
        return self

    def on_status_change(
        self,
        callback: Callable[[HealthStatus, HealthStatus], None],
    ) -> None:
        """Register callback for status changes.

        Args:
            callback: Function called with (old_status, new_status).
        """
        self._on_status_change.append(callback)

    async def check_health(self, use_cache: bool = True) -> HealthReport:
        """Perform all health checks and aggregate results.

        Args:
            use_cache: Whether to use cached results if available.

        Returns:
            Aggregated health report.
        """
        # Check cache
        if use_cache and self._cached_report and self._cache_time:
            age = time.time() - self._cache_time
            if age < self._cache_ttl:
                return self._cached_report

        # Run all checks concurrently
        component_healths: Dict[str, ComponentHealth] = {}

        if self._checks:
            tasks = [check.check() for check in self._checks.values()]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for check, result in zip(self._checks.values(), results, strict=False):
                if isinstance(result, Exception):
                    component_healths[check.name] = ComponentHealth(
                        name=check.name,
                        status=HealthStatus.UNHEALTHY,
                        message=f"Check error: {result}",
                        last_check=datetime.now(timezone.utc),
                    )
                else:
                    component_healths[check.name] = result

        # Determine overall status
        overall_status = self._aggregate_status(component_healths)

        report = HealthReport(
            status=overall_status,
            components=component_healths,
            timestamp=datetime.now(timezone.utc),
            version=self._version,
            uptime_seconds=time.time() - self._start_time,
        )

        # Notify status change
        if self._cached_report and self._cached_report.status != overall_status:
            for callback in self._on_status_change:
                try:
                    callback(self._cached_report.status, overall_status)
                except Exception as e:
                    logger.warning(f"Status change callback error: {e}")

        # Update cache
        self._cached_report = report
        self._cache_time = time.time()

        return report

    def _aggregate_status(
        self,
        components: Dict[str, ComponentHealth],
    ) -> HealthStatus:
        """Aggregate component statuses into overall status.

        Args:
            components: Component health statuses.

        Returns:
            Overall health status.
        """
        if not components:
            return HealthStatus.HEALTHY

        has_unhealthy = False
        has_degraded = False

        for name, health in components.items():
            check = self._checks.get(name)
            is_critical = check.is_critical if check else True

            if health.status == HealthStatus.UNHEALTHY:
                if is_critical:
                    has_unhealthy = True
                else:
                    has_degraded = True
            elif health.status == HealthStatus.DEGRADED:
                has_degraded = True

        if has_unhealthy:
            return HealthStatus.UNHEALTHY
        if has_degraded:
            return HealthStatus.DEGRADED
        return HealthStatus.HEALTHY

    async def is_healthy(self) -> bool:
        """Quick check if system is healthy.

        Returns:
            True if healthy.
        """
        report = await self.check_health()
        return report.status == HealthStatus.HEALTHY

    async def is_ready(self) -> bool:
        """Kubernetes readiness probe.

        Returns True if system can serve traffic.

        Returns:
            True if ready.
        """
        report = await self.check_health()
        return report.status in (HealthStatus.HEALTHY, HealthStatus.DEGRADED)

    async def is_alive(self) -> bool:
        """Kubernetes liveness probe.

        Returns True if system is running (even if degraded).

        Returns:
            True if alive.
        """
        # For liveness, we just check that we can execute
        return True

    def get_check_names(self) -> List[str]:
        """Get list of registered check names.

        Returns:
            List of check names.
        """
        return list(self._checks.keys())


# =============================================================================
# Factory Functions
# =============================================================================


def create_default_health_checker(
    version: str = "1.0.0",
    include_memory: bool = True,
) -> HealthChecker:
    """Create a health checker with default checks.

    Args:
        version: Application version.
        include_memory: Whether to include memory check.

    Returns:
        Configured health checker.
    """
    checker = HealthChecker(version=version)

    if include_memory:
        checker.add_check(MemoryHealthCheck())

    return checker
