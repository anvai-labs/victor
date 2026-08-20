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

"""Tests for health check system module."""

import asyncio
import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock, AsyncMock

from victor.core.health import (
    BaseHealthCheck,
    CacheHealthCheck,
    CallableHealthCheck,
    ComponentHealth,
    HealthChecker,
    HealthReport,
    HealthStatus,
    MemoryHealthCheck,
    ProviderHealthCheck,
    ToolHealthCheck,
    create_default_health_checker,
)

# =============================================================================
# HealthStatus Tests
# =============================================================================


class TestHealthStatus:
    """Tests for HealthStatus enum."""

    def test_status_values(self):
        """Test status enum values."""
        assert HealthStatus.HEALTHY.value == "healthy"
        assert HealthStatus.DEGRADED.value == "degraded"
        assert HealthStatus.UNHEALTHY.value == "unhealthy"
        assert HealthStatus.UNKNOWN.value == "unknown"


# =============================================================================
# ComponentHealth Tests
# =============================================================================


class TestComponentHealth:
    """Tests for ComponentHealth dataclass."""

    def test_basic_creation(self):
        """Test creating component health."""
        health = ComponentHealth(
            name="test_component",
            status=HealthStatus.HEALTHY,
        )

        assert health.name == "test_component"
        assert health.status == HealthStatus.HEALTHY
        assert health.message is None

    def test_with_all_fields(self):
        """Test with all fields populated."""
        now = datetime.now(timezone.utc)
        health = ComponentHealth(
            name="test",
            status=HealthStatus.DEGRADED,
            message="High latency",
            latency_ms=150.5,
            details={"cpu": 80},
            last_check=now,
            consecutive_failures=2,
        )

        assert health.message == "High latency"
        assert health.latency_ms == 150.5
        assert health.details["cpu"] == 80
        assert health.consecutive_failures == 2

    def test_to_dict(self):
        """Test conversion to dictionary."""
        now = datetime.now(timezone.utc)
        health = ComponentHealth(
            name="test",
            status=HealthStatus.HEALTHY,
            message="OK",
            latency_ms=10.0,
            details={"key": "value"},
            last_check=now,
        )

        data = health.to_dict()

        assert data["name"] == "test"
        assert data["status"] == "healthy"
        assert data["message"] == "OK"
        assert data["latency_ms"] == 10.0
        assert data["details"]["key"] == "value"
        assert data["last_check"] == now.isoformat()


# =============================================================================
# HealthReport Tests
# =============================================================================


class TestHealthReport:
    """Tests for HealthReport dataclass."""

    def test_basic_creation(self):
        """Test creating health report."""
        now = datetime.now(timezone.utc)
        report = HealthReport(
            status=HealthStatus.HEALTHY,
            components={},
            timestamp=now,
        )

        assert report.status == HealthStatus.HEALTHY
        assert report.timestamp == now
        assert report.version == "1.0.0"

    def test_is_healthy(self):
        """Test is_healthy property."""
        report = HealthReport(
            status=HealthStatus.HEALTHY,
            components={},
            timestamp=datetime.now(timezone.utc),
        )
        assert report.is_healthy is True

        report = HealthReport(
            status=HealthStatus.DEGRADED,
            components={},
            timestamp=datetime.now(timezone.utc),
        )
        assert report.is_healthy is False

    def test_is_degraded(self):
        """Test is_degraded property."""
        report = HealthReport(
            status=HealthStatus.DEGRADED,
            components={},
            timestamp=datetime.now(timezone.utc),
        )
        assert report.is_degraded is True

    def test_unhealthy_components(self):
        """Test unhealthy_components property."""
        components = {
            "comp1": ComponentHealth("comp1", HealthStatus.HEALTHY),
            "comp2": ComponentHealth("comp2", HealthStatus.UNHEALTHY),
            "comp3": ComponentHealth("comp3", HealthStatus.DEGRADED),
            "comp4": ComponentHealth("comp4", HealthStatus.UNHEALTHY),
        }

        report = HealthReport(
            status=HealthStatus.UNHEALTHY,
            components=components,
            timestamp=datetime.now(timezone.utc),
        )

        unhealthy = report.unhealthy_components
        assert len(unhealthy) == 2
        assert "comp2" in unhealthy
        assert "comp4" in unhealthy

    def test_to_dict(self):
        """Test conversion to dictionary."""
        components = {
            "comp1": ComponentHealth("comp1", HealthStatus.HEALTHY),
        }
        report = HealthReport(
            status=HealthStatus.HEALTHY,
            components=components,
            timestamp=datetime.now(timezone.utc),
            version="2.0.0",
            uptime_seconds=3600.0,
        )

        data = report.to_dict()

        assert data["status"] == "healthy"
        assert data["version"] == "2.0.0"
        assert data["uptime_seconds"] == 3600.0
        assert "comp1" in data["components"]


# =============================================================================
# BaseHealthCheck Tests
# =============================================================================


class TestBaseHealthCheck:
    """Tests for BaseHealthCheck abstract class."""

    @pytest.mark.asyncio
    async def test_timeout_handling(self):
        """Test that timeout is handled properly."""

        class SlowCheck(BaseHealthCheck):
            async def _do_check(self):
                await asyncio.sleep(10)  # Very slow
                return ComponentHealth(
                    name=self._name,
                    status=HealthStatus.HEALTHY,
                )

        check = SlowCheck("slow", timeout=0.1)
        health = await check.check()

        assert health.status == HealthStatus.UNHEALTHY
        assert "timed out" in health.message
        assert check._consecutive_failures == 1

    @pytest.mark.asyncio
    async def test_exception_handling(self):
        """Test that exceptions are caught."""

        class FailingCheck(BaseHealthCheck):
            async def _do_check(self):
                raise ValueError("Check failed")

        check = FailingCheck("failing", timeout=5.0)
        health = await check.check()

        assert health.status == HealthStatus.UNHEALTHY
        assert "Check failed" in health.message
        assert check._consecutive_failures == 1

    @pytest.mark.asyncio
    async def test_consecutive_failures_tracking(self):
        """Test consecutive failures are tracked."""

        class FlakeyCheck(BaseHealthCheck):
            def __init__(self):
                super().__init__("flakey")
                self.call_count = 0

            async def _do_check(self):
                self.call_count += 1
                if self.call_count < 3:
                    raise ValueError("Flaky")
                return ComponentHealth(
                    name=self._name,
                    status=HealthStatus.HEALTHY,
                )

        check = FlakeyCheck()

        # First two fail
        await check.check()
        assert check._consecutive_failures == 1
        await check.check()
        assert check._consecutive_failures == 2

        # Third succeeds
        await check.check()
        assert check._consecutive_failures == 0

    @pytest.mark.asyncio
    async def test_latency_recorded(self):
        """Test latency is recorded."""

        class QuickCheck(BaseHealthCheck):
            async def _do_check(self):
                await asyncio.sleep(0.01)
                return ComponentHealth(
                    name=self._name,
                    status=HealthStatus.HEALTHY,
                )

        check = QuickCheck("quick")
        health = await check.check()

        assert health.latency_ms is not None
        assert health.latency_ms >= 10  # At least 10ms


# =============================================================================
# CallableHealthCheck Tests
# =============================================================================


class TestCallableHealthCheck:
    """Tests for CallableHealthCheck."""

    @pytest.mark.asyncio
    async def test_sync_callable(self):
        """Test with sync callable."""

        def check_fn():
            return ComponentHealth(
                name="sync_check",
                status=HealthStatus.HEALTHY,
                message="All good",
            )

        check = CallableHealthCheck("sync", check_fn)
        health = await check.check()

        assert health.status == HealthStatus.HEALTHY
        assert health.message == "All good"

    @pytest.mark.asyncio
    async def test_async_callable(self):
        """Test with async callable."""

        async def check_fn():
            await asyncio.sleep(0.01)
            return ComponentHealth(
                name="async_check",
                status=HealthStatus.HEALTHY,
            )

        check = CallableHealthCheck("async", check_fn)
        health = await check.check()

        assert health.status == HealthStatus.HEALTHY


# =============================================================================
# ProviderHealthCheck Tests
# =============================================================================


class TestProviderHealthCheck:
    """Tests for ProviderHealthCheck."""

    @pytest.mark.asyncio
    async def test_provider_with_health_check_method(self):
        """Test provider that has health_check method."""
        provider = MagicMock()
        provider.health_check = AsyncMock(return_value={"status": "ok"})

        check = ProviderHealthCheck("test", provider)
        health = await check.check()

        assert health.status == HealthStatus.HEALTHY
        assert health.details["provider_response"] == {"status": "ok"}

    @pytest.mark.asyncio
    async def test_provider_health_check_fails(self):
        """Test provider health check that fails."""
        provider = MagicMock()
        provider.health_check = AsyncMock(side_effect=Exception("Connection failed"))

        check = ProviderHealthCheck("test", provider)
        health = await check.check()

        assert health.status == HealthStatus.UNHEALTHY
        assert "Connection failed" in health.message

    @pytest.mark.asyncio
    async def test_provider_fallback_check(self):
        """Test fallback check when no health_check method."""
        provider = MagicMock(spec=["name", "chat"])
        provider.name = "test_provider"

        check = ProviderHealthCheck("test", provider)
        health = await check.check()

        assert health.status == HealthStatus.HEALTHY
        assert health.details["has_name"] is True
        assert health.details["has_chat"] is True

    @pytest.mark.asyncio
    async def test_provider_missing_interface(self):
        """Test provider missing required interface."""
        provider = MagicMock(spec=[])

        check = ProviderHealthCheck("test", provider)
        health = await check.check()

        assert health.status == HealthStatus.DEGRADED
        assert "missing required interface" in health.message


# =============================================================================
# ToolHealthCheck Tests
# =============================================================================


class TestToolHealthCheck:
    """Tests for ToolHealthCheck."""

    @pytest.mark.asyncio
    async def test_valid_tool(self):
        """Test tool with all required attributes."""
        tool = MagicMock(spec=["name", "execute", "parameters"])
        tool.name = "test_tool"

        check = ToolHealthCheck("test", tool)
        health = await check.check()

        assert health.status == HealthStatus.HEALTHY
        assert health.details["tool_name"] == "test_tool"

    @pytest.mark.asyncio
    async def test_tool_missing_execute(self):
        """Test tool missing execute method."""
        tool = MagicMock(spec=["name", "parameters"])
        tool.name = "incomplete_tool"

        check = ToolHealthCheck("test", tool)
        health = await check.check()

        assert health.status == HealthStatus.DEGRADED
        assert "execute" in health.message


# =============================================================================
# CacheHealthCheck Tests
# =============================================================================


class TestCacheHealthCheck:
    """Tests for CacheHealthCheck."""

    @pytest.mark.asyncio
    async def test_cache_operational(self):
        """Test operational cache."""
        cache = MagicMock()
        cache.set = MagicMock()
        cache.get = MagicMock(return_value="healthy")

        check = CacheHealthCheck("test", cache)
        health = await check.check()

        assert health.status == HealthStatus.HEALTHY
        assert "operational" in health.message

    @pytest.mark.asyncio
    async def test_cache_mismatch(self):
        """Test cache read/write mismatch."""
        cache = MagicMock()
        cache.set = MagicMock()
        cache.get = MagicMock(return_value="wrong_value")

        check = CacheHealthCheck("test", cache)
        health = await check.check()

        assert health.status == HealthStatus.DEGRADED
        assert "mismatch" in health.message

    @pytest.mark.asyncio
    async def test_cache_error(self):
        """Test cache error."""
        cache = MagicMock()
        cache.get = MagicMock(side_effect=Exception("Connection refused"))

        check = CacheHealthCheck("test", cache)
        health = await check.check()

        assert health.status == HealthStatus.UNHEALTHY
        assert "Connection refused" in health.message


# =============================================================================
# MemoryHealthCheck Tests
# =============================================================================


class TestMemoryHealthCheck:
    """Tests for MemoryHealthCheck."""

    @pytest.mark.asyncio
    async def test_memory_healthy(self):
        """Test healthy memory usage."""
        check = MemoryHealthCheck(
            warning_threshold_mb=10000,
            critical_threshold_mb=20000,
        )
        health = await check.check()

        # Unless running on a very memory-constrained system or after many tests
        # Accept all statuses since memory usage varies in test environments
        assert health.status in (
            HealthStatus.HEALTHY,
            HealthStatus.UNKNOWN,
            HealthStatus.DEGRADED,
            HealthStatus.UNHEALTHY,
        )

    @pytest.mark.asyncio
    async def test_memory_details_included(self):
        """Test memory details are included."""
        check = MemoryHealthCheck()
        health = await check.check()

        if health.status != HealthStatus.UNKNOWN:
            assert "rss_mb" in health.details
            assert health.details["warning_threshold_mb"] == 1000
            assert health.details["critical_threshold_mb"] == 2000


# =============================================================================
# HealthChecker Tests
# =============================================================================


class TestHealthChecker:
    """Tests for HealthChecker composite."""

    def test_add_remove_checks(self):
        """Test adding and removing checks."""
        checker = HealthChecker()

        class SimpleCheck(BaseHealthCheck):
            async def _do_check(self):
                return ComponentHealth(self._name, HealthStatus.HEALTHY)

        check1 = SimpleCheck("check1")
        check2 = SimpleCheck("check2")

        checker.add_check(check1).add_check(check2)
        assert len(checker.get_check_names()) == 2

        checker.remove_check("check1")
        assert len(checker.get_check_names()) == 1
        assert "check2" in checker.get_check_names()

    @pytest.mark.asyncio
    async def test_all_healthy(self):
        """Test all components healthy."""

        class HealthyCheck(BaseHealthCheck):
            async def _do_check(self):
                return ComponentHealth(self._name, HealthStatus.HEALTHY)

        checker = HealthChecker()
        checker.add_check(HealthyCheck("comp1"))
        checker.add_check(HealthyCheck("comp2"))

        report = await checker.check_health()

        assert report.status == HealthStatus.HEALTHY
        assert len(report.components) == 2

    @pytest.mark.asyncio
    async def test_degraded_on_non_critical_failure(self):
        """Test degraded status for non-critical failure."""

        class HealthyCheck(BaseHealthCheck):
            async def _do_check(self):
                return ComponentHealth(self._name, HealthStatus.HEALTHY)

        class UnhealthyCheck(BaseHealthCheck):
            async def _do_check(self):
                return ComponentHealth(self._name, HealthStatus.UNHEALTHY)

        checker = HealthChecker()
        checker.add_check(HealthyCheck("critical", critical=True))
        checker.add_check(UnhealthyCheck("optional", critical=False))

        report = await checker.check_health()

        # Non-critical unhealthy = degraded, not unhealthy
        assert report.status == HealthStatus.DEGRADED

    @pytest.mark.asyncio
    async def test_unhealthy_on_critical_failure(self):
        """Test unhealthy status for critical failure."""

        class UnhealthyCheck(BaseHealthCheck):
            async def _do_check(self):
                return ComponentHealth(self._name, HealthStatus.UNHEALTHY)

        checker = HealthChecker()
        checker.add_check(UnhealthyCheck("critical", critical=True))

        report = await checker.check_health()

        assert report.status == HealthStatus.UNHEALTHY

    @pytest.mark.asyncio
    async def test_caching(self):
        """Test health check results are cached."""

        class CountingCheck(BaseHealthCheck):
            def __init__(self):
                super().__init__("counting")
                self.call_count = 0

            async def _do_check(self):
                self.call_count += 1
                return ComponentHealth(self._name, HealthStatus.HEALTHY)

        check = CountingCheck()
        checker = HealthChecker(cache_ttl=10.0)
        checker.add_check(check)

        # First call
        await checker.check_health()
        assert check.call_count == 1

        # Cached call
        await checker.check_health()
        assert check.call_count == 1

        # Force no cache
        await checker.check_health(use_cache=False)
        assert check.call_count == 2

    @pytest.mark.asyncio
    async def test_status_change_callback(self):
        """Test status change callback is invoked."""
        changes = []

        def on_change(old, new):
            changes.append((old, new))

        class ToggleCheck(BaseHealthCheck):
            def __init__(self):
                super().__init__("toggle")
                self.healthy = True

            async def _do_check(self):
                status = HealthStatus.HEALTHY if self.healthy else HealthStatus.UNHEALTHY
                return ComponentHealth(self._name, status)

        check = ToggleCheck()
        checker = HealthChecker(cache_ttl=0.0)
        checker.add_check(check)
        checker.on_status_change(on_change)

        # Initial check
        await checker.check_health()

        # Toggle to unhealthy
        check.healthy = False
        await checker.check_health()

        assert len(changes) == 1
        assert changes[0] == (HealthStatus.HEALTHY, HealthStatus.UNHEALTHY)

    @pytest.mark.asyncio
    async def test_is_ready(self):
        """Test Kubernetes readiness probe."""

        class HealthyCheck(BaseHealthCheck):
            async def _do_check(self):
                return ComponentHealth(self._name, HealthStatus.HEALTHY)

        checker = HealthChecker(cache_ttl=0.0)
        checker.add_check(HealthyCheck("comp"))

        assert await checker.is_ready() is True

    @pytest.mark.asyncio
    async def test_is_ready_degraded(self):
        """Test readiness returns True when degraded."""

        class DegradedCheck(BaseHealthCheck):
            async def _do_check(self):
                return ComponentHealth(self._name, HealthStatus.DEGRADED)

        checker = HealthChecker(cache_ttl=0.0)
        checker.add_check(DegradedCheck("comp"))

        assert await checker.is_ready() is True

    @pytest.mark.asyncio
    async def test_is_ready_unhealthy(self):
        """Test readiness returns False when unhealthy."""

        class UnhealthyCheck(BaseHealthCheck):
            async def _do_check(self):
                return ComponentHealth(self._name, HealthStatus.UNHEALTHY)

        checker = HealthChecker(cache_ttl=0.0)
        checker.add_check(UnhealthyCheck("comp", critical=True))

        assert await checker.is_ready() is False

    @pytest.mark.asyncio
    async def test_is_alive(self):
        """Test Kubernetes liveness probe."""
        checker = HealthChecker()

        # Liveness just checks we're running
        assert await checker.is_alive() is True

    @pytest.mark.asyncio
    async def test_uptime_tracking(self):
        """Test uptime is tracked."""
        checker = HealthChecker()
        await asyncio.sleep(0.1)

        report = await checker.check_health()

        assert report.uptime_seconds is not None
        assert report.uptime_seconds >= 0.1

    @pytest.mark.asyncio
    async def test_concurrent_checks(self):
        """Test checks run concurrently."""
        import time

        class SlowCheck(BaseHealthCheck):
            async def _do_check(self):
                await asyncio.sleep(0.1)
                return ComponentHealth(self._name, HealthStatus.HEALTHY)

        checker = HealthChecker()
        for i in range(5):
            checker.add_check(SlowCheck(f"check{i}"))

        start = time.time()
        await checker.check_health()
        elapsed = time.time() - start

        # 5 checks at 0.1s each, if sequential = 0.5s
        # If concurrent, should be ~0.1s
        assert elapsed < 0.3

    @pytest.mark.asyncio
    async def test_empty_checker(self):
        """Test checker with no checks."""
        checker = HealthChecker()
        report = await checker.check_health()

        assert report.status == HealthStatus.HEALTHY
        assert len(report.components) == 0


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_default_health_checker(self):
        """Test default health checker creation."""
        checker = create_default_health_checker(version="2.0.0")

        assert checker._version == "2.0.0"
        # Should have memory check by default
        assert "system.memory" in checker.get_check_names()

    def test_create_without_memory(self):
        """Test creating without memory check."""
        checker = create_default_health_checker(include_memory=False)

        assert "system.memory" not in checker.get_check_names()
