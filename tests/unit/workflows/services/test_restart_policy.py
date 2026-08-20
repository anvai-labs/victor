# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
# SPDX-License-Identifier: Apache-2.0
"""Tests for RestartPolicyEnforcer and RestartAttempt."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from victor.workflows.services.registry import (
    RestartAttempt,
    RestartPolicyEnforcer,
    ServiceRegistry,
    ServiceRegistryEntry,
)
from victor.workflows.services.definition import (
    ServiceConfig,
    LifecycleConfig,
    ServiceHandle,
    ServiceState,
)


class TestRestartAttempt:
    """Test RestartAttempt dataclass."""

    def test_successful_attempt(self):
        """Should create successful attempt record."""
        attempt = RestartAttempt(
            timestamp=datetime(2025, 1, 1, 12, 0, 0),
            success=True,
        )
        assert attempt.success is True
        assert attempt.error is None

    def test_failed_attempt(self):
        """Should create failed attempt record."""
        attempt = RestartAttempt(
            timestamp=datetime(2025, 1, 1, 12, 0, 0),
            success=False,
            error="Connection refused",
        )
        assert attempt.success is False
        assert attempt.error == "Connection refused"


class TestRestartPolicyEnforcer:
    """Test RestartPolicyEnforcer class."""

    @pytest.fixture
    def registry(self):
        """Create a mock service registry."""
        return ServiceRegistry()

    @pytest.fixture
    def enforcer(self, registry):
        """Create enforcer with registry."""
        return RestartPolicyEnforcer(registry)

    @pytest.fixture
    def service_config_no_restart(self):
        """Service with no restart policy."""
        return ServiceConfig(
            name="no_restart_svc",
            provider="docker",
            image="postgres:15",
            lifecycle=LifecycleConfig(
                restart_policy="no",
                max_restarts=3,
            ),
        )

    @pytest.fixture
    def service_config_on_failure(self):
        """Service with on-failure restart policy."""
        return ServiceConfig(
            name="failure_restart_svc",
            provider="docker",
            image="redis:7",
            lifecycle=LifecycleConfig(
                restart_policy="on-failure",
                max_restarts=5,
            ),
        )

    @pytest.fixture
    def service_config_always(self):
        """Service with always restart policy."""
        return ServiceConfig(
            name="always_restart_svc",
            provider="docker",
            image="nginx:latest",
            lifecycle=LifecycleConfig(
                restart_policy="always",
                max_restarts=3,
            ),
        )

    def test_init(self, registry, enforcer):
        """Should initialize with registry."""
        assert enforcer.registry is registry
        assert enforcer._restart_counts == {}
        assert enforcer._monitoring is False

    def test_should_restart_no_policy(self, registry, enforcer, service_config_no_restart):
        """Should not restart with 'no' policy."""
        registry.add_service(service_config_no_restart)
        assert enforcer.should_restart("no_restart_svc", exit_code=1) is False

    def test_should_restart_manual_stop(self, registry, enforcer, service_config_always):
        """Should not restart if manually stopped."""
        registry.add_service(service_config_always)
        assert enforcer.should_restart("always_restart_svc", was_manual_stop=True) is False

    def test_should_restart_marked_manual(self, registry, enforcer, service_config_always):
        """Should not restart if marked as manually stopped."""
        registry.add_service(service_config_always)
        enforcer.mark_manual_stop("always_restart_svc")
        assert enforcer.should_restart("always_restart_svc", exit_code=1) is False

    def test_should_restart_always_policy(self, registry, enforcer, service_config_always):
        """Should restart with 'always' policy."""
        registry.add_service(service_config_always)
        assert enforcer.should_restart("always_restart_svc", exit_code=0) is True
        assert enforcer.should_restart("always_restart_svc", exit_code=1) is True

    def test_should_restart_on_failure_success(self, registry, enforcer, service_config_on_failure):
        """Should not restart on-failure policy with exit code 0."""
        registry.add_service(service_config_on_failure)
        assert enforcer.should_restart("failure_restart_svc", exit_code=0) is False

    def test_should_restart_on_failure_fail(self, registry, enforcer, service_config_on_failure):
        """Should restart on-failure policy with non-zero exit code."""
        registry.add_service(service_config_on_failure)
        assert enforcer.should_restart("failure_restart_svc", exit_code=1) is True
        assert enforcer.should_restart("failure_restart_svc", exit_code=137) is True

    def test_should_restart_on_failure_none_exit(
        self, registry, enforcer, service_config_on_failure
    ):
        """Should restart on-failure policy with None exit code."""
        registry.add_service(service_config_on_failure)
        assert enforcer.should_restart("failure_restart_svc", exit_code=None) is True

    def test_should_restart_max_reached(self, registry, enforcer, service_config_always):
        """Should not restart when max restarts reached."""
        registry.add_service(service_config_always)
        # Simulate reaching max restarts
        enforcer._restart_counts["always_restart_svc"] = 3
        assert enforcer.should_restart("always_restart_svc", exit_code=1) is False

    def test_should_restart_unknown_service(self, enforcer):
        """Should not restart unknown service."""
        assert enforcer.should_restart("unknown_service", exit_code=1) is False

    def test_get_restart_delay_first(self, enforcer):
        """First restart should have base delay."""
        delay = enforcer.get_restart_delay("test_service")
        assert delay == 1.0

    def test_get_restart_delay_exponential(self, enforcer):
        """Delay should increase exponentially."""
        enforcer._restart_counts["test_service"] = 0
        assert enforcer.get_restart_delay("test_service") == 1.0

        enforcer._restart_counts["test_service"] = 1
        assert enforcer.get_restart_delay("test_service") == 2.0

        enforcer._restart_counts["test_service"] = 2
        assert enforcer.get_restart_delay("test_service") == 4.0

        enforcer._restart_counts["test_service"] = 3
        assert enforcer.get_restart_delay("test_service") == 8.0

    def test_get_restart_delay_max(self, enforcer):
        """Delay should be capped at max."""
        enforcer._restart_counts["test_service"] = 10
        delay = enforcer.get_restart_delay("test_service")
        assert delay == 60.0  # Max delay

    def test_reset_count(self, enforcer):
        """Should reset restart count."""
        enforcer._restart_counts["test_service"] = 5
        enforcer.reset_count("test_service")
        assert "test_service" not in enforcer._restart_counts

    def test_reset_count_nonexistent(self, enforcer):
        """Should handle resetting nonexistent count."""
        enforcer.reset_count("nonexistent")  # Should not raise

    def test_mark_manual_stop(self, enforcer):
        """Should mark service as manually stopped."""
        enforcer.mark_manual_stop("test_service")
        assert "test_service" in enforcer._manually_stopped

    def test_clear_manual_stop(self, enforcer):
        """Should clear manual stop flag."""
        enforcer.mark_manual_stop("test_service")
        enforcer.clear_manual_stop("test_service")
        assert "test_service" not in enforcer._manually_stopped

    def test_clear_manual_stop_nonexistent(self, enforcer):
        """Should handle clearing nonexistent flag."""
        enforcer.clear_manual_stop("nonexistent")  # Should not raise

    def test_get_stats(self, registry, enforcer, service_config_on_failure):
        """Should return statistics."""
        registry.add_service(service_config_on_failure)
        enforcer._restart_counts["failure_restart_svc"] = 2
        enforcer.mark_manual_stop("other_service")
        enforcer._restart_history["failure_restart_svc"].append(
            RestartAttempt(
                timestamp=datetime(2025, 1, 1, 12, 0, 0),
                success=True,
            )
        )

        stats = enforcer.get_stats()

        assert stats["monitoring"] is False
        assert stats["restart_counts"]["failure_restart_svc"] == 2
        assert "other_service" in stats["manually_stopped"]
        assert len(stats["history"]["failure_restart_svc"]) == 1
        assert stats["history"]["failure_restart_svc"][0]["success"] is True

    @pytest.mark.asyncio
    async def test_start_stop_monitoring(self, enforcer):
        """Should start and stop monitoring."""
        # Start monitoring
        await enforcer.start_monitoring(check_interval=0.1)
        assert enforcer._monitoring is True
        assert enforcer._monitor_task is not None

        # Stop monitoring
        await enforcer.stop_monitoring()
        assert enforcer._monitoring is False
        assert enforcer._monitor_task is None

    @pytest.mark.asyncio
    async def test_start_monitoring_idempotent(self, enforcer):
        """Starting monitoring twice should be safe."""
        await enforcer.start_monitoring(check_interval=0.1)
        task1 = enforcer._monitor_task
        await enforcer.start_monitoring(check_interval=0.1)
        task2 = enforcer._monitor_task
        assert task1 is task2  # Same task

        await enforcer.stop_monitoring()


class TestRestartPolicyEnforcerRestart:
    """Test restart_service method of RestartPolicyEnforcer.

    Note: The restart_service method requires stop_service and start_service
    methods on the registry which are currently not implemented. These tests
    mock those methods to test the RestartPolicyEnforcer logic.
    """

    @pytest.fixture
    def registry(self):
        """Create a mock service registry with mocked methods."""
        registry = ServiceRegistry()
        # Mock the stop_service and start_service methods that would be called
        registry.stop_service = AsyncMock()
        registry.start_service = AsyncMock()
        return registry

    @pytest.fixture
    def enforcer(self, registry):
        """Create enforcer with registry."""
        return RestartPolicyEnforcer(registry)

    @pytest.fixture
    def running_service(self, registry):
        """Create a running service config."""
        config = ServiceConfig(
            name="running_svc",
            provider="docker",
            image="redis:7",
            lifecycle=LifecycleConfig(
                restart_policy="always",
                startup_timeout=30.0,
            ),
        )
        registry.add_service(config)
        entry = registry.get_service("running_svc")
        entry.handle = ServiceHandle(
            service_id="handle_123",
            config=config,
            state=ServiceState.HEALTHY,
            host="localhost",
            ports={6379: 6379},
        )
        return config

    @pytest.mark.asyncio
    async def test_restart_unknown_service(self, enforcer):
        """Should return False for unknown service."""
        result = await enforcer.restart_service("unknown")
        assert result is False

    @pytest.mark.asyncio
    async def test_restart_increments_count(self, registry, enforcer, running_service):
        """Restart should increment restart count."""
        assert enforcer._restart_counts.get("running_svc", 0) == 0
        await enforcer.restart_service("running_svc")
        assert enforcer._restart_counts["running_svc"] == 1

    @pytest.mark.asyncio
    async def test_restart_records_history(self, registry, enforcer, running_service):
        """Restart should record in history."""
        await enforcer.restart_service("running_svc")
        history = enforcer._restart_history["running_svc"]
        assert len(history) == 1
        assert history[0].success is True

    @pytest.mark.asyncio
    async def test_restart_with_delay(self, registry, enforcer, running_service):
        """Restart should respect delay."""
        import time

        start = time.time()
        await enforcer.restart_service("running_svc", delay=0.1)
        elapsed = time.time() - start
        assert elapsed >= 0.1

    @pytest.mark.asyncio
    async def test_restart_failure_recorded(self, registry, enforcer):
        """Failed restart should record error in history."""
        config = ServiceConfig(
            name="fail_svc",
            provider="docker",
            image="redis:7",
        )
        registry.add_service(config)
        registry.start_service = AsyncMock(side_effect=Exception("Start failed"))

        result = await enforcer.restart_service("fail_svc")

        assert result is False
        history = enforcer._restart_history["fail_svc"]
        assert len(history) == 1
        assert history[0].success is False
        assert "Start failed" in history[0].error
