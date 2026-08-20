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

"""Provider health monitoring.

This module provides ProviderHealthMonitor, which handles provider health
monitoring and periodic checks. Extracted from ProviderManager to follow
the Single Responsibility Principle (SRP).

Part of SOLID-based refactoring to eliminate god class anti-pattern.
"""

import asyncio
import logging
from typing import Any, List, Optional

from victor.agent.protocols import IProviderHealthMonitor
from victor.providers.base import BaseProvider

logger = logging.getLogger(__name__)


class ProviderHealthMonitor(IProviderHealthMonitor):
    """Monitors provider health and triggers fallbacks.

    This class is responsible for:
    - Checking provider health
    - Starting/stopping periodic health checks
    - Managing health check task lifecycle
    - Coordinating with ProviderHealthChecker

    SRP Compliance: Focuses only on health monitoring, delegating
    provider switching and tool coordination to specialized components.

    Attributes:
        _settings: Application settings for configuration
        _health_checker: Optional ProviderHealthChecker instance
        _health_check_task: Background task for periodic checks
        _enable_health_checks: Whether health checks are enabled
        _health_check_interval: Interval between checks (seconds)
    """

    def __init__(
        self,
        settings: Any,
        enable_health_checks: bool = True,
        health_check_interval: float = 60.0,
    ):
        """Initialize the health monitor.

        Args:
            settings: Application settings
            enable_health_checks: Whether to enable health checks
            health_check_interval: Interval between health checks in seconds
        """
        self._settings = settings
        self._health_checker: Optional[Any] = None
        self._health_check_task: Optional[asyncio.Task] = None
        self._enable_health_checks = enable_health_checks
        self._health_check_interval = health_check_interval

    async def check_health(self, provider: BaseProvider) -> bool:
        """Check if provider is healthy.

        Args:
            provider: Provider instance to check

        Returns:
            True if provider is healthy, False otherwise
        """
        # Skip health check if disabled
        if not self._enable_health_checks:
            return True

        try:
            # Lazy initialize health checker
            if not self._health_checker:
                from victor.providers.health import ProviderHealthChecker

                self._health_checker = ProviderHealthChecker()

            # Infer provider name and model from provider object
            provider_name = getattr(provider, "name", "unknown")
            provider_model = getattr(provider, "model", "default")

            # Register and check
            self._health_checker.register_provider(provider_name, provider)
            result = await self._health_checker.check_provider(
                provider_name,
                provider_model,
                check_connectivity=False,  # Skip connectivity for faster checks
            )

            from victor.providers.health import HealthStatus

            return result.status in (HealthStatus.HEALTHY, HealthStatus.DEGRADED)

        except ImportError:
            logger.debug("Health checker not available, skipping health check")
            return True
        except Exception as e:
            logger.debug(f"Health check failed: {e}")
            return True  # Assume healthy on error

    async def start_health_checks(
        self,
        interval: Optional[float] = None,
        provider: Optional[BaseProvider] = None,
        provider_name: Optional[str] = None,
    ) -> None:
        """Start periodic health checks.

        Args:
            interval: Interval between checks (uses default if not provided)
            provider: Provider to monitor (optional, for caller tracking)
            provider_name: Provider name (optional, for logging)
        """
        if not self._enable_health_checks:
            return

        if self._health_check_task and not self._health_check_task.done():
            logger.debug("Health monitoring already running")
            return

        check_interval = interval or self._health_check_interval

        async def monitor_loop():
            """Background health check loop."""
            while True:
                try:
                    await asyncio.sleep(check_interval)
                    # Health check is performed by caller
                    # This loop just provides timing
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.debug(f"Health monitoring error: {e}")

        self._health_check_task = asyncio.create_task(monitor_loop())
        logger.info(f"Started provider health monitoring (interval: {check_interval}s)")

    async def stop_health_checks(self) -> None:
        """Stop health checks."""
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
            self._health_check_task = None
            logger.debug("Stopped provider health monitoring")

    def is_monitoring(self) -> bool:
        """Check if health monitoring is currently active.

        Returns:
            True if monitoring is active, False otherwise
        """
        return self._health_check_task is not None and not self._health_check_task.done()

    async def get_healthy_providers(self) -> List[str]:
        """Get list of healthy providers.

        Returns:
            List of healthy provider names sorted by latency
        """
        if not self._health_checker:
            return []

        return self._health_checker.get_healthy_providers()
