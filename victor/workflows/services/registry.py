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

"""Service registry for workflow service lifecycle management.

Tracks running services, manages startup/shutdown order, and provides
connection info to workflow nodes.

Example:
    registry = ServiceRegistry()

    # Register providers
    registry.register_provider("docker", DockerServiceProvider())
    registry.register_provider("kubernetes", KubernetesServiceProvider())

    # Start services
    await registry.start_all(service_configs)

    # Get connection info
    postgres_url = registry.get_export("postgres", "DATABASE_URL")

    # Stop all services (reverse order)
    await registry.stop_all()
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Set

from victor.workflows.services.definition import (
    ServiceConfig,
    ServiceDependencyError,
    ServiceError,
    ServiceHandle,
    ServiceHealthError,
    ServiceProvider,
    ServiceProviderType,
    ServiceState,
)

logger = logging.getLogger(__name__)


@dataclass
class ServiceRegistryEntry:
    """Entry in the service registry."""

    config: ServiceConfig
    handle: Optional[ServiceHandle] = None
    provider: Optional[ServiceProvider] = None
    started_at: Optional[datetime] = None
    stopped_at: Optional[datetime] = None
    error: Optional[str] = None


class ServiceRegistry:
    """Registry for managing workflow service lifecycle.

    Responsibilities:
    1. Track service providers by type
    2. Manage service startup order (respecting dependencies)
    3. Track running services and their handles
    4. Provide connection info to workflow nodes
    5. Ensure cleanup on workflow completion/failure

    Thread-safety: This class is designed for single-threaded async use.
    For multi-workflow scenarios, create separate registry instances.
    """

    def __init__(self):
        self._providers: Dict[ServiceProviderType, ServiceProvider] = {}
        self._services: Dict[str, ServiceRegistryEntry] = {}
        self._startup_order: List[str] = []
        self._on_service_started: List[Callable[[ServiceHandle], None]] = []
        self._on_service_stopped: List[Callable[[str], None]] = []

    def register_provider(
        self, provider_type: ServiceProviderType, provider: ServiceProvider
    ) -> None:
        """Register a service provider.

        Args:
            provider_type: Provider type (docker, kubernetes, etc.)
            provider: Provider instance
        """
        self._providers[provider_type] = provider
        logger.debug(f"Registered provider: {provider_type}")

    def get_provider(self, provider_type: ServiceProviderType) -> Optional[ServiceProvider]:
        """Get provider by type."""
        return self._providers.get(provider_type)

    def add_service(self, config: ServiceConfig) -> None:
        """Add a service to the registry.

        Args:
            config: Service configuration
        """
        self._services[config.name] = ServiceRegistryEntry(config=config)
        logger.debug(f"Added service to registry: {config.name}")

    def get_service(self, name: str) -> Optional[ServiceRegistryEntry]:
        """Get service entry by name."""
        return self._services.get(name)

    def get_handle(self, name: str) -> Optional[ServiceHandle]:
        """Get service handle by name."""
        entry = self._services.get(name)
        return entry.handle if entry else None

    def get_export(self, service_name: str, export_key: str) -> Optional[str]:
        """Get an export value from a service.

        Args:
            service_name: Service name
            export_key: Export key (e.g., DATABASE_URL)

        Returns:
            Export value or None if not found
        """
        handle = self.get_handle(service_name)
        if handle:
            return handle.connection_info.get(export_key)
        return None

    def get_all_exports(self) -> Dict[str, Dict[str, str]]:
        """Get all exports from all services.

        Returns:
            Dict mapping service name to export dict
        """
        exports = {}
        for name, entry in self._services.items():
            if entry.handle:
                exports[name] = entry.handle.connection_info
        return exports

    def is_healthy(self, name: str) -> bool:
        """Check if a service is healthy."""
        handle = self.get_handle(name)
        return handle is not None and handle.state == ServiceState.HEALTHY

    def all_healthy(self) -> bool:
        """Check if all services are healthy."""
        return all(
            entry.handle and entry.handle.state == ServiceState.HEALTHY
            for entry in self._services.values()
        )

    async def start_all(
        self,
        configs: Optional[List[ServiceConfig]] = None,
        timeout: float = 300.0,
    ) -> None:
        """Start all services in dependency order.

        Args:
            configs: Optional list of configs to add
            timeout: Overall timeout for all services

        Raises:
            ServiceStartError: If any service fails to start
            ServiceDependencyError: If dependencies can't be resolved
        """
        if configs:
            for config in configs:
                self.add_service(config)

        # Compute startup order (topological sort)
        self._startup_order = self._compute_startup_order()

        start_time = datetime.now(timezone.utc)

        for service_name in self._startup_order:
            elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
            remaining = timeout - elapsed

            if remaining <= 0:
                raise ServiceHealthError(
                    service_name, "Timeout starting services", len(self._startup_order)
                )

            await self._start_service(service_name, remaining)

        logger.info(f"Started {len(self._startup_order)} services")

    async def _start_service(self, name: str, timeout: float) -> None:
        """Start a single service."""
        entry = self._services.get(name)
        if not entry:
            raise ServiceError(name, "Service not found in registry")

        config = entry.config

        # Get provider
        provider = self._providers.get(config.provider)
        if not provider:
            raise ServiceError(name, f"No provider registered for type: {config.provider}")

        entry.provider = provider

        # Check dependencies are healthy
        for dep_name in config.depends_on:
            if not self.is_healthy(dep_name):
                raise ServiceDependencyError(name, dep_name)

        logger.info(f"Starting service: {name}")

        try:
            # Start service
            handle = await asyncio.wait_for(
                provider.start(config),
                timeout=timeout,
            )

            entry.handle = handle
            entry.started_at = datetime.now(timezone.utc)

            # Notify listeners
            for callback in self._on_service_started:
                try:
                    callback(handle)
                except Exception as e:
                    logger.warning(f"Service started callback failed: {e}")

            logger.info(
                f"Service '{name}' started successfully "
                f"(host={handle.host}, ports={handle.ports})"
            )

        except asyncio.TimeoutError:
            entry.error = "Timeout"
            raise ServiceHealthError(name, "Timeout starting service", 1)
        except Exception as e:
            entry.error = str(e)
            raise

    async def stop_all(self, grace_period: float = 30.0) -> None:
        """Stop all services in reverse startup order.

        Args:
            grace_period: Grace period for each service
        """
        # Stop in reverse order
        for service_name in reversed(self._startup_order):
            await self._stop_service(service_name, grace_period)

        logger.info("All services stopped")

    async def _stop_service(self, name: str, grace_period: float) -> None:
        """Stop a single service."""
        entry = self._services.get(name)
        if not entry or not entry.handle:
            return

        logger.info(f"Stopping service: {name}")

        try:
            if entry.provider:
                await asyncio.wait_for(
                    entry.provider.stop(entry.handle, grace_period),
                    timeout=grace_period + 10,
                )

            entry.stopped_at = datetime.now(timezone.utc)

            # Notify listeners
            for callback in self._on_service_stopped:
                try:
                    callback(name)
                except Exception as e:
                    logger.warning(f"Service stopped callback failed: {e}")

        except asyncio.TimeoutError:
            logger.warning(f"Timeout stopping service '{name}', forcing cleanup")
            if entry.provider:
                await entry.provider.cleanup(entry.handle)
        except Exception as e:
            logger.error(f"Error stopping service '{name}': {e}")
            if entry.provider:
                await entry.provider.cleanup(entry.handle)

    async def cleanup_all(self) -> None:
        """Force cleanup all services."""
        for name, entry in self._services.items():
            if entry.handle and entry.provider:
                try:
                    await entry.provider.cleanup(entry.handle)
                except Exception as e:
                    logger.warning(f"Cleanup failed for '{name}': {e}")

    def _compute_startup_order(self) -> List[str]:
        """Compute startup order using topological sort.

        Returns:
            List of service names in startup order

        Raises:
            ServiceDependencyError: If circular dependency detected
        """
        # Build dependency graph
        in_degree: Dict[str, int] = dict.fromkeys(self._services, 0)
        dependents: Dict[str, List[str]] = defaultdict(list)

        for name, entry in self._services.items():
            for dep in entry.config.depends_on:
                if dep not in self._services:
                    raise ServiceDependencyError(name, dep)
                in_degree[name] += 1
                dependents[dep].append(name)

        # Add startup_order as secondary sort key
        services_by_order = sorted(
            self._services.keys(),
            key=lambda n: self._services[n].config.lifecycle.startup_order,
        )

        # Kahn's algorithm
        queue = [n for n in services_by_order if in_degree[n] == 0]
        result = []

        while queue:
            # Pop in startup_order priority
            queue.sort(key=lambda n: self._services[n].config.lifecycle.startup_order)
            current = queue.pop(0)
            result.append(current)

            for dependent in dependents[current]:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)

        if len(result) != len(self._services):
            # Circular dependency
            remaining = set(self._services.keys()) - set(result)
            raise ServiceDependencyError(list(remaining)[0], "Circular dependency detected")

        return result

    def on_service_started(self, callback: Callable[[ServiceHandle], None]) -> None:
        """Register callback for service started events."""
        self._on_service_started.append(callback)

    def on_service_stopped(self, callback: Callable[[str], None]) -> None:
        """Register callback for service stopped events."""
        self._on_service_stopped.append(callback)

    def list_services(self) -> List[Dict[str, Any]]:
        """List all services with their status."""
        return [
            {
                "name": name,
                "provider": entry.config.provider,
                "state": entry.handle.state.name if entry.handle else "PENDING",
                "host": entry.handle.host if entry.handle else None,
                "ports": entry.handle.ports if entry.handle else {},
                "started_at": (entry.started_at.isoformat() if entry.started_at else None),
                "error": entry.error,
            }
            for name, entry in self._services.items()
        ]


class ServiceContext:
    """Context manager for service lifecycle in workflows.

    Ensures services are started before workflow execution and
    stopped afterward (even on failure).

    Example:
        async with ServiceContext(registry, configs) as ctx:
            # Services are running
            db_url = ctx.get_export("postgres", "DATABASE_URL")
            # ... run workflow ...
        # Services automatically stopped
    """

    def __init__(
        self,
        registry: ServiceRegistry,
        configs: List[ServiceConfig],
        timeout: float = 300.0,
        cleanup_on_failure: bool = True,
    ):
        self.registry = registry
        self.configs = configs
        self.timeout = timeout
        self.cleanup_on_failure = cleanup_on_failure
        self._started = False

    async def __aenter__(self) -> "ServiceContext":
        await self.registry.start_all(self.configs, self.timeout)
        self._started = True
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._started:
            if exc_type and self.cleanup_on_failure:
                await self.registry.cleanup_all()
            else:
                await self.registry.stop_all()
        return False

    def get_export(self, service_name: str, export_key: str) -> Optional[str]:
        """Get export value from a service."""
        return self.registry.get_export(service_name, export_key)

    def get_all_exports(self) -> Dict[str, Dict[str, str]]:
        """Get all exports."""
        return self.registry.get_all_exports()

    def is_healthy(self, service_name: str) -> bool:
        """Check if a service is healthy."""
        return self.registry.is_healthy(service_name)


# =============================================================================
# Restart Policy Enforcement
# =============================================================================


@dataclass
class RestartAttempt:
    """Record of a service restart attempt."""

    timestamp: datetime
    success: bool
    error: Optional[str] = None


class RestartPolicyEnforcer:
    """Enforces restart policies for services.

    Monitors service health and automatically restarts services
    according to their configured restart policy. All verticals
    (including third-party plugins) benefit from automatic service
    recovery without manual intervention.

    Restart policies:
    - "no": Never restart (default)
    - "on-failure": Restart only when service fails
    - "always": Always restart unless explicitly stopped

    Example:
        enforcer = RestartPolicyEnforcer(registry)

        # Check if service should restart
        if enforcer.should_restart("postgres", exit_code=1):
            await enforcer.restart_service("postgres")

        # Start monitoring all services
        await enforcer.start_monitoring(check_interval=10.0)
    """

    def __init__(self, registry: ServiceRegistry):
        """Initialize restart policy enforcer.

        Args:
            registry: Service registry to monitor
        """
        self.registry = registry
        self._restart_counts: Dict[str, int] = {}
        self._restart_history: Dict[str, List[RestartAttempt]] = defaultdict(list)
        self._monitoring = False
        self._monitor_task: Optional[asyncio.Task] = None
        self._manually_stopped: Set[str] = set()

    def should_restart(
        self,
        service_name: str,
        exit_code: Optional[int] = None,
        was_manual_stop: bool = False,
    ) -> bool:
        """Determine if service should be restarted.

        Args:
            service_name: Service to check
            exit_code: Exit code if service stopped (0 = success)
            was_manual_stop: True if service was stopped intentionally

        Returns:
            True if service should be restarted
        """
        if was_manual_stop or service_name in self._manually_stopped:
            return False

        entry = self.registry.get_service(service_name)
        if not entry:
            return False

        lifecycle = entry.config.lifecycle
        policy = lifecycle.restart_policy
        max_restarts = lifecycle.max_restarts

        # Check restart limit
        current_count = self._restart_counts.get(service_name, 0)
        if current_count >= max_restarts:
            logger.warning(
                f"Service '{service_name}' reached max restarts "
                f"({max_restarts}), not restarting"
            )
            return False

        # Apply policy
        if policy == "no":
            return False
        elif policy == "always":
            return True
        elif policy == "on-failure":
            # Restart only if exit code indicates failure
            return exit_code is None or exit_code != 0

        return False

    async def restart_service(
        self,
        service_name: str,
        delay: float = 0.0,
    ) -> bool:
        """Restart a service.

        Args:
            service_name: Service to restart
            delay: Delay before restart (for backoff)

        Returns:
            True if restart succeeded
        """
        entry = self.registry.get_service(service_name)
        if not entry or not entry.config:
            logger.error(f"Cannot restart unknown service: {service_name}")
            return False

        if delay > 0:
            logger.info(f"Restarting '{service_name}' in {delay:.1f}s")
            await asyncio.sleep(delay)

        try:
            # Increment restart count
            self._restart_counts[service_name] = self._restart_counts.get(service_name, 0) + 1

            # Stop if still running
            if entry.handle and entry.handle.state == ServiceState.HEALTHY:
                await self.registry.stop_service(service_name)

            # Start again
            await self.registry.start_service(
                entry.config,
                timeout=entry.config.lifecycle.startup_timeout,
            )

            # Record success
            self._restart_history[service_name].append(
                RestartAttempt(
                    timestamp=datetime.now(),
                    success=True,
                )
            )

            logger.info(
                f"Service '{service_name}' restarted successfully "
                f"(attempt {self._restart_counts[service_name]})"
            )
            return True

        except Exception as e:
            self._restart_history[service_name].append(
                RestartAttempt(
                    timestamp=datetime.now(),
                    success=False,
                    error=str(e),
                )
            )
            logger.error(f"Failed to restart '{service_name}': {e}")
            return False

    def get_restart_delay(self, service_name: str) -> float:
        """Calculate delay before next restart (exponential backoff).

        Args:
            service_name: Service name

        Returns:
            Delay in seconds before next restart
        """
        count = self._restart_counts.get(service_name, 0)
        base_delay = 1.0
        max_delay = 60.0
        delay = min(base_delay * (2**count), max_delay)
        return delay

    def reset_count(self, service_name: str) -> None:
        """Reset restart count for a service.

        Call when service has been healthy for a period.

        Args:
            service_name: Service to reset
        """
        self._restart_counts.pop(service_name, None)
        logger.debug(f"Reset restart count for '{service_name}'")

    def mark_manual_stop(self, service_name: str) -> None:
        """Mark service as manually stopped (don't auto-restart).

        Args:
            service_name: Service that was manually stopped
        """
        self._manually_stopped.add(service_name)

    def clear_manual_stop(self, service_name: str) -> None:
        """Clear manual stop flag (allow auto-restart again).

        Args:
            service_name: Service to allow restarts for
        """
        self._manually_stopped.discard(service_name)

    async def start_monitoring(
        self,
        check_interval: float = 10.0,
        healthy_threshold: int = 3,
    ) -> None:
        """Start monitoring services for restart.

        Args:
            check_interval: Seconds between health checks
            healthy_threshold: Consecutive healthy checks to reset count
        """
        if self._monitoring:
            return

        self._monitoring = True
        self._monitor_task = asyncio.create_task(
            self._monitor_loop(check_interval, healthy_threshold)
        )
        logger.info("Started service restart monitoring")

    async def stop_monitoring(self) -> None:
        """Stop monitoring services."""
        self._monitoring = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            self._monitor_task = None
        logger.info("Stopped service restart monitoring")

    async def _monitor_loop(
        self,
        check_interval: float,
        healthy_threshold: int,
    ) -> None:
        """Main monitoring loop."""
        healthy_counts: Dict[str, int] = defaultdict(int)

        while self._monitoring:
            await asyncio.sleep(check_interval)

            for name, entry in self.registry._services.items():
                if not entry.handle:
                    continue

                try:
                    is_healthy = await self.registry._check_service_health(name)

                    if is_healthy:
                        healthy_counts[name] += 1
                        # Reset restart count after sustained health
                        if healthy_counts[name] >= healthy_threshold:
                            self.reset_count(name)
                            healthy_counts[name] = 0
                    else:
                        healthy_counts[name] = 0
                        # Check if we should restart
                        if self.should_restart(name, exit_code=1):
                            delay = self.get_restart_delay(name)
                            await self.restart_service(name, delay=delay)

                except Exception as e:
                    logger.error(f"Error checking health of '{name}': {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get restart policy enforcement statistics."""
        return {
            "monitoring": self._monitoring,
            "restart_counts": dict(self._restart_counts),
            "manually_stopped": list(self._manually_stopped),
            "history": {
                name: [
                    {
                        "timestamp": a.timestamp.isoformat(),
                        "success": a.success,
                        "error": a.error,
                    }
                    for a in attempts
                ]
                for name, attempts in self._restart_history.items()
            },
        }


__all__ = [
    "ServiceRegistryEntry",
    "ServiceRegistry",
    "ServiceContext",
    "RestartAttempt",
    "RestartPolicyEnforcer",
]
