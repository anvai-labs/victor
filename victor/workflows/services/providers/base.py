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

"""Base service provider with common functionality."""

from __future__ import annotations

import asyncio
import logging
import socket
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Optional

import aiohttp

from victor.workflows.services.definition import (
    HealthCheckConfig,
    HealthCheckType,
    ServiceConfig,
    ServiceHandle,
    ServiceHealthError,
    ServiceState,
)

logger = logging.getLogger(__name__)


class BaseServiceProvider(ABC):
    """Base class for service providers with common health check logic.

    Subclasses must implement:
    - _do_start: Start the service
    - _do_stop: Stop the service
    - _do_cleanup: Force cleanup
    - get_logs: Retrieve logs

    The base class provides:
    - Health check implementations (TCP, HTTP, command)
    - Retry logic with backoff
    - State management
    """

    @abstractmethod
    async def _do_start(self, config: ServiceConfig) -> ServiceHandle:
        """Implementation-specific start logic."""
        ...

    @abstractmethod
    async def _do_stop(self, handle: ServiceHandle, grace_period: float) -> None:
        """Implementation-specific stop logic."""
        ...

    @abstractmethod
    async def _do_cleanup(self, handle: ServiceHandle) -> None:
        """Implementation-specific cleanup logic."""
        ...

    @abstractmethod
    async def get_logs(self, handle: ServiceHandle, tail: int = 100) -> str:
        """Get service logs."""
        ...

    async def _run_command_in_service(
        self,
        handle: ServiceHandle,
        command: str,
    ) -> tuple[int, str]:
        """Run a command inside the service (for health checks).

        Returns:
            Tuple of (exit_code, output)
        """
        # Default implementation - subclasses should override
        raise NotImplementedError("Command execution not supported by this provider")

    async def start(self, config: ServiceConfig) -> ServiceHandle:
        """Start the service and wait for healthy status.

        Args:
            config: Service configuration

        Returns:
            ServiceHandle with runtime info

        Raises:
            ServiceStartError: If start fails
            ServiceHealthError: If health check fails
        """
        handle = await self._do_start(config)
        handle.state = ServiceState.STARTING
        handle.started_at = datetime.now(timezone.utc)

        # Wait for health check if configured
        if config.health_check:
            try:
                await self._wait_for_healthy(handle, config.health_check)
                handle.state = ServiceState.HEALTHY
                handle.healthy_at = datetime.now(timezone.utc)
            except ServiceHealthError:
                handle.state = ServiceState.UNHEALTHY
                # Try to cleanup
                try:
                    await self._do_cleanup(handle)
                except Exception as e:
                    logger.warning(f"Cleanup failed after health check failure: {e}")
                raise
        else:
            # No health check - assume healthy immediately
            handle.state = ServiceState.HEALTHY
            handle.healthy_at = datetime.now(timezone.utc)

        # Resolve export templates
        handle.connection_info = handle.resolve_exports()

        logger.info(
            f"Service '{config.name}' started successfully "
            f"(id={handle.service_id}, host={handle.host})"
        )
        return handle

    async def stop(self, handle: ServiceHandle, grace_period: float = 30.0) -> None:
        """Stop the service gracefully.

        Args:
            handle: Service handle
            grace_period: Time to wait for graceful shutdown
        """
        if handle.state in (ServiceState.STOPPED, ServiceState.FAILED):
            logger.debug(f"Service '{handle.config.name}' already stopped")
            return

        handle.state = ServiceState.STOPPING
        logger.info(f"Stopping service '{handle.config.name}'...")

        try:
            await self._do_stop(handle, grace_period)
            handle.state = ServiceState.STOPPED
            logger.info(f"Service '{handle.config.name}' stopped")
        except Exception as e:
            logger.error(f"Failed to stop service '{handle.config.name}': {e}")
            handle.state = ServiceState.FAILED
            handle.error = str(e)
            raise

    async def cleanup(self, handle: ServiceHandle) -> None:
        """Force cleanup service resources."""
        try:
            await self._do_cleanup(handle)
            handle.state = ServiceState.STOPPED
        except Exception as e:
            logger.warning(f"Cleanup failed for '{handle.config.name}': {e}")
            handle.state = ServiceState.FAILED
            handle.error = str(e)

    async def health_check(self, handle: ServiceHandle) -> bool:
        """Check if the service is healthy.

        Args:
            handle: Service handle

        Returns:
            True if healthy
        """
        config = handle.config.health_check
        if not config:
            return True

        try:
            return await self._do_health_check(handle, config)
        except Exception as e:
            logger.debug(f"Health check failed for '{handle.config.name}': {e}")
            return False

    async def _wait_for_healthy(
        self,
        handle: ServiceHandle,
        config: HealthCheckConfig,
    ) -> None:
        """Wait for service to become healthy.

        Args:
            handle: Service handle
            config: Health check configuration

        Raises:
            ServiceHealthError: If health check times out
        """
        # Wait for start period
        if config.start_period > 0:
            logger.debug(f"Waiting {config.start_period}s start period for '{handle.config.name}'")
            await asyncio.sleep(config.start_period)

        attempts = 0
        last_error = None

        while attempts < config.retries:
            attempts += 1

            try:
                if await self._do_health_check(handle, config):
                    logger.debug(
                        f"Health check passed for '{handle.config.name}' "
                        f"after {attempts} attempts"
                    )
                    return
            except Exception as e:
                last_error = e
                logger.debug(
                    f"Health check attempt {attempts}/{config.retries} failed "
                    f"for '{handle.config.name}': {e}"
                )

            if attempts < config.retries:
                await asyncio.sleep(config.interval)

        raise ServiceHealthError(
            service_name=handle.config.name,
            message=f"Health check failed: {last_error}",
            attempts=attempts,
        )

    async def _do_health_check(
        self,
        handle: ServiceHandle,
        config: HealthCheckConfig,
    ) -> bool:
        """Perform a single health check.

        Args:
            handle: Service handle
            config: Health check configuration

        Returns:
            True if healthy
        """
        check_type = config.type

        if check_type == HealthCheckType.TCP:
            return await self._check_tcp(handle, config)

        elif check_type in (HealthCheckType.HTTP, HealthCheckType.HTTPS):
            return await self._check_http(handle, config)

        elif check_type == HealthCheckType.COMMAND:
            return await self._check_command(handle, config)

        elif check_type == HealthCheckType.POSTGRES:
            return await self._check_postgres(handle, config)

        elif check_type == HealthCheckType.REDIS:
            return await self._check_redis(handle, config)

        elif check_type == HealthCheckType.MYSQL:
            return await self._check_mysql(handle, config)

        elif check_type == HealthCheckType.KAFKA:
            return await self._check_kafka(handle, config)

        elif check_type == HealthCheckType.GRPC:
            return await self._check_grpc(handle, config)

        else:
            logger.warning(f"Unknown health check type: {check_type}")
            return True

    async def _check_tcp(
        self,
        handle: ServiceHandle,
        config: HealthCheckConfig,
    ) -> bool:
        """Check TCP port connectivity."""
        host = config.host or handle.host
        port = handle.get_port(config.port) or config.port

        if not port:
            raise ValueError("No port configured for TCP health check")

        try:
            # Use asyncio to check port
            _, writer = await asyncio.wait_for(
                asyncio.open_connection(host, port),
                timeout=config.timeout,
            )
            writer.close()
            await writer.wait_closed()
            return True
        except (asyncio.TimeoutError, ConnectionRefusedError, OSError) as e:
            logger.debug(f"TCP check failed for {host}:{port}: {e}")
            return False

    async def _check_http(
        self,
        handle: ServiceHandle,
        config: HealthCheckConfig,
    ) -> bool:
        """Check HTTP endpoint."""
        host = config.host or handle.host
        port = handle.get_port(config.port) or config.port

        if not port:
            raise ValueError("No port configured for HTTP health check")

        scheme = "https" if config.type == HealthCheckType.HTTPS else "http"
        url = f"{scheme}://{host}:{port}{config.path}"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    timeout=aiohttp.ClientTimeout(total=config.timeout),
                    ssl=False if scheme == "https" else None,
                ) as response:
                    if response.status in config.expected_status:
                        if config.expected_output:
                            body = await response.text()
                            return config.expected_output in body
                        return True
                    logger.debug(
                        f"HTTP check got status {response.status}, "
                        f"expected {config.expected_status}"
                    )
                    return False
        except Exception as e:
            logger.debug(f"HTTP check failed for {url}: {e}")
            return False

    async def _check_command(
        self,
        handle: ServiceHandle,
        config: HealthCheckConfig,
    ) -> bool:
        """Run command inside service container."""
        if not config.command:
            raise ValueError("No command configured for command health check")

        try:
            exit_code, output = await self._run_command_in_service(handle, config.command)

            if exit_code != 0:
                logger.debug(f"Command check failed with exit code {exit_code}")
                return False

            if config.expected_output:
                return config.expected_output in output

            return True
        except NotImplementedError:
            # Fall back to TCP check on first port
            logger.debug("Command execution not supported, falling back to TCP")
            if config.port:
                return await self._check_tcp(handle, config)
            return True
        except Exception as e:
            logger.debug(f"Command check failed: {e}")
            return False

    async def _check_postgres(
        self,
        handle: ServiceHandle,
        config: HealthCheckConfig,
    ) -> bool:
        """Check PostgreSQL readiness."""
        # Try pg_isready command first
        if hasattr(self, "_run_command_in_service"):
            try:
                exit_code, _ = await self._run_command_in_service(handle, "pg_isready -U postgres")
                return exit_code == 0
            except NotImplementedError:
                pass

        # Fall back to TCP check
        return await self._check_tcp(handle, config)

    async def _check_redis(
        self,
        handle: ServiceHandle,
        config: HealthCheckConfig,
    ) -> bool:
        """Check Redis readiness."""
        # Try redis-cli ping first
        if hasattr(self, "_run_command_in_service"):
            try:
                exit_code, output = await self._run_command_in_service(handle, "redis-cli ping")
                return exit_code == 0 and "PONG" in output
            except NotImplementedError:
                pass

        # Fall back to TCP check
        return await self._check_tcp(handle, config)

    async def _check_mysql(
        self,
        handle: ServiceHandle,
        config: HealthCheckConfig,
    ) -> bool:
        """Check MySQL readiness."""
        if hasattr(self, "_run_command_in_service"):
            try:
                exit_code, _ = await self._run_command_in_service(
                    handle, "mysqladmin ping -h localhost"
                )
                return exit_code == 0
            except NotImplementedError:
                pass

        return await self._check_tcp(handle, config)

    async def _check_kafka(
        self,
        handle: ServiceHandle,
        config: HealthCheckConfig,
    ) -> bool:
        """Check Kafka readiness."""
        # For Kafka, we just check TCP since kafka-topics requires the broker
        return await self._check_tcp(handle, config)

    async def _check_grpc(
        self,
        handle: ServiceHandle,
        config: HealthCheckConfig,
    ) -> bool:
        """Check gRPC health endpoint using standard gRPC health checking protocol.

        The gRPC health checking v1 protocol defines a Check service that returns
        the serving status of a service. This implementation uses the standard
        health check request with empty payload.

        Returns:
            True if service is SERVING, False otherwise
        """
        # First check TCP connectivity
        if not await self._check_tcp(handle, config):
            return False

        # Try gRPC health check if grpcio is available
        try:
            import grpc
            from grpc_health.v1 import health, health_pb2, health_pb2_grpc

            # Create channel to the health check endpoint
            # Default health check port is usually the service port + 1 or same port
            host = handle.config.get("health_check_host", handle.config.get("host", "localhost"))
            port = handle.config.get("health_check_port", handle.config.get("port", 50051))

            channel = grpc.aio.insecure_channel(f"{host}:{port}")
            try:
                # Create health check stub
                health_stub = health_pb2_grpc.HealthStub(channel)

                # Call Check with empty request (standard health check)
                request = health_pb2.HealthCheckRequest(service="")
                response = await asyncio.wait_for(
                    health_stub.Check(request, timeout=config.timeout),
                    timeout=config.timeout + 1,
                )

                # Check response status
                # SERVING = 1, NOT_SERVING = 2, SERVICE_UNKNOWN = 3
                return response.status == health.HealthCheckResponse.SERVING

            except asyncio.TimeoutError:
                logger.warning(f"gRPC health check timeout for {handle.name}")
                return False
            except grpc.aio.AioRpcError as e:
                # Service might not implement health check, fall back to TCP
                if e.code() == grpc.StatusCode.UNIMPLEMENTED:
                    logger.debug(f"gRPC health check not implemented for {handle.name}, using TCP")
                    return True  # TCP already passed
                logger.warning(f"gRPC health check failed for {handle.name}: {e.code()}")
                return False
            finally:
                await channel.close()

        except ImportError:
            # grpcio-health-checking not available, use TCP only
            logger.debug("grpcio-health-checking not available, using TCP check")
            return True
        except Exception as e:
            logger.warning(f"gRPC health check error for {handle.name}: {e}")
            # Fall back to TCP which already passed
            return True
