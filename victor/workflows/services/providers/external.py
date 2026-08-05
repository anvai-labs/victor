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

"""External service provider for connecting to already-running services.

Used for managed cloud services (RDS, ElastiCache, Azure SQL, etc.)
or any externally managed infrastructure where Victor only needs to
verify connectivity, not manage the lifecycle.

Example:
    provider = ExternalServiceProvider()

    config = ServiceConfig(
        name="prod-db",
        provider="external",
        endpoint="mydb.cluster-xxx.us-east-1.rds.amazonaws.com",
        ports=[PortMapping(5432)],
        health_check=HealthCheckConfig.for_postgres(5432),
        exports={"DATABASE_URL": "postgresql://..."},
    )

    handle = await provider.start(config)  # Only verifies connectivity
    await provider.stop(handle)  # No-op
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlparse

from victor.workflows.services.definition import (
    HealthCheckConfig,
    ServiceConfig,
    ServiceHandle,
    ServiceHealthError,
    ServiceStartError,
    ServiceState,
)
from victor.workflows.services.providers.base import BaseServiceProvider

logger = logging.getLogger(__name__)


class ExternalServiceProvider(BaseServiceProvider):
    """Service provider for externally managed services.

    This provider doesn't start or stop services - it only:
    1. Verifies connectivity via health checks
    2. Resolves connection info and exports

    Use for:
    - Managed databases (RDS, Cloud SQL, Azure SQL)
    - Managed caches (ElastiCache, Memorystore)
    - External APIs and services
    - Shared infrastructure
    """

    def __init__(self, verify_on_start: bool = True):
        """Initialize external service provider.

        Args:
            verify_on_start: Whether to verify connectivity on start
        """
        self._verify_on_start = verify_on_start

    async def _do_start(self, config: ServiceConfig) -> ServiceHandle:
        """Verify external service connectivity."""
        handle = ServiceHandle.create(config)

        # Parse endpoint
        if config.endpoint:
            parsed = self._parse_endpoint(config.endpoint)
            handle.host = parsed["host"]
            if parsed.get("port"):
                for pm in config.ports:
                    handle.ports[pm.container_port] = parsed["port"]
        elif config.connection_string:
            parsed = self._parse_connection_string(config.connection_string)
            handle.host = parsed.get("host", "localhost")
            if parsed.get("port"):
                for pm in config.ports:
                    handle.ports[pm.container_port] = parsed["port"]

        # Default port mappings (external services use same port)
        for pm in config.ports:
            if pm.container_port not in handle.ports:
                handle.ports[pm.container_port] = pm.container_port

        handle.state = ServiceState.STARTING
        handle.started_at = datetime.now(timezone.utc)

        logger.info(f"Connecting to external service '{config.name}' at {handle.host}")

        return handle

    async def _do_stop(self, handle: ServiceHandle, grace_period: float) -> None:
        """No-op for external services."""
        logger.debug(f"External service '{handle.config.name}' - no stop action needed")

    async def _do_cleanup(self, handle: ServiceHandle) -> None:
        """No-op for external services."""
        pass

    async def get_logs(self, handle: ServiceHandle, tail: int = 100) -> str:
        """External services don't provide logs through this interface."""
        return "[External service - logs not available through Victor]"

    async def _run_command_in_service(
        self,
        handle: ServiceHandle,
        command: str,
    ) -> Tuple[int, str]:
        """Cannot run commands in external services."""
        raise NotImplementedError("Cannot run commands in external services")

    def _parse_endpoint(self, endpoint: str) -> Dict[str, Any]:
        """Parse endpoint string (host:port or URL)."""
        if "://" in endpoint:
            parsed = urlparse(endpoint)
            return {
                "host": parsed.hostname or "localhost",
                "port": parsed.port,
                "scheme": parsed.scheme,
                "path": parsed.path,
            }
        elif ":" in endpoint:
            host, port = endpoint.rsplit(":", 1)
            return {"host": host, "port": int(port)}
        else:
            return {"host": endpoint}

    def _parse_connection_string(self, conn_str: str) -> Dict[str, Any]:
        """Parse database connection string."""
        parsed = urlparse(conn_str)
        return {
            "host": parsed.hostname,
            "port": parsed.port,
            "database": parsed.path.lstrip("/") if parsed.path else None,
            "username": parsed.username,
            "password": parsed.password,
            "scheme": parsed.scheme,
        }


class ManagedPostgresProvider(ExternalServiceProvider):
    """Provider for managed PostgreSQL services (RDS, Cloud SQL, Azure)."""

    async def _do_start(self, config: ServiceConfig) -> ServiceHandle:
        """Start connection to managed PostgreSQL."""
        handle = await super()._do_start(config)

        # Set default exports if not provided
        if not config.exports and config.endpoint:
            user = config.environment.get("POSTGRES_USER", "postgres")
            password = config.environment.get("POSTGRES_PASSWORD", "")
            database = config.environment.get("POSTGRES_DB", "postgres")
            port = handle.ports.get(5432, 5432)

            handle.connection_info["DATABASE_URL"] = (
                f"postgresql://{user}:{password}@{handle.host}:{port}/{database}"
            )
            handle.connection_info["POSTGRES_HOST"] = handle.host
            handle.connection_info["POSTGRES_PORT"] = str(port)

        return handle


class ManagedRedisProvider(ExternalServiceProvider):
    """Provider for managed Redis services (ElastiCache, Memorystore)."""

    async def _do_start(self, config: ServiceConfig) -> ServiceHandle:
        """Start connection to managed Redis."""
        handle = await super()._do_start(config)

        # Set default exports
        if not config.exports and config.endpoint:
            port = handle.ports.get(6379, 6379)
            password = config.environment.get("REDIS_PASSWORD", "")

            if password:
                handle.connection_info["REDIS_URL"] = f"redis://:{password}@{handle.host}:{port}/0"
            else:
                handle.connection_info["REDIS_URL"] = f"redis://{handle.host}:{port}/0"
            handle.connection_info["REDIS_HOST"] = handle.host
            handle.connection_info["REDIS_PORT"] = str(port)

        return handle


class ManagedMongoProvider(ExternalServiceProvider):
    """Provider for managed MongoDB services (Atlas, DocumentDB)."""

    async def _do_start(self, config: ServiceConfig) -> ServiceHandle:
        """Start connection to managed MongoDB."""
        handle = await super()._do_start(config)

        # MongoDB Atlas uses connection strings directly
        if config.connection_string:
            handle.connection_info["MONGODB_URL"] = config.connection_string

        return handle


class ManagedKafkaProvider(ExternalServiceProvider):
    """Provider for managed Kafka services (MSK, Confluent Cloud)."""

    async def _do_start(self, config: ServiceConfig) -> ServiceHandle:
        """Start connection to managed Kafka."""
        handle = await super()._do_start(config)

        if config.endpoint:
            handle.connection_info["KAFKA_BOOTSTRAP_SERVERS"] = config.endpoint

        return handle
