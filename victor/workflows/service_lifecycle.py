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

"""Workflow Service Lifecycle Management.

Manages external service dependencies for workflows: databases (PostgreSQL, MySQL),
caches (Redis, Memcached), message queues (RabbitMQ, Kafka), and external APIs.

Provides service initialization, health checks, and cleanup hooks.

Example:
    from victor.workflows.services import (
        ServiceManager,
        ServiceConfig,
        PostgresService,
        RedisService,
    )

    # Configure services
    services = [
        ServiceConfig(
            name="db",
            service_type="postgres",
            config={
                "host": "localhost",
                "port": 5432,
                "database": "workflows",
            },
        ),
        ServiceConfig(
            name="cache",
            service_type="redis",
            config={"host": "localhost", "port": 6379},
        ),
    ]

    # Initialize services
    manager = ServiceManager()
    handles = await manager.initialize_services(services)

    # Use in workflow
    db = handles["db"]
    cache = handles["cache"]

    # Cleanup
    await manager.cleanup_services()
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

logger = logging.getLogger(__name__)


@dataclass
class ServiceConfig:
    """Configuration for a workflow service.

    Attributes:
        name: Service identifier (used to access in workflow)
        service_type: Type of service (postgres, redis, rabbitmq, etc.)
        config: Service-specific configuration
        required: Whether workflow fails if service unavailable
        init_timeout: Timeout for service initialization (seconds)
        cleanup_timeout: Timeout for service cleanup (seconds)
    """

    name: str
    service_type: str
    config: Dict[str, Any] = field(default_factory=dict)
    required: bool = True
    init_timeout: float = 30.0
    cleanup_timeout: float = 10.0


class ServiceLifecycle(ABC):
    """Abstract base for service lifecycle management."""

    @abstractmethod
    async def initialize(self, config: ServiceConfig) -> Any:
        """Initialize the service, return service handle/client."""
        pass

    @abstractmethod
    async def cleanup(self, handle: Any, config: ServiceConfig) -> None:
        """Cleanup/shutdown the service."""
        pass

    @abstractmethod
    async def health_check(self, handle: Any) -> bool:
        """Check if service is healthy."""
        pass


class PostgresService(ServiceLifecycle):
    """Lifecycle manager for PostgreSQL connections."""

    async def initialize(self, config: ServiceConfig) -> Any:
        """Initialize PostgreSQL connection pool."""
        logger.info(f"Initializing PostgreSQL service: {config.name}")

        # Use asyncpg if available
        try:
            import asyncpg

            pool = await asyncpg.create_pool(
                host=config.config.get("host", "localhost"),
                port=config.config.get("port", 5432),
                database=config.config.get("database", "postgres"),
                user=config.config.get("user", "postgres"),
                password=config.config.get("password", ""),
                min_size=config.config.get("min_connections", 1),
                max_size=config.config.get("max_connections", 10),
            )
            logger.info(f"PostgreSQL pool created for {config.name}")
            return {"pool": pool, "type": "asyncpg"}
        except ImportError:
            logger.warning("asyncpg not installed, using placeholder")
            return {
                "connection": f"postgres://{config.config.get('host', 'localhost')}",
                "type": "placeholder",
            }

    async def cleanup(self, handle: Any, config: ServiceConfig) -> None:
        """Close PostgreSQL connections."""
        logger.info(f"Cleaning up PostgreSQL service: {config.name}")
        if handle.get("type") == "asyncpg" and handle.get("pool"):
            await handle["pool"].close()

    async def health_check(self, handle: Any) -> bool:
        """Check PostgreSQL connectivity."""
        if handle.get("type") == "asyncpg" and handle.get("pool"):
            try:
                async with handle["pool"].acquire() as conn:
                    await conn.execute("SELECT 1")
                return True
            except Exception:
                return False
        return True


class RedisService(ServiceLifecycle):
    """Lifecycle manager for Redis connections."""

    async def initialize(self, config: ServiceConfig) -> Any:
        """Initialize Redis client."""
        logger.info(f"Initializing Redis service: {config.name}")

        try:
            import redis.asyncio as aioredis

            client = aioredis.Redis(
                host=config.config.get("host", "localhost"),
                port=config.config.get("port", 6379),
                db=config.config.get("db", 0),
                password=config.config.get("password"),
                decode_responses=config.config.get("decode_responses", True),
            )
            logger.info(f"Redis client created for {config.name}")
            return {"client": client, "type": "aioredis"}
        except ImportError:
            logger.warning("redis not installed, using placeholder")
            return {
                "client": f"redis://{config.config.get('host', 'localhost')}",
                "type": "placeholder",
            }

    async def cleanup(self, handle: Any, config: ServiceConfig) -> None:
        """Close Redis connection."""
        logger.info(f"Cleaning up Redis service: {config.name}")
        if handle.get("type") == "aioredis" and handle.get("client"):
            await handle["client"].close()

    async def health_check(self, handle: Any) -> bool:
        """Check Redis connectivity."""
        if handle.get("type") == "aioredis" and handle.get("client"):
            try:
                await handle["client"].ping()
                return True
            except Exception:
                return False
        return True


class RabbitMQService(ServiceLifecycle):
    """Lifecycle manager for RabbitMQ connections."""

    async def initialize(self, config: ServiceConfig) -> Any:
        """Initialize RabbitMQ connection."""
        logger.info(f"Initializing RabbitMQ service: {config.name}")

        try:
            import aio_pika

            connection = await aio_pika.connect_robust(
                host=config.config.get("host", "localhost"),
                port=config.config.get("port", 5672),
                login=config.config.get("user", "guest"),
                password=config.config.get("password", "guest"),
                virtualhost=config.config.get("vhost", "/"),
            )
            channel = await connection.channel()
            logger.info(f"RabbitMQ connection created for {config.name}")
            return {"connection": connection, "channel": channel, "type": "aio_pika"}
        except ImportError:
            logger.warning("aio_pika not installed, using placeholder")
            return {
                "connection": f"amqp://{config.config.get('host', 'localhost')}",
                "type": "placeholder",
            }

    async def cleanup(self, handle: Any, config: ServiceConfig) -> None:
        """Close RabbitMQ connection."""
        logger.info(f"Cleaning up RabbitMQ service: {config.name}")
        if handle.get("type") == "aio_pika":
            if handle.get("channel"):
                await handle["channel"].close()
            if handle.get("connection"):
                await handle["connection"].close()

    async def health_check(self, handle: Any) -> bool:
        """Check RabbitMQ connectivity."""
        if handle.get("type") == "aio_pika" and handle.get("connection"):
            return not handle["connection"].is_closed
        return True


class KafkaService(ServiceLifecycle):
    """Lifecycle manager for Kafka connections."""

    async def initialize(self, config: ServiceConfig) -> Any:
        """Initialize Kafka producer/consumer."""
        logger.info(f"Initializing Kafka service: {config.name}")

        try:
            from aiokafka import AIOKafkaProducer, AIOKafkaConsumer

            bootstrap_servers = config.config.get("bootstrap_servers", "localhost:9092")

            producer = AIOKafkaProducer(bootstrap_servers=bootstrap_servers)
            await producer.start()

            logger.info(f"Kafka producer created for {config.name}")
            return {"producer": producer, "type": "aiokafka"}
        except ImportError:
            logger.warning("aiokafka not installed, using placeholder")
            return {
                "producer": f"kafka://{config.config.get('bootstrap_servers', 'localhost:9092')}",
                "type": "placeholder",
            }

    async def cleanup(self, handle: Any, config: ServiceConfig) -> None:
        """Close Kafka connection."""
        logger.info(f"Cleaning up Kafka service: {config.name}")
        if handle.get("type") == "aiokafka" and handle.get("producer"):
            await handle["producer"].stop()

    async def health_check(self, handle: Any) -> bool:
        """Check Kafka connectivity."""
        # Producer health check is implicit - it reconnects automatically
        return True


class HTTPClientService(ServiceLifecycle):
    """Lifecycle manager for HTTP API clients."""

    async def initialize(self, config: ServiceConfig) -> Any:
        """Initialize HTTP client session."""
        logger.info(f"Initializing HTTP client service: {config.name}")

        try:
            import httpx

            client = httpx.AsyncClient(
                base_url=config.config.get("base_url", "http://localhost:8080"),
                timeout=config.config.get("timeout", 30.0),
                headers=config.config.get("headers", {}),
            )
            logger.info(f"HTTP client created for {config.name}")
            return {"client": client, "type": "httpx"}
        except ImportError:
            logger.warning("httpx not installed, using placeholder")
            return {
                "base_url": config.config.get("base_url", "http://localhost:8080"),
                "type": "placeholder",
            }

    async def cleanup(self, handle: Any, config: ServiceConfig) -> None:
        """Close HTTP client session."""
        logger.info(f"Cleaning up HTTP client service: {config.name}")
        if handle.get("type") == "httpx" and handle.get("client"):
            await handle["client"].aclose()

    async def health_check(self, handle: Any) -> bool:
        """Check HTTP client health (optional health endpoint)."""
        return True


class S3Service(ServiceLifecycle):
    """Lifecycle manager for AWS S3 client."""

    async def initialize(self, config: ServiceConfig) -> Any:
        """Initialize S3 client."""
        logger.info(f"Initializing S3 service: {config.name}")

        try:
            import aioboto3

            session = aioboto3.Session(
                aws_access_key_id=config.config.get("aws_access_key_id"),
                aws_secret_access_key=config.config.get("aws_secret_access_key"),
                region_name=config.config.get("region_name", "us-east-1"),
            )
            logger.info(f"S3 session created for {config.name}")
            return {"session": session, "type": "aioboto3"}
        except ImportError:
            logger.warning("aioboto3 not installed, using placeholder")
            return {
                "bucket": config.config.get("bucket"),
                "type": "placeholder",
            }

    async def cleanup(self, handle: Any, config: ServiceConfig) -> None:
        """No cleanup needed for S3 (stateless)."""
        logger.info(f"Cleaning up S3 service: {config.name}")

    async def health_check(self, handle: Any) -> bool:
        """S3 is stateless, always healthy."""
        return True


class SQLiteService(ServiceLifecycle):
    """Lifecycle manager for SQLite databases (project.db, conversation.db).

    Supports LanceSQL pattern: SQLite for structured data (graph, symbols)
    + LanceDB for vector embeddings.
    """

    async def initialize(self, config: ServiceConfig) -> Any:
        """Initialize SQLite database connection.

        Args:
            config: ServiceConfig with:
                - db_path: Path to SQLite database (required)
                - readonly: Open in read-only mode (default: False for agents)
                - enable_wal: Enable WAL mode (default: True)
        """
        db_path = config.config.get("db_path")
        if not db_path:
            raise ValueError(f"SQLite service '{config.name}' missing 'db_path' in config")

        logger.info(f"Initializing SQLite service: {config.name} at {db_path}")

        try:
            import aiosqlite

            # Expand path and ensure parent directory exists
            db_path = Path(db_path).expanduser()
            if not config.config.get("readonly", False):
                db_path.parent.mkdir(parents=True, exist_ok=True)

            # Open connection
            conn = await aiosqlite.connect(
                str(db_path),
                isolation_level=None,  # Autocommit mode
            )

            # Enable WAL mode for better concurrency
            if config.config.get("enable_wal", True):
                await conn.execute("PRAGMA journal_mode=WAL")
                await conn.execute("PRAGMA synchronous=NORMAL")
                await conn.execute("PRAGMA cache_size=-64000")  # 64MB cache
                await conn.execute("PRAGMA temp_store=MEMORY")

            # Read-only mode for agents
            if config.config.get("readonly", False):
                logger.info(f"SQLite service '{config.name}' in READ-ONLY mode")
                # Can't set read-only on aiosqlite after connection, but we can prevent writes
                # via service layer (agent doesn't get write handle)

            logger.info(f"SQLite service '{config.name}' connected successfully")
            return {"connection": conn, "type": "aiosqlite", "path": str(db_path)}

        except ImportError:
            logger.warning("aiosqlite not installed, using placeholder")
            return {
                "path": str(db_path),
                "type": "placeholder",
            }

    async def cleanup(self, handle: Any, config: ServiceConfig) -> None:
        """Close SQLite connection."""
        logger.info(f"Cleaning up SQLite service: {config.name}")
        if handle.get("type") == "aiosqlite" and handle.get("connection"):
            await handle["connection"].close()

    async def health_check(self, handle: Any) -> bool:
        """Check SQLite connectivity."""
        if handle.get("type") == "aiosqlite" and handle.get("connection"):
            try:
                async with handle["connection"].execute("SELECT 1") as cursor:
                    await cursor.fetchone()
                return True
            except Exception:
                return False
        return True


class LanceDBService(ServiceLifecycle):
    """Lifecycle manager for LanceDB vector database.

    Part of LanceSQL pattern: Works with SQLiteService to provide
    both structured queries (SQLite) and semantic search (LanceDB).
    """

    async def initialize(self, config: ServiceConfig) -> Any:
        """Initialize LanceDB connection.

        Args:
            config: ServiceConfig with:
                - persist_directory: Path to LanceDB storage (required)
                - embedding_model: Model name (default: BAAI/bge-small-en-v1.5)
                - table_name: Table name (default: embeddings)
                - readonly: Read-only mode for agents (default: False)
        """
        from victor.storage.vector_stores import EmbeddingConfig, EmbeddingRegistry

        persist_dir = config.config.get("persist_directory")
        if not persist_dir:
            raise ValueError(
                f"LanceDB service '{config.name}' missing 'persist_directory' in config"
            )

        logger.info(f"Initializing LanceDB service: {config.name} at {persist_dir}")

        # Create embedding config
        embedding_config = EmbeddingConfig(
            vector_store="lancedb",
            persist_directory=persist_dir,
            embedding_model_name=config.config.get("embedding_model", "BAAI/bge-small-en-v1.5"),
            extra_config={
                "table_name": config.config.get("table_name", "embeddings"),
                "readonly": config.config.get("readonly", False),
            },
        )

        # Use singleton registry - all agents share same instance!
        provider = EmbeddingRegistry.create(embedding_config)
        await provider.initialize()

        # Get stats
        stats = await provider.get_stats()
        logger.info(
            f"LanceDB service '{config.name}' ready: "
            f"{stats.get('total_documents', 0)} documents, "
            f"table={stats.get('table_name', 'embeddings')}"
        )

        if config.config.get("readonly", False):
            logger.info(f"LanceDB service '{config.name}' in READ-ONLY mode")

        return {
            "provider": provider,
            "config": embedding_config,
            "type": "lancedb",
        }

    async def cleanup(self, handle: Any, config: ServiceConfig) -> None:
        """Cleanup LanceDB connection.

        Note: We don't close the singleton provider as other agents
        may be using it. Just cleanup resources.
        """
        logger.info(f"Cleaning up LanceDB service: {config.name}")
        if handle.get("provider"):
            # Don't close singleton - just log
            logger.debug(f"LanceDB service '{config.name}' provider remains cached for reuse")

    async def health_check(self, handle: Any) -> bool:
        """Check LanceDB is accessible."""
        if handle.get("provider"):
            try:
                stats = await handle["provider"].get_stats()
                return stats.get("total_documents", 0) >= 0
            except Exception:
                return False
        return True


class ServiceManager:
    """Manages service lifecycle for workflow execution.

    Handles initialization, health checks, and cleanup of services
    required by workflows.
    """

    # Registry of lifecycle handlers by service type
    _lifecycle_handlers: Dict[str, Type[ServiceLifecycle]] = {
        "postgres": PostgresService,
        "postgresql": PostgresService,
        "redis": RedisService,
        "rabbitmq": RabbitMQService,
        "kafka": KafkaService,
        "http": HTTPClientService,
        "api": HTTPClientService,
        "s3": S3Service,
        "sqlite": SQLiteService,
        "sqlite3": SQLiteService,
        "lancedb": LanceDBService,
    }

    def __init__(self):
        self._services: Dict[str, Any] = {}
        self._configs: Dict[str, ServiceConfig] = {}
        self._handlers: Dict[str, ServiceLifecycle] = {}

    @classmethod
    def register_service(
        cls,
        service_type: str,
        handler: Type[ServiceLifecycle],
    ) -> None:
        """Register a lifecycle handler for a service type.

        Args:
            service_type: Service type identifier (e.g., "postgres")
            handler: ServiceLifecycle class to handle this type
        """
        cls._lifecycle_handlers[service_type] = handler

    async def initialize_services(
        self,
        configs: List[ServiceConfig],
    ) -> Dict[str, Any]:
        """Initialize all configured services.

        Args:
            configs: List of service configurations

        Returns:
            Dictionary of service name -> handle

        Raises:
            RuntimeError: If a required service fails to initialize
        """
        for config in configs:
            handler_class = self._lifecycle_handlers.get(config.service_type)
            if not handler_class:
                logger.warning(f"No lifecycle handler for service type: {config.service_type}")
                continue

            handler = handler_class()
            self._handlers[config.name] = handler
            self._configs[config.name] = config

            try:
                handle = await asyncio.wait_for(
                    handler.initialize(config),
                    timeout=config.init_timeout,
                )
                self._services[config.name] = handle
                logger.info(f"Service '{config.name}' initialized successfully")

            except asyncio.TimeoutError:
                msg = f"Service '{config.name}' initialization timed out"
                logger.error(msg)
                if config.required:
                    raise RuntimeError(msg)

            except Exception as e:
                msg = f"Service '{config.name}' initialization failed: {e}"
                logger.error(msg, exc_info=True)
                if config.required:
                    raise RuntimeError(msg)

        return self._services

    async def cleanup_services(self) -> None:
        """Cleanup all initialized services."""
        for name in list(self._services.keys()):
            handle = self._services.get(name)
            handler = self._handlers.get(name)
            config = self._configs.get(name)

            if handler and config and handle:
                try:
                    await asyncio.wait_for(
                        handler.cleanup(handle, config),
                        timeout=config.cleanup_timeout,
                    )
                    logger.info(f"Service '{name}' cleaned up successfully")

                except asyncio.TimeoutError:
                    logger.warning(f"Service '{name}' cleanup timed out")

                except Exception as e:
                    logger.warning(f"Service '{name}' cleanup failed: {e}")

        self._services.clear()
        self._handlers.clear()
        self._configs.clear()

    async def health_check_all(self) -> Dict[str, bool]:
        """Check health of all services.

        Returns:
            Dictionary of service name -> health status
        """
        results = {}
        for name, handle in self._services.items():
            handler = self._handlers.get(name)
            if handler:
                try:
                    results[name] = await handler.health_check(handle)
                except Exception:
                    results[name] = False
            else:
                results[name] = True
        return results

    def get_service(self, name: str) -> Optional[Any]:
        """Get a service handle by name.

        Args:
            name: Service name

        Returns:
            Service handle or None if not found
        """
        return self._services.get(name)


__all__ = [
    # Config
    "ServiceConfig",
    # Lifecycle base
    "ServiceLifecycle",
    # Built-in services
    "PostgresService",
    "RedisService",
    "RabbitMQService",
    "KafkaService",
    "HTTPClientService",
    "S3Service",
    "SQLiteService",
    "LanceDBService",
    # Manager
    "ServiceManager",
]
