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

"""ServiceNode definitions for workflow infrastructure lifecycle management.

ServiceNodes manage infrastructure services (databases, caches, message queues)
that need to be running before workflow nodes execute and cleaned up afterward.

Key concepts:
- ServiceConfig: Declarative service configuration
- ServiceHandle: Runtime handle to a running service
- HealthCheck: Service readiness verification
- ServiceProvider: Backend-specific lifecycle management

Lifecycle:
    SETUP PHASE     →    EXECUTION PHASE    →    TEARDOWN PHASE
    ─────────────────────────────────────────────────────────────
    start services       run workflow nodes      stop services
    wait for healthy     inject connections      cleanup resources
    inject exports       handle failures         (guaranteed)

Example:
    services:
      postgres:
        provider: docker
        image: postgres:15
        health_check:
          type: tcp
          port: 5432
        exports:
          DATABASE_URL: postgresql://...

    nodes:
      - id: migrate
        requires_services: [postgres]
        ...
"""

from __future__ import annotations

import logging
import secrets
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Protocol,
    Tuple,
    Type,
    Union,
    runtime_checkable,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Type Aliases
# =============================================================================


class ServiceState(Enum):
    """Service lifecycle states."""

    PENDING = auto()  # Not yet started
    STARTING = auto()  # Start initiated
    HEALTHY = auto()  # Health check passed
    UNHEALTHY = auto()  # Health check failed
    STOPPING = auto()  # Stop initiated
    STOPPED = auto()  # Fully stopped
    FAILED = auto()  # Fatal error


class HealthCheckType(Enum):
    """Types of health checks."""

    TCP = "tcp"  # TCP port connectivity
    HTTP = "http"  # HTTP endpoint check
    HTTPS = "https"  # HTTPS endpoint check
    COMMAND = "command"  # Run command in container
    POSTGRES = "postgres"  # pg_isready
    REDIS = "redis"  # redis-cli ping
    MYSQL = "mysql"  # mysqladmin ping
    KAFKA = "kafka"  # kafka-topics --list
    GRPC = "grpc"  # gRPC health check


class NetworkProtocol(str, Enum):
    """Network protocol for port mappings."""

    TCP = "tcp"  # TCP protocol (reliable, connection-based)
    UDP = "udp"  # UDP protocol (unreliable, connectionless)


class RestartPolicy(str, Enum):
    """Container restart policy.

    Determines when to restart containers:
    - NO: Never restart
    - ON_FAILURE: Restart only on failure
    - ALWAYS: Always restart
    """

    NO = "no"  # Never restart
    ON_FAILURE = "on-failure"  # Restart on failure
    ALWAYS = "always"  # Always restart


class VolumeType(str, Enum):
    """Volume mount type.

    Type of volume mount:
    - BIND: Bind mount host path
    - VOLUME: Named volume
    - TMPFS: Temporary filesystem
    """

    BIND = "bind"  # Bind mount host path
    VOLUME = "volume"  # Named volume
    TMPFS = "tmpfs"  # Temporary filesystem


class ServiceProviderType(str, Enum):
    """Service provider backend type.

    Determines which backend infrastructure to use:
    - docker: Local Docker container
    - kubernetes: Kubernetes deployment
    - local: Local OS process
    - external: External service (verify only)
    - aws_rds: AWS RDS instance
    - aws_elasticache: AWS ElastiCache
    - aws_msk: AWS MSK (Kafka)
    - aws_sqs: AWS SQS
    - aws_dynamodb: AWS DynamoDB
    - gcp_cloudsql: GCP Cloud SQL
    - gcp_memorystore: GCP Memorystore
    - azure_postgres: Azure Database for PostgreSQL
    - azure_redis: Azure Cache for Redis
    """

    DOCKER = "docker"  # Local Docker container
    KUBERNETES = "kubernetes"  # Kubernetes deployment
    LOCAL = "local"  # Local OS process
    EXTERNAL = "external"  # External service (verify only)
    AWS_RDS = "aws_rds"  # AWS RDS instance
    AWS_ELASTICACHE = "aws_elasticache"  # AWS ElastiCache
    AWS_MSK = "aws_msk"  # AWS MSK (Kafka)
    AWS_SQS = "aws_sqs"  # AWS SQS
    AWS_DYNAMODB = "aws_dynamodb"  # AWS DynamoDB
    GCP_CLOUDSQL = "gcp_cloudsql"  # GCP Cloud SQL
    GCP_MEMORYSTORE = "gcp_memorystore"  # GCP Memorystore
    AZURE_POSTGRES = "azure_postgres"  # Azure Database for PostgreSQL
    AZURE_REDIS = "azure_redis"  # Azure Cache for Redis


# =============================================================================
# Health Check Configuration
# =============================================================================


@dataclass
class HealthCheckConfig:
    """Configuration for service health verification.

    Health checks determine when a service is ready to accept connections.
    Different check types are optimized for different services.

    Attributes:
        type: Type of health check to perform
        port: Port to check (for TCP/HTTP types)
        path: HTTP path to check (for HTTP/HTTPS types)
        command: Command to run (for COMMAND type)
        interval: Time between check attempts
        timeout: Maximum time for single check
        retries: Number of retries before failing
        start_period: Grace period before first check
        expected_status: Expected HTTP status code(s)
        expected_output: Expected command output substring

    Example (TCP):
        HealthCheckConfig(type=HealthCheckType.TCP, port=5432)

    Example (HTTP):
        HealthCheckConfig(
            type=HealthCheckType.HTTP,
            port=8080,
            path="/health",
            expected_status=[200, 204],
        )

    Example (Command):
        HealthCheckConfig(
            type=HealthCheckType.COMMAND,
            command="pg_isready -U postgres",
        )
    """

    type: HealthCheckType = HealthCheckType.TCP
    port: Optional[int] = None
    host: Optional[str] = None  # Override host (default: service host)
    path: str = "/health"
    command: Optional[str] = None
    interval: float = 2.0  # seconds
    timeout: float = 5.0  # seconds per attempt
    retries: int = 30
    start_period: float = 0.0  # grace period before first check
    expected_status: List[int] = field(default_factory=lambda: [200])
    expected_output: Optional[str] = None

    @property
    def max_wait_time(self) -> float:
        """Maximum time to wait for healthy status."""
        return self.start_period + (self.interval * self.retries)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type.value,
            "port": self.port,
            "host": self.host,
            "path": self.path,
            "command": self.command,
            "interval": self.interval,
            "timeout": self.timeout,
            "retries": self.retries,
            "start_period": self.start_period,
            "expected_status": self.expected_status,
            "expected_output": self.expected_output,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HealthCheckConfig":
        check_type = data.get("type", "tcp")
        if isinstance(check_type, str):
            check_type = HealthCheckType(check_type)

        return cls(
            type=check_type,
            port=data.get("port"),
            host=data.get("host"),
            path=data.get("path", "/health"),
            command=data.get("command"),
            interval=data.get("interval", 2.0),
            timeout=data.get("timeout", 5.0),
            retries=data.get("retries", 30),
            start_period=data.get("start_period", 0.0),
            expected_status=data.get("expected_status", [200]),
            expected_output=data.get("expected_output"),
        )

    @classmethod
    def for_postgres(cls, port: int = 5432) -> "HealthCheckConfig":
        """Preset for PostgreSQL."""
        return cls(
            type=HealthCheckType.POSTGRES,
            port=port,
            command="pg_isready -U postgres",
            interval=2.0,
            retries=30,
        )

    @classmethod
    def for_redis(cls, port: int = 6379) -> "HealthCheckConfig":
        """Preset for Redis."""
        return cls(
            type=HealthCheckType.REDIS,
            port=port,
            command="redis-cli ping",
            expected_output="PONG",
            interval=1.0,
            retries=30,
        )

    @classmethod
    def for_mysql(cls, port: int = 3306) -> "HealthCheckConfig":
        """Preset for MySQL."""
        return cls(
            type=HealthCheckType.MYSQL,
            port=port,
            command="mysqladmin ping -h localhost",
            interval=2.0,
            retries=30,
        )

    @classmethod
    def for_kafka(cls, port: int = 9092) -> "HealthCheckConfig":
        """Preset for Kafka."""
        return cls(
            type=HealthCheckType.KAFKA,
            port=port,
            interval=3.0,
            retries=20,
        )

    @classmethod
    def for_http(cls, port: int, path: str = "/health") -> "HealthCheckConfig":
        """Preset for HTTP services."""
        return cls(
            type=HealthCheckType.HTTP,
            port=port,
            path=path,
            expected_status=[200, 204],
            interval=2.0,
            retries=30,
        )


# =============================================================================
# Service Lifecycle Configuration
# =============================================================================


@dataclass
class LifecycleConfig:
    """Service lifecycle management configuration.

    Controls startup order, shutdown behavior, and failure handling.

    Attributes:
        startup_order: Order for starting services (lower = earlier)
        shutdown_order: Order for stopping (default: reverse of startup)
        startup_timeout: Max time to wait for healthy status
        shutdown_grace: Grace period for graceful shutdown
        cleanup_on_failure: Clean up even if workflow fails
        restart_policy: How to handle service crashes
        max_restarts: Maximum restart attempts

    Example:
        LifecycleConfig(
            startup_order=1,  # Start first
            shutdown_grace=30.0,  # 30s to shut down gracefully
            cleanup_on_failure=True,
        )
    """

    startup_order: int = 100  # Lower = start earlier
    shutdown_order: Optional[int] = None  # Default: reverse of startup
    startup_timeout: float = 120.0  # Max wait for healthy
    shutdown_grace: float = 30.0  # Grace period for stop
    cleanup_on_failure: bool = True  # Cleanup even on workflow failure
    restart_policy: RestartPolicy = RestartPolicy.NO
    max_restarts: int = 3

    def to_dict(self) -> Dict[str, Any]:
        return {
            "startup_order": self.startup_order,
            "shutdown_order": self.shutdown_order,
            "startup_timeout": self.startup_timeout,
            "shutdown_grace": self.shutdown_grace,
            "cleanup_on_failure": self.cleanup_on_failure,
            "restart_policy": self.restart_policy.value,
            "max_restarts": self.max_restarts,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LifecycleConfig":
        restart_str = data.get("restart_policy", "no")
        restart_policy = RestartPolicy(restart_str) if isinstance(restart_str, str) else restart_str

        return cls(
            startup_order=data.get("startup_order", 100),
            shutdown_order=data.get("shutdown_order"),
            startup_timeout=data.get("startup_timeout", 120.0),
            shutdown_grace=data.get("shutdown_grace", 30.0),
            cleanup_on_failure=data.get("cleanup_on_failure", True),
            restart_policy=restart_policy,
            max_restarts=data.get("max_restarts", 3),
        )


# =============================================================================
# Port Mapping
# =============================================================================


@dataclass
class PortMapping:
    """Container port mapping configuration.

    Maps container ports to host ports for service access.

    Attributes:
        container_port: Port inside the container
        host_port: Port on the host (0 = random)
        protocol: Protocol (tcp/udp)
        host_ip: Host IP to bind (default: 0.0.0.0)
    """

    container_port: int
    host_port: int = 0  # 0 = auto-assign
    protocol: NetworkProtocol = NetworkProtocol.TCP
    host_ip: str = "0.0.0.0"

    @classmethod
    def parse(cls, spec: Union[str, int, Dict, "PortMapping"]) -> "PortMapping":
        """Parse port specification.

        Formats:
            5432         → container 5432, host auto
            "5432"       → container 5432, host auto
            "5432:5432"  → container 5432, host 5432
            {"container_port": 5432, "host_port": 5433}
        """
        if isinstance(spec, PortMapping):
            return spec

        if isinstance(spec, int):
            return cls(container_port=spec)

        if isinstance(spec, str):
            if ":" in spec:
                host, container = spec.split(":")
                return cls(container_port=int(container), host_port=int(host))
            return cls(container_port=int(spec))

        if isinstance(spec, dict):
            protocol_str = spec.get("protocol", "tcp")
            protocol = (
                NetworkProtocol(protocol_str) if isinstance(protocol_str, str) else protocol_str
            )

            return cls(
                container_port=spec["container_port"],
                host_port=spec.get("host_port", 0),
                protocol=protocol,
                host_ip=spec.get("host_ip", "0.0.0.0"),
            )

        raise ValueError(f"Invalid port spec: {spec}")


# =============================================================================
# Volume Mount
# =============================================================================


@dataclass
class VolumeMount:
    """Volume mount configuration for containers.

    Attributes:
        source: Host path or volume name
        target: Container mount path
        read_only: Mount as read-only
        type: Mount type (bind, volume, tmpfs)
    """

    source: str
    target: str
    read_only: bool = False
    type: VolumeType = VolumeType.BIND

    @classmethod
    def parse(cls, spec: Union[str, Dict, "VolumeMount"]) -> "VolumeMount":
        """Parse volume specification.

        Formats:
            "/host/path:/container/path"
            "/host/path:/container/path:ro"
            {"source": "/host", "target": "/container"}
        """
        if isinstance(spec, VolumeMount):
            return spec

        if isinstance(spec, str):
            parts = spec.split(":")
            if len(parts) == 2:
                return cls(source=parts[0], target=parts[1])
            elif len(parts) == 3:
                return cls(
                    source=parts[0],
                    target=parts[1],
                    read_only=parts[2] == "ro",
                )
            raise ValueError(f"Invalid volume spec: {spec}")

        if isinstance(spec, dict):
            type_str = spec.get("type", "bind")
            volume_type = VolumeType(type_str) if isinstance(type_str, str) else type_str

            return cls(
                source=spec["source"],
                target=spec["target"],
                read_only=spec.get("read_only", False),
                type=volume_type,
            )

        raise ValueError(f"Invalid volume spec: {spec}")


# =============================================================================
# Service Configuration
# =============================================================================


@dataclass
class ServiceConfig:
    """Complete service configuration.

    Defines everything needed to start, verify, and connect to a service.

    Attributes:
        name: Unique service name (referenced by nodes)
        provider: Backend provider type
        image: Container image (for docker/k8s providers)
        command: Override container command
        ports: Port mappings
        environment: Environment variables
        volumes: Volume mounts
        health_check: Health verification config
        lifecycle: Lifecycle management config
        exports: Connection info to inject into context
        labels: Metadata labels
        network: Docker network name
        depends_on: Other services this depends on

        # Provider-specific settings
        # AWS
        aws_region: AWS region
        aws_instance_class: RDS instance class
        aws_cluster_id: ElastiCache cluster ID

        # Kubernetes
        k8s_namespace: Kubernetes namespace
        k8s_manifest: Path to K8s manifest
        k8s_replicas: Number of replicas

        # External
        endpoint: External service endpoint
        connection_string: Direct connection string

    Example:
        ServiceConfig(
            name="postgres",
            provider=ServiceProviderType.DOCKER,
            image="postgres:15",
            ports=[PortMapping(5432)],
            environment={"POSTGRES_PASSWORD": "secret"},
            health_check=HealthCheckConfig.for_postgres(),
            exports={"DATABASE_URL": "postgresql://..."},
        )
    """

    name: str
    provider: ServiceProviderType = ServiceProviderType.DOCKER

    # Container settings
    image: Optional[str] = None
    command: Optional[List[str]] = None
    entrypoint: Optional[List[str]] = None
    ports: List[PortMapping] = field(default_factory=list)
    environment: Dict[str, str] = field(default_factory=dict)
    volumes: List[VolumeMount] = field(default_factory=list)
    working_dir: Optional[str] = None
    user: Optional[str] = None

    # Health and lifecycle
    health_check: Optional[HealthCheckConfig] = None
    lifecycle: LifecycleConfig = field(default_factory=LifecycleConfig)

    # Connection exports
    exports: Dict[str, str] = field(default_factory=dict)

    # Metadata
    labels: Dict[str, str] = field(default_factory=dict)
    depends_on: List[str] = field(default_factory=list)

    # Docker-specific
    network: Optional[str] = None
    network_mode: Optional[str] = None
    privileged: bool = False
    cap_add: List[str] = field(default_factory=list)
    cap_drop: List[str] = field(default_factory=list)

    # AWS-specific
    aws_region: Optional[str] = None
    aws_instance_class: Optional[str] = None
    aws_cluster_id: Optional[str] = None
    aws_db_name: Optional[str] = None
    aws_engine: Optional[str] = None
    aws_engine_version: Optional[str] = None

    # Kubernetes-specific
    k8s_namespace: str = "default"
    k8s_manifest: Optional[str] = None
    k8s_replicas: int = 1
    k8s_service_account: Optional[str] = None
    k8s_node_selector: Dict[str, str] = field(default_factory=dict)

    # External service
    endpoint: Optional[str] = None
    connection_string: Optional[str] = None

    # Resource limits
    memory_limit: Optional[str] = None  # e.g., "512m", "2g"
    cpu_limit: Optional[float] = None  # e.g., 1.0, 0.5

    def __post_init__(self):
        # Parse port specs
        if self.ports:
            self.ports = [PortMapping.parse(p) for p in self.ports]

        # Parse volume specs
        if self.volumes:
            self.volumes = [VolumeMount.parse(v) for v in self.volumes]

        # Default health check based on provider/image
        if self.health_check is None:
            self.health_check = self._infer_health_check()

    def _infer_health_check(self) -> Optional[HealthCheckConfig]:
        """Infer health check from image name."""
        if not self.image:
            return None

        image_lower = self.image.lower()

        if "postgres" in image_lower:
            port = self._get_first_port(5432)
            return HealthCheckConfig.for_postgres(port)

        if "redis" in image_lower:
            port = self._get_first_port(6379)
            return HealthCheckConfig.for_redis(port)

        if "mysql" in image_lower or "mariadb" in image_lower:
            port = self._get_first_port(3306)
            return HealthCheckConfig.for_mysql(port)

        if "kafka" in image_lower:
            port = self._get_first_port(9092)
            return HealthCheckConfig.for_kafka(port)

        if "mongo" in image_lower:
            port = self._get_first_port(27017)
            return HealthCheckConfig(type=HealthCheckType.TCP, port=port)

        if "elasticsearch" in image_lower:
            port = self._get_first_port(9200)
            return HealthCheckConfig.for_http(port, "/_cluster/health")

        if "rabbitmq" in image_lower:
            port = self._get_first_port(15672)
            return HealthCheckConfig.for_http(port, "/api/health/checks/alarms")

        # Default to TCP check on first port
        if self.ports:
            return HealthCheckConfig(
                type=HealthCheckType.TCP,
                port=self.ports[0].container_port,
            )

        return None

    def _get_first_port(self, default: int) -> int:
        """Get first mapped port or default."""
        if self.ports:
            return self.ports[0].container_port
        return default

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "provider": self.provider,
            "image": self.image,
            "command": self.command,
            "ports": [
                {"container_port": p.container_port, "host_port": p.host_port} for p in self.ports
            ],
            "environment": self.environment,
            "volumes": [
                {"source": v.source, "target": v.target, "read_only": v.read_only}
                for v in self.volumes
            ],
            "health_check": self.health_check.to_dict() if self.health_check else None,
            "lifecycle": self.lifecycle.to_dict(),
            "exports": self.exports,
            "depends_on": self.depends_on,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ServiceConfig":
        """Create ServiceConfig from dictionary (YAML parsing)."""
        health_data = data.get("health_check")
        lifecycle_data = data.get("lifecycle", {})

        return cls(
            name=data["name"],
            provider=data.get("provider", "docker"),
            image=data.get("image"),
            command=data.get("command"),
            entrypoint=data.get("entrypoint"),
            ports=data.get("ports", []),
            environment=data.get("environment", {}),
            volumes=data.get("volumes", []),
            working_dir=data.get("working_dir"),
            user=data.get("user"),
            health_check=(HealthCheckConfig.from_dict(health_data) if health_data else None),
            lifecycle=LifecycleConfig.from_dict(lifecycle_data),
            exports=data.get("exports", {}),
            labels=data.get("labels", {}),
            depends_on=data.get("depends_on", []),
            network=data.get("network"),
            network_mode=data.get("network_mode"),
            aws_region=data.get("aws_region"),
            aws_instance_class=data.get("aws_instance_class"),
            aws_cluster_id=data.get("aws_cluster_id"),
            aws_db_name=data.get("aws_db_name"),
            aws_engine=data.get("aws_engine"),
            k8s_namespace=data.get("k8s_namespace", "default"),
            k8s_manifest=data.get("k8s_manifest"),
            k8s_replicas=data.get("k8s_replicas", 1),
            endpoint=data.get("endpoint"),
            connection_string=data.get("connection_string"),
            memory_limit=data.get("memory_limit"),
            cpu_limit=data.get("cpu_limit"),
        )


# =============================================================================
# Service Handle (Runtime State)
# =============================================================================


@dataclass
class ServiceHandle:
    """Runtime handle to a running service.

    Created when a service starts, used to track state and stop the service.

    Attributes:
        service_id: Unique runtime identifier
        config: Original service configuration
        state: Current service state
        host: Host address for connections
        ports: Resolved port mappings (with actual host ports)
        container_id: Container ID (for docker provider)
        process_id: Process ID (for local provider)
        started_at: When the service started
        healthy_at: When health check passed
        connection_info: Resolved connection strings
        metadata: Provider-specific metadata
    """

    service_id: str
    config: ServiceConfig
    state: ServiceState = ServiceState.PENDING
    host: str = "localhost"
    ports: Dict[int, int] = field(default_factory=dict)  # container_port → host_port
    container_id: Optional[str] = None
    process_id: Optional[int] = None
    started_at: Optional[datetime] = None
    healthy_at: Optional[datetime] = None
    connection_info: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    @classmethod
    def create(cls, config: ServiceConfig) -> "ServiceHandle":
        """Create a new handle for a service."""
        return cls(
            service_id=f"{config.name}-{uuid.uuid4().hex[:8]}",
            config=config,
        )

    def get_port(self, container_port: int) -> Optional[int]:
        """Get the host port mapped to a container port."""
        return self.ports.get(container_port)

    def get_host_address(self, container_port: int) -> Optional[str]:
        """Get host:port address for a container port."""
        host_port = self.get_port(container_port)
        if host_port:
            return f"{self.host}:{host_port}"
        return None

    def resolve_exports(self) -> Dict[str, str]:
        """Resolve export templates with actual connection info."""
        resolved = {}

        for key, template in self.config.exports.items():
            value = template

            # Replace {host} placeholder
            value = value.replace("{host}", self.host)

            # Replace {port:NNNN} placeholders
            for container_port, host_port in self.ports.items():
                value = value.replace(f"{{port:{container_port}}}", str(host_port))
                # Also support just {port} for first port
                if container_port == (
                    self.config.ports[0].container_port if self.config.ports else 0
                ):
                    value = value.replace("{port}", str(host_port))

            # Replace {container_id}
            if self.container_id:
                value = value.replace("{container_id}", self.container_id)

            resolved[key] = value

        return resolved

    def to_dict(self) -> Dict[str, Any]:
        return {
            "service_id": self.service_id,
            "name": self.config.name,
            "state": self.state.name,
            "host": self.host,
            "ports": self.ports,
            "container_id": self.container_id,
            "process_id": self.process_id,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "healthy_at": self.healthy_at.isoformat() if self.healthy_at else None,
            "connection_info": self.connection_info,
            "error": self.error,
        }


# =============================================================================
# Service Provider Protocol
# =============================================================================


@runtime_checkable
class ServiceProvider(Protocol):
    """Protocol for service lifecycle providers.

    Implementations handle backend-specific start/stop/health operations.

    Methods:
        start: Start the service, return handle
        stop: Stop the service gracefully
        health_check: Check if service is ready
        get_logs: Retrieve service logs
        cleanup: Force cleanup resources
    """

    async def start(self, config: ServiceConfig) -> ServiceHandle:
        """Start the service and return a handle.

        Args:
            config: Service configuration

        Returns:
            ServiceHandle with runtime info

        Raises:
            ServiceStartError: If start fails
        """
        ...

    async def stop(self, handle: ServiceHandle, grace_period: float = 30.0) -> None:
        """Stop the service gracefully.

        Args:
            handle: Service handle from start()
            grace_period: Time to wait for graceful shutdown

        Raises:
            ServiceStopError: If stop fails
        """
        ...

    async def health_check(self, handle: ServiceHandle) -> bool:
        """Check if the service is healthy.

        Args:
            handle: Service handle

        Returns:
            True if healthy, False otherwise
        """
        ...

    async def get_logs(
        self,
        handle: ServiceHandle,
        tail: int = 100,
    ) -> str:
        """Get service logs.

        Args:
            handle: Service handle
            tail: Number of lines from end

        Returns:
            Log output as string
        """
        ...

    async def cleanup(self, handle: ServiceHandle) -> None:
        """Force cleanup service resources.

        Called when graceful stop fails. Should not raise.

        Args:
            handle: Service handle
        """
        ...


# =============================================================================
# Exceptions
# =============================================================================


class ServiceError(Exception):
    """Base exception for service errors."""

    def __init__(self, service_name: str, message: str):
        self.service_name = service_name
        self.message = message
        super().__init__(f"[{service_name}] {message}")


class ServiceStartError(ServiceError):
    """Failed to start a service."""

    pass


class ServiceStopError(ServiceError):
    """Failed to stop a service."""

    pass


class ServiceHealthError(ServiceError):
    """Service health check failed."""

    def __init__(self, service_name: str, message: str, attempts: int = 0):
        self.attempts = attempts
        super().__init__(service_name, f"{message} (after {attempts} attempts)")


class ServiceDependencyError(ServiceError):
    """Service dependency not satisfied."""

    def __init__(self, service_name: str, dependency: str):
        self.dependency = dependency
        super().__init__(service_name, f"Dependency '{dependency}' not available")


# =============================================================================
# Presets for Common Services
# =============================================================================


def _generate_dev_password() -> str:
    """Generate a strong random password for an ephemeral dev service.

    Used as the default for service presets so no credential is hardcoded.
    Callers that need a fixed password (e.g. to connect from another process)
    should pass ``password=`` explicitly.
    """
    return secrets.token_urlsafe(16)


class ServicePresets:
    """Factory methods for common service configurations."""

    @staticmethod
    def postgres(
        name: str = "postgres",
        version: str = "15",
        database: str = "app",
        user: str = "postgres",
        password: Optional[str] = None,
        port: int = 5432,
    ) -> ServiceConfig:
        """PostgreSQL database preset."""
        password = password or _generate_dev_password()
        return ServiceConfig(
            name=name,
            provider="docker",
            image=f"postgres:{version}",
            ports=[PortMapping(container_port=5432, host_port=port)],
            environment={
                "POSTGRES_DB": database,
                "POSTGRES_USER": user,
                "POSTGRES_PASSWORD": password,
            },
            health_check=HealthCheckConfig.for_postgres(5432),
            exports={
                "DATABASE_URL": f"postgresql://{user}:{password}@{{host}}:{{port}}/{database}",
                "POSTGRES_HOST": "{host}",
                "POSTGRES_PORT": "{port}",
            },
        )

    @staticmethod
    def redis(
        name: str = "redis",
        version: str = "7-alpine",
        port: int = 6379,
        password: Optional[str] = None,
    ) -> ServiceConfig:
        """Redis cache preset."""
        env = {}
        if password:
            env["REDIS_PASSWORD"] = password

        url_template = "redis://{host}:{port}/0"
        if password:
            url_template = f"redis://:{password}@{{host}}:{{port}}/0"

        return ServiceConfig(
            name=name,
            provider="docker",
            image=f"redis:{version}",
            ports=[PortMapping(container_port=6379, host_port=port)],
            environment=env,
            command=["redis-server", "--appendonly", "yes"]
            + (["--requirepass", password] if password else []),
            health_check=HealthCheckConfig.for_redis(6379),
            exports={
                "REDIS_URL": url_template,
                "REDIS_HOST": "{host}",
                "REDIS_PORT": "{port}",
            },
        )

    @staticmethod
    def mysql(
        name: str = "mysql",
        version: str = "8",
        database: str = "app",
        user: str = "app",
        password: Optional[str] = None,
        root_password: Optional[str] = None,
        port: int = 3306,
    ) -> ServiceConfig:
        """MySQL database preset."""
        password = password or _generate_dev_password()
        root_password = root_password or _generate_dev_password()
        return ServiceConfig(
            name=name,
            provider="docker",
            image=f"mysql:{version}",
            ports=[PortMapping(container_port=3306, host_port=port)],
            environment={
                "MYSQL_DATABASE": database,
                "MYSQL_USER": user,
                "MYSQL_PASSWORD": password,
                "MYSQL_ROOT_PASSWORD": root_password,
            },
            health_check=HealthCheckConfig.for_mysql(3306),
            exports={
                "DATABASE_URL": f"mysql://{user}:{password}@{{host}}:{{port}}/{database}",
                "MYSQL_HOST": "{host}",
                "MYSQL_PORT": "{port}",
            },
        )

    @staticmethod
    def mongodb(
        name: str = "mongodb",
        version: str = "6",
        database: str = "app",
        user: str = "app",
        password: Optional[str] = None,
        port: int = 27017,
    ) -> ServiceConfig:
        """MongoDB database preset."""
        password = password or _generate_dev_password()
        return ServiceConfig(
            name=name,
            provider="docker",
            image=f"mongo:{version}",
            ports=[PortMapping(container_port=27017, host_port=port)],
            environment={
                "MONGO_INITDB_DATABASE": database,
                "MONGO_INITDB_ROOT_USERNAME": user,
                "MONGO_INITDB_ROOT_PASSWORD": password,
            },
            health_check=HealthCheckConfig(
                type=HealthCheckType.COMMAND,
                command="mongosh --eval \"db.adminCommand('ping')\"",
            ),
            exports={
                "MONGODB_URL": f"mongodb://{user}:{password}@{{host}}:{{port}}/{database}",
                "MONGODB_HOST": "{host}",
                "MONGODB_PORT": "{port}",
            },
        )

    @staticmethod
    def kafka(
        name: str = "kafka",
        version: str = "latest",
        port: int = 9092,
    ) -> ServiceConfig:
        """Kafka message broker preset (using KRaft mode)."""
        return ServiceConfig(
            name=name,
            provider="docker",
            image=f"confluentinc/cp-kafka:{version}",
            ports=[
                PortMapping(container_port=9092, host_port=port),
                PortMapping(container_port=9093, host_port=port + 1),
            ],
            environment={
                "KAFKA_NODE_ID": "1",
                "KAFKA_LISTENER_SECURITY_PROTOCOL_MAP": "CONTROLLER:PLAINTEXT,PLAINTEXT:PLAINTEXT",
                "KAFKA_LISTENERS": "PLAINTEXT://:9092,CONTROLLER://:9093",
                "KAFKA_ADVERTISED_LISTENERS": f"PLAINTEXT://localhost:{port}",
                "KAFKA_CONTROLLER_QUORUM_VOTERS": "1@localhost:9093",
                "KAFKA_PROCESS_ROLES": "broker,controller",
                "KAFKA_CONTROLLER_LISTENER_NAMES": "CONTROLLER",
                "KAFKA_LOG_DIRS": "/var/lib/kafka/data",
                "CLUSTER_ID": "MkU3OEVBNTcwNTJENDM2Qk",
            },
            health_check=HealthCheckConfig.for_kafka(9092),
            exports={
                "KAFKA_BOOTSTRAP_SERVERS": "{host}:{port}",
            },
        )

    @staticmethod
    def elasticsearch(
        name: str = "elasticsearch",
        version: str = "8.11.0",
        port: int = 9200,
    ) -> ServiceConfig:
        """Elasticsearch search engine preset."""
        return ServiceConfig(
            name=name,
            provider="docker",
            image=f"docker.elastic.co/elasticsearch/elasticsearch:{version}",
            ports=[
                PortMapping(container_port=9200, host_port=port),
                PortMapping(container_port=9300, host_port=port + 100),
            ],
            environment={
                "discovery.type": "single-node",
                "xpack.security.enabled": "false",
                "ES_JAVA_OPTS": "-Xms512m -Xmx512m",
            },
            health_check=HealthCheckConfig.for_http(9200, "/_cluster/health"),
            memory_limit="1g",
            exports={
                "ELASTICSEARCH_URL": "http://{host}:{port}",
                "ELASTICSEARCH_HOST": "{host}",
                "ELASTICSEARCH_PORT": "{port}",
            },
        )

    @staticmethod
    def qdrant(
        name: str = "qdrant",
        version: str = "latest",
        port: int = 6333,
    ) -> ServiceConfig:
        """Qdrant vector database preset (for RAG)."""
        return ServiceConfig(
            name=name,
            provider="docker",
            image=f"qdrant/qdrant:{version}",
            ports=[
                PortMapping(container_port=6333, host_port=port),
                PortMapping(container_port=6334, host_port=port + 1),  # gRPC
            ],
            health_check=HealthCheckConfig.for_http(6333, "/"),
            exports={
                "QDRANT_URL": "http://{host}:{port}",
                "QDRANT_HOST": "{host}",
                "QDRANT_PORT": "{port}",
                "QDRANT_GRPC_PORT": "{port:6334}",
            },
        )

    @staticmethod
    def milvus(
        name: str = "milvus",
        version: str = "latest",
        port: int = 19530,
    ) -> ServiceConfig:
        """Milvus vector database preset (for RAG)."""
        return ServiceConfig(
            name=name,
            provider="docker",
            image=f"milvusdb/milvus:{version}",
            ports=[
                PortMapping(container_port=19530, host_port=port),
                PortMapping(container_port=9091, host_port=9091),
            ],
            command=["milvus", "run", "standalone"],
            environment={
                "ETCD_USE_EMBED": "true",
                "ETCD_DATA_DIR": "/var/lib/milvus/etcd",
                "COMMON_STORAGETYPE": "local",
            },
            health_check=HealthCheckConfig(
                type=HealthCheckType.HTTP,
                port=9091,
                path="/healthz",
            ),
            exports={
                "MILVUS_HOST": "{host}",
                "MILVUS_PORT": "{port}",
            },
        )

    @staticmethod
    def rabbitmq(
        name: str = "rabbitmq",
        version: str = "3-management",
        port: int = 5672,
        management_port: int = 15672,
        user: str = "guest",
        password: Optional[str] = None,
    ) -> ServiceConfig:
        """RabbitMQ message broker preset."""
        password = password or _generate_dev_password()
        return ServiceConfig(
            name=name,
            provider="docker",
            image=f"rabbitmq:{version}",
            ports=[
                PortMapping(container_port=5672, host_port=port),
                PortMapping(container_port=15672, host_port=management_port),
            ],
            environment={
                "RABBITMQ_DEFAULT_USER": user,
                "RABBITMQ_DEFAULT_PASS": password,
            },
            health_check=HealthCheckConfig.for_http(15672, "/api/health/checks/alarms"),
            exports={
                "RABBITMQ_URL": f"amqp://{user}:{password}@{{host}}:{{port}}",
                "RABBITMQ_HOST": "{host}",
                "RABBITMQ_PORT": "{port}",
                "RABBITMQ_MANAGEMENT_URL": f"http://{{host}}:{management_port}",
            },
        )


__all__ = [
    # Enums
    "ServiceState",
    "HealthCheckType",
    "ServiceProviderType",
    # Configurations
    "HealthCheckConfig",
    "LifecycleConfig",
    "PortMapping",
    "VolumeMount",
    "ServiceConfig",
    # Runtime
    "ServiceHandle",
    # Protocol
    "ServiceProvider",
    # Exceptions
    "ServiceError",
    "ServiceStartError",
    "ServiceStopError",
    "ServiceHealthError",
    "ServiceDependencyError",
    # Presets
    "ServicePresets",
]
