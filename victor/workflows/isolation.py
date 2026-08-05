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

"""Constraint-to-sandbox mapping for workflow execution.

Maps TaskConstraints to execution environments, providing per-vertical
isolation defaults with constraint-based overrides.

Isolation Levels:
- none: Direct inline execution (fastest, least isolation)
- process: Subprocess with resource limits via rlimit
- docker: Full container isolation (safest, slower startup)

Per-Vertical Defaults:
- coding: process (balance of speed and safety)
- research: none (read-heavy, low risk)
- devops: docker (infrastructure ops need isolation)
- dataanalysis: process (computation-focused)
- rag: docker (document processing needs isolation)

Example:
    from victor.workflows.isolation import IsolationMapper, IsolationConfig

    # Get isolation config from constraints
    isolation = IsolationMapper.from_constraints(
        constraints=node.constraints,
        vertical="dataanalysis",
    )

    if isolation.sandbox_type == "docker":
        result = await execute_in_docker(node, isolation)
    elif isolation.sandbox_type == "process":
        result = await execute_in_subprocess(node, isolation)
    else:
        result = await execute_inline(node)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Type, Union

if TYPE_CHECKING:
    from victor.workflows.definition import ConstraintsProtocol

logger = logging.getLogger(__name__)


# =============================================================================
# Sandbox Provider Protocol and Registry (Extensibility)
# =============================================================================


@dataclass
class SandboxResult:
    """Result from sandbox execution.

    Attributes:
        success: Whether execution succeeded
        output: Output from the execution
        error: Error message if failed
        exit_code: Process exit code (if applicable)
        metadata: Additional execution metadata
    """

    success: bool
    output: Any = None
    error: Optional[str] = None
    exit_code: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class SandboxProvider(ABC):
    """Abstract base for sandbox implementations.

    Implement this to add custom sandbox types (kubernetes, wasm, etc.)
    that can be used for workflow execution.

    All verticals (including third-party plugins) can register custom
    sandbox providers for their specific execution needs.

    Example:
        class KubernetesSandboxProvider(SandboxProvider):
            @property
            def sandbox_type(self) -> str:
                return "kubernetes"

            async def execute(self, code, context, timeout=None):
                # Run in K8s pod
                ...

            async def cleanup(self):
                # Delete pod
                ...

        # Register the provider
        registry = SandboxProviderRegistry.get_instance()
        registry.register("kubernetes", KubernetesSandboxProvider)
    """

    @property
    @abstractmethod
    def sandbox_type(self) -> str:
        """Return sandbox type identifier (e.g., 'none', 'docker', 'kubernetes')."""
        ...

    @abstractmethod
    async def execute(
        self,
        code: str,
        context: Dict[str, Any],
        config: "IsolationConfig",
        timeout: Optional[float] = None,
    ) -> SandboxResult:
        """Execute code in sandbox.

        Args:
            code: Code or command to execute
            context: Execution context
            config: Isolation configuration
            timeout: Execution timeout in seconds

        Returns:
            SandboxResult with execution outcome
        """
        ...

    @abstractmethod
    async def cleanup(self) -> None:
        """Clean up sandbox resources."""
        ...

    def supports_feature(self, feature: str) -> bool:
        """Check if sandbox supports a feature.

        Standard features: 'networking', 'filesystem', 'gpu', 'secrets'

        Args:
            feature: Feature name to check

        Returns:
            True if feature is supported
        """
        return False

    def get_capabilities(self) -> Dict[str, bool]:
        """Get all sandbox capabilities."""
        return {
            "networking": self.supports_feature("networking"),
            "filesystem": self.supports_feature("filesystem"),
            "gpu": self.supports_feature("gpu"),
            "secrets": self.supports_feature("secrets"),
        }


class SandboxProviderRegistry:
    """Registry for sandbox providers - enables extensibility.

    Third-party verticals can register custom sandbox providers
    (e.g., Kubernetes, WASM, custom container runtimes) without
    modifying core Victor code.

    Example:
        registry = SandboxProviderRegistry.get_instance()

        # Register custom provider
        registry.register("kubernetes", KubernetesSandboxProvider)

        # Get provider for use
        provider = registry.get_provider("kubernetes")
        result = await provider.execute(code, context, config)
    """

    _instance: Optional["SandboxProviderRegistry"] = None

    def __init__(self):
        self._providers: Dict[str, Type[SandboxProvider]] = {}
        self._instances: Dict[str, SandboxProvider] = {}

    @classmethod
    def get_instance(cls) -> "SandboxProviderRegistry":
        """Get singleton registry instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton (for testing)."""
        if cls._instance is not None:
            cls._instance._instances.clear()
            cls._instance._providers.clear()
        cls._instance = None

    def register(
        self,
        sandbox_type: str,
        provider_class: Type[SandboxProvider],
    ) -> None:
        """Register a sandbox provider.

        Args:
            sandbox_type: Sandbox type identifier (e.g., "kubernetes")
            provider_class: Provider class (not instance)
        """
        self._providers[sandbox_type] = provider_class
        # Clear cached instance if re-registering
        self._instances.pop(sandbox_type, None)
        logger.info(f"Registered sandbox provider: {sandbox_type}")

    def get_provider(self, sandbox_type: str) -> Optional[SandboxProvider]:
        """Get or create sandbox provider instance.

        Args:
            sandbox_type: Sandbox type to get

        Returns:
            SandboxProvider instance or None if not registered
        """
        if sandbox_type not in self._instances:
            provider_class = self._providers.get(sandbox_type)
            if provider_class:
                self._instances[sandbox_type] = provider_class()
        return self._instances.get(sandbox_type)

    def list_types(self) -> List[str]:
        """List available sandbox types."""
        return list(self._providers.keys())

    def is_registered(self, sandbox_type: str) -> bool:
        """Check if sandbox type is registered."""
        return sandbox_type in self._providers

    def get_all_capabilities(self) -> Dict[str, Dict[str, bool]]:
        """Get capabilities for all registered providers."""
        result = {}
        for sandbox_type in self._providers:
            provider = self.get_provider(sandbox_type)
            if provider:
                result[sandbox_type] = provider.get_capabilities()
        return result


# =============================================================================
# Built-in Sandbox Providers
# =============================================================================


class NoneSandboxProvider(SandboxProvider):
    """No sandboxing - direct inline execution.

    Fastest execution but no isolation. Suitable for trusted code
    or read-only operations.
    """

    @property
    def sandbox_type(self) -> str:
        return "none"

    async def execute(
        self,
        code: str,
        context: Dict[str, Any],
        config: "IsolationConfig",
        timeout: Optional[float] = None,
    ) -> SandboxResult:
        # Direct execution - implement in concrete usage
        return SandboxResult(
            success=True,
            output=None,
            metadata={"sandbox_type": "none", "executed_inline": True},
        )

    async def cleanup(self) -> None:
        pass  # No cleanup needed

    def supports_feature(self, feature: str) -> bool:
        # Inline execution has full access
        return feature in ("networking", "filesystem")


class ProcessSandboxProvider(SandboxProvider):
    """Process-based sandboxing with resource limits.

    Uses subprocess with rlimit for resource control.
    Moderate isolation with reasonable performance.
    """

    @property
    def sandbox_type(self) -> str:
        return "process"

    async def execute(
        self,
        code: str,
        context: Dict[str, Any],
        config: "IsolationConfig",
        timeout: Optional[float] = None,
    ) -> SandboxResult:
        # Subprocess execution - implement in concrete usage
        return SandboxResult(
            success=True,
            output=None,
            metadata={"sandbox_type": "process"},
        )

    async def cleanup(self) -> None:
        pass  # Process cleanup is automatic

    def supports_feature(self, feature: str) -> bool:
        return feature in ("networking", "filesystem")


class DockerSandboxProvider(SandboxProvider):
    """Docker container sandboxing.

    Full container isolation with configurable resources.
    Safest option but has startup overhead.
    """

    @property
    def sandbox_type(self) -> str:
        return "docker"

    async def execute(
        self,
        code: str,
        context: Dict[str, Any],
        config: "IsolationConfig",
        timeout: Optional[float] = None,
    ) -> SandboxResult:
        # Docker execution - implement in concrete usage
        return SandboxResult(
            success=True,
            output=None,
            metadata={
                "sandbox_type": "docker",
                "image": config.docker_image,
            },
        )

    async def cleanup(self) -> None:
        pass  # Container cleanup handled by executor

    def supports_feature(self, feature: str) -> bool:
        # Docker supports most features
        return feature in ("networking", "filesystem", "gpu", "secrets")


def register_builtin_providers() -> None:
    """Register built-in sandbox providers.

    Call this during application initialization to register
    the standard sandbox providers (none, process, docker).
    """
    registry = SandboxProviderRegistry.get_instance()
    registry.register("none", NoneSandboxProvider)
    registry.register("process", ProcessSandboxProvider)
    registry.register("docker", DockerSandboxProvider)
    logger.debug("Registered built-in sandbox providers")


# =============================================================================
# Enums for Workflow Isolation
# =============================================================================


class SandboxType(str, Enum):
    """Sandbox type for workflow execution.

    Isolation levels for workflow node execution:
    - NONE: Direct inline execution (fastest, least isolation)
    - PROCESS: Subprocess with resource limits via rlimit
    - DOCKER: Full container isolation (safest, slower startup)
    """

    NONE = "none"  # Direct inline execution
    PROCESS = "process"  # Subprocess with resource limits
    DOCKER = "docker"  # Full container isolation


# Backward compatibility
_SandboxTypeLiteral = Literal["none", "process", "docker"]


# Deployment target hierarchy:
# - Local: execution on the same machine
#   - inline: direct process (no isolation)
#   - subprocess: OS subprocess with rlimit
#   - docker: local Docker daemon
#   - kubernetes: local K8s (minikube, kind, docker-desktop)
# - Remote: execution on remote infrastructure
#   - docker: remote Docker daemon
#   - kubernetes/{eks,aks,gke}: managed K8s clusters
#   - ecs: AWS ECS Fargate
#   - airflow: Airflow DAG trigger (orchestration API)
#   - api: Generic REST API endpoint


class ExecutionLocality(str, Enum):
    """Execution locality for workflow nodes.

    Determines whether execution happens locally or on remote infrastructure.
    """

    LOCAL = "local"  # Execute on local machine
    REMOTE = "remote"  # Execute on remote infrastructure


# Backward compatibility
_ExecutionLocalityLiteral = Literal["local", "remote"]


class LocalTarget(str, Enum):
    """Local execution target type.

    Determines how to execute workflows locally:
    - inline: Direct process execution (fastest)
    - subprocess: OS subprocess with rlimit
    - docker: Local Docker container
    - kubernetes: Local K8s (minikube, kind)
    """

    INLINE = "inline"  # Direct process execution (fastest)
    SUBPROCESS = "subprocess"  # OS subprocess with rlimit
    DOCKER = "docker"  # Local Docker container
    KUBERNETES = "kubernetes"  # Local K8s (minikube, kind)


class RemoteTarget(str, Enum):
    """Remote execution target type.

    Determines which remote infrastructure to use:
    - docker: Remote Docker daemon
    - kubernetes: Generic remote K8s
    - eks: AWS EKS
    - aks: Azure AKS
    - gke: Google GKE
    - ecs: AWS ECS Fargate
    - airflow: Airflow DAG trigger
    - api: Generic REST API
    """

    DOCKER = "docker"  # Remote Docker daemon
    KUBERNETES = "kubernetes"  # Generic remote K8s
    EKS = "eks"  # AWS EKS
    AKS = "aks"  # Azure AKS
    GKE = "gke"  # Google GKE
    ECS = "ecs"  # AWS ECS Fargate
    AIRFLOW = "airflow"  # Airflow DAG trigger
    API = "api"  # Generic REST API


class NetworkProtocol(str, Enum):
    """Network protocol for connecting to execution targets.

    Determines how to communicate with remote infrastructure:
    - LOCAL: No network connection needed (inline, subprocess)
    - DOCKER: Docker daemon socket/API
    - K8S: Kubernetes API server
    - HTTP: Generic HTTP/HTTPS REST API
    """

    LOCAL = "local"  # No network connection
    DOCKER = "docker"  # Docker daemon socket/API
    K8S = "k8s"  # Kubernetes API server
    HTTP = "http"  # Generic HTTP/HTTPS REST API


@dataclass
class ResourceLimits:
    """Resource limits for sandboxed execution.

    These limits are applied via rlimit (process) or Docker constraints.

    Attributes:
        max_memory_mb: Maximum memory in megabytes
        max_cpu_seconds: Maximum CPU time in seconds
        max_file_descriptors: Maximum open file descriptors
        max_processes: Maximum child processes
        timeout_seconds: Overall execution timeout
    """

    max_memory_mb: int = 512
    max_cpu_seconds: int = 300
    max_file_descriptors: int = 256
    max_processes: int = 32
    timeout_seconds: float = 60.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_memory_mb": self.max_memory_mb,
            "max_cpu_seconds": self.max_cpu_seconds,
            "max_file_descriptors": self.max_file_descriptors,
            "max_processes": self.max_processes,
            "timeout_seconds": self.timeout_seconds,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ResourceLimits":
        return cls(
            max_memory_mb=data.get("max_memory_mb", 512),
            max_cpu_seconds=data.get("max_cpu_seconds", 300),
            max_file_descriptors=data.get("max_file_descriptors", 256),
            max_processes=data.get("max_processes", 32),
            timeout_seconds=data.get("timeout_seconds", 60.0),
        )


@dataclass
class ConnectionConfig:
    """Connection configuration for remote execution targets.

    Supports various protocols for connecting to execution targets:
    - Local: No connection needed (inline, subprocess)
    - Docker: Docker daemon socket/API
    - Kubernetes: K8s API server
    - Cloud: AWS/Azure/GCP credentials
    - API: Generic REST endpoints

    Attributes:
        protocol: Connection protocol (local, docker, k8s, aws, azure, gcp, http)
        endpoint: API endpoint or socket path
        auth_method: Authentication method (token, cert, iam, oauth)
        credentials_secret: Reference to credentials (K8s secret, env var, file)
        region: Cloud region (for AWS, Azure, GCP)
        verify_ssl: Verify SSL certificates
        timeout: Connection timeout in seconds

    Example:
        # Local Docker
        config = ConnectionConfig(protocol=NetworkProtocol.DOCKER, endpoint="unix:///var/run/docker.sock")

        # Remote EKS
        config = ConnectionConfig(
            protocol=NetworkProtocol.K8S,
            auth_method="iam",
            region="us-west-2",
        )
    """

    protocol: Literal["local", "docker", "k8s", "aws", "azure", "gcp", "http"] = "local"
    endpoint: Optional[str] = None
    auth_method: Optional[Literal["token", "cert", "iam", "oauth", "basic"]] = None
    credentials_secret: Optional[str] = None
    region: Optional[str] = None
    verify_ssl: bool = True
    timeout: int = 30

    def to_dict(self) -> Dict[str, Any]:
        return {
            "protocol": self.protocol,
            "endpoint": self.endpoint,
            "auth_method": self.auth_method,
            "credentials_secret": self.credentials_secret,
            "region": self.region,
            "verify_ssl": self.verify_ssl,
            "timeout": self.timeout,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConnectionConfig":
        return cls(
            protocol=data.get("protocol", "local"),
            endpoint=data.get("endpoint"),
            auth_method=data.get("auth_method"),
            credentials_secret=data.get("credentials_secret"),
            region=data.get("region"),
            verify_ssl=data.get("verify_ssl", True),
            timeout=data.get("timeout", 30),
        )


@dataclass
class DeploymentConfig:
    """Deployment target configuration with hierarchical locality.

    Hierarchy:
    - locality: local vs remote
    - target: execution target type
    - connection: how to connect to the target

    Local Targets (execution on same machine):
    - inline: Direct process, no isolation (fastest)
    - subprocess: OS subprocess with rlimit resource limits
    - docker: Local Docker daemon
    - kubernetes: Local K8s (minikube, kind, docker-desktop)

    Remote Targets (execution on remote infrastructure):
    - docker: Remote Docker daemon (TCP/TLS)
    - kubernetes: Generic remote K8s cluster
    - eks: AWS EKS (uses IAM auth)
    - aks: Azure AKS (uses Azure AD)
    - gke: Google GKE (uses GCP service account)
    - ecs: AWS ECS Fargate (serverless containers)
    - airflow: Trigger Airflow DAG (orchestration handoff)
    - api: Generic REST API endpoint

    Attributes:
        locality: Where execution happens (local, remote)
        target: Execution target type
        connection: Connection configuration for remote targets
        cluster_name: Kubernetes/ECS cluster name
        namespace: Kubernetes namespace
        service_account: K8s service account for RBAC
        task_definition: ECS task definition ARN
        node_selector: K8s node selector labels
        tolerations: K8s tolerations for scheduling
        gpu_required: Whether GPU is needed
        spot_instance: Use spot/preemptible instances

    Example (Local Docker):
        config = DeploymentConfig(
            locality=ExecutionLocality.LOCAL,
            target=LocalTarget.DOCKER,
        )

    Example (Remote EKS):
        config = DeploymentConfig(
            locality=ExecutionLocality.REMOTE,
            target=RemoteTarget.EKS,
            connection=ConnectionConfig(
                protocol=NetworkProtocol.K8S,
                auth_method="iam",
                region="us-west-2",
            ),
            cluster_name="ml-cluster",
            namespace="workflows",
            gpu_required=True,
        )

    Example (Airflow handoff):
        config = DeploymentConfig(
            locality=ExecutionLocality.REMOTE,
            target=RemoteTarget.AIRFLOW,
            connection=ConnectionConfig(
                protocol="http",
                endpoint="https://airflow.company.com/api/v1",
                auth_method="oauth",
            ),
        )
    """

    locality: ExecutionLocality = ExecutionLocality.LOCAL
    target: Union[LocalTarget, RemoteTarget] = LocalTarget.INLINE  # type: ignore[assignment]
    connection: Optional[ConnectionConfig] = None
    cluster_name: Optional[str] = None
    namespace: Optional[str] = "default"
    service_account: Optional[str] = None
    task_definition: Optional[str] = None  # ECS
    dag_id: Optional[str] = None  # Airflow
    api_path: Optional[str] = None  # Generic API
    node_selector: Dict[str, str] = field(default_factory=dict)
    tolerations: List[Dict[str, str]] = field(default_factory=list)
    gpu_required: bool = False
    spot_instance: bool = False

    @property
    def is_local(self) -> bool:
        return self.locality == "local"

    @property
    def is_kubernetes(self) -> bool:
        return self.target in ("kubernetes", "eks", "aks", "gke")

    @property
    def is_container(self) -> bool:
        return self.target in ("docker", "kubernetes", "eks", "aks", "gke", "ecs")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "locality": self.locality,
            "target": self.target,
            "connection": self.connection.to_dict() if self.connection else None,
            "cluster_name": self.cluster_name,
            "namespace": self.namespace,
            "service_account": self.service_account,
            "task_definition": self.task_definition,
            "dag_id": self.dag_id,
            "api_path": self.api_path,
            "node_selector": self.node_selector,
            "tolerations": self.tolerations,
            "gpu_required": self.gpu_required,
            "spot_instance": self.spot_instance,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DeploymentConfig":
        connection_data = data.get("connection")
        connection = ConnectionConfig.from_dict(connection_data) if connection_data else None

        return cls(
            locality=data.get("locality", "local"),
            target=data.get("target", "inline"),
            connection=connection,
            cluster_name=data.get("cluster_name"),
            namespace=data.get("namespace", "default"),
            service_account=data.get("service_account"),
            task_definition=data.get("task_definition"),
            dag_id=data.get("dag_id"),
            api_path=data.get("api_path"),
            node_selector=data.get("node_selector", {}),
            tolerations=data.get("tolerations", []),
            gpu_required=data.get("gpu_required", False),
            spot_instance=data.get("spot_instance", False),
        )

    @classmethod
    def local_inline(cls) -> "DeploymentConfig":
        """Create config for direct inline execution."""
        return cls(locality=ExecutionLocality.LOCAL, target=LocalTarget.INLINE)

    @classmethod
    def local_subprocess(cls) -> "DeploymentConfig":
        """Create config for sandboxed subprocess execution."""
        return cls(locality=ExecutionLocality.LOCAL, target=LocalTarget.SUBPROCESS)

    @classmethod
    def local_docker(cls, image: Optional[str] = None) -> "DeploymentConfig":
        """Create config for local Docker execution."""
        return cls(locality=ExecutionLocality.LOCAL, target=LocalTarget.DOCKER)

    @classmethod
    def remote_eks(
        cls,
        cluster_name: str,
        region: str,
        namespace: str = "default",
        gpu: bool = False,
    ) -> "DeploymentConfig":
        """Create config for AWS EKS execution."""
        return cls(
            locality=ExecutionLocality.REMOTE,
            target=RemoteTarget.EKS,
            connection=ConnectionConfig(
                protocol=NetworkProtocol.K8S, auth_method="iam", region=region
            ),
            cluster_name=cluster_name,
            namespace=namespace,
            gpu_required=gpu,
        )

    @classmethod
    def remote_airflow(
        cls,
        endpoint: str,
        dag_id: str,
        auth_method: str = "oauth",
    ) -> "DeploymentConfig":
        """Create config for Airflow DAG trigger."""
        return cls(
            locality=ExecutionLocality.REMOTE,
            target=RemoteTarget.AIRFLOW,
            connection=ConnectionConfig(
                protocol=NetworkProtocol.HTTP,
                endpoint=endpoint,
                auth_method=auth_method,
            ),
            dag_id=dag_id,
        )


@dataclass
class IsolationConfig:
    """Execution environment configuration.

    Defines the sandbox type, deployment target, and associated settings
    for executing workflow nodes safely across different environments.

    Key Differentiators from Step Functions/Airflow:
    - LLM-native: AgentNodes execute with full LLM reasoning
    - Hybrid execution: Mix deterministic (ComputeNode) and LLM (AgentNode)
    - Cost-aware: Built-in cost tier routing and budget control
    - Tool integration: Deep binding to tool registries, not just containers

    Attributes:
        sandbox_type: Level of isolation (none, process, docker)
        deployment: Deployment target configuration
        network_allowed: Whether network access is permitted
        filesystem_readonly: Mount filesystem as read-only
        resource_limits: Resource constraints for execution
        working_directory: Working directory for execution
        environment: Additional environment variables
        docker_image: Docker image for container execution
        docker_volumes: Volume mounts for Docker

    Example:
        # Local Docker execution
        config = IsolationConfig(
            sandbox_type="docker",
            network_allowed=False,
            resource_limits=ResourceLimits(max_memory_mb=1024),
        )

        # Kubernetes deployment
        config = IsolationConfig(
            sandbox_type="docker",
            deployment=DeploymentConfig(
                target="eks",
                cluster_name="ml-cluster",
                gpu_required=True,
            ),
        )
    """

    sandbox_type: SandboxType = "none"
    deployment: Optional[DeploymentConfig] = None
    network_allowed: bool = True
    filesystem_readonly: bool = False
    resource_limits: Optional[ResourceLimits] = None
    working_directory: Optional[str] = None
    environment: Dict[str, str] = field(default_factory=dict)
    docker_image: Optional[str] = None
    docker_volumes: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if self.resource_limits is None:
            self.resource_limits = ResourceLimits()

    @property
    def is_isolated(self) -> bool:
        """Check if any isolation is enabled."""
        return self.sandbox_type != "none"

    @property
    def is_docker(self) -> bool:
        """Check if Docker isolation is used."""
        return self.sandbox_type == "docker"

    @property
    def is_process(self) -> bool:
        """Check if process isolation is used."""
        return self.sandbox_type == "process"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sandbox_type": self.sandbox_type,
            "network_allowed": self.network_allowed,
            "filesystem_readonly": self.filesystem_readonly,
            "resource_limits": (self.resource_limits.to_dict() if self.resource_limits else None),
            "working_directory": self.working_directory,
            "environment": self.environment,
            "docker_image": self.docker_image,
            "docker_volumes": self.docker_volumes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IsolationConfig":
        resource_data = data.get("resource_limits")
        resource_limits = ResourceLimits.from_dict(resource_data) if resource_data else None

        return cls(
            sandbox_type=data.get("sandbox_type", "none"),
            network_allowed=data.get("network_allowed", True),
            filesystem_readonly=data.get("filesystem_readonly", False),
            resource_limits=resource_limits,
            working_directory=data.get("working_directory"),
            environment=data.get("environment", {}),
            docker_image=data.get("docker_image"),
            docker_volumes=data.get("docker_volumes", {}),
        )


class IsolationMapper:
    """Maps constraints to execution environments.

    Provides a centralized way to determine the appropriate isolation
    level based on task constraints and vertical configuration.

    The mapper applies this precedence:
    1. Explicit constraint settings (highest priority)
    2. Vertical-specific defaults
    3. Global defaults (lowest priority)

    Example:
        isolation = IsolationMapper.from_constraints(
            constraints=node.constraints,
            vertical="devops",
        )
        # Returns IsolationConfig with docker sandbox for devops
    """

    # Per-vertical default isolation configurations
    VERTICAL_DEFAULTS: Dict[str, IsolationConfig] = {
        "coding": IsolationConfig(
            sandbox_type="process",
            network_allowed=True,
            filesystem_readonly=False,
            resource_limits=ResourceLimits(max_memory_mb=1024, timeout_seconds=120.0),
        ),
        "research": IsolationConfig(
            sandbox_type="none",
            network_allowed=True,
            filesystem_readonly=False,
        ),
        "devops": IsolationConfig(
            sandbox_type="docker",
            network_allowed=True,
            filesystem_readonly=False,
            docker_image="python:3.11-slim",
            resource_limits=ResourceLimits(max_memory_mb=2048, timeout_seconds=300.0),
        ),
        "dataanalysis": IsolationConfig(
            sandbox_type="process",
            network_allowed=True,
            filesystem_readonly=False,
            resource_limits=ResourceLimits(max_memory_mb=2048, timeout_seconds=180.0),
        ),
        "rag": IsolationConfig(
            sandbox_type="docker",
            network_allowed=False,
            filesystem_readonly=True,
            docker_image="python:3.11-slim",
            resource_limits=ResourceLimits(max_memory_mb=1024, timeout_seconds=120.0),
        ),
    }

    # Default config when no vertical specified
    DEFAULT_CONFIG = IsolationConfig(
        sandbox_type="process",
        network_allowed=True,
        filesystem_readonly=False,
    )

    @classmethod
    def apply_settings(cls, settings: Any) -> None:
        """Apply SandboxSettings from Victor config to default isolation.

        Reads settings.sandbox (SandboxSettings) and updates the
        DEFAULT_CONFIG accordingly.

        Args:
            settings: Victor Settings instance.
        """
        sandbox = getattr(settings, "sandbox", None)
        if sandbox is None:
            return

        enabled = getattr(sandbox, "sandbox_enabled", False)
        if not enabled:
            cls.DEFAULT_CONFIG = IsolationConfig(
                sandbox_type="none",
                network_allowed=True,
                filesystem_readonly=False,
            )
            return

        mode = getattr(sandbox, "sandbox_filesystem_mode", "workspace-only")
        network = not getattr(sandbox, "sandbox_network_isolation", False)
        readonly = mode == "workspace-only"

        cls.DEFAULT_CONFIG = IsolationConfig(
            sandbox_type="process",
            network_allowed=network,
            filesystem_readonly=readonly,
        )

    @classmethod
    def from_constraints(
        cls,
        constraints: "ConstraintsProtocol",
        vertical: Optional[str] = None,
        override_config: Optional[Dict[str, Any]] = None,
    ) -> IsolationConfig:
        """Map constraints to isolation configuration.

        Args:
            constraints: Task constraints from the node
            vertical: Vertical name for default lookup
            override_config: Optional explicit config overrides

        Returns:
            IsolationConfig appropriate for the constraints
        """
        from victor.workflows.definition import (
            AirgappedConstraints,
            ComputeOnlyConstraints,
            FullAccessConstraints,
        )

        # Start with vertical defaults or global default
        if vertical and vertical in cls.VERTICAL_DEFAULTS:
            base_config = cls.VERTICAL_DEFAULTS[vertical]
        else:
            base_config = cls.DEFAULT_CONFIG

        # Apply constraint-based overrides
        sandbox_type = base_config.sandbox_type
        network_allowed = base_config.network_allowed
        filesystem_readonly = base_config.filesystem_readonly
        resource_limits = base_config.resource_limits

        # AirgappedConstraints: Maximum isolation without network
        if isinstance(constraints, AirgappedConstraints):
            from victor.tools.write_path_policy import (
                WritePathPolicy,
                set_active_write_policy,
            )

            set_active_write_policy(WritePathPolicy.read_only())
            sandbox_type = "none"  # No external dependencies
            network_allowed = False
            filesystem_readonly = True
            logger.debug("AirgappedConstraints: sandbox=none, network=off")

        # ComputeOnlyConstraints: Process isolation, no LLM
        elif isinstance(constraints, ComputeOnlyConstraints):
            from victor.tools.write_path_policy import (
                WritePathPolicy,
                set_active_write_policy,
            )

            set_active_write_policy(WritePathPolicy.read_only())
            if sandbox_type == "none":
                sandbox_type = "process"  # Upgrade to process isolation
            network_allowed = constraints.network_allowed
            logger.debug(f"ComputeOnlyConstraints: sandbox={sandbox_type}")

        # FullAccessConstraints: Docker if available
        elif isinstance(constraints, FullAccessConstraints):
            from victor.tools.write_path_policy import (
                WritePathPolicy,
                set_active_write_policy,
            )

            set_active_write_policy(WritePathPolicy.full_access())
            sandbox_type = "docker"
            network_allowed = True
            filesystem_readonly = False
            resource_limits = ResourceLimits(
                max_memory_mb=2048,
                timeout_seconds=constraints.timeout,
            )
            logger.debug("FullAccessConstraints: sandbox=docker, full access")

        # Standard constraints: Apply specific settings
        else:
            # Check network setting from constraints
            if hasattr(constraints, "network_allowed"):
                network_allowed = constraints.network_allowed

            # Path-tiered write policy takes precedence over binary write_allowed
            write_path_policy = getattr(constraints, "write_path_policy", None)
            if write_path_policy is not None:
                from victor.tools.write_path_policy import set_active_write_policy

                set_active_write_policy(write_path_policy)
                filesystem_readonly = False  # per-path enforcement governs
            elif hasattr(constraints, "write_allowed"):
                from victor.tools.write_path_policy import (
                    WritePathPolicy,
                    set_active_write_policy,
                )

                if constraints.write_allowed:
                    set_active_write_policy(WritePathPolicy.full_access())
                else:
                    set_active_write_policy(WritePathPolicy.read_only())
                filesystem_readonly = not constraints.write_allowed

            # Update timeout from constraints
            if resource_limits and hasattr(constraints, "timeout"):
                resource_limits = ResourceLimits(
                    max_memory_mb=resource_limits.max_memory_mb,
                    max_cpu_seconds=resource_limits.max_cpu_seconds,
                    max_file_descriptors=resource_limits.max_file_descriptors,
                    max_processes=resource_limits.max_processes,
                    timeout_seconds=constraints.timeout,
                )

        # Build final config
        config = IsolationConfig(
            sandbox_type=sandbox_type,
            network_allowed=network_allowed,
            filesystem_readonly=filesystem_readonly,
            resource_limits=resource_limits,
            working_directory=base_config.working_directory,
            environment=dict(base_config.environment),
            docker_image=base_config.docker_image,
            docker_volumes=dict(base_config.docker_volumes),
        )

        # Apply explicit overrides
        if override_config:
            if "sandbox_type" in override_config:
                config.sandbox_type = override_config["sandbox_type"]
            if "network_allowed" in override_config:
                config.network_allowed = override_config["network_allowed"]
            if "filesystem_readonly" in override_config:
                config.filesystem_readonly = override_config["filesystem_readonly"]
            if "docker_image" in override_config:
                config.docker_image = override_config["docker_image"]
            if "working_directory" in override_config:
                config.working_directory = override_config["working_directory"]
            if "environment" in override_config:
                config.environment.update(override_config["environment"])

        return config

    @classmethod
    def get_vertical_default(cls, vertical: str) -> IsolationConfig:
        """Get the default isolation config for a vertical.

        Args:
            vertical: Vertical name

        Returns:
            IsolationConfig for the vertical (or default)
        """
        return cls.VERTICAL_DEFAULTS.get(vertical, cls.DEFAULT_CONFIG)

    @classmethod
    def register_vertical_default(
        cls,
        vertical: str,
        config: IsolationConfig,
    ) -> None:
        """Register or update default config for a vertical.

        Allows verticals to customize their default isolation settings.

        Args:
            vertical: Vertical name
            config: Isolation configuration
        """
        cls.VERTICAL_DEFAULTS[vertical] = config
        logger.debug(f"Registered isolation default for vertical: {vertical}")

    @classmethod
    def list_verticals(cls) -> list:
        """List all registered vertical names."""
        return list(cls.VERTICAL_DEFAULTS.keys())


__all__ = [
    # Sandbox providers (extensibility)
    "SandboxResult",
    "SandboxProvider",
    "SandboxProviderRegistry",
    "NoneSandboxProvider",
    "ProcessSandboxProvider",
    "DockerSandboxProvider",
    "register_builtin_providers",
    # Type aliases
    "SandboxType",
    "ExecutionLocality",
    "LocalTarget",
    "RemoteTarget",
    # Configuration classes
    "ResourceLimits",
    "ConnectionConfig",
    "DeploymentConfig",
    "IsolationConfig",
    # Mapper
    "IsolationMapper",
]
