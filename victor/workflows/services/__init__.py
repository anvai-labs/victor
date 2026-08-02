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

"""ServiceNode infrastructure for workflow lifecycle management.

Provides infrastructure services (databases, caches, message queues) that
start before workflow execution and stop afterward. Supports local Docker,
Kubernetes, and cloud-managed services.

Key Components:
- ServiceConfig: Declarative service configuration
- ServiceHandle: Runtime handle to running service
- ServiceRegistry: Manages service lifecycle
- ServiceProvider: Backend-specific implementations
- CredentialManager: Unified credential management with MFA/SSO

Providers:
- Docker: Local container management
- Kubernetes: K8s deployment management
- Local: OS subprocess management
- External: Connect to existing services
- AWS: RDS, ElastiCache, MSK, DynamoDB, SQS

Security Features:
- MFA: TOTP, Touch ID, Passkeys
- SSO: Okta, Azure AD, Google, OIDC/SAML
- Smart Cards: PIV, CAC (FedRAMP)

Example:
    from victor.workflows.services import (
        ServiceConfig,
        ServicePresets,
        ServiceRegistry,
        DockerServiceProvider,
    )

    # Create registry and register provider
    registry = ServiceRegistry()
    registry.register_provider("docker", DockerServiceProvider())

    # Use preset for PostgreSQL
    configs = [
        ServicePresets.postgres(name="db", password="secret"),
        ServicePresets.redis(name="cache"),
    ]

    # Start services
    async with ServiceContext(registry, configs) as ctx:
        db_url = ctx.get_export("db", "DATABASE_URL")
        # ... run workflow with db_url ...
    # Services automatically stopped

YAML Workflow Example:
    workflows:
      data_pipeline:
        services:
          postgres:
            provider: docker
            image: postgres:15
            ports: [5432]
            health_check:
              type: postgres
            exports:
              DATABASE_URL: "postgresql://..."

        nodes:
          - id: migrate
            requires_services: [postgres]
            ...
"""

# Core definitions
from victor.workflows.services.definition import (
    # Enums
    ServiceState,
    HealthCheckType,
    ServiceProviderType,
    # Configurations
    HealthCheckConfig,
    LifecycleConfig,
    PortMapping,
    VolumeMount,
    ServiceConfig,
    # Runtime
    ServiceHandle,
    # Protocol
    ServiceProvider,
    # Exceptions
    ServiceError,
    ServiceStartError,
    ServiceStopError,
    ServiceHealthError,
    ServiceDependencyError,
    # Presets
    ServicePresets,
)

# Registry
from victor.workflows.services.registry import (
    ServiceRegistryEntry,
    ServiceRegistry,
    ServiceContext,
)

# Credentials and Security
from victor.workflows.services.credentials import (
    # Credential types
    CredentialType,
    AWSCredentials,
    AzureCredentials,
    GCPCredentials,
    DockerCredentials,
    KubernetesCredentials,
    DatabaseCredentials,
    APIKeyCredentials,
    # MFA
    MFAMethod,
    MFAConfig,
    MFAVerifier,
    get_mfa_verifier,
    # Smart Card / PIV / CAC
    SmartCardType,
    SmartCardCredentials,
    # SSO / OAuth / OIDC
    SSOProvider,
    SSOConfig,
    SSOTokens,
    SSOAuthenticator,
    # System Authentication (PAM, NTLM, Kerberos, AD)
    SystemAuthType,
    SystemAuthConfig,
    SystemAuthenticator,
    get_system_authenticator,
    # Manager
    CredentialManager,
    get_credential_manager,
)


# Providers (lazy import to avoid heavy dependencies)
def get_docker_provider(**kwargs):
    """Get Docker service provider."""
    from victor.workflows.services.providers.docker import DockerServiceProvider

    return DockerServiceProvider(**kwargs)


def get_kubernetes_provider(**kwargs):
    """Get Kubernetes service provider."""
    from victor.workflows.services.providers.kubernetes import KubernetesServiceProvider

    return KubernetesServiceProvider(**kwargs)


def get_local_provider(**kwargs):
    """Get local process provider."""
    from victor.workflows.services.providers.local import LocalProcessProvider

    return LocalProcessProvider(**kwargs)


def get_external_provider(**kwargs):
    """Get external service provider."""
    from victor.workflows.services.providers.external import ExternalServiceProvider

    return ExternalServiceProvider(**kwargs)


def get_aws_provider(**kwargs):
    """Get AWS service provider."""
    from victor.workflows.services.providers.aws import AWSServiceProvider

    return AWSServiceProvider(**kwargs)


def create_default_registry() -> ServiceRegistry:
    """Create a registry with all default providers registered.

    Returns:
        ServiceRegistry with docker, kubernetes, local, external, aws providers
    """
    registry = ServiceRegistry()

    # Register providers (will fail gracefully if deps missing)
    try:
        from victor.workflows.services.providers.docker import DockerServiceProvider

        registry.register_provider("docker", DockerServiceProvider())
    except ImportError:
        pass

    try:
        from victor.workflows.services.providers.local import LocalProcessProvider

        registry.register_provider("local", LocalProcessProvider())
    except ImportError:
        pass

    try:
        from victor.workflows.services.providers.external import ExternalServiceProvider

        registry.register_provider("external", ExternalServiceProvider())
    except ImportError:
        pass

    try:
        from victor.workflows.services.providers.kubernetes import (
            KubernetesServiceProvider,
        )

        registry.register_provider("kubernetes", KubernetesServiceProvider())
    except ImportError:
        pass

    try:
        from victor.workflows.services.providers.aws import AWSServiceProvider

        registry.register_provider("aws_rds", AWSServiceProvider())
        registry.register_provider("aws_elasticache", AWSServiceProvider())
        registry.register_provider("aws_msk", AWSServiceProvider())
        registry.register_provider("aws_dynamodb", AWSServiceProvider())
        registry.register_provider("aws_sqs", AWSServiceProvider())
    except ImportError:
        pass

    return registry


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
    # Registry
    "ServiceRegistryEntry",
    "ServiceRegistry",
    "ServiceContext",
    # Credentials
    "CredentialType",
    "AWSCredentials",
    "AzureCredentials",
    "GCPCredentials",
    "DockerCredentials",
    "KubernetesCredentials",
    "DatabaseCredentials",
    "APIKeyCredentials",
    # MFA
    "MFAMethod",
    "MFAConfig",
    "MFAVerifier",
    "get_mfa_verifier",
    # Smart Card
    "SmartCardType",
    "SmartCardCredentials",
    # SSO
    "SSOProvider",
    "SSOConfig",
    "SSOTokens",
    "SSOAuthenticator",
    # System Authentication
    "SystemAuthType",
    "SystemAuthConfig",
    "SystemAuthenticator",
    "get_system_authenticator",
    # Credential Manager
    "CredentialManager",
    "get_credential_manager",
    # Provider factories
    "get_docker_provider",
    "get_kubernetes_provider",
    "get_local_provider",
    "get_external_provider",
    "get_aws_provider",
    "create_default_registry",
]
