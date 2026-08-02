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

"""Dynamic capability definitions for the DevOps vertical.

This module provides capability declarations that can be loaded
dynamically by the CapabilityLoader, enabling runtime extension
of the DevOps vertical with custom functionality.

The module follows the CapabilityLoader's discovery patterns:
1. CAPABILITIES list for batch registration
2. @capability decorator for function-based capabilities
3. Capability classes for complex implementations

Example:
    # Register capabilities with loader
    from victor.framework import CapabilityLoader
    loader = CapabilityLoader()
    loader.load_from_module("victor_devops.capabilities")

    # Or use directly
    from victor_devops.capabilities import (
        get_devops_capabilities,
        DevOpsCapabilityProvider,
    )
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Set

from victor_contracts.capabilities import (
    BaseCapabilityProvider,
    CapabilityEntry,
    CapabilityMetadata,
    CapabilityType,
    OrchestratorCapability,
    build_capability_loader,
    capability,
    load_capability_config,
    store_capability_config,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Capability Config Helpers (P1: Framework CapabilityConfigService Migration)
# =============================================================================


_DEPLOYMENT_SAFETY_DEFAULTS: Dict[str, Any] = {
    "require_approval_for_production": True,
    "require_backup_before_deploy": True,
    "enable_rollback": True,
    "protected_environments": ["production", "staging"],
}
_CONTAINER_DEFAULTS: Dict[str, Any] = {
    "runtime": "docker",
    "default_registry": None,
    "security_scan_enabled": True,
    "max_image_size_mb": 2000,
}
_INFRASTRUCTURE_DEFAULTS: Dict[str, Any] = {
    "iac_tool": "terraform",
    "auto_approve_non_destructive": False,
    "require_plan_before_apply": True,
    "state_backend": None,
}
_CICD_DEFAULTS: Dict[str, Any] = {
    "platform": "github_actions",
    "run_tests_before_deploy": True,
    "require_passing_checks": True,
    "enable_security_scan": True,
}
_MONITORING_DEFAULTS: Dict[str, Any] = {
    "metrics_backend": "prometheus",
    "logging_backend": "loki",
    "alerting_enabled": True,
    "dashboard_tool": "grafana",
}

# =============================================================================
# Capability Handlers
# =============================================================================


def configure_deployment_safety(
    orchestrator: Any,
    *,
    require_approval_for_production: bool = True,
    require_backup_before_deploy: bool = True,
    enable_rollback: bool = True,
    protected_environments: Optional[List[str]] = None,
) -> None:
    """Configure deployment safety rules for the orchestrator.

    This capability configures the orchestrator's deployment safety
    checks to prevent dangerous operations.

    Args:
        orchestrator: Target orchestrator
        require_approval_for_production: Require approval for production deployments
        require_backup_before_deploy: Require backup before deployment
        enable_rollback: Enable automatic rollback on failure
        protected_environments: List of environments that require extra caution
    """
    config = {
        "require_approval_for_production": require_approval_for_production,
        "require_backup_before_deploy": require_backup_before_deploy,
        "enable_rollback": enable_rollback,
        "protected_environments": protected_environments or ["production", "staging"],
    }
    service_stored = store_capability_config(orchestrator, "deployment_safety", config)

    # Legacy compatibility while runtime consumers migrate.
    if not service_stored and hasattr(orchestrator, "safety_config"):
        orchestrator.safety_config["deployment"] = {
            "require_approval_for_production": require_approval_for_production,
            "require_backup_before_deploy": require_backup_before_deploy,
            "enable_rollback": enable_rollback,
            "protected_environments": protected_environments or ["production", "staging"],
        }

    logger.info("Configured deployment safety rules")


def configure_container_settings(
    orchestrator: Any,
    *,
    runtime: str = "docker",
    default_registry: Optional[str] = None,
    security_scan_enabled: bool = True,
    max_image_size_mb: int = 2000,
) -> None:
    """Configure container settings for the orchestrator.

    Args:
        orchestrator: Target orchestrator
        runtime: Container runtime to use (docker, podman)
        default_registry: Default container registry URL
        security_scan_enabled: Enable security scanning for images
        max_image_size_mb: Maximum allowed image size in MB
    """
    store_capability_config(
        orchestrator,
        "container_config",
        {
            "runtime": runtime,
            "default_registry": default_registry,
            "security_scan_enabled": security_scan_enabled,
            "max_image_size_mb": max_image_size_mb,
        },
    )

    logger.info(f"Configured container settings: runtime={runtime}")


def get_container_settings(orchestrator: Any) -> Dict[str, Any]:
    """Get current container configuration.

    Args:
        orchestrator: Target orchestrator

    Returns:
        Container configuration dict
    """
    return load_capability_config(orchestrator, "container_config", _CONTAINER_DEFAULTS)


def configure_infrastructure_settings(
    orchestrator: Any,
    *,
    iac_tool: str = "terraform",
    auto_approve_non_destructive: bool = False,
    require_plan_before_apply: bool = True,
    state_backend: Optional[str] = None,
) -> None:
    """Configure Infrastructure as Code settings.

    Args:
        orchestrator: Target orchestrator
        iac_tool: IaC tool to use (terraform, opentofu, pulumi, cloudformation)
        auto_approve_non_destructive: Auto-approve non-destructive changes
        require_plan_before_apply: Require plan step before apply
        state_backend: Backend for storing IaC state
    """
    store_capability_config(
        orchestrator,
        "infrastructure_config",
        {
            "iac_tool": iac_tool,
            "auto_approve_non_destructive": auto_approve_non_destructive,
            "require_plan_before_apply": require_plan_before_apply,
            "state_backend": state_backend,
        },
        fallback_attr="infra_config",
    )

    logger.info(f"Configured infrastructure settings: iac_tool={iac_tool}")


def configure_cicd_settings(
    orchestrator: Any,
    *,
    platform: str = "github_actions",
    run_tests_before_deploy: bool = True,
    require_passing_checks: bool = True,
    enable_security_scan: bool = True,
) -> None:
    """Configure CI/CD pipeline settings.

    Args:
        orchestrator: Target orchestrator
        platform: CI/CD platform (github_actions, gitlab_ci, jenkins)
        run_tests_before_deploy: Run tests before deployment
        require_passing_checks: Require all checks to pass
        enable_security_scan: Enable security scanning in pipeline
    """
    store_capability_config(
        orchestrator,
        "cicd_config",
        {
            "platform": platform,
            "run_tests_before_deploy": run_tests_before_deploy,
            "require_passing_checks": require_passing_checks,
            "enable_security_scan": enable_security_scan,
        },
    )

    logger.info(f"Configured CI/CD settings: platform={platform}")


def configure_monitoring_settings(
    orchestrator: Any,
    *,
    metrics_backend: str = "prometheus",
    logging_backend: str = "loki",
    alerting_enabled: bool = True,
    dashboard_tool: str = "grafana",
) -> None:
    """Configure monitoring and observability settings.

    Args:
        orchestrator: Target orchestrator
        metrics_backend: Metrics collection backend
        logging_backend: Log aggregation backend
        alerting_enabled: Enable alerting
        dashboard_tool: Dashboard visualization tool
    """
    store_capability_config(
        orchestrator,
        "monitoring_config",
        {
            "metrics_backend": metrics_backend,
            "logging_backend": logging_backend,
            "alerting_enabled": alerting_enabled,
            "dashboard_tool": dashboard_tool,
        },
    )

    logger.info(f"Configured monitoring settings: metrics={metrics_backend}")


# =============================================================================
# Decorated Capability Functions
# =============================================================================


@capability(
    name="devops_deployment_safety",
    capability_type=CapabilityType.SAFETY,
    version="1.0",
    description="Deployment safety rules for preventing dangerous operations",
)
def devops_deployment_safety(
    require_approval_for_production: bool = True,
    require_backup_before_deploy: bool = True,
    **kwargs: Any,
) -> Callable:
    """Deployment safety capability handler."""

    def handler(orchestrator: Any) -> None:
        configure_deployment_safety(
            orchestrator,
            require_approval_for_production=require_approval_for_production,
            require_backup_before_deploy=require_backup_before_deploy,
            **kwargs,
        )

    return handler


@capability(
    name="devops_container",
    capability_type=CapabilityType.TOOL,
    version="1.0",
    description="Container management configuration",
    getter="get_container_settings",
)
def devops_container(
    runtime: str = "docker",
    security_scan_enabled: bool = True,
    **kwargs: Any,
) -> Callable:
    """Container capability handler."""

    def handler(orchestrator: Any) -> None:
        configure_container_settings(
            orchestrator,
            runtime=runtime,
            security_scan_enabled=security_scan_enabled,
            **kwargs,
        )

    return handler


@capability(
    name="devops_infrastructure",
    capability_type=CapabilityType.TOOL,
    version="1.0",
    description="Infrastructure as Code configuration",
)
def devops_infrastructure(
    iac_tool: str = "terraform",
    require_plan_before_apply: bool = True,
    **kwargs: Any,
) -> Callable:
    """Infrastructure capability handler."""

    def handler(orchestrator: Any) -> None:
        configure_infrastructure_settings(
            orchestrator,
            iac_tool=iac_tool,
            require_plan_before_apply=require_plan_before_apply,
            **kwargs,
        )

    return handler


@capability(
    name="devops_cicd",
    capability_type=CapabilityType.TOOL,
    version="1.0",
    description="CI/CD pipeline configuration",
)
def devops_cicd(
    platform: str = "github_actions",
    **kwargs: Any,
) -> Callable:
    """CI/CD capability handler."""

    def handler(orchestrator: Any) -> None:
        configure_cicd_settings(
            orchestrator,
            platform=platform,
            **kwargs,
        )

    return handler


@capability(
    name="devops_monitoring",
    capability_type=CapabilityType.TOOL,
    version="1.0",
    description="Monitoring and observability configuration",
)
def devops_monitoring(
    metrics_backend: str = "prometheus",
    **kwargs: Any,
) -> Callable:
    """Monitoring capability handler."""

    def handler(orchestrator: Any) -> None:
        configure_monitoring_settings(
            orchestrator,
            metrics_backend=metrics_backend,
            **kwargs,
        )

    return handler


# =============================================================================
# Capability Provider Class
# =============================================================================


class DevOpsCapabilityProvider(BaseCapabilityProvider[Callable[..., None]]):
    """Provider for DevOps-specific capabilities.

    This class provides a structured way to access and apply
    DevOps capabilities to an orchestrator. It inherits from
    BaseCapabilityProvider for consistent capability registration
    and discovery across all verticals.

    Example:
        provider = DevOpsCapabilityProvider()

        # List available capabilities
        print(provider.list_capabilities())

        # Apply specific capabilities
        provider.apply_deployment_safety(orchestrator)
        provider.apply_container_settings(orchestrator, runtime="podman")

        # Use BaseCapabilityProvider interface
        cap = provider.get_capability("deployment_safety")
        if cap:
            cap(orchestrator)
    """

    def __init__(self):
        """Initialize the capability provider."""
        self._applied: Set[str] = set()
        # Map capability names to their handler functions
        self._capabilities: Dict[str, Callable[..., None]] = {
            "deployment_safety": configure_deployment_safety,
            "container_settings": configure_container_settings,
            "infrastructure_settings": configure_infrastructure_settings,
            "cicd_settings": configure_cicd_settings,
            "monitoring_settings": configure_monitoring_settings,
        }
        # Capability metadata for discovery
        self._metadata: Dict[str, CapabilityMetadata] = {
            "deployment_safety": CapabilityMetadata(
                name="deployment_safety",
                description="Deployment safety rules for preventing dangerous operations",
                version="1.0",
                tags=["safety", "deployment", "production"],
            ),
            "container_settings": CapabilityMetadata(
                name="container_settings",
                description="Container management and configuration",
                version="1.0",
                tags=["docker", "container", "podman"],
            ),
            "infrastructure_settings": CapabilityMetadata(
                name="infrastructure_settings",
                description="Infrastructure as Code configuration",
                version="1.0",
                tags=["terraform", "iac", "infrastructure"],
            ),
            "cicd_settings": CapabilityMetadata(
                name="cicd_settings",
                description="CI/CD pipeline configuration",
                version="1.0",
                tags=["cicd", "pipeline", "automation"],
            ),
            "monitoring_settings": CapabilityMetadata(
                name="monitoring_settings",
                description="Monitoring and observability configuration",
                version="1.0",
                dependencies=["deployment_safety"],
                tags=["monitoring", "observability", "metrics", "logging"],
            ),
        }

    def get_capabilities(self) -> Dict[str, Callable[..., None]]:
        """Return all registered capabilities.

        Returns:
            Dictionary mapping capability names to handler functions.
        """
        return self._capabilities.copy()

    def get_capability_metadata(self) -> Dict[str, CapabilityMetadata]:
        """Return metadata for all registered capabilities.

        Returns:
            Dictionary mapping capability names to their metadata.
        """
        return self._metadata.copy()

    def apply_deployment_safety(
        self,
        orchestrator: Any,
        **kwargs: Any,
    ) -> None:
        """Apply deployment safety capability.

        Args:
            orchestrator: Target orchestrator
            **kwargs: Deployment safety options
        """
        configure_deployment_safety(orchestrator, **kwargs)
        self._applied.add("deployment_safety")

    def apply_container_settings(
        self,
        orchestrator: Any,
        **kwargs: Any,
    ) -> None:
        """Apply container settings capability.

        Args:
            orchestrator: Target orchestrator
            **kwargs: Container options
        """
        configure_container_settings(orchestrator, **kwargs)
        self._applied.add("container_settings")

    def apply_infrastructure_settings(
        self,
        orchestrator: Any,
        **kwargs: Any,
    ) -> None:
        """Apply infrastructure settings capability.

        Args:
            orchestrator: Target orchestrator
            **kwargs: Infrastructure options
        """
        configure_infrastructure_settings(orchestrator, **kwargs)
        self._applied.add("infrastructure_settings")

    def apply_cicd_settings(
        self,
        orchestrator: Any,
        **kwargs: Any,
    ) -> None:
        """Apply CI/CD settings capability.

        Args:
            orchestrator: Target orchestrator
            **kwargs: CI/CD options
        """
        configure_cicd_settings(orchestrator, **kwargs)
        self._applied.add("cicd_settings")

    def apply_monitoring_settings(
        self,
        orchestrator: Any,
        **kwargs: Any,
    ) -> None:
        """Apply monitoring settings capability.

        Args:
            orchestrator: Target orchestrator
            **kwargs: Monitoring options
        """
        configure_monitoring_settings(orchestrator, **kwargs)
        self._applied.add("monitoring_settings")

    def apply_all(
        self,
        orchestrator: Any,
        **kwargs: Any,
    ) -> None:
        """Apply all DevOps capabilities with defaults.

        Args:
            orchestrator: Target orchestrator
            **kwargs: Shared options
        """
        self.apply_deployment_safety(orchestrator)
        self.apply_container_settings(orchestrator)
        self.apply_infrastructure_settings(orchestrator)
        self.apply_cicd_settings(orchestrator)
        self.apply_monitoring_settings(orchestrator)

    def get_applied(self) -> Set[str]:
        """Get set of applied capability names.

        Returns:
            Set of applied capability names
        """
        return self._applied.copy()


# =============================================================================
# CAPABILITIES List for CapabilityLoader Discovery
# =============================================================================


CAPABILITIES: List[CapabilityEntry] = [
    CapabilityEntry(
        capability=OrchestratorCapability(
            name="devops_deployment_safety",
            capability_type=CapabilityType.SAFETY,
            version="1.0",
            setter="configure_deployment_safety",
            description="Deployment safety rules for preventing dangerous operations",
        ),
        handler=configure_deployment_safety,
    ),
    CapabilityEntry(
        capability=OrchestratorCapability(
            name="devops_container",
            capability_type=CapabilityType.TOOL,
            version="1.0",
            setter="configure_container_settings",
            getter="get_container_settings",
            description="Container management and configuration",
        ),
        handler=configure_container_settings,
        getter_handler=get_container_settings,
    ),
    CapabilityEntry(
        capability=OrchestratorCapability(
            name="devops_infrastructure",
            capability_type=CapabilityType.TOOL,
            version="1.0",
            setter="configure_infrastructure_settings",
            description="Infrastructure as Code configuration",
        ),
        handler=configure_infrastructure_settings,
    ),
    CapabilityEntry(
        capability=OrchestratorCapability(
            name="devops_cicd",
            capability_type=CapabilityType.TOOL,
            version="1.0",
            setter="configure_cicd_settings",
            description="CI/CD pipeline configuration",
        ),
        handler=configure_cicd_settings,
    ),
    CapabilityEntry(
        capability=OrchestratorCapability(
            name="devops_monitoring",
            capability_type=CapabilityType.TOOL,
            version="1.0",
            setter="configure_monitoring_settings",
            description="Monitoring and observability configuration",
        ),
        handler=configure_monitoring_settings,
    ),
]


# =============================================================================
# Convenience Functions
# =============================================================================


def get_devops_capabilities() -> List[CapabilityEntry]:
    """Get all DevOps capability entries.

    Returns:
        List of capability entries for loader registration
    """
    return CAPABILITIES.copy()


def create_devops_capability_loader(
    loader_factory: Optional[Callable[[], Any]] = None,
) -> Any:
    """Create and populate a host capability loader for the DevOps vertical.

    Args:
        loader_factory: Host-side factory that returns a compatible capability loader.

    Returns:
        Host capability loader populated with DevOps capability entries.
    """
    if loader_factory is None:
        raise RuntimeError(
            "create_devops_capability_loader() requires a host loader_factory, such as "
            "victor.framework.CapabilityLoader."
        )

    return build_capability_loader(
        CAPABILITIES,
        loader_factory=loader_factory,
        source_module="victor_devops.capabilities",
    )


# =============================================================================
# SOLID: Centralized Config Storage
# =============================================================================


def get_capability_configs() -> Dict[str, Any]:
    """Get DevOps capability configurations for centralized storage.

    Returns default DevOps configuration for VerticalContext storage.
    This replaces direct orchestrator deployment/container/cicd_config assignment.

    Returns:
        Dict with default DevOps capability configurations
    """
    return {
        "deployment_safety": {
            "require_approval_for_production": True,
            "require_backup_before_deploy": True,
            "enable_rollback": True,
            "protected_environments": ["production", "staging"],
        },
        "container_config": {
            "runtime": "docker",
            "default_registry": None,
            "security_scan_enabled": True,
            "max_image_size_mb": 2000,
        },
        "infrastructure_config": {
            "iac_tool": "terraform",
            "auto_approve_non_destructive": False,
            "require_plan_before_apply": True,
            "state_backend": None,
        },
        "cicd_config": {
            "platform": "github_actions",
            "run_tests_before_deploy": True,
            "require_passing_checks": True,
            "enable_security_scan": True,
        },
        "monitoring_config": {
            "metrics_backend": "prometheus",
            "logging_backend": "loki",
            "alerting_enabled": True,
            "dashboard_tool": "grafana",
        },
    }


__all__ = [
    # Handlers
    "configure_deployment_safety",
    "configure_container_settings",
    "configure_infrastructure_settings",
    "configure_cicd_settings",
    "configure_monitoring_settings",
    "get_container_settings",
    # Provider class and base types
    "DevOpsCapabilityProvider",
    "CapabilityMetadata",  # Re-exported from framework for convenience
    # Capability list for loader
    "CAPABILITIES",
    # Convenience functions
    "get_devops_capabilities",
    "create_devops_capability_loader",
    # SOLID: Centralized config storage
    "get_capability_configs",
]
