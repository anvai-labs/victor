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

"""Service provider for workflow dependencies.

Registers all services required by the workflow system with the DI container.
This enables:
- Centralized workflow service configuration
- Consistent lifecycle management
- Easy testing via override_services
- Type-safe service resolution

Design Pattern: Service Provider
- Groups related workflow service registrations
- Separates singleton vs scoped vs transient lifetimes
- Provides factory functions for complex service creation

Usage:
    from victor.core.container import ServiceContainer
    from victor.workflows.services.workflow_service_provider import (
        WorkflowServiceProvider,
        configure_workflow_services,
    )

    # Option 1: Full registration
    container = ServiceContainer()
    configure_workflow_services(container, settings)

    # Option 2: Selective registration
    provider = WorkflowServiceProvider(settings)
    provider.register_singleton_services(container)  # Only singletons
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

from victor.core.container import ServiceContainer, ServiceLifetime
from victor.core.service_registration import (
    ServiceRegistrationSpec,
    register_services_from_specs,
)
from victor.core.service_specs import (
    WORKFLOW_SCOPED_SPECS,
    WORKFLOW_SINGLETON_SPECS,
    WORKFLOW_TRANSIENT_SPECS,
)

if TYPE_CHECKING:
    from victor.config.settings import Settings

logger = logging.getLogger(__name__)


class WorkflowServiceProvider:
    """Service provider for workflow dependencies.

    Manages registration of all services required by the workflow system.
    Services are categorized by lifetime:

    Singleton Services (application lifetime):
        - YAMLWorkflowLoader: Loads and parses YAML workflow definitions
        - WorkflowValidator: Validates workflow structure and semantics
        - NodeExecutorFactory: Creates executor functions for workflow nodes
        - ComputeHandlerRegistry: Registry for compute node handlers

    Scoped Services (per-workflow-execution):
        - ExecutionContext: Execution context for workflow nodes
        - OrchestratorPool: Pool of orchestrators for multi-provider workflows

    Transient Services (created on each request):
        - WorkflowCompiler: Compiles YAML to executable graphs
        - WorkflowExecutor: Executes compiled workflow graphs

    Attributes:
        _settings: Application settings for service configuration

    Example:
        container = ServiceContainer()
        provider = WorkflowServiceProvider(settings)
        provider.register_services(container)

        # Resolve singletons directly
        factory = container.get(NodeExecutorFactoryProtocol)

        # Resolve scoped services within a scope
        with container.create_scope() as scope:
            context = scope.get(ExecutionContextProtocol)
    """

    def __init__(self, settings: "Settings"):
        """Initialize the service provider.

        Args:
            settings: Application settings for service configuration
        """
        self._settings = settings

    def register_services(self, container: ServiceContainer) -> None:
        """Register all workflow services.

        Registers singleton, scoped, and transient services. Call this method
        during application bootstrap to set up all workflow dependencies.

        Args:
            container: DI container to register services in
        """
        # Store container reference for factory methods
        self.container = container
        self.register_singleton_services(container)
        self.register_scoped_services(container)
        self.register_transient_services(container)
        logger.info("Registered all workflow services")

    def register_singleton_services(self, container: ServiceContainer) -> None:
        """Register singleton (application-lifetime) services.

        These services are created once and shared across all workflow executions.
        Use for stateless services or those with expensive initialization.

        Args:
            container: DI container to register services in
        """
        specs = []
        for spec in WORKFLOW_SINGLETON_SPECS:
            factory = getattr(self, spec.factory_attr)
            if spec.pass_container:

                def factory_fn(c, method=factory):
                    return method(c)

            else:

                def factory_fn(c, method=factory):
                    return method()

            specs.append(
                ServiceRegistrationSpec(
                    spec.protocol,
                    factory_fn,
                    lifetime=spec.lifetime,
                )
            )

        register_services_from_specs(container, specs)

        logger.debug("Registered singleton workflow services")

    def register_scoped_services(self, container: ServiceContainer) -> None:
        """Register scoped (per-workflow-execution) services.

        These services are created fresh for each workflow execution.
        Use for stateful services that need isolation between executions.

        Args:
            container: DI container to register services in
        """
        scoped_specs = []
        for spec in WORKFLOW_SCOPED_SPECS:
            factory = getattr(self, spec.factory_attr)
            if spec.pass_container:

                def factory_fn(c, method=factory):
                    return method(c)

            else:

                def factory_fn(c, method=factory):
                    return method()

            scoped_specs.append(
                ServiceRegistrationSpec(
                    spec.protocol,
                    factory_fn,
                    lifetime=spec.lifetime,
                )
            )

        register_services_from_specs(container, scoped_specs)

        logger.debug("Registered scoped workflow services")

    def register_transient_services(self, container: ServiceContainer) -> None:
        """Register transient (created on each request) services.

        These services are created fresh each time they are requested.
        Use for lightweight services with no state.

        Args:
            container: DI container to register services in
        """
        # Import concrete implementations for registration
        from victor.workflows.compiler.workflow_compiler_impl import (
            WorkflowCompilerImpl,
        )
        from victor.workflows.compiled_executor import WorkflowExecutor
        from victor.workflows.compiler_protocols import WorkflowCompilerProtocol

        transient_specs = []
        for spec in WORKFLOW_TRANSIENT_SPECS:
            factory = getattr(self, spec.factory_attr)
            if spec.pass_container:

                def factory_fn(c, method=factory):
                    return method(c)

            else:

                def factory_fn(c, method=factory):
                    return method()

            transient_specs.append(
                ServiceRegistrationSpec(
                    spec.protocol,
                    factory_fn,
                    lifetime=spec.lifetime,
                )
            )

        register_services_from_specs(container, transient_specs)

        logger.debug("Registered transient workflow services")

    # =========================================================================
    # Factory methods for singleton services
    # =========================================================================

    def _create_yaml_loader(self) -> Any:
        """Create YAMLWorkflowLoader instance.

        The YAMLWorkflowLoader is responsible for:
        - Loading YAML from file paths or string content
        - Parsing multi-workflow YAML files
        - Resolving workflow includes and imports
        - Caching parsed YAML definitions

        Returns:
            YAMLWorkflowLoader instance
        """
        from victor.workflows.yaml_loader import YAMLWorkflowLoader

        cache_enabled = getattr(self._settings, "enable_workflow_cache", True)
        cache_ttl = getattr(self._settings, "workflow_cache_ttl", 3600)

        return YAMLWorkflowLoader(
            enable_cache=cache_enabled,
            cache_ttl=cache_ttl,
        )

    def _create_workflow_validator(self) -> Any:
        """Create WorkflowValidator instance.

        The WorkflowValidator is responsible for:
        - Validating workflow structure (nodes, edges)
        - Checking node type compatibility
        - Validating node properties (tool_budget, timeout)
        - Checking for cycles and unreachable nodes

        Returns:
            WorkflowValidator instance
        """
        from victor.workflows.validator import WorkflowValidator

        strict_mode = getattr(self._settings, "strict_workflow_validation", False)

        return WorkflowValidator(strict_mode=strict_mode)

    def _create_node_executor_factory(self) -> Any:
        """Create NodeExecutorFactory instance.

        The NodeExecutorFactory is responsible for:
        - Mapping node types to executor functions
        - Creating executor functions for workflow nodes
        - Providing built-in executor registrations for first-party node types
        - Supporting registration of custom node types
        - Providing executor type checking

        This follows the Factory pattern for extensibility (OCP compliance).

        Returns:
            NodeExecutorFactory instance
        """
        from victor.workflows.executors.factory import NodeExecutorFactory

        factory = NodeExecutorFactory(container=self.container)
        logger.debug("Created node executor factory with built-in executor registrations")

        return factory

    def _create_compute_handler_registry(self) -> Any:
        """Create ComputeHandlerRegistry instance.

        The ComputeHandlerRegistry is responsible for:
        - Registering compute node handlers from verticals
        - Resolving handler names to handler instances
        - Providing handler metadata (input schema, output schema)
        - Validating handler registrations

        Returns:
            ComputeHandlerRegistry instance
        """
        from victor.workflows.handlers import ComputeHandlerRegistry

        registry = ComputeHandlerRegistry()

        # Auto-discover and register handlers from all verticals
        # This will be called after vertical initialization
        logger.debug("Created ComputeHandlerRegistry (handlers will be registered by verticals)")

        return registry

    # =========================================================================
    # Factory methods for scoped services
    # =========================================================================

    def _create_execution_context(self, container: ServiceContainer) -> Any:
        """Create ExecutionContext instance.

        The ExecutionContext is responsible for:
        - Providing orchestrator access for agent nodes
        - Providing execution settings
        - Providing service container access
        - Tracking execution metadata

        SEAM BOUNDARY: Workflow node-level execution context (not agent or tool-level)
        - Use WorkflowNodeContext for: Workflow node executors, per-node execution state
        - Do NOT use for: Agent orchestrator (use RuntimeExecutionContext), tools (use ToolExecutionContext)

        Returns:
            WorkflowNodeContext instance
        """
        from victor.workflows.execution_context import WorkflowNodeContext

        return WorkflowNodeContext(
            orchestrator=None,
            settings=self._settings,
            services=container,
        )

    def _create_orchestrator_pool(self, container: ServiceContainer) -> Any:
        """Create OrchestratorPool instance.

        The OrchestratorPool is responsible for:
        - Managing multiple orchestrators for different providers
        - Creating orchestrators on-demand for unique profiles
        - Reusing orchestrators across workflow executions
        - Providing orchestrator lifecycle management

        Returns:
            OrchestratorPool instance
        """
        from victor.workflows.orchestrator_pool import OrchestratorPool

        pool = OrchestratorPool(settings=self._settings, container=container)

        logger.debug("Created OrchestratorPool for multi-provider workflows")

        return pool

    # =========================================================================
    # Factory methods for transient services
    # =========================================================================

    def _create_workflow_compiler(self) -> Any:
        """Create WorkflowCompiler instance.

        The WorkflowCompiler is responsible for:
        - Loading YAML from file/string
        - Validating workflow definition
        - Building StateGraph from definition
        - Returning CompiledGraphProtocol

        This is a pure compiler - NO execution logic (SRP compliance).

        Returns:
            WorkflowCompilerImpl instance (using canonical backend)
        """
        from victor.workflows.compiler.workflow_compiler_impl import (
            WorkflowCompilerImpl,
        )
        from victor.workflows.compiler_protocols import NodeExecutorFactoryProtocol
        from victor.workflows.validator import WorkflowValidator
        from victor.workflows.yaml_loader import YAMLWorkflowLoader

        # Get dependencies from DI container (use actual types, not strings)
        factory = self.container.get(NodeExecutorFactoryProtocol)
        validator = self.container.get(WorkflowValidator)

        # Create YAML loader directly (not registered in container)
        cache_enabled = getattr(self._settings, "enable_workflow_cache", True)
        cache_ttl = getattr(self._settings, "workflow_cache_ttl", 3600)
        yaml_loader = YAMLWorkflowLoader(
            enable_cache=cache_enabled,
            cache_ttl=cache_ttl,
        )

        compiler = WorkflowCompilerImpl(
            yaml_loader=yaml_loader,
            validator=validator,
            node_factory=factory,
        )

        return compiler

    def _create_workflow_compiler_impl(self) -> Any:
        """Create WorkflowCompilerImpl instance.

        The WorkflowCompilerImpl is responsible for:
        - Loading YAML from file/string
        - Validating workflow definition
        - Building StateGraph from definition (via legacy compiler during migration)
        - Returning CompiledGraphProtocol

        This is a concrete wrapper that can be registered in the DI container.
        During migration, it delegates to the legacy implementation.

        Returns:
            WorkflowCompilerImpl instance
        """
        from victor.workflows.compiler.workflow_compiler_impl import (
            WorkflowCompilerImpl,
        )
        from victor.workflows.compiler_protocols import NodeExecutorFactoryProtocol
        from victor.workflows.validator import WorkflowValidator
        from victor.workflows.yaml_loader import YAMLWorkflowLoader

        # Get dependencies from DI container (use actual types, not strings)
        factory = self.container.get(NodeExecutorFactoryProtocol)
        validator = self.container.get(WorkflowValidator)

        # Create YAML loader directly (not registered in container)
        cache_enabled = getattr(self._settings, "enable_workflow_cache", True)
        cache_ttl = getattr(self._settings, "workflow_cache_ttl", 3600)
        yaml_loader = YAMLWorkflowLoader(
            enable_cache=cache_enabled,
            cache_ttl=cache_ttl,
        )

        compiler_impl = WorkflowCompilerImpl(
            yaml_loader=yaml_loader,
            validator=validator,
            node_factory=factory,
        )

        return compiler_impl

    def _create_workflow_executor(self) -> Any:
        """Create WorkflowExecutor instance.

        The WorkflowExecutor is responsible for:
        - Executing compiled graphs
        - Managing execution context
        - Streaming execution events
        - Handling checkpoints

        This is a pure executor - NO compilation logic (SRP compliance).

        Returns:
            WorkflowExecutor instance
        """
        from victor.workflows.compiled_executor import WorkflowExecutor
        from victor.workflows.orchestrator_pool import OrchestratorPool

        # Get dependencies from DI container (use actual type, not string)
        orchestrator_pool = self.container.get(OrchestratorPool)

        executor = WorkflowExecutor(
            orchestrator_pool=orchestrator_pool,
        )

        return executor


# =============================================================================
# Convenience function
# =============================================================================


def configure_workflow_services(
    container: ServiceContainer,
    settings: "Settings",
) -> None:
    """Configure all workflow services in one call.

    Convenience function for application bootstrap. Creates a service
    provider and registers all services.

    Args:
        container: DI container to register services in
        settings: Application settings

    Example:
        from victor.core.container import ServiceContainer
        from victor.workflows.services.workflow_service_provider import configure_workflow_services

        container = ServiceContainer()
        configure_workflow_services(container, settings)
    """
    provider = WorkflowServiceProvider(settings)
    provider.register_services(container)


__all__ = [
    "WorkflowServiceProvider",
    "configure_workflow_services",
]
