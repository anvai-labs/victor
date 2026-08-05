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

"""Plugin protocols for Victor Contracts.

This module defines the interfaces that external plugins and verticals
implement to register themselves with the Victor framework.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional, Protocol, Type, runtime_checkable

if TYPE_CHECKING:
    import typer
    from victor_contracts.verticals.protocols import (
        EmbeddingServiceProtocol,
        GraphStoreProtocol,
        MemoryCoordinatorProtocol,
        ProviderRegistryProtocol,
        VectorStoreProtocol,
    )


@runtime_checkable
class PluginContext(Protocol):
    """Context provided to plugins during registration.

    This interface allows plugins to register tools, verticals, chunkers,
    and CLI commands without depending on framework internal registries.
    """

    def register_tool(self, tool_instance: Any) -> None:
        """Register a tool instance with the framework.

        Args:
            tool_instance: The tool to register (e.g., a function with @tool).
        """
        ...

    # CONSOLIDATION: plugin-vertical unification — see memory plugin_vertical_consolidation.md
    # This is the bridge between the plugin seam and the vertical role. No
    # ``register_plugin`` method exists today — ``register_vertical`` IS the
    # canonical way to expose a plugin's extension surface. S4 will add a
    # ``register_plugin`` alias (behaviour identical) for naming consistency.
    def register_vertical(self, vertical_class: Type[Any]) -> None:
        """Register a vertical class with the framework.

        Args:
            vertical_class: The class implementing VerticalBase.
        """
        ...

    def register_chunker(self, chunker_instance: Any) -> None:
        """Register a specialized chunker for document processing.

        .. deprecated:: 0.7.0
            Use ``register_capability(ChunkerProtocol, chunker_instance)`` instead.
            Will be removed in v1.0.

        Args:
            chunker_instance: The chunker implementation.
        """
        ...

    def register_command(self, name: str, app: "typer.Typer") -> None:
        """Register a Typer CLI application as a sub-command.

        Args:
            name: The sub-command name.
            app: The Typer application to mount.
        """
        ...

    def register_workflow_node_executor(
        self,
        node_type: str,
        executor_factory: Any,
        *,
        replace: bool = False,
    ) -> None:
        """Register a custom workflow node executor type with the host.

        .. deprecated:: 0.7.0
            Use the ``victor.workflow_executors`` entry point group instead.
            Will be removed in v1.0.

        Args:
            node_type: Custom workflow node type identifier.
            executor_factory: Executor class or factory callable for the node type.
            replace: Whether to replace an existing registration.
        """
        ...

    def register_category(
        self,
        name: str,
        tools: set[str],
        *,
        description: Optional[str] = None,
    ) -> None:
        """Register a custom tool category with the host.

        .. deprecated:: 0.7.0
            Tool categories have no active callers. Will be removed in v1.0.

        Args:
            name: Category name.
            tools: Tool names exposed through the category.
            description: Optional human-readable description.
        """
        ...

    def extend_category(self, name: str, tools: set[str]) -> None:
        """Extend an existing tool category with additional tools.

        .. deprecated:: 0.7.0
            Tool categories have no active callers. Will be removed in v1.0.

        Args:
            name: Category name to extend.
            tools: Tool names to merge into the category.
        """
        ...

    def get_service(self, service_type: Type[Any]) -> Optional[Any]:
        """Retrieve a service from the host container.

        Args:
            service_type: The type or interface of the service.

        Returns:
            The service instance or None if not found.
        """
        ...

    def get_provider_registry(self) -> Optional["ProviderRegistryProtocol"]:
        """Retrieve the host's LLM provider registry, if available."""
        ...

    def get_graph_store(self) -> Optional["GraphStoreProtocol"]:
        """Retrieve the host's graph-store service, if available."""
        ...

    def get_vector_store(self) -> Optional["VectorStoreProtocol"]:
        """Retrieve the host's vector-store service, if available."""
        ...

    def get_embedding_service(self) -> Optional["EmbeddingServiceProtocol"]:
        """Retrieve the host's embedding service, if available."""
        ...

    def get_memory_coordinator(self) -> Optional["MemoryCoordinatorProtocol"]:
        """Retrieve the host's shared memory coordinator, if available."""
        ...

    def get_settings(self) -> Any:
        """Retrieve the application settings.

        Returns the framework Settings object. Plugins should use this
        instead of importing victor.config.settings directly.

        Returns:
            The Settings instance.
        """
        ...

    def register_capability(
        self,
        protocol_type: Type[Any],
        provider: Any,
        *,
        lazy: bool = False,
    ) -> None:
        """Register a capability provider with the framework.

        This is the canonical way for plugins to register capabilities
        (e.g., TreeSitterParser, Editor, CodebaseIndex) so they are
        available to framework tools via CapabilityRegistry.

        Args:
            protocol_type: The protocol type to register against.
            provider: The provider instance, or if lazy=True, a zero-arg
                      callable that returns the provider on first access.
            lazy: If True, provider is a factory called on first access.
        """
        ...

    # CONSOLIDATION: plugin-vertical unification — see memory plugin_vertical_consolidation.md
    # These methods let a plugin self-register everything via register(context)
    # without needing sidecar entry-point groups (victor.safety_rules,
    # victor.tool_dependencies, victor.escape_hatches, victor.rl_configs,
    # victor.bootstrap_services, victor.mcp_servers). Hosts running older
    # frameworks may not implement some of these; plugins should probe with
    # hasattr() before calling. Added in SDK minor bump for PluginContext.
    def register_safety_rule(self, rule: Any) -> None:
        """Register a single safety rule with the framework's SafetyEnforcer.

        Args:
            rule: A SafetyRule instance (see ``victor_contracts.safety``).
        """
        ...

    def register_tool_dependency(self, name: str, provider: Any) -> None:
        """Register a tool dependency provider for a named tool group.

        Args:
            name: Tool or tool-group identifier (e.g., ``"devops"``).
            provider: Provider instance implementing the
                ``BaseToolDependencyProvider`` contract.
        """
        ...

    def register_escape_hatch(self, hatch: Any) -> None:
        """Register an escape-hatch (condition or transform) with the framework.

        Args:
            hatch: An escape-hatch descriptor. The host inspects ``kind``
                (``"condition"`` or ``"transform"``) and ``name`` / ``fn``
                attributes to route to the appropriate registry method.
        """
        ...

    def register_rl_config(self, key: str, config: Any) -> None:
        """Register an RL config fragment the framework merges at bootstrap.

        Args:
            key: Config namespace (typically the vertical name).
            config: Dict-like config payload.
        """
        ...

    def register_bootstrap_service(
        self,
        factory: Any,
        *,
        phase: str = "vertical_services",
    ) -> None:
        """Register a bootstrap-service hook that runs during the given phase.

        Args:
            factory: Callable with signature ``(container, settings, context)``.
            phase: Bootstrap phase name (default ``"vertical_services"``).
        """
        ...

    def register_mcp_server(self, spec: Any) -> None:
        """Register an MCP server spec for discovery by the MCP vertical.

        Args:
            spec: MCP server specification object or dict.
        """
        ...

    def register_ccg_builder(self, language: str, builder: Any) -> None:
        """Register a language-specific CCG builder with the capability registry.

        This allows external packages (victor-coding) to provide enhanced
        CCG construction for specific languages.

        Args:
            language: Language identifier (e.g., "python", "javascript").
            builder: CCG builder instance implementing CCGBuilderProtocol.

        Example:
            context.register_ccg_builder("python", PythonCCGBuilder(graph_store))
        """
        ...


@runtime_checkable
class VictorPlugin(Protocol):
    """Interface for Victor plugins.

    Plugins are discovered via the 'victor.plugins' entry point.
    """

    @property
    def name(self) -> str:
        """Return the stable identifier for this plugin (e.g., 'coding')."""
        ...

    def register(self, context: PluginContext) -> None:
        """Register the plugin's components with the host framework.

        Args:
            context: The registration context provided by the host.
        """
        ...

    def get_cli_app(self) -> Optional["typer.Typer"]:
        """Return a Typer application to be registered as a sub-command.

        .. deprecated::
            Use context.register_command() inside register() instead.
        """
        ...

    # CONSOLIDATION: plugin-vertical unification — see memory plugin_vertical_consolidation.md
    # "this plugin's vertical" == the VerticalBase subclass registered via
    # context.register_vertical. Plugin and vertical describe the same
    # extension instance from two angles (runtime seam vs. configuration role).
    def on_activate(self) -> None:
        """Called when this plugin's vertical is activated.

        Optional lifecycle hook. Implement to perform setup when the
        vertical associated with this plugin becomes active.
        """
        ...

    def on_deactivate(self) -> None:
        """Called when this plugin's vertical is being deactivated.

        Optional lifecycle hook. Implement to perform cleanup when
        switching away from this plugin's vertical.
        """
        ...

    def health_check(self) -> Dict[str, Any]:
        """Return health status for this plugin.

        Optional lifecycle hook. Returns a dictionary with at minimum
        a 'healthy' boolean key.

        Returns:
            Dict with 'healthy' key and optional detail keys.
        """
        ...

    async def on_activate_async(self) -> None:
        """Async variant of on_activate.

        When implemented, the framework calls this instead of on_activate
        in async contexts. Useful for plugins that need I/O during setup.
        """
        ...

    async def on_deactivate_async(self) -> None:
        """Async variant of on_deactivate.

        When implemented, the framework calls this instead of on_deactivate
        in async contexts. Useful for plugins that need I/O during teardown.
        """
        ...
