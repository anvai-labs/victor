"""Mock test fixtures for external vertical development.

Provides MockPluginContext — an in-memory implementation of PluginContext
that allows testing VictorPlugin.register() without installing victor-ai.

Usage::

    from victor_contracts.testing import MockPluginContext

    def test_my_plugin_registers_tools():
        ctx = MockPluginContext()
        my_plugin.register(ctx)
        assert "my_tool" in ctx.registered_tools

Also ships the workflow-handler test mocks (``MockToolResult``,
``MockComputeNode``, ``MockWorkflowContext``) that vertical test suites
need to exercise compute handlers without victor-ai installed.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Type


class MockPluginContext:
    """In-memory PluginContext for testing verticals without victor-ai.

    Satisfies the ``PluginContext`` protocol from ``victor_contracts.core.plugins``.
    All registration calls store items in public attributes for assertion.
    """

    def __init__(self) -> None:
        self.registered_tools: List[Any] = []
        self.registered_verticals: List[Type[Any]] = []
        self.registered_commands: Dict[str, Any] = {}
        self.registered_chunkers: List[Any] = []
        self.registered_workflow_node_executors: Dict[str, Any] = {}
        self.registered_categories: Dict[str, set[str]] = {}
        self.extended_categories: Dict[str, set[str]] = {}
        self.registered_safety_rules: List[Any] = []
        self.registered_tool_dependencies: Dict[str, Any] = {}
        self.registered_ccg_builders: Dict[str, Any] = {}
        self.registered_escape_hatches: List[Any] = []
        self.pending_rl_configs: Dict[str, Any] = {}
        self.pending_bootstrap_services: List[tuple[str, Any]] = []
        self.pending_mcp_servers: List[Any] = []
        self._services: Dict[Type[Any], Any] = {}
        self._settings: Dict[str, Any] = {
            "default_provider": "mock",
            "default_model": "mock-model",
            "debug": False,
        }

    def register_tool(self, tool_instance: Any) -> None:
        """Register a tool instance."""
        self.registered_tools.append(tool_instance)

    def register_vertical(self, vertical_class: Type[Any]) -> None:
        """Register a vertical class."""
        self.registered_verticals.append(vertical_class)

    def register_chunker(self, chunker_instance: Any) -> None:
        """Register a chunker."""
        self.registered_chunkers.append(chunker_instance)

    def register_command(self, name: str, app: Any) -> None:
        """Register a CLI sub-command."""
        self.registered_commands[name] = app

    def register_workflow_node_executor(
        self,
        node_type: str,
        executor_factory: Any,
        *,
        replace: bool = False,
    ) -> None:
        """Register a custom workflow node executor."""
        self.registered_workflow_node_executors[node_type] = executor_factory

    def register_category(
        self,
        name: str,
        tools: set[str],
        *,
        description: Optional[str] = None,
    ) -> None:
        """Register a custom tool category."""
        self.registered_categories[name] = set(tools)

    def extend_category(self, name: str, tools: set[str]) -> None:
        """Extend an existing tool category."""
        if name not in self.extended_categories:
            self.extended_categories[name] = set()
        self.extended_categories[name].update(tools)

    def get_service(self, service_type: Type[Any]) -> Optional[Any]:
        """Retrieve a registered service, or None."""
        return self._services.get(service_type)

    def get_provider_registry(self) -> Optional[Any]:
        """Return the registered provider-registry service, if any."""
        from victor_contracts.verticals.protocols import ProviderRegistryProtocol

        return self.get_service(ProviderRegistryProtocol)

    def get_graph_store(self) -> Optional[Any]:
        """Return the registered graph-store service, if any."""
        from victor_contracts.verticals.protocols import GraphStoreProtocol

        return self.get_service(GraphStoreProtocol)

    def get_vector_store(self) -> Optional[Any]:
        """Return the registered vector-store service, if any."""
        from victor_contracts.verticals.protocols import VectorStoreProtocol

        return self.get_service(VectorStoreProtocol)

    def get_embedding_service(self) -> Optional[Any]:
        """Return the registered embedding service, if any."""
        from victor_contracts.verticals.protocols import EmbeddingServiceProtocol

        return self.get_service(EmbeddingServiceProtocol)

    def get_memory_coordinator(self) -> Optional[Any]:
        """Return the registered memory-coordinator service, if any."""
        from victor_contracts.verticals.protocols import MemoryCoordinatorProtocol

        return self.get_service(MemoryCoordinatorProtocol)

    def get_settings(self) -> Any:
        """Return mock settings dict."""
        return self._settings

    def register_capability(
        self,
        protocol_type: Type[Any],
        provider: Any,
        *,
        lazy: bool = False,
    ) -> None:
        """Register a capability provider (stores in _capabilities for testing)."""
        if not hasattr(self, "_capabilities"):
            self._capabilities: Dict[Type[Any], Any] = {}
        self._capabilities[protocol_type] = provider

    def register_safety_rule(self, rule: Any) -> None:
        """Register a safety rule for testing."""
        self.registered_safety_rules.append(rule)

    def register_tool_dependency(self, name: str, provider: Any) -> None:
        """Register a tool dependency provider for testing."""
        self.registered_tool_dependencies[name] = provider

    def register_ccg_builder(self, language: str, builder: Any) -> None:
        """Register a Code Context Graph builder for testing."""
        self.registered_ccg_builders[language] = builder

    def register_escape_hatch(self, hatch: Any) -> None:
        """Register an escape hatch for testing."""
        self.registered_escape_hatches.append(hatch)

    def register_rl_config(self, key: str, config: Any) -> None:
        """Buffer RL config fragments for testing."""
        self.pending_rl_configs[key] = config

    def register_bootstrap_service(
        self,
        factory: Any,
        *,
        phase: str = "vertical_services",
    ) -> None:
        """Buffer bootstrap services for testing."""
        self.pending_bootstrap_services.append((phase, factory))

    def register_mcp_server(self, spec: Any) -> None:
        """Buffer MCP server specs for testing."""
        self.pending_mcp_servers.append(spec)

    # --- Test helpers (not part of PluginContext protocol) ---

    def set_service(self, service_type: Type[Any], instance: Any) -> None:
        """Pre-register a service for testing."""
        self._services[service_type] = instance


class MockToolResult:
    """Mock tool result for handler tests (duck-typed ToolResult)."""

    def __init__(self, success: bool = True, output: Any = None, error: Optional[str] = None):
        self.success = success
        self.output = output
        self.error = error


class MockComputeNode:
    """Mock compute node for handler tests."""

    def __init__(
        self,
        node_id: str = "test_node",
        input_mapping: Optional[Dict[str, Any]] = None,
        output_key: Optional[str] = None,
    ):
        self.id = node_id
        self.input_mapping = input_mapping or {}
        self.output_key = output_key


class MockWorkflowContext:
    """Mock workflow context for handler tests."""

    def __init__(self, data: Optional[Dict[str, Any]] = None):
        self._data = data or {}

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self._data[key] = value
