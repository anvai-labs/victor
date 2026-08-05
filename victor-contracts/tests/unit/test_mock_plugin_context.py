# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for MockPluginContext — written BEFORE implementation (TDD RED phase)."""

from __future__ import annotations


from victor_contracts.core.plugins import PluginContext
from victor_contracts.testing.fixtures import MockPluginContext


class TestMockPluginContextProtocol:
    """MockPluginContext must satisfy the PluginContext protocol."""

    def test_satisfies_plugin_context_protocol(self):
        ctx = MockPluginContext()
        assert isinstance(ctx, PluginContext)

    def test_all_plugin_context_methods_present(self):
        """Every method on PluginContext is implemented on MockPluginContext."""
        import inspect

        protocol_methods = {
            name
            for name, _ in inspect.getmembers(PluginContext, predicate=inspect.isfunction)
            if not name.startswith("_")
        }
        mock_methods = {
            name
            for name, _ in inspect.getmembers(MockPluginContext, predicate=callable)
            if not name.startswith("_")
        }
        missing = protocol_methods - mock_methods
        assert not missing, f"MockPluginContext missing methods: {missing}"


class TestMockPluginContextRegistration:
    """Test that registration methods store items correctly."""

    def test_register_tool_stores_tools(self):
        ctx = MockPluginContext()
        sentinel = object()
        ctx.register_tool(sentinel)
        assert sentinel in ctx.registered_tools

    def test_register_vertical_stores_verticals(self):
        ctx = MockPluginContext()

        class FakeVertical:
            pass

        ctx.register_vertical(FakeVertical)
        assert FakeVertical in ctx.registered_verticals

    def test_register_command_stores_commands(self):
        ctx = MockPluginContext()
        app = object()
        ctx.register_command("test-cmd", app)
        assert "test-cmd" in ctx.registered_commands
        assert ctx.registered_commands["test-cmd"] is app

    def test_register_chunker_stores_chunkers(self):
        ctx = MockPluginContext()
        chunker = object()
        ctx.register_chunker(chunker)
        assert chunker in ctx.registered_chunkers

    def test_register_workflow_node_executor_stores_executors(self):
        ctx = MockPluginContext()
        factory = object()
        ctx.register_workflow_node_executor("custom", factory)
        assert "custom" in ctx.registered_workflow_node_executors
        assert ctx.registered_workflow_node_executors["custom"] is factory

    def test_register_category_stores_categories(self):
        ctx = MockPluginContext()
        ctx.register_category("rag", {"rag_search", "rag_query"})
        assert ctx.registered_categories["rag"] == {"rag_search", "rag_query"}

    def test_extend_category_accumulates_extensions(self):
        ctx = MockPluginContext()
        ctx.extend_category("search", {"semantic_search"})
        ctx.extend_category("search", {"hybrid_search"})
        assert ctx.extended_categories["search"] == {"semantic_search", "hybrid_search"}

    def test_multiple_tools_accumulate(self):
        ctx = MockPluginContext()
        ctx.register_tool("tool_a")
        ctx.register_tool("tool_b")
        assert len(ctx.registered_tools) == 2


class TestMockPluginContextServices:
    """Test service registration and retrieval."""

    def test_get_service_returns_none_by_default(self):
        ctx = MockPluginContext()
        assert ctx.get_service(str) is None

    def test_get_service_returns_registered_service(self):
        ctx = MockPluginContext()
        instance = {"key": "value"}
        ctx.set_service(dict, instance)
        assert ctx.get_service(dict) is instance

    def test_get_settings_returns_dict(self):
        ctx = MockPluginContext()
        settings = ctx.get_settings()
        assert isinstance(settings, dict)

    def test_typed_service_helpers_use_registered_services(self):
        from victor_contracts.verticals.protocols import (
            EmbeddingServiceProtocol,
            GraphStoreProtocol,
            MemoryCoordinatorProtocol,
            ProviderRegistryProtocol,
            VectorStoreProtocol,
        )

        ctx = MockPluginContext()
        provider_registry = object()
        graph_store = object()
        vector_store = object()
        embedding_service = object()
        memory_coordinator = object()

        ctx.set_service(ProviderRegistryProtocol, provider_registry)
        ctx.set_service(GraphStoreProtocol, graph_store)
        ctx.set_service(VectorStoreProtocol, vector_store)
        ctx.set_service(EmbeddingServiceProtocol, embedding_service)
        ctx.set_service(MemoryCoordinatorProtocol, memory_coordinator)

        assert ctx.get_provider_registry() is provider_registry
        assert ctx.get_graph_store() is graph_store
        assert ctx.get_vector_store() is vector_store
        assert ctx.get_embedding_service() is embedding_service
        assert ctx.get_memory_coordinator() is memory_coordinator


class TestMockPluginContextRoundtrip:
    """End-to-end: a VictorPlugin registers via MockPluginContext."""

    def test_plugin_register_roundtrip(self):

        class MyPlugin:
            @property
            def name(self) -> str:
                return "test-plugin"

            def register(self, context: PluginContext) -> None:
                context.register_tool("my_tool")
                context.register_command("my-cmd", "app")

        ctx = MockPluginContext()
        plugin = MyPlugin()
        plugin.register(ctx)

        assert "my_tool" in ctx.registered_tools
        assert "my-cmd" in ctx.registered_commands
