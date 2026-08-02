# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for extended PluginContext methods (S2 of plugin-vertical consolidation)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest


@pytest.fixture(autouse=True)
def reset_tool_category_registry():
    from victor.framework.tools import ToolCategoryRegistry

    ToolCategoryRegistry.reset_instance()
    yield
    ToolCategoryRegistry.reset_instance()


def test_register_rl_config_buffers_payload():
    from victor.core.plugins.context import HostPluginContext

    ctx = HostPluginContext(container=None)
    ctx.register_rl_config("coding", {"alpha": 0.1})
    assert ctx.pending_rl_configs == {"coding": {"alpha": 0.1}}


def test_register_bootstrap_service_buffers_factory():
    from victor.core.plugins.context import HostPluginContext

    def factory(container, settings, context):  # pragma: no cover - inert hook
        return None

    ctx = HostPluginContext(container=None)
    ctx.register_bootstrap_service(factory, phase="vertical_services")
    pending = ctx.pending_bootstrap_services
    assert len(pending) == 1
    assert pending[0] == ("vertical_services", factory)


def test_register_mcp_server_buffers_spec():
    from victor.core.plugins.context import HostPluginContext

    spec = SimpleNamespace(name="example", url="http://localhost:9000")
    ctx = HostPluginContext(container=None)
    ctx.register_mcp_server(spec)
    assert ctx.pending_mcp_servers == [spec]


def test_register_tool_dependency_seeds_cache():
    from victor.core.plugins.context import HostPluginContext
    from victor.core.tool_dependency_loader import (
        _vertical_provider_cache,
        _vertical_provider_cache_lock,
        clear_vertical_tool_dependency_provider_cache,
    )

    clear_vertical_tool_dependency_provider_cache()

    class StubProvider:
        def get_dependencies(self):
            return {}

        def get_recommended_sequence(self, _task):
            return []

    provider = StubProvider()
    ctx = HostPluginContext(container=None)
    ctx.register_tool_dependency("stubvertical", provider)

    with _vertical_provider_cache_lock:
        canon = _vertical_provider_cache.get(("stubvertical", True))
        noncanon = _vertical_provider_cache.get(("stubvertical", False))

    assert canon is provider
    assert noncanon is provider
    clear_vertical_tool_dependency_provider_cache()


def test_register_safety_rule_no_container_is_noop():
    """When SafetyEnforcer isn't registered, the method logs + returns without error."""
    from victor.core.plugins.context import HostPluginContext

    class StubRule:
        name = "stub_rule"

    ctx = HostPluginContext(container=None)
    # Should not raise even though no SafetyEnforcer is registered.
    ctx.register_safety_rule(StubRule())


def test_register_escape_hatch_condition(monkeypatch):
    from victor.core.plugins.context import HostPluginContext

    recorded = {}

    class FakeRegistry:
        def register_condition(self, name, fn, **opts):
            recorded["condition"] = (name, fn, opts)

        def register_transform(self, name, fn, **opts):
            recorded["transform"] = (name, fn, opts)

    monkeypatch.setattr(
        "victor.framework.escape_hatch_registry.get_escape_hatch_registry",
        lambda: FakeRegistry(),
    )

    ctx = HostPluginContext(container=None)
    ctx.register_escape_hatch(
        SimpleNamespace(kind="condition", name="allow_everything", fn=lambda ctx: True)
    )

    assert "condition" in recorded
    assert recorded["condition"][0] == "allow_everything"


def test_pluginctx_protocol_has_new_methods():
    """The SDK Protocol includes the new register_* methods."""
    from victor_contracts.core.plugins import PluginContext

    for name in (
        "register_category",
        "extend_category",
        "get_provider_registry",
        "get_graph_store",
        "get_vector_store",
        "get_embedding_service",
        "get_memory_coordinator",
        "register_safety_rule",
        "register_tool_dependency",
        "register_escape_hatch",
        "register_rl_config",
        "register_bootstrap_service",
        "register_mcp_server",
    ):
        assert hasattr(PluginContext, name), f"SDK PluginContext missing {name}"


def test_register_category_routes_to_tool_category_registry():
    from victor.core.plugins.context import HostPluginContext
    from victor.framework.tools import get_category_registry

    ctx = HostPluginContext(container=None)
    ctx.register_category("rag", {"rag_search", "rag_query"})

    assert get_category_registry().get_tools("rag") == {"rag_search", "rag_query"}


def test_extend_category_routes_to_tool_category_registry():
    from victor.core.plugins.context import HostPluginContext
    from victor.framework.tools import get_category_registry

    ctx = HostPluginContext(container=None)
    ctx.extend_category("search", {"semantic_rag_search"})

    assert "semantic_rag_search" in get_category_registry().get_tools("search")


def test_typed_service_accessors_bridge_internal_and_sdk_service_keys(monkeypatch):
    from victor.agent.protocols import (
        ProviderRegistryProtocol as HostProviderRegistryProtocol,
    )
    from victor.core.plugins.context import HostPluginContext
    from victor.core.protocols import (
        EmbeddingServiceProtocol as HostEmbeddingServiceProtocol,
    )
    from victor.storage.graph.protocol import (
        GraphStoreProtocol as HostGraphStoreProtocol,
    )
    from victor_contracts.verticals.protocols import (
        VectorStoreProtocol as SdkVectorStoreProtocol,
    )

    class FakeProviderRegistry:
        def get(self, name: str) -> str:
            return f"provider:{name}"

        def list_providers(self) -> list[str]:
            return ["openai"]

    class FakeEmbeddingService:
        async def embed_text(self, text: str) -> list[float]:
            return [float(len(text))]

        async def embed_batch(self, texts: list[str]) -> list[list[float]]:
            return [[float(len(text))] for text in texts]

        def embed_text_sync(self, text: str) -> list[float]:
            return [float(len(text))]

    class FakeGraphStore:
        async def upsert_nodes(self, nodes):
            return None

        async def upsert_edges(self, edges):
            return None

        async def get_neighbors(self, node_id: str, edge_types=None, **kwargs):
            return []

        async def find_nodes(self, **kwargs):
            return []

    class FakeVectorStore:
        async def index_document(self, doc_id: str, content: str, embedding, metadata=None):
            return None

        async def search_similar(self, query_embedding, limit: int = 10):
            return []

        async def delete_document(self, doc_id: str):
            return None

    class FakeContainer:
        def __init__(self, services):
            self._services = services

        def get_optional(self, service_type):
            return self._services.get(service_type)

        def get(self, service_type):
            service = self.get_optional(service_type)
            if service is None:
                raise LookupError(service_type)
            return service

        def is_registered(self, service_type):
            return service_type in self._services

    provider_registry = FakeProviderRegistry()
    embedding_service = FakeEmbeddingService()
    graph_store = FakeGraphStore()
    vector_store = FakeVectorStore()

    ctx = HostPluginContext(
        container=FakeContainer(
            {
                HostProviderRegistryProtocol: provider_registry,
                HostEmbeddingServiceProtocol: embedding_service,
                HostGraphStoreProtocol: graph_store,
                SdkVectorStoreProtocol: vector_store,
            }
        )
    )

    provider_registry_service = ctx.get_provider_registry()
    assert provider_registry_service is not None
    assert provider_registry_service.get_provider("openai") == "provider:openai"
    assert provider_registry_service.list_providers() == ["openai"]
    assert ctx.get_embedding_service() is embedding_service
    assert ctx.get_graph_store() is graph_store
    assert ctx.get_vector_store() is vector_store


def test_get_memory_coordinator_falls_back_to_global_accessor(monkeypatch):
    from victor.core.plugins.context import HostPluginContext

    class FakeMemoryCoordinator:
        async def search_all(self, query: str, limit: int = 20, **kwargs):
            return []

        async def search_type(self, memory_type, query: str, limit: int = 20, **kwargs):
            return []

        async def store(self, memory_type, key: str, value, metadata=None) -> bool:
            return True

        async def get(self, memory_type, key: str):
            return None

        def register_provider(self, provider) -> None:
            return None

        def unregister_provider(self, memory_type) -> bool:
            return True

        def get_registered_types(self):
            return []

        def get_stats(self):
            return {}

    sentinel = FakeMemoryCoordinator()
    monkeypatch.setattr("victor.storage.memory.get_memory_coordinator", lambda: sentinel)

    ctx = HostPluginContext(container=SimpleNamespace(get_optional=lambda *_args: None))

    assert ctx.get_memory_coordinator() is sentinel
