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

"""Tests for SharedToolRegistry singleton pattern.

This module tests the SharedToolRegistry class which provides a memory-efficient
way to share tool instances across multiple concurrent orchestrator sessions.
"""

import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

from victor.tools.base import BaseTool, ToolResult


class TestSharedToolRegistrySingleton:
    """Tests for SharedToolRegistry singleton behavior."""

    def test_singleton_instance(self):
        """Test that SharedToolRegistry is a singleton."""
        from victor.agent.shared_tool_registry import SharedToolRegistry

        instance1 = SharedToolRegistry.get_instance()
        instance2 = SharedToolRegistry.get_instance()

        assert instance1 is instance2

    def test_singleton_persists_across_calls(self):
        """Test that singleton instance persists across multiple calls."""
        from victor.agent.shared_tool_registry import SharedToolRegistry

        instance1 = SharedToolRegistry.get_instance()
        # Simulate some operations
        instance1.get_tool_classes()
        instance2 = SharedToolRegistry.get_instance()

        assert instance1 is instance2

    def test_reset_instance_creates_new_singleton(self):
        """Test that reset_instance() allows creating a fresh singleton."""
        from victor.agent.shared_tool_registry import SharedToolRegistry

        instance1 = SharedToolRegistry.get_instance()
        original_id = id(instance1)

        SharedToolRegistry.reset_instance()

        instance2 = SharedToolRegistry.get_instance()
        new_id = id(instance2)

        assert original_id != new_id
        assert instance1 is not instance2

    def test_direct_instantiation_prevented(self):
        """Test that direct instantiation is prevented."""
        from victor.agent.shared_tool_registry import SharedToolRegistry

        # First call to get_instance sets up the singleton
        SharedToolRegistry.get_instance()

        # Direct instantiation should raise an error
        with pytest.raises(RuntimeError, match="Use SharedToolRegistry.get_instance()"):
            SharedToolRegistry()


class TestSharedToolRegistryThreadSafety:
    """Tests for thread-safety of SharedToolRegistry."""

    def test_concurrent_get_instance_returns_same_object(self):
        """Test that concurrent calls to get_instance() return the same object."""
        from victor.agent.shared_tool_registry import SharedToolRegistry

        SharedToolRegistry.reset_instance()

        results = []
        errors = []

        def get_instance():
            try:
                instance = SharedToolRegistry.get_instance()
                results.append(id(instance))
            except Exception as e:
                errors.append(e)

        # Create multiple threads that call get_instance concurrently
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(get_instance) for _ in range(100)]
            for future in futures:
                future.result()

        assert len(errors) == 0, f"Errors during concurrent access: {errors}"
        assert len(set(results)) == 1, "Multiple instances were created"

    def test_tool_instances_are_shared_across_threads(self):
        """Test that tool instances are shared across threads."""
        from victor.agent.shared_tool_registry import SharedToolRegistry

        SharedToolRegistry.reset_instance()
        registry = SharedToolRegistry.get_instance()

        tool_ids = []
        lock = threading.Lock()

        def get_tool_id():
            tools = registry.get_tool_classes()
            # Get the first tool class if available
            if tools:
                first_tool_class = list(tools.values())[0]
                with lock:
                    tool_ids.append(id(first_tool_class))

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(get_tool_id) for _ in range(20)]
            for future in futures:
                future.result()

        # All threads should see the same tool class instances
        if tool_ids:
            assert len(set(tool_ids)) == 1


class TestSharedToolRegistryToolManagement:
    """Tests for tool class management in SharedToolRegistry."""

    def test_get_tool_classes_returns_dict(self):
        """Test that get_tool_classes returns a dictionary."""
        from victor.agent.shared_tool_registry import SharedToolRegistry

        SharedToolRegistry.reset_instance()
        registry = SharedToolRegistry.get_instance()

        tool_classes = registry.get_tool_classes()

        assert isinstance(tool_classes, dict)

    def test_tool_classes_are_discovered(self):
        """Test that tool classes are discovered from victor/tools directory."""
        from victor.agent.shared_tool_registry import SharedToolRegistry

        SharedToolRegistry.reset_instance()
        registry = SharedToolRegistry.get_instance()

        tool_classes = registry.get_tool_classes()

        # Should discover some tools
        assert len(tool_classes) > 0

    def test_tool_classes_are_cached(self):
        """Test that tool classes are cached after first discovery."""
        from victor.agent.shared_tool_registry import SharedToolRegistry

        SharedToolRegistry.reset_instance()
        registry = SharedToolRegistry.get_instance()

        # First call should discover tools
        classes1 = registry.get_tool_classes()

        # Second call should return cached version
        classes2 = registry.get_tool_classes()

        # Should be the same dictionary instance (not a new discovery)
        assert classes1 is classes2

    def test_create_tool_instance_returns_new_instances(self):
        """Test that create_tool_instance returns new instances each time."""
        from victor.agent.shared_tool_registry import SharedToolRegistry

        SharedToolRegistry.reset_instance()
        registry = SharedToolRegistry.get_instance()

        tool_classes = registry.get_tool_classes()
        if not tool_classes:
            pytest.skip("No tools available for testing")

        # Find a tool that can be instantiated (class-based tools)
        # Skip decorated function-based tools which may not be re-instantiable
        tool_name = None
        for name, cls in tool_classes.items():
            try:
                test_instance = cls()
                if test_instance is not None:
                    tool_name = name
                    break
            except Exception:
                continue

        if tool_name is None:
            pytest.skip("No instantiable tools available for testing")

        # Create two instances
        instance1 = registry.create_tool_instance(tool_name)
        instance2 = registry.create_tool_instance(tool_name)

        # Should be different object instances
        assert instance1 is not None
        assert instance2 is not None
        assert instance1 is not instance2
        # But same type
        assert type(instance1) is type(instance2)

    def test_create_tool_instance_returns_none_for_unknown(self):
        """Test that create_tool_instance returns None for unknown tools."""
        from victor.agent.shared_tool_registry import SharedToolRegistry

        SharedToolRegistry.reset_instance()
        registry = SharedToolRegistry.get_instance()

        result = registry.create_tool_instance("nonexistent_tool_xyz")

        assert result is None


class TestSharedToolRegistryWithToolRegistrar:
    """Tests for integration between SharedToolRegistry and ToolRegistrar."""

    def test_multiple_orchestrators_share_tool_classes(self):
        """Test that multiple orchestrators share tool class definitions."""
        from victor.agent.shared_tool_registry import SharedToolRegistry

        SharedToolRegistry.reset_instance()
        registry = SharedToolRegistry.get_instance()

        # Simulate multiple orchestrators getting tool classes
        classes_for_orchestrator1 = registry.get_tool_classes()
        classes_for_orchestrator2 = registry.get_tool_classes()
        classes_for_orchestrator3 = registry.get_tool_classes()

        # Should all point to the same dictionary
        assert classes_for_orchestrator1 is classes_for_orchestrator2
        assert classes_for_orchestrator2 is classes_for_orchestrator3

    def test_tool_discovery_happens_once(self):
        """Test that tool discovery only happens once."""
        from victor.agent.shared_tool_registry import SharedToolRegistry

        SharedToolRegistry.reset_instance()

        with patch(
            "victor.agent.shared_tool_registry.SharedToolRegistry._discover_tools"
        ) as mock_discover:
            mock_discover.return_value = {}

            registry = SharedToolRegistry.get_instance()

            # Call get_tool_classes multiple times
            registry.get_tool_classes()
            registry.get_tool_classes()
            registry.get_tool_classes()

            # Discovery should only be called once
            assert mock_discover.call_count == 1

    def test_graph_tool_is_discovered_for_registration(self):
        """Graph tools should be present in shared discovery for orchestrators."""
        from victor.agent.shared_tool_registry import SharedToolRegistry

        SharedToolRegistry.reset_instance()
        registry = SharedToolRegistry.get_instance()

        tool_names = registry.get_tool_names()

        # The graph tool is split into specialized sub-tools; at least one should be registered
        graph_tools = [name for name in tool_names if name.startswith("graph")]
        assert len(graph_tools) > 0, f"No graph tools found; registered tools: {tool_names}"


class TestSharedToolRegistryAirgappedMode:
    """Tests for airgapped mode filtering in SharedToolRegistry."""

    def test_get_tool_classes_filters_web_tools_in_airgapped_mode(self):
        """Test that web tools are filtered out in airgapped mode."""
        from victor.agent.shared_tool_registry import SharedToolRegistry

        SharedToolRegistry.reset_instance()
        registry = SharedToolRegistry.get_instance()

        # Get all tools
        all_tools = registry.get_tool_classes()

        # Get tools filtered for airgapped mode
        airgapped_tools = registry.get_tool_classes(airgapped_mode=True)

        # Web tools should be filtered out
        web_tool_names = {"web_search", "web_fetch", "http_tool"}
        for tool_name in web_tool_names:
            if tool_name in all_tools:
                assert tool_name not in airgapped_tools


class TestSharedToolRegistryResetFixture:
    """Tests to verify reset_instance works correctly for test isolation."""

    def test_reset_clears_cached_tools(self):
        """Test that reset_instance clears cached tool classes."""
        from victor.agent.shared_tool_registry import SharedToolRegistry

        # Get initial instance and cache tools
        SharedToolRegistry.reset_instance()
        registry1 = SharedToolRegistry.get_instance()
        classes1 = registry1.get_tool_classes()

        # Reset and get new instance
        SharedToolRegistry.reset_instance()
        registry2 = SharedToolRegistry.get_instance()
        classes2 = registry2.get_tool_classes()

        # Should be different dictionary instances (fresh discovery)
        assert classes1 is not classes2

    def test_reset_allows_test_isolation(self):
        """Test that reset allows proper test isolation."""
        from victor.agent.shared_tool_registry import SharedToolRegistry

        # First test modifies state
        SharedToolRegistry.reset_instance()
        registry1 = SharedToolRegistry.get_instance()
        original_id = id(registry1)

        # Simulate test cleanup
        SharedToolRegistry.reset_instance()

        # Second test gets fresh state
        registry2 = SharedToolRegistry.get_instance()
        new_id = id(registry2)

        assert original_id != new_id


class MockBaseTool(BaseTool):
    """Mock tool for testing."""

    @property
    def name(self) -> str:
        return "mock_tool"

    @property
    def description(self) -> str:
        return "A mock tool for testing"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {"type": "object", "properties": {}}

    async def execute(self, _exec_ctx: Dict[str, Any], **kwargs: Any) -> ToolResult:
        return ToolResult(success=True, output="mock result")


class CountingInstantiationTool(BaseTool):
    """Tool that counts how many times its class is instantiated.

    Used to verify that discovery and LazyToolProxy loading do not cause
    double-instantiation of the same tool.
    """

    instantiation_count: int = 0

    def __init__(self) -> None:
        super().__init__()
        CountingInstantiationTool.instantiation_count += 1

    @property
    def name(self) -> str:
        return "counting_instantiation_tool"

    @property
    def description(self) -> str:
        return "Counts instantiations for double-init regression tests"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {"type": "object", "properties": {}}

    async def execute(self, _exec_ctx: Dict[str, Any], **kwargs: Any) -> ToolResult:
        return ToolResult(success=True, output="counting result")


class TestSharedToolRegistryNoDoubleInstantiation:
    """Regression tests: discovery must not cause duplicate tool instantiation."""

    def test_proxy_reuses_discovery_instance_not_reinstantiated(self):
        """LazyToolProxy must reuse the instance created during discovery.

        Regression: previously _discover_tools() created an instance via obj()
        to read .name, then LazyToolProxy._ensure_loaded() called cls() again,
        instantiating every used tool twice.
        """
        from victor.agent.shared_tool_registry import SharedToolRegistry

        CountingInstantiationTool.instantiation_count = 0
        SharedToolRegistry.reset_instance()
        registry = SharedToolRegistry.get_instance()

        # Simulate discovery having instantiated the tool to harvest its name
        cached_instance = CountingInstantiationTool()
        assert CountingInstantiationTool.instantiation_count == 1
        registry._tool_classes["counting_instantiation_tool"] = CountingInstantiationTool
        registry._tool_instances["counting_instantiation_tool"] = cached_instance
        registry._initialized = True

        # Build the registration list (creates LazyToolProxy objects)
        tools = registry.get_all_tools_for_registration()
        proxy = next(t for t in tools if getattr(t, "name", None) == "counting_instantiation_tool")

        # Simulate first use — this must NOT instantiate again
        loaded = proxy._ensure_loaded()

        assert loaded is cached_instance, "Proxy should reuse the cached discovery instance"
        assert (
            CountingInstantiationTool.instantiation_count == 1
        ), "Tool was instantiated more than once (double-instantiation regression)"


class TestSharedToolRegistryGetToolNames:
    """Tests for get_tool_names method."""

    def test_get_tool_names_returns_list(self):
        """Test that get_tool_names returns a list of tool names."""
        from victor.agent.shared_tool_registry import SharedToolRegistry

        SharedToolRegistry.reset_instance()
        registry = SharedToolRegistry.get_instance()

        names = registry.get_tool_names()

        assert isinstance(names, list)
        assert all(isinstance(name, str) for name in names)

    def test_get_tool_names_matches_tool_classes_keys(self):
        """Test that tool names match the keys in tool_classes."""
        from victor.agent.shared_tool_registry import SharedToolRegistry

        SharedToolRegistry.reset_instance()
        registry = SharedToolRegistry.get_instance()

        names = registry.get_tool_names()
        classes = registry.get_tool_classes()

        assert set(names) == set(classes.keys())


class TestGraphDemandSpec:
    """Regression: the graph demand spec must point at a real, loadable module.

    It previously pointed at ``victor.tools.graph_tool`` (which doesn't exist),
    so demand-loading graph silently failed — graph was only ever reachable via
    the victor-coding entry-point, making its registration fragile.
    """

    def test_graph_spec_not_phantom_module(self):
        """Guard: the graph spec must not point at the non-existent core path.

        Runs in every CI environment (no victor-coding needed) — this is the
        cheap regression guard against re-introducing the phantom
        ``victor.tools.graph_tool`` spec.
        """
        from victor.agent.shared_tool_registry import DEMAND_TOOL_SPECS

        module_name, _member = DEMAND_TOOL_SPECS["graph"]
        assert (
            module_name != "victor.tools.graph_tool"
        ), "graph spec points at a phantom module (victor.tools.graph_tool doesn't exist)"
        # graph lives in the optional victor-coding package.
        assert module_name.startswith("victor_coding"), module_name

    def test_graph_spec_module_exists(self):
        pytest.importorskip("victor_coding")  # spec resolves only where graph lives

        import importlib

        from victor.agent.shared_tool_registry import DEMAND_TOOL_SPECS

        module_name, _member = DEMAND_TOOL_SPECS["graph"]
        # The spec's module must actually resolve (no phantom paths).
        importlib.import_module(module_name)

    def test_graph_demand_loadable(self):
        pytest.importorskip("victor_coding")  # graph lives in the optional vertical

        from victor.agent.shared_tool_registry import SharedToolRegistry

        SharedToolRegistry.reset_instance()
        registry = SharedToolRegistry.get_instance()

        tool = registry._load_tool_spec("graph")
        assert tool is not None
        assert getattr(tool, "name", None) == "graph"

        loaded = registry.get_tools_for_names(["graph"])
        assert loaded and getattr(loaded[0], "name", None) == "graph"
