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

"""Integration tests for tool composition framework.

Tests LazyToolRunnable and ToolCompositionBuilder end-to-end with
real tool-like objects and validates lazy loading behavior.
"""

import pytest
from dataclasses import dataclass, field
from typing import Any, Dict, List

from victor.tools.composition import LazyToolRunnable, ToolCompositionBuilder

# =============================================================================
# Mock Tools for Testing
# =============================================================================


@dataclass
class MockTool:
    """A simple mock tool for testing composition."""

    name: str
    initialized: bool = field(default=False, repr=False)
    call_count: int = field(default=0, repr=False)
    call_history: List[Dict[str, Any]] = field(default_factory=list, repr=False)

    def __post_init__(self):
        self.initialized = True

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the tool synchronously."""
        self.call_count += 1
        self.call_history.append(inputs.copy())
        return {
            "success": True,
            "tool": self.name,
            "result": f"Processed {inputs}",
        }

    async def arun(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the tool asynchronously."""
        self.call_count += 1
        self.call_history.append(inputs.copy())
        return {
            "success": True,
            "tool": self.name,
            "result": f"Async processed {inputs}",
        }


class ExpensiveTool:
    """Simulates an expensive-to-initialize tool."""

    # Class-level counter to track initializations
    init_count = 0

    def __init__(self):
        ExpensiveTool.init_count += 1
        self.creation_order = ExpensiveTool.init_count
        self.name = f"expensive_tool_{self.creation_order}"

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "success": True,
            "tool": self.name,
            "creation_order": self.creation_order,
            "inputs": inputs,
        }

    async def arun(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return await self.run(inputs)

    @classmethod
    def reset_counter(cls):
        cls.init_count = 0


class StatefulTool:
    """A tool that maintains state across calls."""

    def __init__(self, initial_value: int = 0):
        self.state = initial_value

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        operation = inputs.get("operation", "get")
        if operation == "increment":
            self.state += inputs.get("amount", 1)
        elif operation == "set":
            self.state = inputs.get("value", 0)
        return {"success": True, "state": self.state}


# =============================================================================
# Test Class: LazyToolRunnable
# =============================================================================


@pytest.mark.integration
class TestLazyToolRunnableIntegration:
    """Integration tests for LazyToolRunnable."""

    @pytest.fixture(autouse=True)
    def reset_expensive_tool_counter(self):
        """Reset the ExpensiveTool counter before each test."""
        ExpensiveTool.reset_counter()
        yield
        ExpensiveTool.reset_counter()

    def test_lazy_loading_defers_initialization(self):
        """Test that lazy loading actually defers tool initialization."""
        # Track whether factory was called
        factory_called = False

        def create_tool():
            nonlocal factory_called
            factory_called = True
            return MockTool(name="deferred")

        # Create lazy wrapper
        lazy = LazyToolRunnable(create_tool, name="test_lazy")

        # Factory should NOT be called yet
        assert not factory_called
        assert not lazy.is_initialized
        assert lazy.name == "test_lazy"

        # Access the tool
        _ = lazy.tool

        # Now factory should have been called
        assert factory_called
        assert lazy.is_initialized

    def test_lazy_tool_run_initializes_on_first_call(self):
        """Test that run() initializes the tool on first call."""
        ExpensiveTool.reset_counter()

        lazy = LazyToolRunnable(ExpensiveTool, name="expensive")

        # Not initialized yet
        assert not lazy.is_initialized
        assert ExpensiveTool.init_count == 0

        # First run should initialize
        result = lazy.run({"input": "test"})

        assert lazy.is_initialized
        assert ExpensiveTool.init_count == 1
        assert result["success"] is True
        assert result["creation_order"] == 1

    def test_lazy_tool_caches_instance_by_default(self):
        """Test that subsequent calls use cached instance."""
        ExpensiveTool.reset_counter()

        lazy = LazyToolRunnable(ExpensiveTool, name="cached")

        # First call
        result1 = lazy.run({"call": 1})
        assert ExpensiveTool.init_count == 1

        # Second call should use cached instance
        result2 = lazy.run({"call": 2})
        assert ExpensiveTool.init_count == 1  # Still 1

        # Both results should reference same creation order
        assert result1["creation_order"] == result2["creation_order"]

    def test_lazy_tool_without_cache_creates_fresh_instances(self):
        """Test that cache=False creates new instances each call."""
        ExpensiveTool.reset_counter()

        lazy = LazyToolRunnable(ExpensiveTool, name="no_cache", cache=False)

        # First call
        result1 = lazy.run({"call": 1})
        assert ExpensiveTool.init_count == 1
        assert result1["creation_order"] == 1

        # Second call should create new instance
        result2 = lazy.run({"call": 2})
        assert ExpensiveTool.init_count == 2
        assert result2["creation_order"] == 2

    def test_lazy_tool_reset_clears_cache(self):
        """Test that reset() clears the cached instance."""
        ExpensiveTool.reset_counter()

        lazy = LazyToolRunnable(ExpensiveTool, name="resettable")

        # Initialize
        lazy.run({"init": True})
        assert lazy.is_initialized
        assert ExpensiveTool.init_count == 1

        # Reset
        lazy.reset()
        assert not lazy.is_initialized

        # Next access creates new instance
        lazy.run({"after_reset": True})
        assert lazy.is_initialized
        assert ExpensiveTool.init_count == 2

    @pytest.mark.asyncio
    async def test_lazy_tool_async_run(self):
        """Test async execution with lazy loading."""
        lazy = LazyToolRunnable(
            lambda: MockTool(name="async_tool"),
            name="async_lazy",
        )

        assert not lazy.is_initialized

        result = await lazy.arun({"async": True})

        assert lazy.is_initialized
        assert result["success"] is True
        assert result["tool"] == "async_tool"

    def test_lazy_tool_name_from_factory(self):
        """Test that name is derived from factory if not provided."""

        def my_custom_factory():
            return MockTool(name="custom")

        lazy = LazyToolRunnable(my_custom_factory)
        assert lazy.name == "my_custom_factory"

    def test_lazy_tool_repr(self):
        """Test string representation of lazy tool."""
        lazy = LazyToolRunnable(
            lambda: MockTool(name="repr_tool"),
            name="repr_test",
        )

        repr_str = repr(lazy)
        assert "repr_test" in repr_str
        assert "pending" in repr_str

        # Initialize
        lazy.run({"input": "test"})

        repr_str = repr(lazy)
        assert "initialized" in repr_str


# =============================================================================
# Test Class: ToolCompositionBuilder
# =============================================================================


@pytest.mark.integration
class TestToolCompositionBuilderIntegration:
    """Integration tests for ToolCompositionBuilder."""

    @pytest.fixture(autouse=True)
    def reset_expensive_tool_counter(self):
        """Reset counters before each test."""
        ExpensiveTool.reset_counter()
        yield

    def test_build_empty_composition(self):
        """Test building an empty tool composition."""
        builder = ToolCompositionBuilder()
        tools = builder.build()

        assert tools == {}
        assert len(builder) == 0

    def test_add_lazy_tool(self):
        """Test adding a lazy-loaded tool."""
        ExpensiveTool.reset_counter()

        builder = ToolCompositionBuilder()
        builder.add("expensive", ExpensiveTool, lazy=True)

        tools = builder.build()

        # Tool should be wrapped in LazyToolRunnable
        assert "expensive" in tools
        assert isinstance(tools["expensive"], LazyToolRunnable)

        # Should not be initialized yet
        assert not tools["expensive"].is_initialized
        assert ExpensiveTool.init_count == 0

    def test_add_eager_tool(self):
        """Test adding an eagerly-initialized tool."""
        ExpensiveTool.reset_counter()

        builder = ToolCompositionBuilder()
        builder.add("eager", ExpensiveTool, lazy=False)

        # Tool should be initialized immediately when building
        tools = builder.build()

        assert "eager" in tools
        assert isinstance(tools["eager"], ExpensiveTool)
        assert ExpensiveTool.init_count == 1

    def test_add_eager_with_instance(self):
        """Test adding a pre-created instance eagerly."""
        existing_tool = MockTool(name="existing")

        builder = ToolCompositionBuilder()
        builder.add_eager("tool", existing_tool)

        tools = builder.build()

        assert tools["tool"] is existing_tool

    def test_builder_chaining(self):
        """Test fluent builder chaining."""
        builder = (
            ToolCompositionBuilder()
            .add("tool1", lambda: MockTool(name="t1"), lazy=True)
            .add("tool2", lambda: MockTool(name="t2"), lazy=True)
            .add("tool3", lambda: MockTool(name="t3"), lazy=True)
        )

        assert len(builder) == 3

        tools = builder.build()
        assert set(tools.keys()) == {"tool1", "tool2", "tool3"}

    def test_composition_with_mixed_lazy_and_eager(self):
        """Test composition with both lazy and eager tools."""
        ExpensiveTool.reset_counter()

        builder = (
            ToolCompositionBuilder()
            .add("lazy1", ExpensiveTool, lazy=True)
            .add("eager1", ExpensiveTool, lazy=False)
            .add("lazy2", ExpensiveTool, lazy=True)
        )

        tools = builder.build()

        # Only eager tool should be initialized
        assert ExpensiveTool.init_count == 1

        # Check types
        assert isinstance(tools["lazy1"], LazyToolRunnable)
        assert isinstance(tools["eager1"], ExpensiveTool)
        assert isinstance(tools["lazy2"], LazyToolRunnable)

    def test_remove_tool(self):
        """Test removing a tool from composition."""
        builder = (
            ToolCompositionBuilder()
            .add("keep", lambda: MockTool(name="keep"))
            .add("remove", lambda: MockTool(name="remove"))
        )

        assert builder.has("remove")

        builder.remove("remove")

        assert not builder.has("remove")
        assert builder.has("keep")

        tools = builder.build()
        assert "remove" not in tools
        assert "keep" in tools

    def test_clear_builder(self):
        """Test clearing all tools from builder."""
        builder = (
            ToolCompositionBuilder()
            .add("tool1", lambda: MockTool(name="t1"))
            .add("tool2", lambda: MockTool(name="t2"))
        )

        assert len(builder) == 2

        builder.clear()

        assert len(builder) == 0
        assert builder.build() == {}

    def test_add_lazy_with_cache_control(self):
        """Test add_lazy with explicit cache control."""
        builder = ToolCompositionBuilder()
        builder.add_lazy("no_cache", ExpensiveTool, cache=False)
        builder.add_lazy("cached", ExpensiveTool, cache=True)

        tools = builder.build()

        # Check that cache setting is preserved
        assert isinstance(tools["no_cache"], LazyToolRunnable)
        assert isinstance(tools["cached"], LazyToolRunnable)

    def test_composition_end_to_end(self):
        """End-to-end test of tool composition with execution."""
        ExpensiveTool.reset_counter()

        # Build composition
        builder = (
            ToolCompositionBuilder()
            .add("search", lambda: MockTool(name="search"))
            .add("analyze", lambda: MockTool(name="analyze"))
            .add("format", lambda: MockTool(name="format"))
        )

        tools = builder.build()

        # No tools initialized yet
        for name, tool in tools.items():
            assert isinstance(tool, LazyToolRunnable)
            assert not tool.is_initialized

        # Execute first tool
        result1 = tools["search"].run({"query": "test"})
        assert result1["success"] is True
        assert tools["search"].is_initialized
        assert not tools["analyze"].is_initialized
        assert not tools["format"].is_initialized

        # Execute second tool
        result2 = tools["analyze"].run({"data": result1})
        assert result2["success"] is True
        assert tools["analyze"].is_initialized
        assert not tools["format"].is_initialized

        # Execute third tool
        result3 = tools["format"].run({"analysis": result2})
        assert result3["success"] is True
        assert tools["format"].is_initialized


# =============================================================================
# Test Class: Composition with Real Tool Patterns
# =============================================================================


@pytest.mark.integration
class TestCompositionWithRealPatterns:
    """Tests that validate composition patterns with realistic tool behaviors."""

    def test_stateful_tool_in_composition(self):
        """Test that stateful tools work correctly in composition."""
        builder = ToolCompositionBuilder()
        builder.add("counter", lambda: StatefulTool(initial_value=10))

        tools = builder.build()

        # Get initial state
        result = tools["counter"].run({"operation": "get"})
        assert result["state"] == 10

        # Increment
        result = tools["counter"].run({"operation": "increment", "amount": 5})
        assert result["state"] == 15

        # State is preserved (cached instance)
        result = tools["counter"].run({"operation": "get"})
        assert result["state"] == 15

    def test_multiple_lazy_tools_init_order(self):
        """Test that lazy tools initialize in correct order when accessed."""
        ExpensiveTool.reset_counter()

        tools = (
            ToolCompositionBuilder()
            .add("first", ExpensiveTool)
            .add("second", ExpensiveTool)
            .add("third", ExpensiveTool)
            .build()
        )

        # Access in reverse order
        r3 = tools["third"].run({})
        r2 = tools["second"].run({})
        r1 = tools["first"].run({})

        # Creation order should match access order
        assert r3["creation_order"] == 1
        assert r2["creation_order"] == 2
        assert r1["creation_order"] == 3

    def test_composition_repr(self):
        """Test the string representation of ToolCompositionBuilder."""
        builder = ToolCompositionBuilder().add("a", MockTool).add("b", MockTool)

        repr_str = repr(builder)
        assert "ToolCompositionBuilder" in repr_str
        assert "a" in repr_str or "b" in repr_str

    @pytest.mark.asyncio
    async def test_async_composition_workflow(self):
        """Test async execution through composed tools."""
        tools = (
            ToolCompositionBuilder()
            .add("step1", lambda: MockTool(name="step1"))
            .add("step2", lambda: MockTool(name="step2"))
            .build()
        )

        # Execute async pipeline
        result1 = await tools["step1"].arun({"input": "start"})
        result2 = await tools["step2"].arun({"input": result1["result"]})

        assert result1["success"] is True
        assert result2["success"] is True
        assert "step1" in result1["tool"]
        assert "step2" in result2["tool"]
