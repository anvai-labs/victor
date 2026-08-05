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

"""Tests for lazy tool composition utilities."""

from typing import Any, Dict
from unittest.mock import MagicMock

import pytest

from victor.tools.composition import LazyToolRunnable, ToolCompositionBuilder


class MockTool:
    """A mock tool for testing lazy initialization."""

    instances_created = 0

    def __init__(self, name: str = "mock_tool"):
        MockTool.instances_created += 1
        self.name = name
        self.run_count = 0
        self.arun_count = 0

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        self.run_count += 1
        return {"result": inputs.get("input", "default"), "run_count": self.run_count}

    async def arun(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        self.arun_count += 1
        return {"result": inputs.get("input", "default"), "arun_count": self.arun_count}

    @classmethod
    def reset_counter(cls):
        cls.instances_created = 0


@pytest.fixture(autouse=True)
def reset_mock_tool():
    """Reset the MockTool instance counter before each test."""
    MockTool.reset_counter()
    yield


class TestLazyToolRunnable:
    """Tests for LazyToolRunnable."""

    def test_lazy_initialization_not_created_until_access(self):
        """Test that tool is not created until first access."""
        lazy = LazyToolRunnable(lambda: MockTool())

        # Tool should not be created yet
        assert not lazy.is_initialized
        assert MockTool.instances_created == 0

        # Access the tool
        _ = lazy.tool

        # Now it should be created
        assert lazy.is_initialized
        assert MockTool.instances_created == 1

    def test_lazy_initialization_on_run(self):
        """Test that tool is created when run() is called."""
        lazy = LazyToolRunnable(lambda: MockTool())

        assert not lazy.is_initialized
        assert MockTool.instances_created == 0

        # Run should trigger initialization
        result = lazy.run({"input": "test"})

        assert lazy.is_initialized
        assert MockTool.instances_created == 1
        assert result["result"] == "test"

    @pytest.mark.asyncio
    async def test_lazy_initialization_on_arun(self):
        """Test that tool is created when arun() is called."""
        lazy = LazyToolRunnable(lambda: MockTool())

        assert not lazy.is_initialized
        assert MockTool.instances_created == 0

        # arun should trigger initialization
        result = await lazy.arun({"input": "async_test"})

        assert lazy.is_initialized
        assert MockTool.instances_created == 1
        assert result["result"] == "async_test"

    def test_caching_behavior_default(self):
        """Test that tool instance is cached by default."""
        lazy = LazyToolRunnable(lambda: MockTool())

        # Multiple accesses should return the same instance
        tool1 = lazy.tool
        tool2 = lazy.tool
        tool3 = lazy.get_tool_instance()

        assert tool1 is tool2
        assert tool2 is tool3
        assert MockTool.instances_created == 1

    def test_caching_behavior_disabled(self):
        """Test that caching can be disabled."""
        lazy = LazyToolRunnable(lambda: MockTool(), cache=False)

        # get_tool_instance should create fresh instances when cache=False
        tool1 = lazy.get_tool_instance()
        tool2 = lazy.get_tool_instance()

        assert tool1 is not tool2
        assert MockTool.instances_created == 2

    def test_reset_clears_cached_instance(self):
        """Test that reset() clears the cached instance."""
        lazy = LazyToolRunnable(lambda: MockTool())

        # Create instance
        tool1 = lazy.tool
        assert lazy.is_initialized
        assert MockTool.instances_created == 1

        # Reset
        lazy.reset()
        assert not lazy.is_initialized

        # Access again - should create new instance
        tool2 = lazy.tool
        assert lazy.is_initialized
        assert MockTool.instances_created == 2
        assert tool1 is not tool2

    def test_name_from_factory(self):
        """Test that name is derived from factory when not explicit."""

        def my_tool_factory():
            return MockTool()

        lazy = LazyToolRunnable(my_tool_factory)
        assert lazy.name == "my_tool_factory"

    def test_name_explicit(self):
        """Test that explicit name overrides factory name."""
        lazy = LazyToolRunnable(lambda: MockTool(), name="custom_name")
        assert lazy.name == "custom_name"

    def test_name_anonymous_lambda(self):
        """Test anonymous lambda gets 'anonymous' name from lambda."""
        lazy = LazyToolRunnable(lambda: MockTool())
        # Lambda functions have __name__ = "<lambda>"
        assert lazy.name == "<lambda>"

    def test_run_method(self):
        """Test the run() method executes tool correctly."""
        lazy = LazyToolRunnable(lambda: MockTool())

        result1 = lazy.run({"input": "first"})
        assert result1["result"] == "first"
        assert result1["run_count"] == 1

        result2 = lazy.run({"input": "second"})
        assert result2["result"] == "second"
        assert result2["run_count"] == 2

    @pytest.mark.asyncio
    async def test_arun_method(self):
        """Test the arun() method executes tool correctly."""
        lazy = LazyToolRunnable(lambda: MockTool())

        result1 = await lazy.arun({"input": "async_first"})
        assert result1["result"] == "async_first"
        assert result1["arun_count"] == 1

        result2 = await lazy.arun({"input": "async_second"})
        assert result2["result"] == "async_second"
        assert result2["arun_count"] == 2

    def test_repr(self):
        """Test string representation."""
        lazy = LazyToolRunnable(lambda: MockTool(), name="test_tool")

        # Before initialization
        repr_str = repr(lazy)
        assert "LazyToolRunnable" in repr_str
        assert "test_tool" in repr_str
        assert "pending" in repr_str

        # After initialization
        _ = lazy.tool
        repr_str = repr(lazy)
        assert "initialized" in repr_str

    def test_is_initialized_property(self):
        """Test the is_initialized property."""
        lazy = LazyToolRunnable(lambda: MockTool())

        assert lazy.is_initialized is False

        _ = lazy.tool

        assert lazy.is_initialized is True


class TestToolCompositionBuilder:
    """Tests for ToolCompositionBuilder."""

    def test_add_lazy_tool(self):
        """Test adding a lazy tool."""
        builder = ToolCompositionBuilder()
        builder.add("search", lambda: MockTool("search"))

        tools = builder.build()

        assert "search" in tools
        assert isinstance(tools["search"], LazyToolRunnable)
        assert MockTool.instances_created == 0  # Not created yet

    def test_add_eager_tool(self):
        """Test adding an eager tool."""
        builder = ToolCompositionBuilder()
        builder.add("search", lambda: MockTool("search"), lazy=False)

        tools = builder.build()

        assert "search" in tools
        assert isinstance(tools["search"], MockTool)
        assert MockTool.instances_created == 1  # Created immediately

    def test_add_eager_instance(self):
        """Test adding a pre-created instance."""
        tool = MockTool("prebuilt")
        builder = ToolCompositionBuilder()
        builder.add_eager("tool", tool)

        tools = builder.build()

        assert tools["tool"] is tool

    def test_add_lazy_with_cache_control(self):
        """Test add_lazy with explicit cache control."""
        builder = ToolCompositionBuilder()
        builder.add_lazy("nocache", lambda: MockTool(), cache=False)

        tools = builder.build()
        lazy_tool = tools["nocache"]

        # Create instances - should not cache
        t1 = lazy_tool.get_tool_instance()
        t2 = lazy_tool.get_tool_instance()

        assert t1 is not t2

    def test_method_chaining(self):
        """Test fluent method chaining."""
        tools = (
            ToolCompositionBuilder()
            .add("tool1", lambda: MockTool("tool1"))
            .add("tool2", lambda: MockTool("tool2"))
            .add("tool3", lambda: MockTool("tool3"))
            .build()
        )

        assert len(tools) == 3
        assert all(isinstance(t, LazyToolRunnable) for t in tools.values())

    def test_remove_tool(self):
        """Test removing a tool from composition."""
        builder = (
            ToolCompositionBuilder()
            .add("keep", lambda: MockTool())
            .add("remove", lambda: MockTool())
        )

        assert builder.has("remove")

        builder.remove("remove")

        assert not builder.has("remove")
        assert builder.has("keep")

    def test_remove_nonexistent_raises(self):
        """Test that removing non-existent tool raises KeyError."""
        builder = ToolCompositionBuilder()

        with pytest.raises(KeyError):
            builder.remove("nonexistent")

    def test_has_method(self):
        """Test the has() method."""
        builder = ToolCompositionBuilder().add("exists", lambda: MockTool())

        assert builder.has("exists")
        assert not builder.has("does_not_exist")

    def test_clear_method(self):
        """Test the clear() method."""
        builder = (
            ToolCompositionBuilder()
            .add("tool1", lambda: MockTool())
            .add("tool2", lambda: MockTool())
        )

        assert len(builder) == 2

        builder.clear()

        assert len(builder) == 0
        assert builder.build() == {}

    def test_len(self):
        """Test __len__ method."""
        builder = ToolCompositionBuilder()
        assert len(builder) == 0

        builder.add("tool1", lambda: MockTool())
        assert len(builder) == 1

        builder.add("tool2", lambda: MockTool())
        assert len(builder) == 2

    def test_repr(self):
        """Test string representation."""
        builder = (
            ToolCompositionBuilder()
            .add("search", lambda: MockTool())
            .add("analyze", lambda: MockTool())
        )

        repr_str = repr(builder)
        assert "ToolCompositionBuilder" in repr_str
        assert "search" in repr_str
        assert "analyze" in repr_str

    def test_build_returns_copy(self):
        """Test that build() returns a copy of the internal dict."""
        builder = ToolCompositionBuilder().add("tool", lambda: MockTool())

        tools1 = builder.build()
        tools2 = builder.build()

        # Should be equal but different objects
        assert tools1 == tools2
        assert tools1 is not tools2

    def test_mixed_lazy_and_eager(self):
        """Test composition with both lazy and eager tools."""
        builder = (
            ToolCompositionBuilder()
            .add("lazy1", lambda: MockTool("lazy1"), lazy=True)
            .add("eager1", lambda: MockTool("eager1"), lazy=False)
            .add_lazy("lazy2", lambda: MockTool("lazy2"))
            .add_eager("eager2", MockTool("eager2"))
        )

        # 2 eager tools should be created during build
        assert MockTool.instances_created == 2

        tools = builder.build()

        assert isinstance(tools["lazy1"], LazyToolRunnable)
        assert isinstance(tools["lazy2"], LazyToolRunnable)
        assert isinstance(tools["eager1"], MockTool)
        assert isinstance(tools["eager2"], MockTool)


class TestIntegration:
    """Integration tests for lazy composition."""

    def test_lazy_tools_in_composition_initialized_on_use(self):
        """Test that lazy tools in a composition are only initialized when used."""
        tools = (
            ToolCompositionBuilder()
            .add("search", lambda: MockTool("search"))
            .add("analyze", lambda: MockTool("analyze"))
            .add("format", lambda: MockTool("format"))
            .build()
        )

        # No tools created yet
        assert MockTool.instances_created == 0

        # Use only one tool
        tools["search"].run({"input": "query"})
        assert MockTool.instances_created == 1

        # Use another
        tools["analyze"].run({"input": "data"})
        assert MockTool.instances_created == 2

        # format never used, never created
        assert MockTool.instances_created == 2

    def test_factory_with_arguments(self):
        """Test factory functions that capture arguments."""

        def make_tool(config_value: str):
            def factory():
                tool = MockTool()
                tool.config = config_value
                return tool

            return factory

        lazy = LazyToolRunnable(make_tool("custom_config"), name="configured_tool")

        tool = lazy.tool
        assert tool.config == "custom_config"

    @pytest.mark.asyncio
    async def test_async_workflow(self):
        """Test a complete async workflow with lazy tools."""
        tools = (
            ToolCompositionBuilder()
            .add("step1", lambda: MockTool("step1"))
            .add("step2", lambda: MockTool("step2"))
            .build()
        )

        # Execute async workflow
        result1 = await tools["step1"].arun({"input": "start"})
        result2 = await tools["step2"].arun({"input": result1["result"]})

        assert result1["result"] == "start"
        assert result2["result"] == "start"
        assert MockTool.instances_created == 2

    def test_reset_and_reinitialize(self):
        """Test resetting and reinitializing lazy tools."""
        lazy = LazyToolRunnable(lambda: MockTool())

        # First use
        tool1 = lazy.tool
        tool1_id = id(tool1)
        assert MockTool.instances_created == 1

        # Reset and reinitialize
        lazy.reset()
        tool2 = lazy.tool
        tool2_id = id(tool2)
        assert MockTool.instances_created == 2

        # Should be different instances
        assert tool1_id != tool2_id
