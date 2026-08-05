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

"""Tests for victor.tools.selection module."""

import pytest
from typing import List, Optional, Dict, Any

from victor.tools.selection import (
    BaseToolSelectionStrategy,
    PerformanceProfile,
    ToolSelectionContext,
    ToolSelectionStrategy,
    ToolSelectionStrategyRegistry,
    ToolSelectorFeatures,
    get_best_strategy,
    get_strategy,
    get_strategy_registry,
    list_strategies,
    register_strategy,
)


class TestPerformanceProfile:
    """Tests for PerformanceProfile dataclass."""

    def test_create_profile(self):
        """Test creating a performance profile."""
        profile = PerformanceProfile(
            avg_latency_ms=10.0,
            requires_embeddings=True,
            requires_model_inference=False,
            memory_usage_mb=50.0,
        )
        assert profile.avg_latency_ms == 10.0
        assert profile.requires_embeddings is True
        assert profile.requires_model_inference is False
        assert profile.memory_usage_mb == 50.0

    def test_keyword_profile(self):
        """Test typical keyword strategy profile."""
        profile = PerformanceProfile(
            avg_latency_ms=1.0,
            requires_embeddings=False,
            requires_model_inference=False,
            memory_usage_mb=5.0,
        )
        assert profile.avg_latency_ms < 5.0
        assert profile.requires_embeddings is False

    def test_semantic_profile(self):
        """Test typical semantic strategy profile."""
        profile = PerformanceProfile(
            avg_latency_ms=30.0,
            requires_embeddings=True,
            requires_model_inference=False,
            memory_usage_mb=100.0,
        )
        assert profile.requires_embeddings is True


class TestToolSelectionContext:
    """Tests for ToolSelectionContext dataclass."""

    def test_minimal_context(self):
        """Test context with only required fields."""
        context = ToolSelectionContext(prompt="Find Python files")
        assert context.prompt == "Find Python files"
        assert context.conversation_history == []
        assert context.current_stage is None
        assert context.max_tools == 10

    def test_full_context(self):
        """Test context with all fields."""
        context = ToolSelectionContext(
            prompt="Refactor the auth module",
            conversation_history=[{"role": "user", "content": "help"}],
            current_stage="PLANNING",
            task_type="refactoring",
            provider_name="anthropic",
            model_name="claude-3-opus",
            cost_budget=0.5,
            enabled_tools=["read", "edit"],
            disabled_tools=["shell"],
            vertical="coding",
            recent_tools=["read", "grep"],
            turn_number=3,
            max_tools=15,
            use_cost_aware=True,
        )
        assert context.current_stage == "PLANNING"
        assert context.task_type == "refactoring"
        assert context.vertical == "coding"
        assert context.max_tools == 15
        assert "read" in context.recent_tools

    def test_from_agent_context(self):
        """Test creating context from agent dict."""
        agent_ctx = {
            "conversation_history": [{"role": "user", "content": "hi"}],
            "stage": "EXECUTING",
            "task_type": "implementation",
            "provider_name": "openai",
            "model_name": "gpt-4",
            "vertical": "devops",
            "recent_tools": ["shell"],
            "turn_number": 5,
        }
        context = ToolSelectionContext.from_agent_context(
            prompt="Deploy to production",
            agent_context=agent_ctx,
        )
        assert context.prompt == "Deploy to production"
        assert context.current_stage == "EXECUTING"
        assert context.task_type == "implementation"
        assert context.vertical == "devops"

    def test_from_empty_agent_context(self):
        """Test creating context from empty agent dict."""
        context = ToolSelectionContext.from_agent_context(
            prompt="Hello",
            agent_context={},
        )
        assert context.prompt == "Hello"
        assert context.conversation_history == []
        assert context.max_tools == 10


class TestToolSelectorFeatures:
    """Tests for ToolSelectorFeatures dataclass."""

    def test_default_features(self):
        """Test default feature values."""
        features = ToolSelectorFeatures()
        assert features.supports_semantic_matching is False
        assert features.supports_context_awareness is False
        assert features.supports_cost_optimization is False
        assert features.supports_usage_learning is False
        assert features.supports_workflow_patterns is False
        assert features.requires_embeddings is False

    def test_semantic_features(self):
        """Test semantic strategy features."""
        features = ToolSelectorFeatures(
            supports_semantic_matching=True,
            supports_context_awareness=True,
            supports_cost_optimization=True,
            supports_usage_learning=True,
            supports_workflow_patterns=True,
            requires_embeddings=True,
        )
        assert features.supports_semantic_matching is True
        assert features.requires_embeddings is True


class MockToolSelector(BaseToolSelectionStrategy):
    """Mock selector for testing."""

    def __init__(self, name: str = "mock", tools: Optional[List[str]] = None):
        super().__init__()
        self._name = name
        self._tools = tools or ["read", "write"]
        self._recorded_executions: List[Dict[str, Any]] = []

    def get_strategy_name(self) -> str:
        return self._name

    def get_performance_profile(self) -> PerformanceProfile:
        return PerformanceProfile(
            avg_latency_ms=5.0,
            requires_embeddings=False,
            requires_model_inference=False,
            memory_usage_mb=10.0,
        )

    async def select_tools(
        self,
        context: ToolSelectionContext,
        max_tools: int = 10,
    ) -> List[str]:
        return self._tools[:max_tools]

    def record_tool_execution(
        self,
        tool_name: str,
        success: bool,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._recorded_executions.append(
            {
                "tool_name": tool_name,
                "success": success,
                "context": context,
            }
        )


class TestBaseToolSelectionStrategy:
    """Tests for BaseToolSelectionStrategy ABC."""

    def test_mock_selector(self):
        """Test mock selector implementation."""
        selector = MockToolSelector()
        assert selector.get_strategy_name() == "mock"

    def test_performance_profile(self):
        """Test getting performance profile."""
        selector = MockToolSelector()
        profile = selector.get_performance_profile()
        assert profile.avg_latency_ms == 5.0
        assert profile.requires_embeddings is False

    @pytest.mark.asyncio
    async def test_select_tools(self):
        """Test tool selection."""
        selector = MockToolSelector(tools=["read", "write", "grep"])
        context = ToolSelectionContext(prompt="test")
        tools = await selector.select_tools(context, max_tools=2)
        assert len(tools) == 2
        assert tools == ["read", "write"]

    def test_default_supports_context(self):
        """Test default supports_context returns True."""
        selector = MockToolSelector()
        context = ToolSelectionContext(prompt="test")
        assert selector.supports_context(context) is True

    def test_default_features(self):
        """Test default get_supported_features."""
        selector = MockToolSelector()
        features = selector.get_supported_features()
        assert features.supports_semantic_matching is False

    def test_record_execution(self):
        """Test recording tool execution."""
        selector = MockToolSelector()
        selector.record_tool_execution("read", True, {"duration": 100})
        assert len(selector._recorded_executions) == 1
        assert selector._recorded_executions[0]["tool_name"] == "read"
        assert selector._recorded_executions[0]["success"] is True

    @pytest.mark.asyncio
    async def test_close(self):
        """Test close method."""
        selector = MockToolSelector()
        assert selector._closed is False
        await selector.close()
        assert selector._closed is True


class TestToolSelectionStrategyRegistry:
    """Tests for ToolSelectionStrategyRegistry."""

    @pytest.fixture
    def registry(self):
        """Create fresh registry for each test."""
        ToolSelectionStrategyRegistry.reset_instance()
        return ToolSelectionStrategyRegistry.get_instance()

    def test_singleton(self):
        """Test registry is singleton."""
        ToolSelectionStrategyRegistry.reset_instance()
        r1 = ToolSelectionStrategyRegistry.get_instance()
        r2 = ToolSelectionStrategyRegistry.get_instance()
        assert r1 is r2

    def test_register_strategy(self, registry):
        """Test registering a strategy."""
        selector = MockToolSelector(name="test")
        registry.register("test", selector)
        assert "test" in registry.list_strategies()

    def test_register_duplicate_fails(self, registry):
        """Test registering duplicate without replace fails."""
        selector = MockToolSelector(name="test")
        registry.register("test", selector)
        with pytest.raises(ValueError, match="already registered"):
            registry.register("test", selector)

    def test_register_duplicate_with_replace(self, registry):
        """Test registering duplicate with replace succeeds."""
        selector1 = MockToolSelector(name="test1")
        selector2 = MockToolSelector(name="test2")
        registry.register("test", selector1)
        registry.register("test", selector2, replace=True)
        assert registry.get("test").get_strategy_name() == "test2"

    def test_get_strategy(self, registry):
        """Test getting a strategy."""
        selector = MockToolSelector(name="test")
        registry.register("test", selector)
        result = registry.get("test")
        assert result is selector

    def test_get_nonexistent(self, registry):
        """Test getting nonexistent strategy returns None."""
        result = registry.get("nonexistent")
        assert result is None

    def test_unregister(self, registry):
        """Test unregistering a strategy."""
        selector = MockToolSelector(name="test")
        registry.register("test", selector)
        assert registry.unregister("test") is True
        assert registry.get("test") is None

    def test_unregister_nonexistent(self, registry):
        """Test unregistering nonexistent returns False."""
        assert registry.unregister("nonexistent") is False

    def test_list_strategies(self, registry):
        """Test listing all strategies."""
        registry.register("a", MockToolSelector(name="a"))
        registry.register("b", MockToolSelector(name="b"))
        names = registry.list_strategies()
        assert "a" in names
        assert "b" in names
        assert names == sorted(names)  # Should be sorted

    def test_register_class(self, registry):
        """Test registering a strategy class."""
        registry.register_class("mock_class", MockToolSelector)
        # Should instantiate on first get
        strategy = registry.get("mock_class")
        assert strategy is not None
        assert strategy.get_strategy_name() == "mock"

    def test_get_strategy_info(self, registry):
        """Test getting strategy info."""
        selector = MockToolSelector(name="test")
        registry.register("test", selector)
        info = registry.get_strategy_info("test")
        assert info is not None
        assert info["name"] == "test"
        assert "performance" in info
        assert "features" in info

    def test_get_best_strategy_default(self, registry):
        """Test get_best_strategy default behavior."""
        hybrid = MockToolSelector(name="hybrid")
        keyword = MockToolSelector(name="keyword")
        registry.register("hybrid", hybrid)
        registry.register("keyword", keyword)

        context = ToolSelectionContext(prompt="test")
        best = registry.get_best_strategy(context)
        assert best is hybrid  # Hybrid preferred by default

    def test_get_best_strategy_prefer_fast(self, registry):
        """Test get_best_strategy with prefer_fast."""
        hybrid = MockToolSelector(name="hybrid")
        keyword = MockToolSelector(name="keyword")
        registry.register("hybrid", hybrid)
        registry.register("keyword", keyword)

        context = ToolSelectionContext(prompt="test")
        best = registry.get_best_strategy(context, prefer_fast=True)
        assert best is keyword

    @pytest.mark.asyncio
    async def test_close_all(self, registry):
        """Test closing all strategies."""
        s1 = MockToolSelector(name="s1")
        s2 = MockToolSelector(name="s2")
        registry.register("s1", s1)
        registry.register("s2", s2)

        await registry.close_all()
        assert s1._closed is True
        assert s2._closed is True


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    @pytest.fixture(autouse=True)
    def reset_registry(self):
        """Reset registry before each test."""
        ToolSelectionStrategyRegistry.reset_instance()
        yield
        ToolSelectionStrategyRegistry.reset_instance()

    def test_get_strategy_registry(self):
        """Test get_strategy_registry returns singleton."""
        r = get_strategy_registry()
        assert r is ToolSelectionStrategyRegistry.get_instance()

    def test_register_strategy(self):
        """Test register_strategy convenience function."""
        selector = MockToolSelector(name="test")
        register_strategy("test", selector)
        assert get_strategy("test") is selector

    def test_get_strategy(self):
        """Test get_strategy convenience function."""
        selector = MockToolSelector(name="test")
        register_strategy("test", selector)
        assert get_strategy("test") is selector

    def test_get_best_strategy(self):
        """Test get_best_strategy convenience function."""
        selector = MockToolSelector(name="hybrid")
        register_strategy("hybrid", selector)
        context = ToolSelectionContext(prompt="test")
        assert get_best_strategy(context) is selector

    def test_list_strategies(self):
        """Test list_strategies convenience function."""
        register_strategy("a", MockToolSelector(name="a"))
        register_strategy("b", MockToolSelector(name="b"))
        names = list_strategies()
        assert "a" in names
        assert "b" in names


class TestProtocolCompliance:
    """Tests for protocol compliance."""

    def test_mock_selector_is_protocol_compliant(self):
        """Test MockToolSelector implements ToolSelectionStrategy protocol."""
        selector = MockToolSelector()
        # Check if it's recognized as implementing the protocol
        assert isinstance(selector, ToolSelectionStrategy)

    def test_base_class_protocol_compliance(self):
        """Test BaseToolSelectionStrategy subclasses implement protocol."""

        class CustomSelector(BaseToolSelectionStrategy):
            def get_strategy_name(self) -> str:
                return "custom"

            def get_performance_profile(self) -> PerformanceProfile:
                return PerformanceProfile(
                    avg_latency_ms=1.0,
                    requires_embeddings=False,
                    requires_model_inference=False,
                    memory_usage_mb=5.0,
                )

            async def select_tools(
                self,
                context: ToolSelectionContext,
                max_tools: int = 10,
            ) -> List[str]:
                return []

        selector = CustomSelector()
        assert isinstance(selector, ToolSelectionStrategy)
