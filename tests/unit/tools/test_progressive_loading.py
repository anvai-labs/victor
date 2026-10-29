"""Tests for progressive tool loading (FEP-0003).

Covers:
- LazyToolProxy defers initialization until first use
- Cost-tier preference in tool selection
- Tool registry supports lazy registration
- Progressive parameter escalation
"""

from __future__ import annotations

from unittest.mock import MagicMock, AsyncMock


from victor.tools.enums import CostTier


class TestLazyToolProxy:
    """LazyToolProxy defers tool instantiation."""

    def test_proxy_does_not_instantiate_on_creation(self):
        from victor.tools.progressive import LazyToolProxy

        factory = MagicMock()
        proxy = LazyToolProxy("test_tool", factory, cost_tier=CostTier.LOW)
        factory.assert_not_called()
        assert proxy.name == "test_tool"

    def test_proxy_instantiates_on_first_execute(self):
        from victor.tools.progressive import LazyToolProxy

        mock_tool = MagicMock()
        mock_tool.name = "test_tool"
        mock_tool.execute = AsyncMock(return_value=MagicMock(success=True))
        factory = MagicMock(return_value=mock_tool)

        proxy = LazyToolProxy("test_tool", factory, cost_tier=CostTier.LOW)

        import asyncio

        asyncio.run(proxy.execute({}, input="test"))
        factory.assert_called_once()

    def test_proxy_caches_after_first_use(self):
        from victor.tools.progressive import LazyToolProxy

        mock_tool = MagicMock()
        mock_tool.execute = AsyncMock(return_value=MagicMock(success=True))
        factory = MagicMock(return_value=mock_tool)

        proxy = LazyToolProxy("test_tool", factory, cost_tier=CostTier.LOW)

        import asyncio

        asyncio.run(proxy.execute({}, a="1"))
        # Re-create since asyncio.run() closes the loop
        asyncio.run(proxy.execute({}, a="2"))
        factory.assert_called_once()  # Only created once

    def test_proxy_exposes_metadata(self):
        from victor.tools.progressive import LazyToolProxy

        proxy = LazyToolProxy(
            "web_search",
            lambda: None,
            cost_tier=CostTier.MEDIUM,
            description="Search the web",
        )
        assert proxy.name == "web_search"
        assert proxy.cost_tier == CostTier.MEDIUM
        assert proxy.description == "Search the web"


class TestCostTierPreference:
    """Cost-aware selection prefers cheaper tools when scores are close."""

    def test_prefer_free_over_medium_when_close(self):
        from victor.tools.progressive import apply_cost_preference

        candidates = [
            ("web_search", 0.80, CostTier.MEDIUM),
            ("grep", 0.78, CostTier.FREE),
        ]
        ranked = apply_cost_preference(candidates, preference_boost=0.05)
        # grep should rank higher because FREE + 0.05 boost > MEDIUM
        assert ranked[0][0] == "grep"

    def test_keep_higher_score_when_gap_is_large(self):
        from victor.tools.progressive import apply_cost_preference

        candidates = [
            ("web_search", 0.90, CostTier.MEDIUM),
            ("grep", 0.60, CostTier.FREE),
        ]
        ranked = apply_cost_preference(candidates, preference_boost=0.05)
        # web_search still wins because 0.90 >> 0.60 + 0.05
        assert ranked[0][0] == "web_search"

    def test_empty_candidates(self):
        from victor.tools.progressive import apply_cost_preference

        assert apply_cost_preference([], preference_boost=0.05) == []


class TestProgressiveParams:
    """Progressive parameter escalation."""

    def test_initial_params(self):
        from victor.tools.progressive import ProgressiveParams

        pp = ProgressiveParams(initial=5, max_value=100, escalation_factor=2.0)
        assert pp.current == 5

    def test_escalate(self):
        from victor.tools.progressive import ProgressiveParams

        pp = ProgressiveParams(initial=5, max_value=100, escalation_factor=2.0)
        pp.escalate()
        assert pp.current == 10
        pp.escalate()
        assert pp.current == 20

    def test_caps_at_max(self):
        from victor.tools.progressive import ProgressiveParams

        pp = ProgressiveParams(initial=50, max_value=100, escalation_factor=3.0)
        pp.escalate()  # 150 → capped to 100
        assert pp.current == 100

    def test_reset(self):
        from victor.tools.progressive import ProgressiveParams

        pp = ProgressiveParams(initial=5, max_value=100, escalation_factor=2.0)
        pp.escalate()
        pp.escalate()
        pp.reset()
        assert pp.current == 5


class TestRegistryAcceptsLazyProxy:
    """ToolRegistry.register() accepts LazyToolProxy."""

    def test_register_lazy_proxy(self):
        from victor.tools.progressive import LazyToolProxy
        from victor.tools.registry import ToolRegistry

        registry = ToolRegistry()
        proxy = LazyToolProxy(
            "lazy_tool",
            lambda: None,
            cost_tier=CostTier.LOW,
            description="A lazy tool",
        )

        registry.register(proxy)
        assert "lazy_tool" in registry._tools

    def test_lazy_proxy_schema_without_loading(self):
        from victor.tools.progressive import LazyToolProxy
        from victor.tools.registry import ToolRegistry

        registry = ToolRegistry()
        proxy = LazyToolProxy(
            "lazy_tool",
            lambda: None,
            cost_tier=CostTier.LOW,
            description="A lazy tool",
        )
        registry.register(proxy)

        # Schema should be available without loading the tool
        assert not proxy.is_loaded

    def test_decorated_tools_still_work(self):
        """@tool decorated functions still register eagerly."""
        from victor.tools.registry import ToolRegistry

        registry = ToolRegistry()
        # Verify registry accepts BaseTool instances (existing path unchanged)
        from victor.tools.base import BaseTool, ToolResult

        class EagerTool(BaseTool):
            @property
            def name(self):
                return "eager_tool"

            @property
            def description(self):
                return "Eager"

            @property
            def parameters(self):
                return {"type": "object", "properties": {}}

            async def execute(self, ctx, **kwargs):
                return ToolResult(success=True, output="ok")

        registry.register(EagerTool())
        assert "eager_tool" in registry._tools
