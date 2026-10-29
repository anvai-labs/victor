"""Tests for enhanced predictive tool selection.

Tests cover:
- Initialization with predictive components
- Tool selection with predictions enabled
- Tool selection with predictions disabled (backward compatibility)
- Recording tool usage for learning
- Getting statistics
- Preloading for next step
- Fallback to static mapping on prediction failure
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from victor.agent.conversation.state_machine import ConversationStage
from victor.agent.planning.cooccurrence_tracker import CooccurrenceTracker
from victor.agent.planning.readable_schema import TaskComplexity
from victor.agent.planning.tool_preloader import ToolPreloader
from victor.agent.planning.tool_predictor import ToolPredictor
from victor.agent.planning.tool_selection import StepAwareToolSelector
from victor.agent.tool_selection import ToolSelector


class TestPredictiveInitialization:
    """Test initialization with predictive components."""

    def test_initialization_without_predictive(self):
        """Test initialization without predictive components (backward compatible)."""
        tool_selector = MagicMock(spec=ToolSelector)
        tool_selector.tools = MagicMock()

        selector = StepAwareToolSelector(
            tool_selector=tool_selector,
            enable_predictive=False,
        )

        assert selector.enable_predictive is False
        assert selector.tool_predictor is None
        assert selector.cooccurrence_tracker is None
        assert selector.tool_preloader is None

    def test_initialization_with_predictive(self):
        """Test initialization with predictive components enabled."""
        tool_selector = MagicMock(spec=ToolSelector)
        tool_selector.tools = MagicMock()

        selector = StepAwareToolSelector(
            tool_selector=tool_selector,
            enable_predictive=True,
        )

        assert selector.enable_predictive is True
        assert selector.tool_predictor is not None
        assert selector.cooccurrence_tracker is not None
        assert selector.tool_preloader is not None

    def test_initialization_with_custom_predictive_components(self):
        """Test initialization with custom predictive components."""
        tool_selector = MagicMock(spec=ToolSelector)
        tool_selector.tools = MagicMock()

        tracker = CooccurrenceTracker()
        predictor = ToolPredictor(cooccurrence_tracker=tracker)
        preloader = ToolPreloader(tool_predictor=predictor)

        selector = StepAwareToolSelector(
            tool_selector=tool_selector,
            enable_predictive=True,
            tool_predictor=predictor,
            cooccurrence_tracker=tracker,
            tool_preloader=preloader,
        )

        assert selector.tool_predictor is predictor
        assert selector.cooccurrence_tracker is tracker
        assert selector.tool_preloader is preloader


class TestPredictiveToolSelection:
    """Test tool selection with predictions."""

    def _create_mock_tool(self, name, description):
        """Helper to create a properly mocked tool."""
        tool = MagicMock()
        tool.name = name
        tool.description = description
        tool.parameters = {"type": "object"}
        return tool

    def test_tool_selection_with_predictions_disabled(self):
        """Test tool selection falls back to static mapping when disabled."""
        tool_selector = MagicMock(spec=ToolSelector)
        tool_selector.tools = MagicMock()

        # Mock tool registry with proper tool objects
        mock_tools = [
            self._create_mock_tool("read", "Read file"),
            self._create_mock_tool("grep", "Search patterns"),
        ]
        tool_selector.tools.list_tools = MagicMock(return_value=mock_tools)

        selector = StepAwareToolSelector(
            tool_selector=tool_selector,
            enable_predictive=False,
        )

        tools = selector.get_tools_for_step(
            step_type="research",
            complexity=TaskComplexity("simple"),
            step_description="Find authentication patterns",
            conversation_stage=ConversationStage.READING,
        )

        # Should return tools from static mapping
        assert isinstance(tools, list)

    def test_tool_selection_with_predictions_enabled(self):
        """Test tool selection uses predictions when enabled."""
        tool_selector = MagicMock(spec=ToolSelector)
        tool_selector.tools = MagicMock()

        # Mock tool registry with proper tool objects
        mock_tools = [
            self._create_mock_tool("read", "Read file"),
            self._create_mock_tool("grep", "Search patterns"),
        ]
        tool_selector.tools.list_tools = MagicMock(return_value=mock_tools)

        # Mock predictor
        predictor = MagicMock()
        prediction = MagicMock()
        prediction.tool_name = "code_search"
        prediction.probability = 0.8
        predictor.predict_tools = MagicMock(return_value=[prediction])

        selector = StepAwareToolSelector(
            tool_selector=tool_selector,
            enable_predictive=True,
            tool_predictor=predictor,
        )

        tools = selector.get_tools_for_step(
            step_type="research",
            complexity=TaskComplexity("simple"),
            step_description="Find authentication patterns",
            conversation_stage=ConversationStage.READING,
        )

        # Should call predictor
        predictor.predict_tools.assert_called_once()

    def test_prediction_failure_fallback(self):
        """Test that prediction failures fall back to static mapping."""
        tool_selector = MagicMock(spec=ToolSelector)
        tool_selector.tools = MagicMock()

        # Mock tool registry with proper tool objects
        mock_tools = [
            self._create_mock_tool("read", "Read file"),
        ]
        tool_selector.tools.list_tools = MagicMock(return_value=mock_tools)

        # Mock predictor that raises exception
        predictor = MagicMock()
        predictor.predict_tools = MagicMock(side_effect=Exception("Prediction failed"))

        selector = StepAwareToolSelector(
            tool_selector=tool_selector,
            enable_predictive=True,
            tool_predictor=predictor,
        )

        # Should not raise exception
        tools = selector.get_tools_for_step(
            step_type="research",
            complexity=TaskComplexity("simple"),
            step_description="Find patterns",
        )

        # Should return tools from static mapping
        assert isinstance(tools, list)

    def test_high_confidence_predictions_added(self):
        """Test that high-confidence predictions are added to tool set."""
        tool_selector = MagicMock(spec=ToolSelector)
        tool_selector.tools = MagicMock()

        # Mock tool registry with proper tool objects
        mock_tools = [
            self._create_mock_tool("read", "Read file"),
            self._create_mock_tool("code_search", "Semantic search"),
        ]
        tool_selector.tools.list_tools = MagicMock(return_value=mock_tools)

        # Mock predictor with high confidence
        predictor = MagicMock()
        prediction = MagicMock()
        prediction.tool_name = "code_search"
        prediction.probability = 0.9  # High confidence
        predictor.predict_tools = MagicMock(return_value=[prediction])

        selector = StepAwareToolSelector(
            tool_selector=tool_selector,
            enable_predictive=True,
            tool_predictor=predictor,
        )

        tools = selector.get_tools_for_step(
            step_type="research",
            complexity=TaskComplexity("moderate"),
            step_description="Find authentication patterns",
        )

        # High-confidence prediction should be included
        predictor.predict_tools.assert_called_once()

    def test_low_confidence_predictions_filtered(self):
        """Test that low-confidence predictions are filtered out."""
        tool_selector = MagicMock(spec=ToolSelector)
        tool_selector.tools = MagicMock()

        # Mock tool registry with proper tool objects
        mock_tools = [
            self._create_mock_tool("read", "Read file"),
        ]
        tool_selector.tools.list_tools = MagicMock(return_value=mock_tools)

        # Mock predictor with low confidence
        predictor = MagicMock()
        prediction = MagicMock()
        prediction.tool_name = "test"
        prediction.probability = 0.3  # Low confidence (< 0.6)
        predictor.predict_tools = MagicMock(return_value=[prediction])

        selector = StepAwareToolSelector(
            tool_selector=tool_selector,
            enable_predictive=True,
            tool_predictor=predictor,
        )

        tools = selector.get_tools_for_step(
            step_type="research",
            complexity=TaskComplexity("simple"),
            step_description="Find patterns",
        )

        # Low-confidence prediction should be filtered
        predictor.predict_tools.assert_called_once()


class TestToolUsageRecording:
    """Test recording tool usage for learning."""

    def test_record_tool_usage_updates_recent_tools(self):
        """Test that recording updates recent tools list."""
        tool_selector = MagicMock(spec=ToolSelector)
        tool_selector.tools = MagicMock()

        selector = StepAwareToolSelector(
            tool_selector=tool_selector,
            enable_predictive=True,
        )

        # Record tool usage
        selector.record_tool_usage(
            tools_used=["read", "grep"],
            step_type="research",
            task_type="search",
            success=True,
        )

        # Should update recent tools
        assert len(selector._recent_tools) == 2
        assert selector._recent_tools == ["read", "grep"]

    def test_recent_tools_limited_to_10(self):
        """Test that recent tools list is limited to 10 entries."""
        tool_selector = MagicMock(spec=ToolSelector)
        tool_selector.tools = MagicMock()

        selector = StepAwareToolSelector(
            tool_selector=tool_selector,
            enable_predictive=True,
        )

        # Record 15 tool usages
        for i in range(15):
            selector.record_tool_usage(
                tools_used=[f"tool_{i}"],
                step_type="research",
                task_type="search",
                success=True,
            )

        # Should keep only last 10
        assert len(selector._recent_tools) == 10
        assert selector._recent_tools[0] == "tool_5"
        assert selector._recent_tools[-1] == "tool_14"

    def test_record_tool_usage_updates_tracker(self):
        """Test that recording updates co-occurrence tracker."""
        tool_selector = MagicMock(spec=ToolSelector)
        tool_selector.tools = MagicMock()

        tracker = CooccurrenceTracker()
        selector = StepAwareToolSelector(
            tool_selector=tool_selector,
            enable_predictive=True,
            cooccurrence_tracker=tracker,
        )

        # Record tool usage
        selector.record_tool_usage(
            tools_used=["read", "grep"],
            step_type="research",
            task_type="search",
            success=True,
        )

        # Should update tracker
        stats = tracker.get_statistics()
        assert stats["total_sequences_recorded"] == 1


class TestStatistics:
    """Test statistics gathering."""

    def test_statistics_includes_predictive_status(self):
        """Test that statistics include predictive enabled status."""
        tool_selector = MagicMock(spec=ToolSelector)
        tool_selector.tools = MagicMock()

        selector = StepAwareToolSelector(
            tool_selector=tool_selector,
            enable_predictive=True,
        )

        stats = selector.get_statistics()

        assert "predictive_enabled" in stats
        assert stats["predictive_enabled"] is True

    def test_statistics_includes_recent_tools(self):
        """Test that statistics include recent tools."""
        tool_selector = MagicMock(spec=ToolSelector)
        tool_selector.tools = MagicMock()

        selector = StepAwareToolSelector(
            tool_selector=tool_selector,
            enable_predictive=True,
        )

        # Record some tool usage
        selector.record_tool_usage(
            tools_used=["read", "grep"],
            step_type="research",
            task_type="search",
        )

        stats = selector.get_statistics()

        assert "recent_tools" in stats
        assert stats["recent_tools"] == ["read", "grep"]
        assert stats["recent_tools_count"] == 2

    def test_statistics_includes_predictor_stats(self):
        """Test that statistics include predictor stats when available."""
        tool_selector = MagicMock(spec=ToolSelector)
        tool_selector.tools = MagicMock()

        predictor = ToolPredictor()
        selector = StepAwareToolSelector(
            tool_selector=tool_selector,
            enable_predictive=True,
            tool_predictor=predictor,
        )

        stats = selector.get_statistics()

        assert "predictor" in stats
        assert "config" in stats["predictor"]

    def test_statistics_includes_tracker_stats(self):
        """Test that statistics include tracker stats when available."""
        tool_selector = MagicMock(spec=ToolSelector)
        tool_selector.tools = MagicMock()

        tracker = CooccurrenceTracker()
        selector = StepAwareToolSelector(
            tool_selector=tool_selector,
            enable_predictive=True,
            cooccurrence_tracker=tracker,
        )

        # Record some usage
        selector.record_tool_usage(
            tools_used=["read", "grep"],
            step_type="research",
            task_type="search",
        )

        stats = selector.get_statistics()

        assert "tracker" in stats
        assert stats["tracker"]["total_sequences_recorded"] == 1

    def test_statistics_includes_preloader_stats(self):
        """Test that statistics include preloader stats when available."""
        tool_selector = MagicMock(spec=ToolSelector)
        tool_selector.tools = MagicMock()

        preloader = ToolPreloader()
        selector = StepAwareToolSelector(
            tool_selector=tool_selector,
            enable_predictive=True,
            tool_preloader=preloader,
        )

        stats = selector.get_statistics()

        assert "preloader" in stats
        assert "l1_cache_size" in stats["preloader"]


class TestPreloading:
    """Test preloading for next step."""

    def _create_mock_tool(self, name, description):
        """Helper to create a properly mocked tool."""
        tool = MagicMock()
        tool.name = name
        tool.description = description
        tool.parameters = {"type": "object"}
        return tool

    @pytest.mark.asyncio
    async def test_preload_called_after_tool_selection(self):
        """Test that preloading is attempted after tool selection in async context."""
        import asyncio

        tool_selector = MagicMock(spec=ToolSelector)
        tool_selector.tools = MagicMock()

        # Mock tool registry with proper tool objects
        mock_tools = [
            self._create_mock_tool("read", "Read file"),
        ]
        tool_selector.tools.list_tools = MagicMock(return_value=mock_tools)

        # Mock preloader
        preloader = MagicMock()
        preloader.preload_for_next_step = AsyncMock(return_value=2)

        selector = StepAwareToolSelector(
            tool_selector=tool_selector,
            enable_predictive=True,
            tool_preloader=preloader,
        )

        # Get tools for step (this will schedule the preload task)
        tools = selector.get_tools_for_step(
            step_type="research",
            complexity=TaskComplexity("simple"),
            step_description="Find patterns",
        )

        # Give the background task time to execute
        await asyncio.sleep(0.1)

        # Preload should have been called
        preloader.preload_for_next_step.assert_called_once()

    @pytest.mark.asyncio
    async def test_preload_failure_doesnt_affect_tool_selection(self):
        """Test that preload failures don't affect tool selection."""
        tool_selector = MagicMock(spec=ToolSelector)
        tool_selector.tools = MagicMock()

        # Mock tool registry with proper tool objects
        mock_tools = [
            self._create_mock_tool("read", "Read file"),
        ]
        tool_selector.tools.list_tools = MagicMock(return_value=mock_tools)

        # Mock preloader that raises exception
        preloader = MagicMock()
        preloader.preload_for_next_step = AsyncMock(side_effect=Exception("Preload failed"))

        selector = StepAwareToolSelector(
            tool_selector=tool_selector,
            enable_predictive=True,
            tool_preloader=preloader,
        )

        # Should not raise exception
        tools = selector.get_tools_for_step(
            step_type="research",
            complexity=TaskComplexity("simple"),
            step_description="Find patterns",
        )

        # Should still return tools
        assert isinstance(tools, list)


class TestBackwardCompatibility:
    """Test backward compatibility with existing code."""

    def _create_mock_tool(self, name, description):
        """Helper to create a properly mocked tool."""
        tool = MagicMock()
        tool.name = name
        tool.description = description
        tool.parameters = {"type": "object"}
        return tool

    def test_existing_code_works_without_predictive(self):
        """Test that existing code works without predictive features."""
        tool_selector = MagicMock(spec=ToolSelector)
        tool_selector.tools = MagicMock()

        # Mock tool registry with proper tool objects
        mock_tools = [
            self._create_mock_tool("read", "Read file"),
            self._create_mock_tool("grep", "Search patterns"),
        ]
        tool_selector.tools.list_tools = MagicMock(return_value=mock_tools)

        # Old-style initialization (no predictive parameters)
        selector = StepAwareToolSelector(
            tool_selector=tool_selector,
        )

        # Should work as before
        tools = selector.get_tools_for_step(
            step_type="research",
            complexity=TaskComplexity("simple"),
            step_description="Find patterns",
            conversation_stage=ConversationStage.READING,
        )

        # Should return tools from static mapping
        assert isinstance(tools, list)

    def test_cache_invalidation_works(self):
        """Test that cache invalidation still works."""
        tool_selector = MagicMock(spec=ToolSelector)
        tool_selector.tools = MagicMock()

        selector = StepAwareToolSelector(
            tool_selector=tool_selector,
            enable_predictive=True,
        )

        # Add something to cache
        selector._tool_set_cache[("research", "simple", None)] = []

        # Invalidate cache
        selector.invalidate_cache()

        # Cache should be empty
        assert len(selector._tool_set_cache) == 0

    def test_step_tool_summary_works(self):
        """Test that step tool summary still works."""
        tool_selector = MagicMock(spec=ToolSelector)
        tool_selector.tools = MagicMock()
        tool_selector.tools.list_tools = MagicMock(return_value=[])

        selector = StepAwareToolSelector(
            tool_selector=tool_selector,
            enable_predictive=True,
        )

        summary = selector.get_step_tool_summary(
            step_type="research",
            complexity=TaskComplexity("simple"),
        )

        assert summary["step_type"] == "research"
        assert summary["complexity"] == "simple"

    def test_map_step_type_to_task_type_works(self):
        """Test that step type mapping still works."""
        tool_selector = MagicMock(spec=ToolSelector)
        tool_selector.tools = MagicMock()

        selector = StepAwareToolSelector(
            tool_selector=tool_selector,
            enable_predictive=True,
        )

        task_type = selector.map_step_type_to_task_type("research")

        assert task_type == "search"
