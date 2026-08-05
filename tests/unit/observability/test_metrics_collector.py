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

"""Comprehensive unit tests for MetricsCollector module."""

import time
import pytest
from unittest.mock import MagicMock

from victor.agent.metrics_collector import (
    MetricsCollector,
    MetricsCollectorConfig,
    ToolSelectionStats,
    CostTracking,
)
from victor.agent.stream_handler import StreamMetrics
from victor.tools.base import CostTier


class TestMetricsCollectorConfig:
    """Tests for MetricsCollectorConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = MetricsCollectorConfig()
        assert config.model == "unknown"
        assert config.provider == "unknown"
        assert config.analytics_enabled is False

    def test_custom_values(self):
        """Test custom configuration values."""
        config = MetricsCollectorConfig(
            model="gpt-4",
            provider="openai",
            analytics_enabled=True,
        )
        assert config.model == "gpt-4"
        assert config.provider == "openai"
        assert config.analytics_enabled is True


class TestToolSelectionStats:
    """Tests for ToolSelectionStats dataclass."""

    def test_default_values(self):
        """Test default statistics values."""
        stats = ToolSelectionStats()
        assert stats.semantic_selections == 0
        assert stats.keyword_selections == 0
        assert stats.fallback_selections == 0
        assert stats.total_tools_selected == 0
        assert stats.total_tools_executed == 0

    def test_to_dict(self):
        """Test dictionary conversion."""
        stats = ToolSelectionStats(
            semantic_selections=5,
            keyword_selections=3,
            fallback_selections=1,
            total_tools_selected=20,
            total_tools_executed=15,
        )
        result = stats.to_dict()
        assert result["semantic_selections"] == 5
        assert result["keyword_selections"] == 3
        assert result["fallback_selections"] == 1
        assert result["total_tools_selected"] == 20
        assert result["total_tools_executed"] == 15


class TestCostTracking:
    """Tests for CostTracking dataclass."""

    def test_default_values(self):
        """Test default cost tracking values."""
        cost = CostTracking()
        assert cost.total_cost_weight == 0.0
        assert cost.cost_by_tier["free"] == 0.0
        assert cost.cost_by_tier["low"] == 0.0
        assert cost.cost_by_tier["medium"] == 0.0
        assert cost.cost_by_tier["high"] == 0.0
        assert cost.calls_by_tier["free"] == 0
        assert cost.calls_by_tier["low"] == 0
        assert cost.calls_by_tier["medium"] == 0
        assert cost.calls_by_tier["high"] == 0

    def test_to_dict(self):
        """Test dictionary conversion."""
        cost = CostTracking(total_cost_weight=10.5)
        cost.cost_by_tier["free"] = 2.0
        cost.calls_by_tier["free"] = 5
        result = cost.to_dict()
        assert result["total_cost_weight"] == 10.5
        assert result["cost_by_tier"]["free"] == 2.0
        assert result["calls_by_tier"]["free"] == 5


class TestMetricsCollectorInit:
    """Tests for MetricsCollector initialization."""

    @pytest.fixture
    def mock_usage_logger(self):
        """Create a mock usage logger."""
        return MagicMock()

    @pytest.fixture
    def mock_debug_logger(self):
        """Create a mock debug logger."""
        return MagicMock()

    @pytest.fixture
    def collector(self, mock_usage_logger, mock_debug_logger):
        """Create a MetricsCollector for testing."""
        config = MetricsCollectorConfig(model="test-model", provider="test-provider")
        return MetricsCollector(
            config=config,
            usage_logger=mock_usage_logger,
            debug_logger=mock_debug_logger,
        )

    def test_initialization(self, collector):
        """Test collector initializes correctly."""
        assert collector.config.model == "test-model"
        assert collector.config.provider == "test-provider"
        assert collector._selection_stats.semantic_selections == 0
        assert collector._cost_tracking.total_cost_weight == 0.0

    def test_initialization_with_streaming_metrics_collector(self, mock_usage_logger):
        """Test initialization with streaming metrics collector."""
        mock_streaming = MagicMock()
        config = MetricsCollectorConfig(analytics_enabled=True)
        collector = MetricsCollector(
            config=config,
            usage_logger=mock_usage_logger,
            streaming_metrics_collector=mock_streaming,
        )
        assert collector.streaming_metrics_collector is mock_streaming

    def test_initialization_with_tool_cost_lookup(self, mock_usage_logger):
        """Test initialization with custom tool cost lookup."""

        def custom_lookup(name):
            return CostTier.HIGH if name == "expensive" else CostTier.FREE

        config = MetricsCollectorConfig()
        collector = MetricsCollector(
            config=config,
            usage_logger=mock_usage_logger,
            tool_cost_lookup=custom_lookup,
        )
        assert collector._tool_cost_lookup("expensive") == CostTier.HIGH
        assert collector._tool_cost_lookup("cheap") == CostTier.FREE


class TestStreamMetricsTracking:
    """Tests for stream metrics functionality."""

    @pytest.fixture
    def collector(self):
        """Create a MetricsCollector for testing."""
        config = MetricsCollectorConfig(model="test", provider="test")
        return MetricsCollector(config=config, usage_logger=MagicMock())

    def test_init_stream_metrics(self, collector):
        """Test initializing stream metrics."""
        metrics = collector.init_stream_metrics()
        assert isinstance(metrics, StreamMetrics)
        assert metrics.start_time > 0
        assert metrics.first_token_time is None
        assert metrics.end_time == 0.0  # Default value before finalization

    def test_record_first_token(self, collector):
        """Test recording first token time."""
        collector.init_stream_metrics()
        collector.record_first_token()
        assert collector._current_stream_metrics.first_token_time is not None

    def test_record_first_token_only_once(self, collector):
        """Test first token is only recorded once."""
        collector.init_stream_metrics()
        collector.record_first_token()
        first_time = collector._current_stream_metrics.first_token_time

        time.sleep(0.01)
        collector.record_first_token()
        # Should not change
        assert collector._current_stream_metrics.first_token_time == first_time

    def test_record_first_token_no_metrics(self, collector):
        """Test recording first token when no metrics initialized."""
        # Should not raise
        collector.record_first_token()

    def test_finalize_stream_metrics(self, collector):
        """Test finalizing stream metrics."""
        metrics = collector.init_stream_metrics()
        metrics.total_chunks = 100
        metrics.total_content_length = 5000

        result = collector.finalize_stream_metrics()
        assert result is not None
        assert result.end_time is not None
        assert result.total_chunks == 100

    def test_finalize_stream_metrics_no_metrics(self, collector):
        """Test finalizing when no metrics initialized."""
        result = collector.finalize_stream_metrics()
        assert result is None

    def test_get_last_stream_metrics(self, collector):
        """Test getting last stream metrics."""
        collector.init_stream_metrics()
        result = collector.get_last_stream_metrics()
        assert result is not None

    def test_get_last_stream_metrics_none(self, collector):
        """Test getting last stream metrics when none initialized."""
        result = collector.get_last_stream_metrics()
        assert result is None


class TestToolSelectionTracking:
    """Tests for tool selection statistics."""

    @pytest.fixture
    def collector(self):
        """Create a MetricsCollector for testing."""
        config = MetricsCollectorConfig()
        return MetricsCollector(config=config, usage_logger=MagicMock())

    def test_record_semantic_selection(self, collector):
        """Test recording semantic tool selection."""
        collector.record_tool_selection("semantic", 5)
        assert collector._selection_stats.semantic_selections == 1
        assert collector._selection_stats.total_tools_selected == 5

    def test_record_keyword_selection(self, collector):
        """Test recording keyword tool selection."""
        collector.record_tool_selection("keyword", 3)
        assert collector._selection_stats.keyword_selections == 1
        assert collector._selection_stats.total_tools_selected == 3

    def test_record_fallback_selection(self, collector):
        """Test recording fallback tool selection."""
        collector.record_tool_selection("fallback", 2)
        assert collector._selection_stats.fallback_selections == 1
        assert collector._selection_stats.total_tools_selected == 2

    def test_record_multiple_selections(self, collector):
        """Test recording multiple tool selections."""
        collector.record_tool_selection("semantic", 5)
        collector.record_tool_selection("semantic", 3)
        collector.record_tool_selection("keyword", 2)

        assert collector._selection_stats.semantic_selections == 2
        assert collector._selection_stats.keyword_selections == 1
        assert collector._selection_stats.total_tools_selected == 10


class TestToolExecutionTracking:
    """Tests for tool execution statistics."""

    @pytest.fixture
    def collector(self):
        """Create a MetricsCollector for testing."""
        config = MetricsCollectorConfig()

        def cost_lookup(name):
            if name == "expensive_tool":
                return CostTier.HIGH
            elif name == "medium_tool":
                return CostTier.MEDIUM
            return CostTier.FREE

        return MetricsCollector(
            config=config,
            usage_logger=MagicMock(),
            tool_cost_lookup=cost_lookup,
        )

    def test_record_successful_execution(self, collector):
        """Test recording successful tool execution."""
        collector.record_tool_execution("test_tool", success=True, elapsed_ms=100.0)

        stats = collector._tool_usage_stats.get("test_tool")
        assert stats is not None
        assert stats["total_calls"] == 1
        assert stats["successful_calls"] == 1
        assert stats["failed_calls"] == 0
        assert stats["total_time_ms"] == 100.0
        assert stats["avg_time_ms"] == 100.0
        assert stats["min_time_ms"] == 100.0
        assert stats["max_time_ms"] == 100.0

    def test_record_failed_execution(self, collector):
        """Test recording failed tool execution."""
        collector.record_tool_execution("failing_tool", success=False, elapsed_ms=50.0)

        stats = collector._tool_usage_stats.get("failing_tool")
        assert stats is not None
        assert stats["total_calls"] == 1
        assert stats["successful_calls"] == 0
        assert stats["failed_calls"] == 1

    def test_record_multiple_executions(self, collector):
        """Test recording multiple tool executions."""
        collector.record_tool_execution("timed_tool", success=True, elapsed_ms=100.0)
        collector.record_tool_execution("timed_tool", success=True, elapsed_ms=200.0)
        collector.record_tool_execution("timed_tool", success=False, elapsed_ms=50.0)

        stats = collector._tool_usage_stats.get("timed_tool")
        assert stats["total_calls"] == 3
        assert stats["successful_calls"] == 2
        assert stats["failed_calls"] == 1
        assert stats["total_time_ms"] == 350.0
        assert stats["avg_time_ms"] == pytest.approx(116.67, rel=0.01)
        assert stats["min_time_ms"] == 50.0
        assert stats["max_time_ms"] == 200.0

    def test_cost_tracking(self, collector):
        """Test cost tracking by tier."""
        collector.record_tool_execution("expensive_tool", success=True, elapsed_ms=100.0)
        collector.record_tool_execution("medium_tool", success=True, elapsed_ms=100.0)
        collector.record_tool_execution("free_tool", success=True, elapsed_ms=100.0)

        assert collector._cost_tracking.calls_by_tier["high"] == 1
        assert collector._cost_tracking.calls_by_tier["medium"] == 1
        assert collector._cost_tracking.calls_by_tier["free"] == 1
        assert collector._cost_tracking.total_cost_weight > 0

    def test_total_tools_executed(self, collector):
        """Test total tools executed counter."""
        collector.record_tool_execution("tool1", success=True, elapsed_ms=100.0)
        collector.record_tool_execution("tool2", success=True, elapsed_ms=100.0)

        assert collector._selection_stats.total_tools_executed == 2


class TestGetToolUsageStats:
    """Tests for get_tool_usage_stats method."""

    @pytest.fixture
    def collector(self):
        """Create a MetricsCollector for testing."""
        config = MetricsCollectorConfig()
        return MetricsCollector(config=config, usage_logger=MagicMock())

    def test_empty_stats(self, collector):
        """Test stats when no tools used."""
        stats = collector.get_tool_usage_stats()
        assert "selection_stats" in stats
        assert "tool_stats" in stats
        assert "cost_tracking" in stats
        assert "top_tools_by_usage" in stats
        assert "top_tools_by_time" in stats
        assert "top_tools_by_cost" in stats

    def test_stats_with_data(self, collector):
        """Test stats with recorded data."""
        collector.record_tool_selection("semantic", 5)
        collector.record_tool_execution("tool_a", success=True, elapsed_ms=100.0)
        collector.record_tool_execution("tool_a", success=True, elapsed_ms=50.0)
        collector.record_tool_execution("tool_b", success=True, elapsed_ms=200.0)

        stats = collector.get_tool_usage_stats()

        assert stats["selection_stats"]["semantic_selections"] == 1
        assert "tool_a" in stats["tool_stats"]
        assert stats["tool_stats"]["tool_a"]["total_calls"] == 2
        assert len(stats["top_tools_by_usage"]) > 0
        assert stats["top_tools_by_usage"][0] == ("tool_a", 2)

    def test_stats_with_conversation_state(self, collector):
        """Test stats with conversation state summary."""
        state_summary = {"stage": "EXPLORING", "turn": 3}
        stats = collector.get_tool_usage_stats(conversation_state_summary=state_summary)

        assert "conversation_state" in stats
        assert stats["conversation_state"]["stage"] == "EXPLORING"


class TestCallbacks:
    """Tests for callback methods."""

    @pytest.fixture
    def collector(self):
        """Create a MetricsCollector with mocked debug logger."""
        config = MetricsCollectorConfig()
        mock_debug_logger = MagicMock()
        return MetricsCollector(
            config=config,
            usage_logger=MagicMock(),
            debug_logger=mock_debug_logger,
        )

    def test_on_tool_start(self, collector):
        """Test on_tool_start callback."""
        collector.on_tool_start("read_file", {"path": "/test.py"}, iteration=1)
        collector.debug_logger.log_tool_call.assert_called_once_with(
            "read_file", {"path": "/test.py"}, 1
        )

    def test_on_tool_start_no_debug_logger(self):
        """Test on_tool_start without debug logger."""
        config = MetricsCollectorConfig()
        collector = MetricsCollector(config=config, usage_logger=MagicMock())
        # Should not raise
        collector.on_tool_start("test_tool", {}, 0)

    def test_on_tool_complete(self, collector):
        """Test on_tool_complete callback."""
        mock_result = MagicMock()
        mock_result.tool_name = "test_tool"
        mock_result.success = True
        mock_result.execution_time_ms = 100.0

        collector.on_tool_complete(mock_result)

        stats = collector._tool_usage_stats.get("test_tool")
        assert stats is not None
        assert stats["total_calls"] == 1
        assert stats["successful_calls"] == 1

    def test_on_streaming_session_complete(self, collector):
        """Test on_streaming_session_complete callback."""
        mock_session = MagicMock()
        mock_session.session_id = "sess-123"
        mock_session.model = "gpt-4"
        mock_session.provider = "openai"
        mock_session.duration = 5.0
        mock_session.cancelled = False

        collector.on_streaming_session_complete(mock_session)

        collector.usage_logger.log_event.assert_called_once()
        call_args = collector.usage_logger.log_event.call_args
        assert call_args[0][0] == "stream_completed"
        assert call_args[0][1]["session_id"] == "sess-123"


class TestConfigurationUpdates:
    """Tests for configuration update methods."""

    @pytest.fixture
    def collector(self):
        """Create a MetricsCollector for testing."""
        config = MetricsCollectorConfig(model="initial-model", provider="initial-provider")
        return MetricsCollector(config=config, usage_logger=MagicMock())

    def test_update_model_info(self, collector):
        """Test updating model info."""
        collector.update_model_info("new-model", "new-provider")
        assert collector.config.model == "new-model"
        assert collector.config.provider == "new-provider"

    def test_reset_stats(self, collector):
        """Test resetting all statistics."""
        # Add some data
        collector.record_tool_selection("semantic", 5)
        collector.record_tool_execution("tool", success=True, elapsed_ms=100.0)
        collector.init_stream_metrics()

        # Reset
        collector.reset_stats()

        # Verify reset
        assert collector._selection_stats.semantic_selections == 0
        assert collector._selection_stats.total_tools_selected == 0
        assert len(collector._tool_usage_stats) == 0
        assert collector._cost_tracking.total_cost_weight == 0.0
        assert collector._current_stream_metrics is None
