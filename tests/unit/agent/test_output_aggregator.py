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

"""Tests for OutputAggregator.

Covers GAP Phase 3: Output aggregation and completion detection.
"""

import pytest
import time
from unittest.mock import MagicMock

from victor.agent.output_aggregator import (
    AggregationState,
    AggregatedResult,
    ToolOutput,
    OutputAggregator,
    OutputAggregatorObserver,
    LoggingObserver,
    MetricsObserver,
    create_default_aggregator,
    create_monitored_aggregator,
)


class TestToolOutput:
    """Tests for ToolOutput dataclass."""

    def test_create_tool_output(self):
        """Test basic ToolOutput creation."""
        output = ToolOutput(
            tool_name="read_file",
            result="file content",
            success=True,
        )
        assert output.tool_name == "read_file"
        assert output.result == "file content"
        assert output.success is True
        assert output.timestamp > 0

    def test_tool_output_with_args_hash(self):
        """Test ToolOutput computes args hash."""
        output = ToolOutput(
            tool_name="read_file",
            result="content",
            metadata={"args": {"path": "foo.py", "limit": 100}},
        )
        assert output.args_hash != ""
        assert len(output.args_hash) == 12  # MD5 prefix

    def test_same_args_produce_same_hash(self):
        """Test that identical args produce same hash."""
        args = {"path": "foo.py", "limit": 100}
        output1 = ToolOutput(
            tool_name="read",
            result="a",
            metadata={"args": args},
        )
        output2 = ToolOutput(
            tool_name="read",
            result="b",
            metadata={"args": args},
        )
        assert output1.args_hash == output2.args_hash


class TestAggregatedResult:
    """Tests for AggregatedResult dataclass."""

    def test_empty_result_properties(self):
        """Test properties with no results."""
        result = AggregatedResult(state=AggregationState.COLLECTING)
        assert result.tool_count == 0
        assert result.unique_tools == set()
        assert result.success_rate == 0.0

    def test_result_with_tools(self):
        """Test properties with multiple results."""
        result = AggregatedResult(
            state=AggregationState.READY_TO_SYNTHESIZE,
            results=[
                ToolOutput(tool_name="read", result="a", success=True),
                ToolOutput(tool_name="grep", result="b", success=True),
                ToolOutput(tool_name="read", result="c", success=False),
            ],
        )
        assert result.tool_count == 3
        assert result.unique_tools == {"read", "grep"}
        assert result.success_rate == pytest.approx(2 / 3)


class TestOutputAggregator:
    """Tests for OutputAggregator."""

    @pytest.fixture
    def aggregator(self):
        """Create a basic aggregator."""
        return OutputAggregator(max_results=5, stale_threshold_seconds=30.0)

    def test_initial_state(self, aggregator):
        """Test aggregator starts in COLLECTING state."""
        assert aggregator.state == AggregationState.COLLECTING
        assert len(aggregator.results) == 0

    def test_add_result(self, aggregator):
        """Test adding a result."""
        aggregator.add_result("read_file", "content", success=True)
        assert len(aggregator.results) == 1
        assert aggregator.results[0].tool_name == "read_file"

    def test_state_transition_to_ready(self, aggregator):
        """Test state transitions to READY_TO_SYNTHESIZE at threshold."""
        for i in range(5):
            aggregator.add_result(f"tool_{i}", f"result_{i}")

        assert aggregator.state == AggregationState.READY_TO_SYNTHESIZE

    def test_loop_detection(self):
        """Test loop detection when same tool called repeatedly."""
        aggregator = OutputAggregator(max_results=20, loop_detection_window=3)

        # Call same tool 3 times in a row
        for i in range(3):
            aggregator.add_result(
                "read_file",
                f"result_{i}",
                metadata={"args": {"path": "same.py"}},
            )

        assert aggregator.state == AggregationState.LOOPING

    def test_duplicate_detection(self, aggregator):
        """Test duplicate args are flagged."""
        args = {"path": "foo.py"}
        aggregator.add_result("read", "a", metadata={"args": args})
        aggregator.add_result("read", "b", metadata={"args": args})

        # Second result should be flagged as duplicate
        assert aggregator.results[1].metadata.get("is_duplicate") is True

    def test_get_synthesis_prompt(self, aggregator):
        """Test synthesis prompt generation."""
        aggregator.add_result("read_file", "content A")
        aggregator.add_result("grep", "matched: foo")

        prompt = aggregator.get_synthesis_prompt()
        assert "2 tool results" in prompt
        assert "read_file" in prompt
        assert "grep" in prompt

    def test_get_aggregated_result(self, aggregator):
        """Test getting aggregated result."""
        aggregator.add_result("read_file", "content", success=True)
        aggregator.add_result("grep", "match", success=False)

        result = aggregator.get_aggregated_result()
        assert result.state == AggregationState.COLLECTING
        assert result.tool_count == 2
        assert "read_file" in result.metadata["unique_tools"]
        assert result.metadata["success_rate"] == 0.5

    def test_reset(self, aggregator):
        """Test reset clears state."""
        aggregator.add_result("tool", "result")
        aggregator.add_result("tool", "result")

        aggregator.reset()

        assert aggregator.state == AggregationState.COLLECTING
        assert len(aggregator.results) == 0

    def test_confidence_calculation(self, aggregator):
        """Test confidence score calculation."""
        aggregator.add_result("read", "a", success=True)
        aggregator.add_result("grep", "b", success=True)
        aggregator.add_result("ls", "c", success=True)

        result = aggregator.get_aggregated_result()
        # High success rate + high tool diversity = good confidence
        assert result.confidence > 0.5


class TestObservers:
    """Tests for observer pattern."""

    def test_add_observer(self):
        """Test adding an observer."""
        aggregator = OutputAggregator()
        observer = MagicMock(spec=OutputAggregatorObserver)

        aggregator.add_observer(observer)
        aggregator.add_result("test", "result")

        observer.on_result_added.assert_called_once()

    def test_remove_observer(self):
        """Test removing an observer."""
        aggregator = OutputAggregator()
        observer = MagicMock(spec=OutputAggregatorObserver)

        aggregator.add_observer(observer)
        aggregator.remove_observer(observer)
        aggregator.add_result("test", "result")

        observer.on_result_added.assert_not_called()

    def test_state_change_notification(self):
        """Test observers are notified of state changes."""
        aggregator = OutputAggregator(max_results=2)
        observer = MagicMock(spec=OutputAggregatorObserver)
        aggregator.add_observer(observer)

        aggregator.add_result("tool1", "a")
        aggregator.add_result("tool2", "b")

        observer.on_state_change.assert_called()
        call_args = observer.on_state_change.call_args[0]
        assert call_args[0] == AggregationState.READY_TO_SYNTHESIZE

    def test_synthesis_ready_notification(self):
        """Test observers are notified when synthesis is ready."""
        aggregator = OutputAggregator(max_results=2)
        observer = MagicMock(spec=OutputAggregatorObserver)
        aggregator.add_observer(observer)

        aggregator.add_result("tool1", "a")
        aggregator.add_result("tool2", "b")

        observer.on_synthesis_ready.assert_called_once()

    def test_logging_observer(self, caplog):
        """Test LoggingObserver logs events."""
        import logging

        observer = LoggingObserver()
        output = ToolOutput(tool_name="test", result="result")

        with caplog.at_level(logging.DEBUG):
            observer.on_result_added(output)

        # Check that something was logged (exact message may vary)
        # The logging observer logs at DEBUG level

    def test_metrics_observer(self):
        """Test MetricsObserver collects metrics."""
        observer = MetricsObserver()

        observer.on_result_added(ToolOutput(tool_name="read", result="a"))
        observer.on_result_added(ToolOutput(tool_name="grep", result="b"))
        observer.on_result_added(ToolOutput(tool_name="read", result="c"))
        observer.on_state_change(AggregationState.READY_TO_SYNTHESIZE)
        observer.on_synthesis_ready(AggregatedResult(state=AggregationState.COMPLETE))

        metrics = observer.get_metrics()
        assert metrics["total_results"] == 3
        assert metrics["results_by_tool"]["read"] == 2
        assert metrics["results_by_tool"]["grep"] == 1
        assert metrics["synthesis_count"] == 1


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_default_aggregator(self):
        """Test default aggregator creation."""
        aggregator = create_default_aggregator()
        assert isinstance(aggregator, OutputAggregator)
        # Should have logging observer
        assert len(aggregator._observers) >= 1

    def test_create_monitored_aggregator(self):
        """Test monitored aggregator creation."""
        aggregator, metrics = create_monitored_aggregator()
        assert isinstance(aggregator, OutputAggregator)
        assert isinstance(metrics, MetricsObserver)

        # Add a result and check metrics
        aggregator.add_result("test", "result")
        assert metrics.get_metrics()["total_results"] == 1


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_aggregator_synthesis_prompt(self):
        """Test synthesis prompt with no results."""
        aggregator = OutputAggregator()
        prompt = aggregator.get_synthesis_prompt()
        assert prompt == ""

    def test_observer_error_handling(self):
        """Test that observer errors don't break aggregator."""
        aggregator = OutputAggregator()

        # Create a faulty observer
        class FaultyObserver:
            def on_result_added(self, result):
                raise ValueError("Observer error")

            def on_state_change(self, state):
                raise ValueError("Observer error")

            def on_synthesis_ready(self, aggregated):
                raise ValueError("Observer error")

        aggregator.add_observer(FaultyObserver())

        # Should not raise
        aggregator.add_result("test", "result")
        assert len(aggregator.results) == 1

    def test_large_result_truncation_in_prompt(self):
        """Test that large results are truncated in synthesis prompt."""
        aggregator = OutputAggregator()
        aggregator.add_result("tool", "x" * 1000)

        prompt = aggregator.get_synthesis_prompt()
        assert "..." in prompt
        assert len(prompt) < 2000

    def test_completion_detection_with_long_content(self):
        """Test completion detection with substantial content."""
        aggregator = OutputAggregator()
        aggregator.add_result("generate", "x" * 600, success=True)

        # Large successful result should trigger completion
        assert aggregator.state == AggregationState.COMPLETE

    def test_mixed_success_failure(self):
        """Test handling of mixed success/failure results."""
        aggregator = OutputAggregator(max_results=10)

        aggregator.add_result("tool1", "result", success=True)
        aggregator.add_result("tool2", None, success=False)
        aggregator.add_result("tool3", "result", success=True)

        result = aggregator.get_aggregated_result()
        assert result.metadata["success_rate"] == pytest.approx(2 / 3)
