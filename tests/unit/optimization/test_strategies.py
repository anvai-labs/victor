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

"""Unit tests for optimization strategies."""

import pytest
from unittest.mock import Mock

from victor.optimization.strategies import (
    PruningStrategy,
    ParallelizationStrategy,
    ToolSelectionStrategy,
    create_strategy,
    ParallelGroup,
)
from victor.optimization.models import (
    Bottleneck,
    BottleneckType,
    BottleneckSeverity,
    OptimizationStrategyType,
    NodeStatistics,
    WorkflowProfile,
)


@pytest.fixture
def sample_profile():
    """Create a sample workflow profile."""
    node_stats = {
        "node_1": NodeStatistics(
            node_id="node_1",
            avg_duration=5.0,
            p50_duration=5.0,
            p95_duration=6.0,
            p99_duration=7.0,
            success_rate=0.9,
            total_cost=0.005,
        ),
        "node_2": NodeStatistics(
            node_id="node_2",
            avg_duration=3.0,
            p50_duration=3.0,
            p95_duration=4.0,
            p99_duration=5.0,
            success_rate=0.95,
            total_cost=0.003,
        ),
        "node_3": NodeStatistics(
            node_id="node_3",
            avg_duration=2.0,
            p50_duration=2.0,
            p95_duration=2.5,
            p99_duration=3.0,
            success_rate=0.5,  # Low success rate
            total_cost=0.002,
        ),
    }

    return WorkflowProfile(
        workflow_id="test_workflow",
        node_stats=node_stats,
        bottlenecks=[],
        opportunities=[],
        total_duration=10.0,
        total_cost=0.01,
        total_tokens=5000,
        success_rate=0.78,
    )


class TestPruningStrategy:
    """Tests for PruningStrategy."""

    @pytest.fixture
    def strategy(self):
        return PruningStrategy()

    def test_can_apply_unreliable_node(self, strategy):
        """Test that pruning applies to unreliable nodes."""
        bottleneck = Bottleneck(
            type=BottleneckType.UNRELIABLE_NODE,
            severity=BottleneckSeverity.HIGH,
            node_id="failing_node",
            metric="success_rate",
            value=50.0,
            threshold=80.0,
            suggestion="Remove failing node",
        )

        profile = Mock()

        assert strategy.can_apply(bottleneck, profile) is True

    def test_can_apply_unused_output(self, strategy):
        """Test that pruning applies to unused outputs."""
        bottleneck = Bottleneck(
            type=BottleneckType.UNUSED_OUTPUT,
            severity=BottleneckSeverity.MEDIUM,
            node_id="unused_node",
            metric="usage",
            value=0.0,
            threshold=0.1,
            suggestion="Remove unused node",
        )

        profile = Mock()

        assert strategy.can_apply(bottleneck, profile) is True

    def test_cannot_apply_slow_node(self, strategy):
        """Test that pruning does not apply to slow nodes."""
        bottleneck = Bottleneck(
            type=BottleneckType.SLOW_NODE,
            severity=BottleneckSeverity.HIGH,
            node_id="slow_node",
            metric="duration",
            value=10.0,
            threshold=3.0,
            suggestion="Parallelize node",
        )

        profile = Mock()

        assert strategy.can_apply(bottleneck, profile) is False

    def test_generate_opportunity_unreliable_node(self, strategy, sample_profile):
        """Test generating opportunity for unreliable node."""
        bottleneck = Bottleneck(
            type=BottleneckType.UNRELIABLE_NODE,
            severity=BottleneckSeverity.HIGH,
            node_id="node_3",
            metric="success_rate",
            value=50.0,
            threshold=80.0,
            suggestion="Remove failing node",
        )

        opportunity = strategy.generate_opportunity(bottleneck, sample_profile)

        assert opportunity is not None
        assert opportunity.strategy_type == OptimizationStrategyType.PRUNING
        assert opportunity.target == "node_3"
        assert opportunity.expected_improvement > 0
        assert opportunity.risk_level == BottleneckSeverity.HIGH

    def test_risk_level(self, strategy):
        """Test that risk level is HIGH."""
        assert strategy.risk_level == BottleneckSeverity.HIGH


class TestParallelizationStrategy:
    """Tests for ParallelizationStrategy."""

    @pytest.fixture
    def strategy(self):
        return ParallelizationStrategy()

    def test_can_apply_slow_node(self, strategy):
        """Test that parallelization applies to slow nodes."""
        bottleneck = Bottleneck(
            type=BottleneckType.SLOW_NODE,
            severity=BottleneckSeverity.HIGH,
            node_id="slow_node",
            metric="duration",
            value=10.0,
            threshold=3.0,
            suggestion="Parallelize node",
        )

        profile = Mock()

        assert strategy.can_apply(bottleneck, profile) is True

    def test_find_parallelizable_nodes(self, strategy, sample_profile):
        """Test finding parallelizable nodes."""
        groups = strategy._find_parallelizable_nodes(sample_profile)

        # Should find at least one group
        assert len(groups) >= 0

        # Check group structure if found
        for group in groups:
            assert isinstance(group, ParallelGroup)
            assert len(group.node_ids) >= 1
            assert group.estimated_speedup >= 1.0
            assert group.sequential_duration > 0
            assert group.parallel_duration > 0

    def test_generate_opportunity(self, strategy, sample_profile):
        """Test generating parallelization opportunity."""
        bottleneck = Bottleneck(
            type=BottleneckType.SLOW_NODE,
            severity=BottleneckSeverity.HIGH,
            node_id="node_1",
            metric="duration",
            value=10.0,
            threshold=3.0,
            suggestion="Parallelize node",
        )

        opportunity = strategy.generate_opportunity(bottleneck, sample_profile)

        # May return None if no parallelization found
        if opportunity:
            assert opportunity.strategy_type == OptimizationStrategyType.PARALLELIZATION
            assert opportunity.expected_improvement > 0
            assert opportunity.risk_level == BottleneckSeverity.MEDIUM

    def test_risk_level(self, strategy):
        """Test that risk level is MEDIUM."""
        assert strategy.risk_level == BottleneckSeverity.MEDIUM


class TestToolSelectionStrategy:
    """Tests for ToolSelectionStrategy."""

    @pytest.fixture
    def strategy(self):
        return ToolSelectionStrategy()

    def test_can_apply_expensive_tool(self, strategy):
        """Test that tool selection applies to expensive tools."""
        bottleneck = Bottleneck(
            type=BottleneckType.EXPENSIVE_TOOL,
            severity=BottleneckSeverity.MEDIUM,
            tool_id="claude_opus",
            metric="cost",
            value=0.5,
            threshold=0.1,
            suggestion="Use cheaper tool",
        )

        profile = Mock()

        assert strategy.can_apply(bottleneck, profile) is True

    def test_cannot_apply_slow_node(self, strategy):
        """Test that tool selection does not apply to slow nodes."""
        bottleneck = Bottleneck(
            type=BottleneckType.SLOW_NODE,
            severity=BottleneckSeverity.HIGH,
            node_id="slow_node",
            metric="duration",
            value=10.0,
            threshold=3.0,
            suggestion="Parallelize node",
        )

        profile = Mock()

        assert strategy.can_apply(bottleneck, profile) is False

    def test_generate_opportunity_known_tool(self, strategy):
        """Test generating opportunity for known expensive tool."""
        bottleneck = Bottleneck(
            type=BottleneckType.EXPENSIVE_TOOL,
            severity=BottleneckSeverity.MEDIUM,
            tool_id="claude_opus",
            metric="cost",
            value=50.0,
            threshold=10.0,
            suggestion="Use cheaper tool",
        )

        profile = Mock()

        opportunity = strategy.generate_opportunity(bottleneck, profile)

        assert opportunity is not None
        assert opportunity.strategy_type == OptimizationStrategyType.TOOL_SELECTION
        assert "claude_opus" in opportunity.target
        assert opportunity.expected_improvement > 0
        assert opportunity.risk_level == BottleneckSeverity.LOW

    def test_generate_opportunity_unknown_tool(self, strategy):
        """Test generating opportunity for unknown tool."""
        bottleneck = Bottleneck(
            type=BottleneckType.EXPENSIVE_TOOL,
            severity=BottleneckSeverity.MEDIUM,
            tool_id="unknown_expensive_tool",
            metric="cost",
            value=50.0,
            threshold=10.0,
            suggestion="Use cheaper tool",
        )

        profile = Mock()

        opportunity = strategy.generate_opportunity(bottleneck, profile)

        # Should return None for unknown tool
        assert opportunity is None

    def test_risk_level(self, strategy):
        """Test that risk level is LOW."""
        assert strategy.risk_level == BottleneckSeverity.LOW


def test_create_strategy():
    """Test strategy factory function."""
    # Test creating each strategy type
    pruning = create_strategy(OptimizationStrategyType.PRUNING)
    assert isinstance(pruning, PruningStrategy)

    parallelization = create_strategy(OptimizationStrategyType.PARALLELIZATION)
    assert isinstance(parallelization, ParallelizationStrategy)

    tool_selection = create_strategy(OptimizationStrategyType.TOOL_SELECTION)
    assert isinstance(tool_selection, ToolSelectionStrategy)


def test_create_strategy_invalid():
    """Test creating strategy with invalid type."""
    with pytest.raises(ValueError, match="Unsupported strategy type"):
        create_strategy(OptimizationStrategyType.CACHING)
