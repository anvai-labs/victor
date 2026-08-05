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

"""Unit tests for WorkflowProfiler."""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

from victor.optimization.profiler import WorkflowProfiler
from victor.optimization.models import (
    BottleneckType,
    BottleneckSeverity,
    NodeStatistics,
    WorkflowProfile,
)


@pytest.fixture
def profiler():
    """Create a WorkflowProfiler instance."""
    return WorkflowProfiler()


@pytest.fixture
def mock_executions():
    """Create mock execution data."""
    return [
        {
            "run_id": "run_1",
            "success": True,
            "duration": 10.0,
            "cost": 0.01,
            "node_metrics": {
                "node_1": {
                    "duration": 3.0,
                    "input_tokens": 1000,
                    "output_tokens": 500,
                    "cost": 0.003,
                    "success": True,
                    "tools": {
                        "tool_a": {"count": 2, "cost": 0.001},
                    },
                },
                "node_2": {
                    "duration": 7.0,
                    "input_tokens": 2000,
                    "output_tokens": 1000,
                    "cost": 0.007,
                    "success": True,
                    "tools": {
                        "tool_b": {"count": 1, "cost": 0.005},
                    },
                },
            },
        },
        {
            "run_id": "run_2",
            "success": True,
            "duration": 12.0,
            "cost": 0.012,
            "node_metrics": {
                "node_1": {
                    "duration": 4.0,
                    "input_tokens": 1200,
                    "output_tokens": 600,
                    "cost": 0.004,
                    "success": True,
                },
                "node_2": {
                    "duration": 8.0,
                    "input_tokens": 2500,
                    "output_tokens": 1200,
                    "cost": 0.008,
                    "success": True,
                },
            },
        },
    ]


@pytest.mark.asyncio
async def test_profile_workflow_success(profiler, mock_executions):
    """Test successful workflow profiling."""
    with patch.object(
        profiler,
        "_fetch_executions",
        return_value=mock_executions,
    ):
        mock_tracker = Mock()
        profile = await profiler.profile_workflow(
            workflow_id="test_workflow",
            experiment_tracker=mock_tracker,
            min_executions=2,
        )

        assert profile is not None
        assert profile.workflow_id == "test_workflow"
        assert len(profile.node_stats) == 2
        assert profile.num_executions == 2
        assert "node_1" in profile.node_stats
        assert "node_2" in profile.node_stats


@pytest.mark.asyncio
async def test_profile_workflow_insufficient_data(profiler):
    """Test profiling with insufficient execution data."""
    with patch.object(profiler, "_fetch_executions", return_value=[]):
        mock_tracker = Mock()
        profile = await profiler.profile_workflow(
            workflow_id="test_workflow",
            experiment_tracker=mock_tracker,
            min_executions=3,
        )

        assert profile is None


@pytest.mark.asyncio
async def test_calculate_node_statistics(profiler, mock_executions):
    """Test calculation of node statistics."""
    node_stats = profiler._calculate_node_statistics(mock_executions)

    assert len(node_stats) == 2
    assert "node_1" in node_stats
    assert "node_2" in node_stats

    # Check node_1 stats
    stats_1 = node_stats["node_1"]
    assert stats_1.node_id == "node_1"
    assert stats_1.avg_duration > 0
    assert stats_1.p50_duration > 0
    assert stats_1.success_rate > 0
    assert stats_1.avg_input_tokens > 0
    assert stats_1.avg_output_tokens > 0


def test_detect_bottlenecks_slow_nodes(profiler):
    """Test detection of slow nodes."""
    # Create profiler with lower threshold for testing
    test_profiler = WorkflowProfiler(slow_threshold_multiplier=1.5)

    node_stats = {
        "fast_node": NodeStatistics(
            node_id="fast_node",
            avg_duration=1.0,
            p50_duration=1.0,
            p95_duration=1.5,
            p99_duration=2.0,
            success_rate=1.0,
            total_cost=0.001,
        ),
        "slow_node": NodeStatistics(
            node_id="slow_node",
            avg_duration=12.0,  # > 1.5x median (6.5), will trigger slow node detection
            p50_duration=12.0,
            p95_duration=14.0,
            p99_duration=16.0,
            success_rate=1.0,
            total_cost=0.01,
        ),
    }

    bottlenecks = test_profiler._detect_bottlenecks(node_stats)

    # Should detect slow node
    slow_bottlenecks = [b for b in bottlenecks if b.type == BottleneckType.SLOW_NODE]
    assert len(slow_bottlenecks) > 0
    assert any(b.node_id == "slow_node" for b in slow_bottlenecks)


def test_detect_bottlenecks_failing_nodes(profiler):
    """Test detection of failing nodes."""
    node_stats = {
        "reliable_node": NodeStatistics(
            node_id="reliable_node",
            avg_duration=2.0,
            p50_duration=2.0,
            p95_duration=2.5,
            p99_duration=3.0,
            success_rate=0.95,
            total_cost=0.002,
        ),
        "unreliable_node": NodeStatistics(
            node_id="unreliable_node",
            avg_duration=2.0,
            p50_duration=2.0,
            p95_duration=2.5,
            p99_duration=3.0,
            success_rate=0.5,  # Below threshold
            total_cost=0.002,
        ),
    }

    bottlenecks = profiler._detect_bottlenecks(node_stats)

    # Should detect unreliable node
    unreliable_bottlenecks = [b for b in bottlenecks if b.type == BottleneckType.UNRELIABLE_NODE]
    assert len(unreliable_bottlenecks) > 0
    assert any(b.node_id == "unreliable_node" for b in unreliable_bottlenecks)


def test_detect_bottlenecks_expensive_tools(profiler):
    """Test detection of expensive tools."""
    node_stats = {
        "node_with_expensive_tool": NodeStatistics(
            node_id="node_with_expensive_tool",
            avg_duration=2.0,
            p50_duration=2.0,
            p95_duration=2.5,
            p99_duration=3.0,
            success_rate=1.0,
            total_cost=0.1,  # High cost
            tool_calls={
                "expensive_tool": {
                    "count": 1,
                    "cost": 0.09,  # 90% of total
                },
            },
        ),
    }

    bottlenecks = profiler._detect_bottlenecks(node_stats)

    # Should detect expensive tool
    expensive_bottlenecks = [b for b in bottlenecks if b.type == BottleneckType.EXPENSIVE_TOOL]
    assert len(expensive_bottlenecks) > 0
    assert any(b.tool_id == "expensive_tool" for b in expensive_bottlenecks)


def test_generate_opportunities(profiler):
    """Test generation of optimization opportunities."""
    bottlenecks = [
        Mock(
            type=BottleneckType.UNRELIABLE_NODE,
            node_id="failing_node",
            tool_id=None,
            severity=BottleneckSeverity.HIGH,
            value=50.0,
        ),
        Mock(
            type=BottleneckType.EXPENSIVE_TOOL,
            node_id=None,
            tool_id="expensive_tool",
            severity=BottleneckSeverity.MEDIUM,
            value=20.0,
        ),
    ]

    node_stats = {
        "failing_node": NodeStatistics(
            node_id="failing_node",
            avg_duration=2.0,
            p50_duration=2.0,
            p95_duration=2.5,
            p99_duration=3.0,
            success_rate=0.5,
            total_cost=0.01,
        ),
    }

    opportunities = profiler._generate_opportunities(bottlenecks, node_stats)

    assert len(opportunities) > 0
    # Should have opportunities for both bottlenecks
    assert any(opp.target == "failing_node" for opp in opportunities)
