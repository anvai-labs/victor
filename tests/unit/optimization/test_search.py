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

"""Unit tests for search algorithms."""

import pytest
from unittest.mock import Mock, AsyncMock, patch

from victor.optimization.search import (
    HillClimbingOptimizer,
    SimulatedAnnealingOptimizer,
    OptimizationResult,
)
from victor.optimization.models import (
    OptimizationOpportunity,
    OptimizationStrategyType,
    BottleneckSeverity,
    WorkflowProfile,
    NodeStatistics,
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
    }

    return WorkflowProfile(
        workflow_id="test_workflow",
        node_stats=node_stats,
        bottlenecks=[],
        opportunities=[],
        total_duration=10.0,
        total_cost=0.01,
        total_tokens=5000,
        success_rate=0.9,
    )


@pytest.fixture
def sample_opportunities():
    """Create sample optimization opportunities."""
    return [
        OptimizationOpportunity(
            strategy_type=OptimizationStrategyType.PRUNING,
            target="node_1",
            description="Remove slow node",
            expected_improvement=0.2,
            risk_level=BottleneckSeverity.MEDIUM,
            estimated_cost_reduction=0.005,
            estimated_duration_reduction=5.0,
            confidence=0.8,
        ),
        OptimizationOpportunity(
            strategy_type=OptimizationStrategyType.PARALLELIZATION,
            target="node_2,node_3",
            description="Parallelize independent nodes",
            expected_improvement=0.3,
            risk_level=BottleneckSeverity.LOW,
            estimated_duration_reduction=3.0,
            confidence=0.7,
        ),
    ]


@pytest.fixture
def sample_config():
    """Create sample workflow config."""
    return {
        "nodes": {
            "node_1": {
                "type": "agent",
                "goal": "Analyze data",
            },
            "node_2": {
                "type": "agent",
                "goal": "Process results",
            },
        },
        "edges": [
            {"source": "node_1", "target": "node_2"},
        ],
    }


class TestHillClimbingOptimizer:
    """Tests for HillClimbingOptimizer."""

    @pytest.fixture
    def optimizer(self):
        return HillClimbingOptimizer()

    @pytest.mark.asyncio
    async def test_optimize_workflow_success(
        self,
        optimizer,
        sample_config,
        sample_profile,
        sample_opportunities,
    ):
        """Test successful optimization."""
        with patch.object(
            optimizer.variant_generator,
            "generate_variant",
            new=AsyncMock(),
        ) as mock_generate:
            # Mock variant generation
            from victor.optimization.generator import WorkflowVariant

            mock_variant = WorkflowVariant(
                variant_id="test_variant",
                base_workflow_id="test_workflow",
                changes=[],
                expected_improvement=0.2,
                risk_level="medium",
                config=sample_config,
            )
            mock_generate.return_value = mock_variant

            result = await optimizer.optimize_workflow(
                workflow_config=sample_config,
                profile=sample_profile,
                opportunities=sample_opportunities,
                max_iterations=10,
            )

            assert result is not None
            assert isinstance(result, OptimizationResult)
            assert result.iterations > 0
            assert len(result.score_history) > 0

    @pytest.mark.asyncio
    async def test_optimize_workflow_no_opportunities(
        self,
        optimizer,
        sample_config,
        sample_profile,
    ):
        """Test optimization with no opportunities."""
        result = await optimizer.optimize_workflow(
            workflow_config=sample_config,
            profile=sample_profile,
            opportunities=[],
            max_iterations=10,
        )

        assert result is not None
        # Should complete after first iteration since no neighbors
        assert result.iterations >= 0

    def test_evaluate_config_default_scoring(self, optimizer, sample_profile):
        """Test default config evaluation."""
        score = optimizer._evaluate_config(
            config={},
            profile=sample_profile,
            score_function=None,
        )

        assert score > 0
        assert isinstance(score, float)

    def test_evaluate_config_custom_scoring(self, optimizer, sample_profile):
        """Test custom scoring function."""

        def custom_score(variant):
            return 42.0

        score = optimizer._evaluate_config(
            config={},
            profile=sample_profile,
            score_function=custom_score,
        )

        assert score == 42.0


class TestSimulatedAnnealingOptimizer:
    """Tests for SimulatedAnnealingOptimizer."""

    @pytest.fixture
    def optimizer(self):
        return SimulatedAnnealingOptimizer(
            initial_temperature=10.0,
            cooling_rate=0.9,
            min_temperature=0.5,
        )

    @pytest.mark.asyncio
    async def test_optimize_workflow_success(
        self,
        optimizer,
        sample_config,
        sample_profile,
        sample_opportunities,
    ):
        """Test successful optimization."""
        with patch.object(
            optimizer.variant_generator,
            "generate_variant",
            new=AsyncMock(),
        ) as mock_generate:
            # Mock variant generation
            from victor.optimization.generator import WorkflowVariant

            mock_variant = WorkflowVariant(
                variant_id="test_variant",
                base_workflow_id="test_workflow",
                changes=[],
                expected_improvement=0.2,
                risk_level="medium",
                config=sample_config,
            )
            mock_generate.return_value = mock_variant

            result = await optimizer.optimize_workflow(
                workflow_config=sample_config,
                profile=sample_profile,
                opportunities=sample_opportunities,
            )

            assert result is not None
            assert isinstance(result, OptimizationResult)
            assert result.iterations > 0
            assert len(result.score_history) > 0

    @pytest.mark.asyncio
    async def test_optimize_workflow_no_opportunities(
        self,
        optimizer,
        sample_config,
        sample_profile,
    ):
        """Test optimization with no opportunities."""
        result = await optimizer.optimize_workflow(
            workflow_config=sample_config,
            profile=sample_profile,
            opportunities=[],
        )

        assert result is not None
        # Should finish quickly with no opportunities
        assert result.iterations == 0


def test_optimization_result_to_dict():
    """Test OptimizationResult serialization."""
    from victor.optimization.generator import WorkflowVariant

    variant = WorkflowVariant(
        variant_id="test_variant",
        base_workflow_id="test_workflow",
        changes=[],
        expected_improvement=0.2,
        risk_level="medium",
    )

    result = OptimizationResult(
        best_variant=variant,
        best_score=0.85,
        iterations=20,
        converged=True,
        score_history=[0.5, 0.6, 0.75, 0.85],
    )

    result_dict = result.to_dict()

    assert result_dict["best_variant_id"] == "test_variant"
    assert result_dict["best_score"] == 0.85
    assert result_dict["iterations"] == 20
    assert result_dict["converged"] is True
    assert len(result_dict["score_history"]) == 4
