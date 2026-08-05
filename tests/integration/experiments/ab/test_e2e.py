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

"""Integration tests for A/B testing."""

import asyncio
import tempfile
from pathlib import Path

import pytest

from victor.experiments.ab_testing import (
    ABTestManager,
    AllocationStrategy,
    ExperimentConfig,
    ExperimentMetric,
    ExecutionMetrics,
    ExperimentVariant,
    MetricsCollector,
    StatisticalAnalyzer,
    WorkflowInterceptor,
)


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".db", delete=False) as f:
        db_path = f.name

    yield db_path

    # Cleanup
    Path(db_path).unlink(missing_ok=True)


@pytest.fixture
def sample_experiment_config():
    """Create a sample experiment configuration."""
    control = ExperimentVariant(
        variant_id="control",
        name="Control Variant",
        description="Baseline configuration",
        workflow_type="yaml",
        workflow_config={"path": "test_workflow.yaml"},
        parameter_overrides={"model": "claude-sonnet-3-5", "tool_budget": 10},
        traffic_weight=0.5,
        is_control=True,
    )

    treatment = ExperimentVariant(
        variant_id="treatment",
        name="Treatment Variant",
        description="Experimental configuration",
        workflow_type="yaml",
        workflow_config={"path": "test_workflow.yaml"},
        parameter_overrides={"model": "claude-opus-4-5", "tool_budget": 20},
        traffic_weight=0.5,
        is_control=False,
    )

    primary_metric = ExperimentMetric(
        metric_id="execution_time",
        name="Execution Time",
        description="Total workflow execution time",
        metric_type="execution_time",
        optimization_goal="minimize",
        relative_improvement=0.1,
    )

    return ExperimentConfig(
        name="Model Comparison Test",
        description="Compare Claude Sonnet vs Opus",
        hypothesis="Opus will be 10% faster",
        variants=[control, treatment],
        primary_metric=primary_metric,
        min_sample_size=10,
        significance_level=0.05,
    )


@pytest.mark.asyncio
class TestABTestManager:
    """Integration tests for ABTestManager."""

    async def test_create_and_start_experiment(self, temp_db, sample_experiment_config):
        """Test creating and starting an experiment."""
        manager = ABTestManager(storage_path=temp_db)

        # Create experiment
        experiment_id = await manager.create_experiment(sample_experiment_config)
        assert experiment_id is not None

        # Load experiment
        loaded = await manager.get_experiment(experiment_id)
        assert loaded is not None
        assert loaded.name == "Model Comparison Test"
        assert len(loaded.variants) == 2

        # Start experiment
        await manager.start_experiment(experiment_id)

        # Check status
        status = await manager.get_status(experiment_id)
        assert status.status == "running"
        assert status.started_at is not None

    async def test_allocate_variants(self, temp_db, sample_experiment_config):
        """Test variant allocation."""
        manager = ABTestManager(storage_path=temp_db)

        # Create and start experiment
        experiment_id = await manager.create_experiment(sample_experiment_config)
        await manager.start_experiment(experiment_id)

        # Allocate users
        allocations = {}
        for i in range(100):
            user_id = f"user_{i}"
            variant_id = await manager.allocate_variant(experiment_id, user_id)
            allocations[variant_id] = allocations.get(variant_id, 0) + 1

        # Both variants should be allocated
        assert "control" in allocations
        assert "treatment" in allocations

        # Total should be 100
        assert sum(allocations.values()) == 100

    async def test_sticky_allocation_consistency(self, temp_db, sample_experiment_config):
        """Test that sticky allocation is consistent."""
        manager = ABTestManager(storage_path=temp_db, allocation_strategy=AllocationStrategy.STICKY)

        # Create and start experiment
        experiment_id = await manager.create_experiment(sample_experiment_config)
        await manager.start_experiment(experiment_id)

        # Allocate same user multiple times
        variant1 = await manager.allocate_variant(experiment_id, "user_123")
        variant2 = await manager.allocate_variant(experiment_id, "user_123")
        variant3 = await manager.allocate_variant(experiment_id, "user_123")

        # Should always get same variant
        assert variant1 == variant2 == variant3

    async def test_record_execution(self, temp_db, sample_experiment_config):
        """Test recording execution metrics."""
        manager = ABTestManager(storage_path=temp_db)

        # Create and start experiment
        experiment_id = await manager.create_experiment(sample_experiment_config)
        await manager.start_experiment(experiment_id)

        # Record some executions
        for i in range(10):
            metrics = ExecutionMetrics(
                execution_id=f"exec_{i}",
                experiment_id=experiment_id,
                variant_id="control",
                user_id=f"user_{i}",
                execution_time=10.0 + i * 0.1,
                total_tokens=1000 + i * 10,
                tool_calls_count=5,
                success=True,
                estimated_cost=0.01,
            )
            await manager.record_execution(metrics)

        # Check status updated
        status = await manager.get_status(experiment_id)
        assert status.total_samples == 10
        assert status.variant_samples.get("control") == 10


@pytest.mark.asyncio
class TestMetricsCollector:
    """Integration tests for MetricsCollector."""

    async def test_collect_and_aggregate_metrics(self, temp_db, sample_experiment_config):
        """Test collecting and aggregating metrics."""
        manager = ABTestManager(storage_path=temp_db)
        collector = MetricsCollector(storage_path=temp_db)

        # Create and start experiment
        experiment_id = await manager.create_experiment(sample_experiment_config)
        await manager.start_experiment(experiment_id)

        # Record executions for both variants
        for i in range(20):
            variant_id = "control" if i < 10 else "treatment"
            metrics = ExecutionMetrics(
                execution_id=f"exec_{i}",
                experiment_id=experiment_id,
                variant_id=variant_id,
                user_id=f"user_{i}",
                execution_time=10.0 if variant_id == "control" else 8.0,
                total_tokens=1000,
                tool_calls_count=5,
                success=True,
                estimated_cost=0.01,
            )
            await manager.record_execution(metrics)

        # Get aggregated metrics
        control_metrics = await collector.get_variant_metrics(experiment_id, "control")
        treatment_metrics = await collector.get_variant_metrics(experiment_id, "treatment")

        # Check control metrics
        assert control_metrics is not None
        assert control_metrics.sample_count == 10
        assert control_metrics.execution_time_mean == 10.0

        # Check treatment metrics
        assert treatment_metrics is not None
        assert treatment_metrics.sample_count == 10
        assert treatment_metrics.execution_time_mean == 8.0

        # Get all metrics
        all_metrics = await collector.get_all_variant_metrics(experiment_id)
        assert len(all_metrics) == 2
        assert "control" in all_metrics
        assert "treatment" in all_metrics


@pytest.mark.asyncio
class TestStatisticalAnalysis:
    """Integration tests for statistical analysis."""

    async def test_full_experiment_analysis(self, temp_db, sample_experiment_config):
        """Test complete experiment with statistical analysis."""
        manager = ABTestManager(storage_path=temp_db)
        collector = MetricsCollector(storage_path=temp_db)
        analyzer = StatisticalAnalyzer()

        # Create and start experiment
        experiment_id = await manager.create_experiment(sample_experiment_config)
        await manager.start_experiment(experiment_id)

        # Simulate executions - treatment is faster
        import random

        random.seed(42)
        for i in range(50):
            variant_id = "control" if i < 25 else "treatment"

            # Treatment is consistently faster
            if variant_id == "control":
                execution_time = 10.0 + random.gauss(0, 1.0)
            else:
                execution_time = 8.0 + random.gauss(0, 1.0)

            metrics = ExecutionMetrics(
                execution_id=f"exec_{i}",
                experiment_id=experiment_id,
                variant_id=variant_id,
                user_id=f"user_{i}",
                execution_time=execution_time,
                total_tokens=1000,
                tool_calls_count=5,
                success=True,
                estimated_cost=0.01,
            )
            await manager.record_execution(metrics)

        # Stop experiment
        await manager.stop_experiment(experiment_id)

        # Get metrics
        control_metrics = await collector.get_variant_metrics(experiment_id, "control")
        treatment_metrics = await collector.get_variant_metrics(experiment_id, "treatment")

        # Perform statistical test
        control_data = [10.0 + random.gauss(0, 1.0) for _ in range(25)]
        treatment_data = [8.0 + random.gauss(0, 1.0) for _ in range(25)]

        result = analyzer.compare_means(control_data, treatment_data, alpha=0.05)

        # Treatment should be significantly faster
        assert result["treatment_mean"] < result["control_mean"]
        # With this effect size, should be significant
        assert result["significant"] is True


@pytest.mark.asyncio
class TestWorkflowInterceptor:
    """Integration tests for WorkflowInterceptor."""

    async def test_interceptor_allocation(self, temp_db, sample_experiment_config):
        """Test that interceptor allocates variants."""
        manager = ABTestManager(storage_path=temp_db)
        interceptor = WorkflowInterceptor(manager)

        # Create and start experiment
        experiment_id = await manager.create_experiment(sample_experiment_config)
        await manager.start_experiment(experiment_id)

        # Mock workflow function
        async def mock_workflow(*args, **kwargs):
            class MockResult:
                duration_seconds = 10.0

            return MockResult()

        # Execute with experiment
        result = await interceptor.execute_with_experiment(
            workflow_func=mock_workflow,
            experiment_id=experiment_id,
            user_id="user_123",
            context={},
        )

        # Should have executed
        assert result is not None

        # Check execution was recorded
        status = await manager.get_status(experiment_id)
        assert status.total_samples == 1

    async def test_interceptor_no_experiment_runs_normally(self, temp_db):
        """Test that interceptor runs workflow normally without experiment."""
        manager = ABTestManager(storage_path=temp_db)
        interceptor = WorkflowInterceptor(manager)

        # Mock workflow function
        async def mock_workflow(*args, **kwargs):
            class MockResult:
                duration_seconds = 10.0

            return MockResult()

        # Execute without experiment
        result = await interceptor.execute_with_experiment(
            workflow_func=mock_workflow,
            experiment_id="nonexistent",
            user_id="user_123",
            context={},
        )

        # Should have executed normally
        assert result is not None


@pytest.mark.asyncio
class TestEndToEndWorkflow:
    """End-to-end integration tests."""

    async def test_complete_experiment_lifecycle(self, temp_db):
        """Test complete experiment from creation to analysis."""
        manager = ABTestManager(storage_path=temp_db)
        collector = MetricsCollector(storage_path=temp_db)
        analyzer = StatisticalAnalyzer()

        # 1. Create experiment
        config = ExperimentConfig(
            name="E2E Test",
            hypothesis="Treatment is better",
            variants=[
                ExperimentVariant(
                    variant_id="control",
                    name="Control",
                    workflow_type="yaml",
                    workflow_config={},
                    traffic_weight=0.5,
                    is_control=True,
                ),
                ExperimentVariant(
                    variant_id="treatment",
                    name="Treatment",
                    workflow_type="yaml",
                    workflow_config={},
                    traffic_weight=0.5,
                ),
            ],
            primary_metric=ExperimentMetric(
                metric_id="time",
                name="Time",
                metric_type="execution_time",
            ),
            min_sample_size=10,
        )

        experiment_id = await manager.create_experiment(config)
        assert experiment_id is not None

        # 2. Start experiment
        await manager.start_experiment(experiment_id)
        status = await manager.get_status(experiment_id)
        assert status.status == "running"

        # 3. Allocate and record
        for i in range(20):
            variant_id = await manager.allocate_variant(experiment_id, f"user_{i}")

            metrics = ExecutionMetrics(
                execution_id=f"exec_{i}",
                experiment_id=experiment_id,
                variant_id=variant_id,
                user_id=f"user_{i}",
                execution_time=10.0 if variant_id == "control" else 9.0,
                total_tokens=1000,
                tool_calls_count=5,
                success=True,
                estimated_cost=0.01,
            )
            await manager.record_execution(metrics)

        # 4. Check collection
        all_metrics = await collector.get_all_variant_metrics(experiment_id)
        assert len(all_metrics) == 2
        assert all_metrics["control"].sample_count > 0
        assert all_metrics["treatment"].sample_count > 0

        # 5. Stop experiment
        await manager.stop_experiment(experiment_id)
        status = await manager.get_status(experiment_id)
        assert status.status == "completed"

        # 6. Analyze results
        control_data = [10.0] * all_metrics["control"].sample_count
        treatment_data = [9.0] * all_metrics["treatment"].sample_count

        result = analyzer.compare_means(control_data, treatment_data)
        assert result["test"] == "t-test"
        assert result["treatment_mean"] < result["control_mean"]
