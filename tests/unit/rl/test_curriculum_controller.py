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

"""Unit tests for CurriculumController.

Tests the curriculum learning controller for progressive complexity.
"""

import pytest

from victor.framework.rl.curriculum_controller import (
    CurriculumController,
    CurriculumStage,
    StageConfig,
    StageMetrics,
)


@pytest.fixture
def controller() -> CurriculumController:
    """Fixture for CurriculumController without database."""
    return CurriculumController()


class TestStageMetrics:
    """Tests for StageMetrics."""

    def test_success_rate_empty(self) -> None:
        """Test success rate with no samples."""
        metrics = StageMetrics(stage=CurriculumStage.WARM_UP)
        assert metrics.success_rate == 0.0

    def test_success_rate_calculation(self) -> None:
        """Test success rate calculation."""
        metrics = StageMetrics(
            stage=CurriculumStage.WARM_UP,
            sample_count=10,
            success_count=8,
        )
        assert metrics.success_rate == 0.8

    def test_update_increments_counts(self) -> None:
        """Test that update increments counts."""
        metrics = StageMetrics(stage=CurriculumStage.WARM_UP)

        metrics.update(success=True, quality=0.8, iterations=5, tools_used=3)

        assert metrics.sample_count == 1
        assert metrics.success_count == 1
        assert metrics.avg_quality == 0.8

    def test_update_running_averages(self) -> None:
        """Test running averages in update."""
        metrics = StageMetrics(stage=CurriculumStage.WARM_UP)

        metrics.update(success=True, quality=0.8, iterations=4, tools_used=2)
        metrics.update(success=False, quality=0.6, iterations=6, tools_used=4)

        assert metrics.sample_count == 2
        assert metrics.success_count == 1
        assert metrics.avg_quality == pytest.approx(0.7, abs=0.01)
        assert metrics.avg_iterations == pytest.approx(5.0, abs=0.01)
        assert metrics.avg_tools_used == pytest.approx(3.0, abs=0.01)


class TestCurriculumController:
    """Tests for CurriculumController."""

    def test_default_stage_is_warmup(self, controller: CurriculumController) -> None:
        """Test that unknown contexts start at warm-up."""
        stage = controller.get_stage("unknown:context:key")
        assert stage == CurriculumStage.WARM_UP

    def test_get_stage_config(self, controller: CurriculumController) -> None:
        """Test getting stage configuration."""
        config = controller.get_stage_config("test:context:key")

        assert config.stage == CurriculumStage.WARM_UP
        assert config.max_tools == 3
        assert config.max_iterations == 5

    def test_get_constraints(self, controller: CurriculumController) -> None:
        """Test getting constraints for a context."""
        constraints = controller.get_constraints("test:context:key")

        assert "max_tools" in constraints
        assert "max_iterations" in constraints
        assert "tool_budget" in constraints
        assert constraints["stage"] == CurriculumStage.WARM_UP.value

    def test_is_task_allowed_warmup(self, controller: CurriculumController) -> None:
        """Test task type checking at warm-up stage."""
        context = "test:context:key"

        assert controller.is_task_allowed(context, "search") is True
        assert controller.is_task_allowed(context, "read") is True
        assert controller.is_task_allowed(context, "analysis") is True
        # Edit not allowed at warm-up by default
        assert controller.is_task_allowed(context, "edit") is False

    def test_record_outcome_creates_metrics(self, controller: CurriculumController) -> None:
        """Test that recording outcome creates metrics."""
        context = "test:context:key"

        controller.record_outcome(context, success=True, quality=0.8, iterations=3, tools_used=2)

        stage = controller.get_stage(context)
        metrics = controller._metrics.get(context, {}).get(stage)

        assert metrics is not None
        assert metrics.sample_count == 1

    def test_no_advancement_insufficient_samples(self, controller: CurriculumController) -> None:
        """Test no advancement with insufficient samples."""
        context = "test:context:key"

        # Record a few successful outcomes
        for _ in range(5):
            controller.record_outcome(
                context, success=True, quality=0.9, iterations=3, tools_used=2
            )

        # Should not advance (need min_samples, default 10 for warm-up)
        assert controller.get_stage(context) == CurriculumStage.WARM_UP

    def test_advancement_with_sufficient_samples(self, controller: CurriculumController) -> None:
        """Test advancement with sufficient samples and success rate."""
        context = "test:context:key"

        # Record enough successful outcomes to advance
        config = controller.get_stage_config(context)
        for _ in range(config.min_samples + 5):
            controller.record_outcome(
                context, success=True, quality=0.9, iterations=3, tools_used=2
            )

        # Should have advanced to BASIC
        assert controller.get_stage(context) == CurriculumStage.BASIC

    def test_no_advancement_low_success_rate(self, controller: CurriculumController) -> None:
        """Test no advancement with low success rate."""
        context = "test:context:key"

        # Record mix of outcomes with <70% success
        for i in range(20):
            controller.record_outcome(
                context,
                success=i % 3 == 0,  # ~33% success
                quality=0.5,
                iterations=3,
                tools_used=2,
            )

        # Should not advance
        assert controller.get_stage(context) == CurriculumStage.WARM_UP

    def test_regression_on_poor_recent_performance(self, controller: CurriculumController) -> None:
        """Test regression when recent performance is poor."""
        context = "test:context:key"

        # Manually set to BASIC stage
        controller._context_stages[context] = CurriculumStage.BASIC
        controller._metrics[context] = {
            CurriculumStage.BASIC: StageMetrics(stage=CurriculumStage.BASIC)
        }

        # Record many failures (triggers regression)
        for _ in range(controller.REGRESSION_WINDOW + 2):
            controller.record_outcome(
                context, success=False, quality=0.3, iterations=5, tools_used=4
            )

        # Should have regressed to WARM_UP
        assert controller.get_stage(context) == CurriculumStage.WARM_UP

    def test_no_regression_below_warmup(self, controller: CurriculumController) -> None:
        """Test that we can't regress below warm-up."""
        context = "test:context:key"

        # Record many failures
        for _ in range(20):
            controller.record_outcome(
                context, success=False, quality=0.2, iterations=5, tools_used=4
            )

        # Should still be at WARM_UP (can't go lower)
        assert controller.get_stage(context) == CurriculumStage.WARM_UP

    def test_progress_summary(self, controller: CurriculumController) -> None:
        """Test progress summary."""
        context = "test:context:key"

        # Record some outcomes
        for i in range(5):
            controller.record_outcome(
                context, success=True, quality=0.8, iterations=3, tools_used=2
            )

        summary = controller.get_progress_summary(context)

        assert summary["context_key"] == context
        assert summary["current_stage"] == CurriculumStage.WARM_UP.value
        assert summary["sample_count"] == 5
        assert summary["success_rate"] == 1.0
        assert "progress_to_next" in summary

    def test_export_metrics(self, controller: CurriculumController) -> None:
        """Test metrics export."""
        controller.record_outcome("ctx1", success=True, quality=0.8, iterations=3, tools_used=2)
        controller.record_outcome("ctx2", success=True, quality=0.7, iterations=4, tools_used=3)

        metrics = controller.export_metrics()

        assert metrics["total_contexts"] == 2
        assert "stage_distribution" in metrics
        assert metrics["stage_distribution"]["WARM_UP"] == 2

    def test_expert_stage_allows_all_tasks(self, controller: CurriculumController) -> None:
        """Test that expert stage allows all task types."""
        context = "expert:context:key"
        controller._context_stages[context] = CurriculumStage.EXPERT

        assert controller.is_task_allowed(context, "any_task") is True
        assert controller.is_task_allowed(context, "debug") is True
        assert controller.is_task_allowed(context, "action") is True

    def test_stage_configs_have_increasing_limits(self, controller: CurriculumController) -> None:
        """Test that stage configs have increasing complexity limits."""
        prev_tools = 0
        prev_iters = 0

        for stage in CurriculumStage:
            config = controller.stages[stage]
            assert config.max_tools >= prev_tools
            assert config.max_iterations >= prev_iters
            prev_tools = config.max_tools
            prev_iters = config.max_iterations
