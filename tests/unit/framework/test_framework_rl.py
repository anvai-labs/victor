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

"""Tests for victor.framework.rl module."""

import pytest
from unittest.mock import MagicMock, patch

from victor.framework.rl import (
    analyze_prompt_rollout_experiment,
    analyze_prompt_rollout_experiment_async,
    apply_prompt_rollout_recommendation,
    apply_prompt_rollout_recommendation_async,
    process_prompt_candidate_evaluation_suite,
    process_prompt_candidate_evaluation_suite_async,
    create_prompt_rollout_experiment,
    create_prompt_rollout_experiment_async,
    LearnerType,
    LearnerStats,
    RLStats,
    RLManager,
    RLOutcome,
    RLRecommendation,
    create_outcome,
    record_tool_success,
    get_rl_coordinator,
    get_rl_coordinator_async,
)


class TestLearnerType:
    """Tests for LearnerType enum."""

    def test_tool_selector(self):
        """Test TOOL_SELECTOR value."""
        assert LearnerType.TOOL_SELECTOR.value == "tool_selector"

    def test_continuation_patience(self):
        """Test CONTINUATION_PATIENCE value."""
        assert LearnerType.CONTINUATION_PATIENCE.value == "continuation_patience"

    def test_grounding_threshold(self):
        """Test GROUNDING_THRESHOLD value."""
        assert LearnerType.GROUNDING_THRESHOLD.value == "grounding_threshold"

    def test_model_selector(self):
        """Test MODEL_SELECTOR value."""
        assert LearnerType.MODEL_SELECTOR.value == "model_selector"

    def test_all_learner_types(self):
        """Test all learner types have string values."""
        for learner in LearnerType:
            assert isinstance(learner.value, str)
            assert len(learner.value) > 0


class TestLearnerStats:
    """Tests for LearnerStats dataclass."""

    def test_default_values(self):
        """Test default values."""
        stats = LearnerStats(name="test")
        assert stats.name == "test"
        assert stats.total_records == 0
        assert stats.success_rate == 0.0
        assert stats.last_updated is None
        assert stats.parameters == {}

    def test_with_values(self):
        """Test with custom values."""
        stats = LearnerStats(
            name="tool_selector",
            total_records=100,
            success_rate=0.85,
            last_updated=1234567890.0,
            parameters={"learning_rate": 0.1},
        )
        assert stats.name == "tool_selector"
        assert stats.total_records == 100
        assert stats.success_rate == 0.85
        assert stats.last_updated == 1234567890.0
        assert stats.parameters == {"learning_rate": 0.1}


class TestRLStats:
    """Tests for RLStats dataclass."""

    def test_default_values(self):
        """Test default values."""
        stats = RLStats()
        assert stats.total_outcomes == 0
        assert stats.active_learners == 0
        assert stats.learner_stats == {}
        assert stats.database_path is None

    def test_with_learner_stats(self):
        """Test with learner statistics."""
        learner_stat = LearnerStats(name="test", total_records=50)
        stats = RLStats(
            total_outcomes=50,
            active_learners=1,
            learner_stats={"test": learner_stat},
            database_path="/path/to/db",
        )
        assert stats.total_outcomes == 50
        assert stats.active_learners == 1
        assert "test" in stats.learner_stats
        assert stats.database_path == "/path/to/db"


class TestRLManager:
    """Tests for RLManager class."""

    @pytest.fixture
    def mock_coordinator(self):
        """Create mock RLCoordinator."""
        coordinator = MagicMock()
        coordinator.record_outcome = MagicMock()
        coordinator.get_recommendation = MagicMock(return_value=None)
        coordinator.list_learners = MagicMock(return_value=["tool_selector"])
        return coordinator

    def test_init_default(self):
        """Test initialization with default coordinator."""
        with patch("victor.framework.rl.get_rl_coordinator") as mock_get:
            mock_coordinator = MagicMock()
            mock_get.return_value = mock_coordinator
            manager = RLManager()
            assert manager.coordinator is mock_coordinator

    def test_init_with_coordinator(self):
        """Test initialization with custom coordinator."""
        mock_coordinator = MagicMock()
        manager = RLManager(coordinator=mock_coordinator)
        assert manager.coordinator is mock_coordinator

    def test_from_agent(self):
        """Test from_agent class method."""
        with patch("victor.framework.rl.get_rl_coordinator") as mock_get:
            mock_coordinator = MagicMock()
            mock_get.return_value = mock_coordinator
            mock_agent = MagicMock()
            manager = RLManager.from_agent(mock_agent)
            assert manager is not None

    def test_from_orchestrator(self):
        """Test from_orchestrator class method."""
        with patch("victor.framework.rl.get_rl_coordinator") as mock_get:
            mock_coordinator = MagicMock()
            mock_get.return_value = mock_coordinator
            mock_orch = MagicMock()
            manager = RLManager.from_orchestrator(mock_orch)
            assert manager is not None

    def test_record_outcome(self, mock_coordinator):
        """Test record_outcome method."""
        manager = RLManager(coordinator=mock_coordinator)
        outcome = RLOutcome(
            provider="anthropic",
            model="claude-3",
            task_type="analysis",
            success=True,
            quality_score=0.9,
        )
        manager.record_outcome(LearnerType.TOOL_SELECTOR, outcome, vertical="coding")
        mock_coordinator.record_outcome.assert_called_once_with(
            learner_name="tool_selector",
            outcome=outcome,
            vertical="coding",
        )

    def test_record_outcome_string_learner(self, mock_coordinator):
        """Test record_outcome with string learner name."""
        manager = RLManager(coordinator=mock_coordinator)
        outcome = RLOutcome(
            provider="anthropic",
            model="claude-3",
            task_type="analysis",
            success=True,
            quality_score=0.9,
        )
        manager.record_outcome("custom_learner", outcome)
        mock_coordinator.record_outcome.assert_called_once()
        call_args = mock_coordinator.record_outcome.call_args
        assert call_args.kwargs["learner_name"] == "custom_learner"

    def test_record_success(self, mock_coordinator):
        """Test record_success convenience method."""
        manager = RLManager(coordinator=mock_coordinator)
        manager.record_success(
            LearnerType.TOOL_SELECTOR,
            provider="anthropic",
            model="claude-3",
            task_type="analysis",
            quality_score=0.95,
            metadata={"tool": "code_search"},
        )
        mock_coordinator.record_outcome.assert_called_once()
        call_args = mock_coordinator.record_outcome.call_args
        outcome = call_args.kwargs["outcome"]
        assert outcome.success is True
        assert outcome.provider == "anthropic"
        assert outcome.model == "claude-3"
        assert outcome.quality_score == 0.95
        assert outcome.metadata == {"tool": "code_search"}

    def test_record_failure(self, mock_coordinator):
        """Test record_failure convenience method."""
        manager = RLManager(coordinator=mock_coordinator)
        manager.record_failure(
            LearnerType.TOOL_SELECTOR,
            provider="anthropic",
            model="claude-3",
            task_type="analysis",
            error="Tool not found",
        )
        mock_coordinator.record_outcome.assert_called_once()
        call_args = mock_coordinator.record_outcome.call_args
        outcome = call_args.kwargs["outcome"]
        assert outcome.success is False
        assert outcome.metadata["error"] == "Tool not found"

    def test_get_recommendation(self, mock_coordinator):
        """Test get_recommendation method."""
        mock_rec = MagicMock()
        mock_rec.value = 0.8
        mock_rec.confidence = 0.9
        mock_coordinator.get_recommendation.return_value = mock_rec
        manager = RLManager(coordinator=mock_coordinator)
        rec = manager.get_recommendation(
            LearnerType.CONTINUATION_PATIENCE,
            provider="anthropic",
            model="claude-3",
        )
        assert rec is mock_rec
        mock_coordinator.get_recommendation.assert_called_once()

    def test_get_recommendation_none(self, mock_coordinator):
        """Test get_recommendation returns None when no recommendation."""
        mock_coordinator.get_recommendation.return_value = None
        manager = RLManager(coordinator=mock_coordinator)
        rec = manager.get_recommendation(LearnerType.TOOL_SELECTOR)
        assert rec is None

    def test_get_tool_recommendation(self, mock_coordinator):
        """Test get_tool_recommendation convenience method."""
        mock_rec = MagicMock()
        mock_rec.value = ["code_search", "edit"]
        mock_rec.confidence = 0.8
        mock_coordinator.get_recommendation.return_value = mock_rec
        manager = RLManager(coordinator=mock_coordinator)
        tools = manager.get_tool_recommendation(
            task_type="analysis",
            available_tools=["code_search", "edit", "bash"],
        )
        assert tools == ["code_search", "edit"]

    def test_get_tool_recommendation_none(self, mock_coordinator):
        """Test get_tool_recommendation returns None when no rec."""
        mock_coordinator.get_recommendation.return_value = None
        manager = RLManager(coordinator=mock_coordinator)
        tools = manager.get_tool_recommendation(task_type="analysis")
        assert tools is None

    def test_get_patience_recommendation(self, mock_coordinator):
        """Test get_patience_recommendation convenience method."""
        mock_rec = MagicMock()
        mock_rec.value = 5
        mock_rec.confidence = 0.9
        mock_coordinator.get_recommendation.return_value = mock_rec
        manager = RLManager(coordinator=mock_coordinator)
        patience = manager.get_patience_recommendation(
            provider="deepseek",
            model="deepseek-chat",
        )
        assert patience == 5

    def test_get_patience_recommendation_float(self, mock_coordinator):
        """Test get_patience_recommendation with float value."""
        mock_rec = MagicMock()
        mock_rec.value = 5.7
        mock_rec.confidence = 0.9
        mock_coordinator.get_recommendation.return_value = mock_rec
        manager = RLManager(coordinator=mock_coordinator)
        patience = manager.get_patience_recommendation(
            provider="deepseek",
            model="deepseek-chat",
        )
        assert patience == 5  # Should be int

    def test_list_learners(self, mock_coordinator):
        """Test list_learners method."""
        manager = RLManager(coordinator=mock_coordinator)
        learners = manager.list_learners()
        assert "tool_selector" in learners

    def test_list_learners_fallback(self, mock_coordinator):
        """Test list_learners fallback to LearnerType enum."""
        del mock_coordinator.list_learners
        manager = RLManager(coordinator=mock_coordinator)
        learners = manager.list_learners()
        assert LearnerType.TOOL_SELECTOR.value in learners

    def test_repr(self, mock_coordinator):
        """Test __repr__ method."""
        manager = RLManager(coordinator=mock_coordinator)
        repr_str = repr(manager)
        assert "RLManager" in repr_str
        assert "learners=" in repr_str


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_create_outcome_success(self):
        """Test create_outcome with success."""
        outcome = create_outcome(
            success=True,
            provider="anthropic",
            model="claude-3",
            task_type="analysis",
            metadata={"tool": "test"},
        )
        assert outcome.success is True
        assert outcome.provider == "anthropic"
        assert outcome.model == "claude-3"
        assert outcome.quality_score == 1.0  # Default for success
        assert outcome.metadata == {"tool": "test"}

    def test_create_outcome_failure(self):
        """Test create_outcome with failure."""
        outcome = create_outcome(success=False)
        assert outcome.success is False
        assert outcome.quality_score == 0.0  # Default for failure
        assert outcome.metadata == {}

    def test_create_outcome_custom_quality(self):
        """Test create_outcome with custom quality score."""
        outcome = create_outcome(success=True, quality_score=0.75)
        assert outcome.success is True
        assert outcome.quality_score == 0.75

    def test_record_tool_success(self):
        """Test record_tool_success function."""
        with patch("victor.framework.rl.get_rl_coordinator") as mock_get:
            mock_coordinator = MagicMock()
            mock_get.return_value = mock_coordinator

            record_tool_success(
                tool_name="code_search",
                task_type="analysis",
                provider="anthropic",
                model="claude-3",
                duration_ms=100.0,
                vertical="coding",
            )

            mock_coordinator.record_outcome.assert_called_once()
            call_args = mock_coordinator.record_outcome.call_args
            assert call_args.kwargs["learner_name"] == "tool_selector"
            outcome = call_args.kwargs["outcome"]
            assert outcome.success is True
            assert outcome.task_type == "analysis"
            assert outcome.metadata["tool"] == "code_search"
            assert outcome.metadata["duration_ms"] == 100.0


class TestFrameworkExports:
    """Tests for framework module exports."""

    def test_rl_manager_exported(self):
        """Test RLManager is exported from framework."""
        from victor.framework import RLManager as ExportedRLManager

        assert ExportedRLManager is RLManager

    def test_learner_type_exported(self):
        """Test LearnerType is exported from framework."""
        from victor.framework import LearnerType as ExportedLearnerType

        assert ExportedLearnerType is LearnerType

    def test_rl_outcome_exported(self):
        """Test RLOutcome is exported from framework."""
        from victor.framework import RLOutcome as ExportedRLOutcome

        assert ExportedRLOutcome is RLOutcome

    def test_get_rl_coordinator_exported(self):
        """Test get_rl_coordinator is exported from framework."""
        from victor.framework import get_rl_coordinator as ExportedGetCoordinator

        assert ExportedGetCoordinator is get_rl_coordinator

    def test_get_rl_coordinator_async_exported(self):
        """Test get_rl_coordinator_async is exported from framework."""
        from victor.framework import (
            get_rl_coordinator_async as ExportedGetCoordinatorAsync,
        )

        assert ExportedGetCoordinatorAsync is get_rl_coordinator_async

    def test_create_outcome_exported(self):
        """Test create_outcome is exported from framework."""
        from victor.framework import create_outcome as ExportedCreateOutcome

        assert ExportedCreateOutcome is create_outcome

    def test_create_prompt_rollout_experiment_exported(self):
        """Test create_prompt_rollout_experiment is exported from framework."""
        from victor.framework import (
            create_prompt_rollout_experiment as ExportedCreatePromptRolloutExperiment,
        )

        assert ExportedCreatePromptRolloutExperiment is create_prompt_rollout_experiment

    def test_create_prompt_rollout_experiment_async_exported(self):
        """Test async prompt rollout helper is exported from framework."""
        from victor.framework import (
            create_prompt_rollout_experiment_async as ExportedCreatePromptRolloutExperimentAsync,
        )

        assert ExportedCreatePromptRolloutExperimentAsync is create_prompt_rollout_experiment_async

    def test_analyze_prompt_rollout_experiment_exported(self):
        """Test rollout analysis helper is exported from framework."""
        from victor.framework import (
            analyze_prompt_rollout_experiment as ExportedAnalyzePromptRolloutExperiment,
        )

        assert ExportedAnalyzePromptRolloutExperiment is analyze_prompt_rollout_experiment

    def test_analyze_prompt_rollout_experiment_async_exported(self):
        """Test async rollout analysis helper is exported from framework."""
        from victor.framework import (
            analyze_prompt_rollout_experiment_async as ExportedAnalyzePromptRolloutExperimentAsync,
        )

        assert (
            ExportedAnalyzePromptRolloutExperimentAsync is analyze_prompt_rollout_experiment_async
        )

    def test_apply_prompt_rollout_recommendation_exported(self):
        """Test rollout decision helper is exported from framework."""
        from victor.framework import (
            apply_prompt_rollout_recommendation as ExportedApplyPromptRolloutRecommendation,
        )

        assert ExportedApplyPromptRolloutRecommendation is apply_prompt_rollout_recommendation

    def test_apply_prompt_rollout_recommendation_async_exported(self):
        """Test async rollout decision helper is exported from framework."""
        from victor.framework import (
            apply_prompt_rollout_recommendation_async as ExportedApplyPromptRolloutRecommendationAsync,
        )

        assert (
            ExportedApplyPromptRolloutRecommendationAsync
            is apply_prompt_rollout_recommendation_async
        )

    def test_process_prompt_candidate_evaluation_suite_exported(self):
        """Test suite workflow helper is exported from framework."""
        from victor.framework import (
            process_prompt_candidate_evaluation_suite as ExportedProcessPromptCandidateEvaluationSuite,
        )

        assert (
            ExportedProcessPromptCandidateEvaluationSuite
            is process_prompt_candidate_evaluation_suite
        )

    def test_process_prompt_candidate_evaluation_suite_async_exported(self):
        """Test async suite workflow helper is exported from framework."""
        from victor.framework import (
            process_prompt_candidate_evaluation_suite_async as ExportedProcessPromptCandidateEvaluationSuiteAsync,
        )

        assert (
            ExportedProcessPromptCandidateEvaluationSuiteAsync
            is process_prompt_candidate_evaluation_suite_async
        )
