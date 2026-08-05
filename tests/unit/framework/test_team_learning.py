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

"""Unit tests for team learning and metrics modules."""

import tempfile
from pathlib import Path
import pytest

from victor.core.database import reset_database, get_database
from victor.agent.subagents import SubAgentRole
from victor.agent.teams import (
    TeamMember,
    TeamConfig,
    TaskCategory,
    TeamMetrics,
    CompositionStats,
    categorize_task,
    TeamRecommendation,
    TeamCompositionLearner,
    DEFAULT_COMPOSITIONS,
)
from victor.teams import (
    TeamFormation,
    MemberResult,
    TeamResult,
)


class TestTaskCategory:
    """Test TaskCategory enum."""

    def test_all_categories_defined(self):
        """All expected categories are defined."""
        assert TaskCategory.EXPLORATION.value == "exploration"
        assert TaskCategory.IMPLEMENTATION.value == "implementation"
        assert TaskCategory.REVIEW.value == "review"
        assert TaskCategory.TESTING.value == "testing"
        assert TaskCategory.REFACTORING.value == "refactoring"
        assert TaskCategory.DOCUMENTATION.value == "documentation"
        assert TaskCategory.DEBUGGING.value == "debugging"
        assert TaskCategory.PLANNING.value == "planning"
        assert TaskCategory.MIXED.value == "mixed"


class TestCategorizeTask:
    """Test categorize_task function."""

    def test_exploration_keywords(self):
        """Exploration keywords categorized correctly."""
        assert categorize_task("Find all API endpoints") == TaskCategory.EXPLORATION
        assert categorize_task("Search for auth code") == TaskCategory.EXPLORATION
        assert categorize_task("Explore the codebase") == TaskCategory.EXPLORATION

    def test_implementation_keywords(self):
        """Implementation keywords categorized correctly."""
        assert categorize_task("Implement user login") == TaskCategory.IMPLEMENTATION
        assert categorize_task("Create new feature") == TaskCategory.IMPLEMENTATION
        assert categorize_task("Build the API") == TaskCategory.IMPLEMENTATION

    def test_review_keywords(self):
        """Review keywords categorized correctly."""
        assert categorize_task("Review the changes") == TaskCategory.REVIEW
        assert categorize_task("Check code quality") == TaskCategory.REVIEW
        assert categorize_task("Validate the implementation") == TaskCategory.REVIEW

    def test_testing_keywords(self):
        """Testing keywords categorized correctly."""
        assert categorize_task("Test the function") == TaskCategory.TESTING
        assert categorize_task("Write unit specs") == TaskCategory.TESTING

    def test_debugging_keywords(self):
        """Debugging keywords categorized correctly."""
        assert categorize_task("Debug the error") == TaskCategory.DEBUGGING
        assert categorize_task("Fix the bug") == TaskCategory.DEBUGGING

    def test_mixed_default(self):
        """Unrecognized tasks return MIXED."""
        assert categorize_task("Do something random") == TaskCategory.MIXED


class TestTeamMetrics:
    """Test TeamMetrics class."""

    def _create_sample_result(self, success: bool = True) -> TeamResult:
        """Create a sample TeamResult for testing."""
        return TeamResult(
            success=success,
            final_output="Team completed work",
            member_results={
                "researcher_1": MemberResult(
                    member_id="researcher_1",
                    success=True,
                    output="Found patterns",
                    tool_calls_used=5,
                    duration_seconds=10.0,
                    discoveries=["Pattern A found"],
                ),
                "executor_1": MemberResult(
                    member_id="executor_1",
                    success=success,
                    output="Made changes",
                    tool_calls_used=8,
                    duration_seconds=15.0,
                    discoveries=["Issue B identified"],
                ),
            },
            total_tool_calls=13,
            total_duration=25.0,
            formation=TeamFormation.SEQUENTIAL,
        )

    def _create_sample_config(self) -> TeamConfig:
        """Create a sample TeamConfig for testing."""
        return TeamConfig(
            name="Test Team",
            goal="Test goal",
            members=[
                TeamMember(
                    id="researcher_1",
                    role=SubAgentRole.RESEARCHER,
                    name="Researcher",
                    goal="Research",
                    tool_budget=10,
                ),
                TeamMember(
                    id="executor_1",
                    role=SubAgentRole.EXECUTOR,
                    name="Executor",
                    goal="Execute",
                    tool_budget=15,
                ),
            ],
            formation=TeamFormation.SEQUENTIAL,
            total_tool_budget=25,
        )

    def test_from_result(self):
        """from_result creates correct metrics."""
        config = self._create_sample_config()
        result = self._create_sample_result()

        metrics = TeamMetrics.from_result(config, result, TaskCategory.IMPLEMENTATION)

        assert metrics.team_id.startswith("team_")
        assert metrics.task_category == TaskCategory.IMPLEMENTATION
        assert metrics.formation == TeamFormation.SEQUENTIAL
        assert metrics.member_count == 2
        assert metrics.role_distribution == {"researcher": 1, "executor": 1}
        assert metrics.total_tool_budget == 25
        assert metrics.tools_used == 13
        assert metrics.success is True
        assert metrics.quality_score == 0.8  # Default for success
        assert metrics.discoveries_count == 2

    def test_compute_efficiency(self):
        """compute_efficiency returns valid score."""
        config = self._create_sample_config()
        result = self._create_sample_result()
        metrics = TeamMetrics.from_result(config, result, TaskCategory.IMPLEMENTATION)

        efficiency = metrics.compute_efficiency()
        assert 0.0 <= efficiency <= 1.0

    def test_compute_speed_score(self):
        """compute_speed_score returns valid score."""
        config = self._create_sample_config()
        result = self._create_sample_result()
        metrics = TeamMetrics.from_result(config, result, TaskCategory.IMPLEMENTATION)

        speed = metrics.compute_speed_score(target_seconds=60.0)
        assert speed == 1.0  # 25s is under 60s target

        speed = metrics.compute_speed_score(target_seconds=10.0)
        assert 0.0 < speed < 1.0  # 25s is over 10s target

    def test_to_dict_from_dict(self):
        """Serialization round-trip works."""
        config = self._create_sample_config()
        result = self._create_sample_result()
        metrics = TeamMetrics.from_result(config, result, TaskCategory.IMPLEMENTATION)

        data = metrics.to_dict()
        restored = TeamMetrics.from_dict(data)

        assert restored.team_id == metrics.team_id
        assert restored.task_category == metrics.task_category
        assert restored.formation == metrics.formation
        assert restored.success == metrics.success


class TestCompositionStats:
    """Test CompositionStats class."""

    def test_success_rate(self):
        """success_rate calculates correctly."""
        stats = CompositionStats(
            formation=TeamFormation.SEQUENTIAL,
            role_counts={"researcher": 1, "executor": 1},
            task_category=TaskCategory.IMPLEMENTATION,
            total_executions=10,
            successes=8,
        )
        assert stats.success_rate == 0.8

    def test_avg_quality(self):
        """avg_quality calculates correctly."""
        stats = CompositionStats(
            formation=TeamFormation.SEQUENTIAL,
            role_counts={"researcher": 1},
            task_category=TaskCategory.EXPLORATION,
            total_executions=4,
            total_quality=3.2,
        )
        assert stats.avg_quality == 0.8

    def test_get_composition_key(self):
        """get_composition_key returns correct format."""
        stats = CompositionStats(
            formation=TeamFormation.PARALLEL,
            role_counts={"researcher": 2, "executor": 1},
            task_category=TaskCategory.EXPLORATION,
        )
        key = stats.get_composition_key()
        assert "parallel:" in key
        assert "researcher=2" in key
        assert "executor=1" in key

    def test_update(self):
        """update adds metrics correctly."""
        stats = CompositionStats(
            formation=TeamFormation.SEQUENTIAL,
            role_counts={"executor": 1},
            task_category=TaskCategory.IMPLEMENTATION,
        )

        # Create minimal metrics
        metrics = TeamMetrics(
            team_id="test",
            task_category=TaskCategory.IMPLEMENTATION,
            formation=TeamFormation.SEQUENTIAL,
            member_count=1,
            role_distribution={"executor": 1},
            total_tool_budget=20,
            tools_used=10,
            success=True,
            quality_score=0.9,
            duration_seconds=30.0,
        )

        stats.update(metrics)

        assert stats.total_executions == 1
        assert stats.successes == 1
        assert stats.total_quality == 0.9

    def test_compute_q_value(self):
        """compute_q_value returns valid value."""
        stats = CompositionStats(
            formation=TeamFormation.SEQUENTIAL,
            role_counts={"executor": 1},
            task_category=TaskCategory.IMPLEMENTATION,
            total_executions=10,
            successes=8,
            total_quality=8.0,
            total_duration=300.0,
            total_tools_used=80,
            total_budget=100,
        )

        q_value = stats.compute_q_value()
        assert 0.0 <= q_value <= 1.0


class TestTeamRecommendation:
    """Test TeamRecommendation class."""

    def test_basic_recommendation(self):
        """Create basic recommendation."""
        rec = TeamRecommendation(
            formation=TeamFormation.SEQUENTIAL,
            role_distribution={"researcher": 1, "executor": 1},
            suggested_budget=30,
            confidence=0.8,
            reason="Test reason",
        )
        assert rec.formation == TeamFormation.SEQUENTIAL
        assert rec.suggested_budget == 30
        assert rec.confidence == 0.8

    def test_to_team_config(self):
        """to_team_config creates valid config."""
        rec = TeamRecommendation(
            formation=TeamFormation.PARALLEL,
            role_distribution={"researcher": 2},
            suggested_budget=30,
            confidence=0.7,
            reason="Parallel research",
        )

        config = rec.to_team_config(
            name="Test Team",
            goal="Test goal",
        )

        assert config.name == "Test Team"
        assert config.goal == "Test goal"
        assert config.formation == TeamFormation.PARALLEL
        assert len(config.members) == 2
        assert all(m.role == SubAgentRole.RESEARCHER for m in config.members)

    def test_hierarchical_has_manager(self):
        """Hierarchical formation sets manager flag."""
        rec = TeamRecommendation(
            formation=TeamFormation.HIERARCHICAL,
            role_distribution={"researcher": 2, "executor": 1},
            suggested_budget=45,
            confidence=0.6,
            reason="Hierarchical team",
        )

        config = rec.to_team_config("Team", "Goal")

        # First member should be manager
        managers = [m for m in config.members if m.is_manager]
        assert len(managers) == 1


class TestDefaultCompositions:
    """Test default compositions."""

    def test_all_categories_have_default(self):
        """All task categories have default compositions."""
        for category in TaskCategory:
            assert category in DEFAULT_COMPOSITIONS

    def test_defaults_are_baselines(self):
        """Default compositions are marked as baselines."""
        for rec in DEFAULT_COMPOSITIONS.values():
            assert rec.is_baseline is True


class TestTeamCompositionLearner:
    """Test TeamCompositionLearner class."""

    @pytest.fixture
    def learner(self, tmp_path: Path):
        """Create learner with temporary database."""
        reset_database()
        db_path = tmp_path / "test_learning.db"
        get_database(db_path)
        learner = TeamCompositionLearner(db_path=db_path)
        yield learner
        reset_database()

    def test_initialization(self, learner):
        """Learner initializes correctly."""
        assert learner.learning_rate == 0.1
        assert learner.epsilon == 0.1
        assert learner.db is not None

    def test_suggest_team_cold_start(self, learner):
        """suggest_team returns baseline for cold start."""
        # Force exploit (no exploration)
        learner.epsilon = 0.0

        rec = learner.suggest_team("Find all API endpoints")

        # Should get exploration baseline
        assert rec.formation == TeamFormation.PARALLEL
        assert rec.is_baseline is True

    def test_suggest_team_explore(self, learner):
        """suggest_team explores with high epsilon."""
        # Force exploration
        learner.epsilon = 1.0

        rec = learner.suggest_team("Implement feature")

        # Should be exploration (not baseline)
        assert rec.confidence == 0.3
        assert "Exploration" in rec.reason

    def test_record_outcome(self, learner):
        """record_outcome stores data."""
        config = TeamConfig(
            name="Test",
            goal="Test",
            members=[
                TeamMember(
                    id="exec_1",
                    role=SubAgentRole.EXECUTOR,
                    name="Executor",
                    goal="Execute",
                )
            ],
            formation=TeamFormation.SEQUENTIAL,
            total_tool_budget=20,
        )

        result = TeamResult(
            success=True,
            final_output="Done",
            member_results={
                "exec_1": MemberResult(
                    member_id="exec_1",
                    success=True,
                    output="Done",
                    tool_calls_used=10,
                    duration_seconds=30.0,
                )
            },
            total_tool_calls=10,
            total_duration=30.0,
            formation=TeamFormation.SEQUENTIAL,
        )

        learner.record_team_outcome("Implement feature", config, result)

        stats = learner.get_stats(TaskCategory.IMPLEMENTATION)
        assert stats["total_executions"] == 1

    def test_q_value_updates(self, learner):
        """Q-values update after recording outcomes."""
        config = TeamConfig(
            name="Test",
            goal="Test",
            members=[
                TeamMember(
                    id="exec_1",
                    role=SubAgentRole.EXECUTOR,
                    name="Executor",
                    goal="Execute",
                )
            ],
            formation=TeamFormation.SEQUENTIAL,
            total_tool_budget=20,
        )

        # Record multiple successful outcomes
        for i in range(5):
            result = TeamResult(
                success=True,
                final_output="Done",
                member_results={
                    "exec_1": MemberResult(
                        member_id="exec_1",
                        success=True,
                        output="Done",
                        tool_calls_used=8,
                        duration_seconds=25.0,
                    )
                },
                total_tool_calls=8,
                total_duration=25.0,
                formation=TeamFormation.SEQUENTIAL,
            )
            learner.record_team_outcome("Implement feature", config, result)

        # Q-value should be relatively high after successes
        stats = learner.get_stats(TaskCategory.IMPLEMENTATION)
        assert stats["total_executions"] == 5
        if stats["top_compositions"]:
            assert stats["top_compositions"][0]["q_value"] > 0.5

    def test_get_stats(self, learner):
        """get_stats returns valid structure."""
        stats = learner.get_stats()

        assert "composition_count" in stats
        assert "total_executions" in stats
        assert "avg_q_value" in stats
        assert "top_compositions" in stats
        assert "learning_rate" in stats

    def test_reset(self, learner):
        """reset clears all data."""
        # Add some data first
        config = TeamConfig(
            name="Test",
            goal="Test",
            members=[
                TeamMember(
                    id="exec_1",
                    role=SubAgentRole.EXECUTOR,
                    name="Executor",
                    goal="Execute",
                )
            ],
            formation=TeamFormation.SEQUENTIAL,
            total_tool_budget=20,
        )

        result = TeamResult(
            success=True,
            final_output="Done",
            member_results={
                "exec_1": MemberResult(
                    member_id="exec_1",
                    success=True,
                    output="Done",
                    tool_calls_used=10,
                    duration_seconds=30.0,
                )
            },
            total_tool_calls=10,
            total_duration=30.0,
            formation=TeamFormation.SEQUENTIAL,
        )

        learner.record_team_outcome("Create task", config, result)

        # Reset
        learner.reset()

        stats = learner.get_stats()
        assert stats["composition_count"] == 0
        assert stats["total_executions"] == 0


class TestModuleExports:
    """Test module exports."""

    def test_teams_exports_learning(self):
        """Teams module exports learning components."""
        from victor.agent.teams import (
            TaskCategory,
            TeamMetrics,
            CompositionStats,
            categorize_task,
            TeamRecommendation,
            TeamCompositionLearner,
            get_team_learner,
            DEFAULT_COMPOSITIONS,
        )

        # If we get here without ImportError, all exports work
        assert True
