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

"""Tests for team formation selector (Phase 4 consolidation)."""

import pytest

from victor.framework.agentic_graph.state import (
    AgenticLoopStateModel,
    create_initial_state,
)
from victor.framework.agentic_graph.team_selector import (
    select_formation,
    FormationCriteria,
    DEFAULT_FORMATION,
)
from victor.teams.types import TeamFormation


class TestFormationCriteria:
    """Tests for FormationCriteria dataclass."""

    def test_create_default_criteria(self):
        """Test creating default formation criteria."""
        criteria = FormationCriteria()
        assert criteria.complexity == "medium"
        assert criteria.task_type == "general"
        assert criteria.team_size == 1
        assert criteria.has_dependencies is False
        assert criteria.requires_consensus is False

    def test_create_custom_criteria(self):
        """Test creating custom formation criteria."""
        criteria = FormationCriteria(
            complexity="high",
            task_type="feature",
            team_size=3,
            has_dependencies=True,
            requires_consensus=False,
        )
        assert criteria.complexity == "high"
        assert criteria.task_type == "feature"
        assert criteria.team_size == 3
        assert criteria.has_dependencies is True

    def test_criteria_from_state(self):
        """Test creating criteria from state."""
        state = AgenticLoopStateModel(
            query="Build feature",
            context={
                "complexity": "high",
                "task_type": "feature",
                "team_size": 2,
            },
        )

        criteria = FormationCriteria.from_state(state)
        assert criteria.complexity == "high"
        assert criteria.task_type == "feature"
        assert criteria.team_size == 2


class TestSelectFormation:
    """Tests for select_formation function."""

    def test_select_single_agent(self):
        """Test selection for single agent (default)."""
        criteria = FormationCriteria(team_size=1)
        formation = select_formation(criteria)
        assert formation == TeamFormation.SEQUENTIAL

    def test_select_parallel_for_independent_tasks(self):
        """Test parallel selection for independent tasks."""
        criteria = FormationCriteria(
            team_size=3,
            has_dependencies=False,
        )
        formation = select_formation(criteria)
        assert formation == TeamFormation.PARALLEL

    def test_select_sequential_for_dependent_tasks(self):
        """Test sequential selection for dependent tasks."""
        criteria = FormationCriteria(
            team_size=3,
            has_dependencies=True,
        )
        formation = select_formation(criteria)
        assert formation == TeamFormation.SEQUENTIAL

    def test_select_pipeline_for_workflow_tasks(self):
        """Test pipeline selection for workflow-style tasks."""
        criteria = FormationCriteria(
            task_type="workflow",
            team_size=3,
            has_dependencies=True,
        )
        formation = select_formation(criteria)
        assert formation == TeamFormation.PIPELINE

    def test_select_hierarchical_for_complex_tasks(self):
        """Test hierarchical selection for complex tasks."""
        criteria = FormationCriteria(
            complexity="high",
            team_size=4,
        )
        formation = select_formation(criteria)
        assert formation == TeamFormation.HIERARCHICAL

    def test_select_consensus_when_required(self):
        """Test consensus selection when explicitly required."""
        criteria = FormationCriteria(
            team_size=3,
            requires_consensus=True,
        )
        formation = select_formation(criteria)
        assert formation == TeamFormation.CONSENSUS

    def test_select_with_state(self):
        """Test formation selection directly from state."""
        state = create_initial_state(query="Complex multi-agent task")
        state = state.model_copy(
            update={
                "context": {
                    "complexity": "high",
                    "team_size": 3,
                }
            }
        )

        formation = select_formation(state)
        assert formation == TeamFormation.HIERARCHICAL

    def test_select_default_for_no_context(self):
        """Test default formation when no context provided."""
        state = create_initial_state(query="Simple task")
        formation = select_formation(state)
        assert formation == DEFAULT_FORMATION

    def test_select_debugging_uses_sequential(self):
        """Test debugging tasks prefer sequential execution."""
        criteria = FormationCriteria(
            task_type="debugging",
            team_size=2,
        )
        formation = select_formation(criteria)
        assert formation == TeamFormation.SEQUENTIAL

    def test_select_research_uses_parallel(self):
        """Test research tasks prefer parallel execution."""
        criteria = FormationCriteria(
            task_type="research",
            team_size=3,
        )
        formation = select_formation(criteria)
        assert formation == TeamFormation.PARALLEL

    def test_select_code_generation_uses_pipeline(self):
        """Test code generation tasks prefer pipeline."""
        criteria = FormationCriteria(
            task_type="code_generation",
            team_size=3,
            has_dependencies=True,
        )
        formation = select_formation(criteria)
        assert formation == TeamFormation.PIPELINE

    def test_complexity_overrides_for_large_teams(self):
        """Test that large teams default to hierarchical regardless of complexity."""
        criteria = FormationCriteria(
            complexity="low",
            team_size=5,
        )
        formation = select_formation(criteria)
        # Large teams need coordination
        assert formation in (TeamFormation.HIERARCHICAL, TeamFormation.PARALLEL)


class TestFormationEdgeCases:
    """Tests for edge cases in formation selection."""

    def test_zero_team_size_defaults_to_sequential(self):
        """Test that zero team size defaults to sequential."""
        criteria = FormationCriteria(team_size=0)
        formation = select_formation(criteria)
        assert formation == TeamFormation.SEQUENTIAL

    def test_negative_team_size_treated_as_single(self):
        """Test that negative team size is treated as single agent."""
        criteria = FormationCriteria(team_size=-1)
        formation = select_formation(criteria)
        assert formation == TeamFormation.SEQUENTIAL

    def test_very_large_team_forces_hierarchical(self):
        """Test that very large teams force hierarchical formation."""
        criteria = FormationCriteria(team_size=10)
        formation = select_formation(criteria)
        assert formation == TeamFormation.HIERARCHICAL

    def test_unknown_complexity_treated_as_medium(self):
        """Test that unknown complexity defaults to medium."""
        criteria = FormationCriteria(complexity="unknown")
        formation = select_formation(criteria)
        # Should not crash, should return a valid formation
        assert isinstance(formation, TeamFormation)

    def test_consensus_trumps_other_selections(self):
        """Test that consensus requirement overrides other logic."""
        criteria = FormationCriteria(
            team_size=10,
            complexity="high",
            requires_consensus=True,
        )
        formation = select_formation(criteria)
        assert formation == TeamFormation.CONSENSUS
