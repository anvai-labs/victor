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

"""Tests for victor.framework.multi_agent.teams module.

These tests verify the TeamTemplate, TeamMember, and TeamSpec classes
that provide generic team structures for multi-agent collaboration.
"""

import pytest

from victor.framework.multi_agent.personas import (
    CommunicationStyle,
    ExpertiseLevel,
    PersonaTraits,
)
from victor.framework.multi_agent.teams import (
    TaskAssignmentStrategy,
    TeamAgentCategory,
    TeamMember,
    TeamSpec,
    TeamTemplate,
    TeamTopology,
)

# =============================================================================
# TeamTopology Enum Tests
# =============================================================================


class TestTeamTopology:
    """Tests for TeamTopology enum."""

    def test_hierarchy_value(self):
        """HIERARCHY should have correct string value."""
        assert TeamTopology.HIERARCHY.value == "hierarchy"

    def test_mesh_value(self):
        """MESH should have correct string value."""
        assert TeamTopology.MESH.value == "mesh"

    def test_pipeline_value(self):
        """PIPELINE should have correct string value."""
        assert TeamTopology.PIPELINE.value == "pipeline"

    def test_hub_spoke_value(self):
        """HUB_SPOKE should have correct string value."""
        assert TeamTopology.HUB_SPOKE.value == "hub_spoke"

    def test_all_values_unique(self):
        """All enum values should be unique."""
        values = [t.value for t in TeamTopology]
        assert len(values) == len(set(values))


# =============================================================================
# TaskAssignmentStrategy Enum Tests
# =============================================================================


class TestTaskAssignmentStrategy:
    """Tests for TaskAssignmentStrategy enum."""

    def test_round_robin_value(self):
        """ROUND_ROBIN should have correct string value."""
        assert TaskAssignmentStrategy.ROUND_ROBIN.value == "round_robin"

    def test_skill_match_value(self):
        """SKILL_MATCH should have correct string value."""
        assert TaskAssignmentStrategy.SKILL_MATCH.value == "skill_match"

    def test_load_balanced_value(self):
        """LOAD_BALANCED should have correct string value."""
        assert TaskAssignmentStrategy.LOAD_BALANCED.value == "load_balanced"

    def test_all_values_unique(self):
        """All enum values should be unique."""
        values = [s.value for s in TaskAssignmentStrategy]
        assert len(values) == len(set(values))


# =============================================================================
# TeamMember Tests
# =============================================================================


class TestTeamMember:
    """Tests for TeamMember dataclass."""

    @pytest.fixture
    def sample_persona(self):
        """Create a sample persona for testing."""
        return PersonaTraits(
            name="Test Agent",
            role="tester",
            description="A test agent",
            expertise_level=ExpertiseLevel.EXPERT,
        )

    def test_required_fields(self, sample_persona):
        """TeamMember should require persona and role_in_team."""
        member = TeamMember(
            persona=sample_persona,
            role_in_team="lead_tester",
        )

        assert member.persona == sample_persona
        assert member.role_in_team == "lead_tester"

    def test_default_is_leader_false(self, sample_persona):
        """TeamMember should default to is_leader=False."""
        member = TeamMember(
            persona=sample_persona,
            role_in_team="worker",
        )

        assert member.is_leader is False
        assert member.is_supervisor is False
        assert member.agent_category == TeamAgentCategory.SPECIALIST

    def test_supervisor_category_sets_legacy_leader(self, sample_persona):
        """Supervisor category should imply the legacy leader flag."""
        member = TeamMember(
            persona=sample_persona,
            role_in_team="lead_tester",
            agent_category=TeamAgentCategory.SUPERVISOR,
        )

        assert member.is_supervisor is True
        assert member.is_leader is True

    def test_default_max_concurrent_tasks(self, sample_persona):
        """TeamMember should default to max_concurrent_tasks=1."""
        member = TeamMember(
            persona=sample_persona,
            role_in_team="worker",
        )

        assert member.max_concurrent_tasks == 1

    def test_default_tool_access_empty(self, sample_persona):
        """TeamMember should default to empty tool_access."""
        member = TeamMember(
            persona=sample_persona,
            role_in_team="worker",
        )

        assert member.tool_access == []

    def test_all_fields(self, sample_persona):
        """TeamMember should accept all fields together."""
        member = TeamMember(
            persona=sample_persona,
            role_in_team="lead_tester",
            is_leader=True,
            max_concurrent_tasks=5,
            tool_access=["read_file", "write_file"],
        )

        assert member.is_leader is True
        assert member.max_concurrent_tasks == 5
        assert member.tool_access == ["read", "write"]

    def test_name_property(self, sample_persona):
        """name property should return persona name."""
        member = TeamMember(
            persona=sample_persona,
            role_in_team="worker",
        )

        assert member.name == "Test Agent"

    def test_expertise_level_property(self, sample_persona):
        """expertise_level property should return persona expertise level."""
        member = TeamMember(
            persona=sample_persona,
            role_in_team="worker",
        )

        assert member.expertise_level == ExpertiseLevel.EXPERT


# =============================================================================
# TeamTemplate Creation Tests
# =============================================================================


class TestTeamTemplateCreation:
    """Tests for TeamTemplate dataclass creation."""

    def test_required_fields(self):
        """TeamTemplate should require name and description."""
        template = TeamTemplate(
            name="Test Team",
            description="A team for testing",
        )

        assert template.name == "Test Team"
        assert template.description == "A team for testing"

    def test_default_topology(self):
        """TeamTemplate should default to HIERARCHY topology."""
        template = TeamTemplate(
            name="Team",
            description="Desc",
        )

        assert template.topology == TeamTopology.HIERARCHY

    def test_default_assignment_strategy(self):
        """TeamTemplate should default to SKILL_MATCH assignment strategy."""
        template = TeamTemplate(
            name="Team",
            description="Desc",
        )

        assert template.assignment_strategy == TaskAssignmentStrategy.SKILL_MATCH

    def test_default_member_slots_empty(self):
        """TeamTemplate should default to empty member_slots."""
        template = TeamTemplate(
            name="Team",
            description="Desc",
        )

        assert template.member_slots == {}

    def test_default_shared_context_keys_empty(self):
        """TeamTemplate should default to empty shared_context_keys."""
        template = TeamTemplate(
            name="Team",
            description="Desc",
        )

        assert template.shared_context_keys == []

    def test_default_escalation_threshold(self):
        """TeamTemplate should default to 0.8 escalation_threshold."""
        template = TeamTemplate(
            name="Team",
            description="Desc",
        )

        assert template.escalation_threshold == 0.8

    def test_default_max_iterations(self):
        """TeamTemplate should default to 10 max_iterations."""
        template = TeamTemplate(
            name="Team",
            description="Desc",
        )

        assert template.max_iterations == 10

    def test_default_config_empty(self):
        """TeamTemplate should default to empty config."""
        template = TeamTemplate(
            name="Team",
            description="Desc",
        )

        assert template.config == {}

    def test_all_fields(self):
        """TeamTemplate should accept all fields together."""
        template = TeamTemplate(
            name="Full Team",
            description="Full description",
            topology=TeamTopology.PIPELINE,
            assignment_strategy=TaskAssignmentStrategy.LOAD_BALANCED,
            member_slots={"researcher": 1, "executor": 2},
            shared_context_keys=["project_id", "codebase_path"],
            escalation_threshold=0.6,
            max_iterations=25,
            config={"custom_key": "custom_value"},
        )

        assert template.name == "Full Team"
        assert template.topology == TeamTopology.PIPELINE
        assert template.assignment_strategy == TaskAssignmentStrategy.LOAD_BALANCED
        assert template.member_slots == {"researcher": 1, "executor": 2}
        assert template.shared_context_keys == ["project_id", "codebase_path"]
        assert template.escalation_threshold == 0.6
        assert template.max_iterations == 25
        assert template.config == {"custom_key": "custom_value"}


# =============================================================================
# TeamTemplate Validation Tests
# =============================================================================


class TestTeamTemplateValidation:
    """Tests for TeamTemplate validation."""

    def test_escalation_threshold_below_zero_raises(self):
        """TeamTemplate should raise for escalation_threshold < 0."""
        with pytest.raises(ValueError, match="escalation_threshold"):
            TeamTemplate(
                name="Team",
                description="Desc",
                escalation_threshold=-0.1,
            )

    def test_escalation_threshold_above_one_raises(self):
        """TeamTemplate should raise for escalation_threshold > 1."""
        with pytest.raises(ValueError, match="escalation_threshold"):
            TeamTemplate(
                name="Team",
                description="Desc",
                escalation_threshold=1.5,
            )

    def test_max_iterations_zero_raises(self):
        """TeamTemplate should raise for max_iterations < 1."""
        with pytest.raises(ValueError, match="max_iterations"):
            TeamTemplate(
                name="Team",
                description="Desc",
                max_iterations=0,
            )

    def test_max_iterations_negative_raises(self):
        """TeamTemplate should raise for negative max_iterations."""
        with pytest.raises(ValueError, match="max_iterations"):
            TeamTemplate(
                name="Team",
                description="Desc",
                max_iterations=-5,
            )

    def test_boundary_values_valid(self):
        """TeamTemplate should accept boundary values."""
        template = TeamTemplate(
            name="Team",
            description="Desc",
            escalation_threshold=0.0,
            max_iterations=1,
        )

        assert template.escalation_threshold == 0.0
        assert template.max_iterations == 1


# =============================================================================
# TeamTemplate Serialization Tests
# =============================================================================


class TestTeamTemplateSerialization:
    """Tests for TeamTemplate serialization methods."""

    def test_to_dict_includes_all_fields(self):
        """to_dict should include all template fields."""
        template = TeamTemplate(
            name="Serializable Team",
            description="Team for serialization",
            topology=TeamTopology.MESH,
            assignment_strategy=TaskAssignmentStrategy.ROUND_ROBIN,
            member_slots={"worker": 3},
            shared_context_keys=["key1"],
            escalation_threshold=0.7,
            max_iterations=15,
            config={"extra": "data"},
        )

        data = template.to_dict()

        assert data["name"] == "Serializable Team"
        assert data["description"] == "Team for serialization"
        assert data["topology"] == "mesh"
        assert data["assignment_strategy"] == "round_robin"
        assert data["member_slots"] == {"worker": 3}
        assert data["shared_context_keys"] == ["key1"]
        assert data["escalation_threshold"] == 0.7
        assert data["max_iterations"] == 15
        assert data["config"] == {"extra": "data"}


# =============================================================================
# TeamSpec Tests
# =============================================================================


class TestTeamSpec:
    """Tests for TeamSpec dataclass."""

    @pytest.fixture
    def sample_template(self):
        """Create a sample template for testing."""
        return TeamTemplate(
            name="Sample Team",
            description="A sample team",
            topology=TeamTopology.PIPELINE,
            member_slots={"researcher": 1, "executor": 1},
        )

    @pytest.fixture
    def sample_members(self):
        """Create sample members for testing."""
        researcher_persona = PersonaTraits(
            name="Researcher",
            role="researcher",
            description="Researches code",
        )
        executor_persona = PersonaTraits(
            name="Executor",
            role="executor",
            description="Executes changes",
        )
        return [
            TeamMember(
                persona=researcher_persona,
                role_in_team="researcher",
                is_leader=True,
            ),
            TeamMember(
                persona=executor_persona,
                role_in_team="executor",
            ),
        ]

    def test_required_template(self, sample_template):
        """TeamSpec should require template."""
        spec = TeamSpec(template=sample_template)

        assert spec.template == sample_template

    def test_default_members_empty(self, sample_template):
        """TeamSpec should default to empty members."""
        spec = TeamSpec(template=sample_template)

        assert spec.members == []

    def test_with_members(self, sample_template, sample_members):
        """TeamSpec should accept members."""
        spec = TeamSpec(
            template=sample_template,
            members=sample_members,
        )

        assert len(spec.members) == 2

    def test_leader_property_returns_leader(self, sample_template, sample_members):
        """leader property should return the member marked as leader."""
        spec = TeamSpec(
            template=sample_template,
            members=sample_members,
        )

        leader = spec.leader

        assert leader is not None
        assert leader.is_leader is True
        assert leader.name == "Researcher"
        assert spec.supervisor is leader

    def test_leader_property_returns_none_when_no_leader(self, sample_template):
        """leader property should return None when no leader is set."""
        persona = PersonaTraits(
            name="Worker",
            role="worker",
            description="A worker",
        )
        member = TeamMember(
            persona=persona,
            role_in_team="worker",
            is_leader=False,
        )
        spec = TeamSpec(
            template=sample_template,
            members=[member],
        )

        assert spec.leader is None

    def test_name_property(self, sample_template, sample_members):
        """name property should return template name."""
        spec = TeamSpec(
            template=sample_template,
            members=sample_members,
        )

        assert spec.name == "Sample Team"

    def test_topology_property(self, sample_template, sample_members):
        """topology property should return template topology."""
        spec = TeamSpec(
            template=sample_template,
            members=sample_members,
        )

        assert spec.topology == TeamTopology.PIPELINE

    def test_get_members_by_role(self, sample_template, sample_members):
        """get_members_by_role should return members with matching role."""
        spec = TeamSpec(
            template=sample_template,
            members=sample_members,
        )

        researchers = spec.get_members_by_role("researcher")

        assert len(researchers) == 1
        assert researchers[0].name == "Researcher"

    def test_get_members_by_role_returns_empty_for_unknown(self, sample_template, sample_members):
        """get_members_by_role should return empty list for unknown role."""
        spec = TeamSpec(
            template=sample_template,
            members=sample_members,
        )

        unknown = spec.get_members_by_role("unknown_role")

        assert unknown == []

    def test_get_members_by_role_returns_multiple(self, sample_template):
        """get_members_by_role should return all members with matching role."""
        persona1 = PersonaTraits(
            name="Worker 1",
            role="worker",
            description="Worker 1",
        )
        persona2 = PersonaTraits(
            name="Worker 2",
            role="worker",
            description="Worker 2",
        )
        spec = TeamSpec(
            template=sample_template,
            members=[
                TeamMember(persona=persona1, role_in_team="worker"),
                TeamMember(persona=persona2, role_in_team="worker"),
            ],
        )

        workers = spec.get_members_by_role("worker")

        assert len(workers) == 2


# =============================================================================
# TeamSpec Validation Tests
# =============================================================================


class TestTeamSpecValidation:
    """Tests for TeamSpec.validate_slots() method."""

    @pytest.fixture
    def template_with_slots(self):
        """Create a template with specific member slots."""
        return TeamTemplate(
            name="Team",
            description="Team with slots",
            member_slots={"researcher": 1, "executor": 2},
        )

    def test_validate_slots_returns_empty_when_valid(self, template_with_slots):
        """validate_slots should return empty list when slots are satisfied."""
        researcher = PersonaTraits(name="R", role="r", description="d")
        executor1 = PersonaTraits(name="E1", role="e", description="d")
        executor2 = PersonaTraits(name="E2", role="e", description="d")

        spec = TeamSpec(
            template=template_with_slots,
            members=[
                TeamMember(persona=researcher, role_in_team="researcher"),
                TeamMember(persona=executor1, role_in_team="executor"),
                TeamMember(persona=executor2, role_in_team="executor"),
            ],
        )

        errors = spec.validate_slots()

        assert errors == []

    def test_validate_slots_returns_errors_when_missing(self, template_with_slots):
        """validate_slots should return errors when slots are not filled."""
        researcher = PersonaTraits(name="R", role="r", description="d")

        spec = TeamSpec(
            template=template_with_slots,
            members=[
                TeamMember(persona=researcher, role_in_team="researcher"),
            ],
        )

        errors = spec.validate_slots()

        assert len(errors) == 1
        assert "executor" in errors[0]
        assert "2" in errors[0]

    def test_validate_slots_returns_multiple_errors(self, template_with_slots):
        """validate_slots should return all missing slot errors."""
        spec = TeamSpec(
            template=template_with_slots,
            members=[],
        )

        errors = spec.validate_slots()

        assert len(errors) == 2
        # Should have errors for both researcher and executor


# =============================================================================
# TeamSpec Serialization Tests
# =============================================================================


class TestTeamSpecSerialization:
    """Tests for TeamSpec serialization methods."""

    def test_to_dict_includes_template_and_members(self):
        """to_dict should include template and members."""
        template = TeamTemplate(
            name="Team",
            description="Team description",
            topology=TeamTopology.HIERARCHY,
        )
        persona = PersonaTraits(
            name="Agent",
            role="worker",
            description="A worker agent",
            communication_style=CommunicationStyle.TECHNICAL,
        )
        member = TeamMember(
            persona=persona,
            role_in_team="worker",
            is_leader=True,
            max_concurrent_tasks=3,
            tool_access=["read_file"],
        )
        spec = TeamSpec(
            template=template,
            members=[member],
        )

        data = spec.to_dict()

        assert "template" in data
        assert data["template"]["name"] == "Team"
        assert data["template"]["topology"] == "hierarchy"
        assert "members" in data
        assert len(data["members"]) == 1
        assert data["members"][0]["role_in_team"] == "worker"
        assert data["members"][0]["is_leader"] is True
        assert data["members"][0]["max_concurrent_tasks"] == 3
        assert data["members"][0]["tool_access"] == ["read"]
        assert data["members"][0]["persona"]["name"] == "Agent"


# =============================================================================
# Module Export Tests
# =============================================================================


class TestTeamsExports:
    """Tests for module exports."""

    def test_all_exports_from_module(self):
        """All expected items should be exported from the module."""
        from victor.framework.multi_agent.teams import (
            TaskAssignmentStrategy,
            TeamMember,
            TeamSpec,
            TeamTemplate,
            TeamTopology,
        )

        assert TaskAssignmentStrategy is not None
        assert TeamMember is not None
        assert TeamSpec is not None
        assert TeamTemplate is not None
        assert TeamTopology is not None

    def test_exports_from_package(self):
        """All expected items should be exported from the package."""
        from victor.framework.multi_agent import (
            TaskAssignmentStrategy,
            TeamMember,
            TeamSpec,
            TeamTemplate,
            TeamTopology,
        )

        assert TaskAssignmentStrategy is not None
        assert TeamMember is not None
        assert TeamSpec is not None
        assert TeamTemplate is not None
        assert TeamTopology is not None
