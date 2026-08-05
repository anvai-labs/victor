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

"""Integration tests for persona and team systems.

Tests PersonaTraits integration with coding vertical,
TeamSpec creation, and multi-agent team composition.
"""

import pytest
from typing import List

from victor.core.verticals.import_resolver import import_module_with_fallback

from victor.framework.multi_agent.personas import (
    PersonaTraits,
    PersonaTemplate,
    CommunicationStyle,
    ExpertiseLevel,
)
from victor.framework.multi_agent.teams import (
    TeamSpec,
    TeamTemplate,
    TeamMember,
    TeamTopology,
    TaskAssignmentStrategy,
)

_PERSONA_MODULE, _resolved = import_module_with_fallback("victor.coding.teams.personas")
if _PERSONA_MODULE is None:
    pytest.skip("Coding persona module not available", allow_module_level=True)


def _persona_attr(name: str):
    if not hasattr(_PERSONA_MODULE, name):
        pytest.skip(
            f"Missing coding persona attribute: {name}",
            allow_module_level=True,
        )
    return getattr(_PERSONA_MODULE, name)


CodingPersona = _persona_attr("CodingPersona")
CodingPersonaTraits = _persona_attr("PersonaTraits")
ExpertiseCategory = _persona_attr("ExpertiseCategory")
CodingCommunicationStyle = _persona_attr("CommunicationStyle")
DecisionStyle = _persona_attr("DecisionStyle")
CODING_PERSONAS = _persona_attr("CODING_PERSONAS")
get_persona = _persona_attr("get_persona")
get_personas_for_role = _persona_attr("get_personas_for_role")
get_persona_by_expertise = _persona_attr("get_persona_by_expertise")
apply_persona_to_spec = _persona_attr("apply_persona_to_spec")
list_personas = _persona_attr("list_personas")

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_persona_traits():
    """Create sample PersonaTraits for testing."""
    return PersonaTraits(
        name="Test Developer",
        role="developer",
        description="A test developer persona for integration testing",
        communication_style=CommunicationStyle.TECHNICAL,
        expertise_level=ExpertiseLevel.EXPERT,
        verbosity=0.7,
        strengths=["testing", "debugging", "documentation"],
        preferred_tools=["pytest", "git", "read"],
        risk_tolerance=0.3,
        creativity=0.6,
    )


@pytest.fixture
def sample_team_template():
    """Create sample TeamTemplate for testing."""
    return TeamTemplate(
        name="Code Review Team",
        description="A team for comprehensive code review",
        topology=TeamTopology.PIPELINE,
        assignment_strategy=TaskAssignmentStrategy.SKILL_MATCH,
        member_slots={"researcher": 1, "reviewer": 2, "approver": 1},
        shared_context_keys=["codebase", "review_criteria"],
        escalation_threshold=0.7,
        max_iterations=15,
    )


@pytest.fixture
def mock_team_member_spec():
    """Create a mock TeamMemberSpec-like object for persona application."""

    class MockSpec:
        def __init__(self):
            self.expertise = []
            self.backstory = ""
            self.personality = ""

    return MockSpec()


# =============================================================================
# Test Class: PersonaTraits Integration
# =============================================================================


@pytest.mark.integration
class TestPersonaTraitsIntegration:
    """Integration tests for PersonaTraits with framework components."""

    def test_persona_traits_creation(self, sample_persona_traits):
        """Test creating PersonaTraits with all attributes."""
        persona = sample_persona_traits

        assert persona.name == "Test Developer"
        assert persona.role == "developer"
        assert persona.communication_style == CommunicationStyle.TECHNICAL
        assert persona.expertise_level == ExpertiseLevel.EXPERT
        assert persona.verbosity == 0.7
        assert "testing" in persona.strengths
        assert persona.risk_tolerance == 0.3

    def test_persona_traits_validation(self):
        """Test PersonaTraits validation for out-of-range values."""
        with pytest.raises(ValueError, match="verbosity"):
            PersonaTraits(
                name="Invalid",
                role="test",
                description="Invalid verbosity",
                verbosity=1.5,  # Out of range
            )

        with pytest.raises(ValueError, match="risk_tolerance"):
            PersonaTraits(
                name="Invalid",
                role="test",
                description="Invalid risk",
                risk_tolerance=-0.1,  # Out of range
            )

        with pytest.raises(ValueError, match="creativity"):
            PersonaTraits(
                name="Invalid",
                role="test",
                description="Invalid creativity",
                creativity=2.0,  # Out of range
            )

    def test_persona_to_system_prompt_fragment(self, sample_persona_traits):
        """Test generating system prompt fragment from persona."""
        persona = sample_persona_traits
        fragment = persona.to_system_prompt_fragment()

        assert "Test Developer" in fragment
        assert "developer" in fragment
        assert "technical" in fragment.lower()
        assert "testing" in fragment or "debugging" in fragment

    def test_persona_serialization(self, sample_persona_traits):
        """Test PersonaTraits serialization to dict and back."""
        persona = sample_persona_traits
        data = persona.to_dict()

        assert data["name"] == "Test Developer"
        assert data["communication_style"] == "technical"
        assert data["expertise_level"] == "expert"

        # Recreate from dict
        restored = PersonaTraits.from_dict(data)

        assert restored.name == persona.name
        assert restored.communication_style == persona.communication_style
        assert restored.expertise_level == persona.expertise_level

    def test_persona_template_creation(self, sample_persona_traits):
        """Test creating PersonaTemplate and generating instances."""
        base = sample_persona_traits
        template = PersonaTemplate(base_traits=base)

        # Create variations
        senior = template.create(
            name="Senior Developer",
            expertise_level=ExpertiseLevel.SPECIALIST,
            verbosity=0.8,
        )

        assert senior.name == "Senior Developer"
        assert senior.expertise_level == ExpertiseLevel.SPECIALIST
        assert senior.verbosity == 0.8
        # Should inherit other properties
        assert senior.role == base.role
        assert senior.risk_tolerance == base.risk_tolerance


# =============================================================================
# Test Class: TeamSpec and TeamTemplate Integration
# =============================================================================


@pytest.mark.integration
class TestTeamSpecIntegration:
    """Integration tests for TeamSpec with team composition."""

    def test_team_template_creation(self, sample_team_template):
        """Test TeamTemplate creation with all attributes."""
        template = sample_team_template

        assert template.name == "Code Review Team"
        assert template.topology == TeamTopology.PIPELINE
        assert template.assignment_strategy == TaskAssignmentStrategy.SKILL_MATCH
        assert template.member_slots["researcher"] == 1
        assert template.member_slots["reviewer"] == 2
        assert template.escalation_threshold == 0.7

    def test_team_template_validation(self):
        """Test TeamTemplate validation."""
        with pytest.raises(ValueError, match="escalation_threshold"):
            TeamTemplate(
                name="Invalid",
                description="Invalid threshold",
                escalation_threshold=1.5,  # Out of range
            )

        with pytest.raises(ValueError, match="max_iterations"):
            TeamTemplate(
                name="Invalid",
                description="Invalid iterations",
                max_iterations=0,  # Must be >= 1
            )

    def test_team_template_to_dict(self, sample_team_template):
        """Test TeamTemplate serialization."""
        template = sample_team_template
        data = template.to_dict()

        assert data["name"] == "Code Review Team"
        assert data["topology"] == "pipeline"
        assert data["assignment_strategy"] == "skill_match"
        assert data["member_slots"]["reviewer"] == 2

    def test_team_spec_with_members(self, sample_team_template, sample_persona_traits):
        """Test TeamSpec with actual team members."""
        template = sample_team_template

        # Create team members
        researcher = TeamMember(
            persona=sample_persona_traits,
            role_in_team="researcher",
            is_leader=False,
            tool_access=["code_search", "read"],
        )

        reviewer1 = TeamMember(
            persona=PersonaTraits(
                name="Code Reviewer 1",
                role="reviewer",
                description="First code reviewer",
            ),
            role_in_team="reviewer",
            is_leader=False,
        )

        reviewer2 = TeamMember(
            persona=PersonaTraits(
                name="Code Reviewer 2",
                role="reviewer",
                description="Second code reviewer",
            ),
            role_in_team="reviewer",
            is_leader=False,
        )

        approver = TeamMember(
            persona=PersonaTraits(
                name="Lead Approver",
                role="approver",
                description="Final approver",
            ),
            role_in_team="approver",
            is_leader=True,
            max_concurrent_tasks=2,
        )

        spec = TeamSpec(
            template=template,
            members=[researcher, reviewer1, reviewer2, approver],
        )

        assert spec.name == "Code Review Team"
        assert spec.topology == TeamTopology.PIPELINE
        assert len(spec.members) == 4
        assert spec.leader is approver

    def test_team_spec_leader_detection(self, sample_team_template, sample_persona_traits):
        """Test leader detection in TeamSpec."""
        template = sample_team_template

        member1 = TeamMember(
            persona=sample_persona_traits,
            role_in_team="worker",
            is_leader=False,
        )

        member2 = TeamMember(
            persona=PersonaTraits(
                name="Leader",
                role="lead",
                description="Team leader",
            ),
            role_in_team="lead",
            is_leader=True,
        )

        spec = TeamSpec(template=template, members=[member1, member2])

        assert spec.leader is member2
        assert spec.leader.name == "Leader"

    def test_team_spec_no_leader(self, sample_team_template, sample_persona_traits):
        """Test TeamSpec with no leader assigned."""
        template = sample_team_template

        member = TeamMember(
            persona=sample_persona_traits,
            role_in_team="worker",
            is_leader=False,
        )

        spec = TeamSpec(template=template, members=[member])

        assert spec.leader is None

    def test_team_spec_get_members_by_role(self, sample_team_template, sample_persona_traits):
        """Test filtering team members by role."""
        template = sample_team_template

        members = [
            TeamMember(
                persona=sample_persona_traits,
                role_in_team="researcher",
            ),
            TeamMember(
                persona=PersonaTraits(name="R1", role="r", description=""),
                role_in_team="reviewer",
            ),
            TeamMember(
                persona=PersonaTraits(name="R2", role="r", description=""),
                role_in_team="reviewer",
            ),
        ]

        spec = TeamSpec(template=template, members=members)

        researchers = spec.get_members_by_role("researcher")
        reviewers = spec.get_members_by_role("reviewer")
        approvers = spec.get_members_by_role("approver")

        assert len(researchers) == 1
        assert len(reviewers) == 2
        assert len(approvers) == 0

    def test_team_spec_validate_slots(self, sample_team_template, sample_persona_traits):
        """Test slot validation in TeamSpec."""
        template = sample_team_template  # Requires 1 researcher, 2 reviewers, 1 approver

        # Create team with missing roles
        spec = TeamSpec(
            template=template,
            members=[
                TeamMember(
                    persona=sample_persona_traits,
                    role_in_team="researcher",
                ),
            ],
        )

        errors = spec.validate_slots()

        assert len(errors) > 0
        assert any("reviewer" in error.lower() for error in errors)
        assert any("approver" in error.lower() for error in errors)

    def test_team_spec_serialization(self, sample_team_template, sample_persona_traits):
        """Test TeamSpec serialization to dict."""
        template = sample_team_template

        member = TeamMember(
            persona=sample_persona_traits,
            role_in_team="developer",
            is_leader=True,
            tool_access=["git", "read"],
        )

        spec = TeamSpec(template=template, members=[member])
        data = spec.to_dict()

        assert "template" in data
        assert "members" in data
        assert len(data["members"]) == 1
        assert data["members"][0]["role_in_team"] == "developer"
        assert data["members"][0]["is_leader"] is True


# =============================================================================
# Test Class: Coding Vertical Personas Integration
# =============================================================================


@pytest.mark.integration
class TestCodingPersonasIntegration:
    """Integration tests for coding-specific personas."""

    def test_builtin_coding_personas_exist(self):
        """Test that built-in coding personas are available."""
        assert len(CODING_PERSONAS) > 0
        assert "code_archaeologist" in CODING_PERSONAS
        assert "security_auditor" in CODING_PERSONAS
        assert "architect" in CODING_PERSONAS
        assert "craftsman" in CODING_PERSONAS
        assert "quality_guardian" in CODING_PERSONAS

    def test_get_persona(self):
        """Test retrieving persona by name."""
        archaeologist = get_persona("code_archaeologist")

        assert archaeologist is not None
        assert archaeologist.name == "Code Archaeologist"
        assert archaeologist.role == "researcher"
        assert ExpertiseCategory.CODE_ANALYSIS in archaeologist.expertise

    def test_get_persona_not_found(self):
        """Test retrieving non-existent persona."""
        result = get_persona("nonexistent_persona")
        assert result is None

    def test_get_personas_for_role(self):
        """Test retrieving all personas for a role."""
        researchers = get_personas_for_role("researcher")
        planners = get_personas_for_role("planner")
        executors = get_personas_for_role("executor")
        reviewers = get_personas_for_role("reviewer")

        assert len(researchers) >= 1
        assert len(planners) >= 1
        assert len(executors) >= 1
        assert len(reviewers) >= 1

        # Verify roles match
        for persona in researchers:
            assert persona.role == "researcher"

    def test_get_persona_by_expertise(self):
        """Test retrieving personas by expertise."""
        security_experts = get_persona_by_expertise(ExpertiseCategory.SECURITY)
        testing_experts = get_persona_by_expertise(ExpertiseCategory.TESTING)

        assert len(security_experts) >= 1
        assert len(testing_experts) >= 1

        # Verify expertise
        for persona in security_experts:
            expertise_list = persona.expertise + persona.secondary_expertise
            assert ExpertiseCategory.SECURITY in expertise_list

    def test_list_personas(self):
        """Test listing all persona names."""
        names = list_personas()

        assert isinstance(names, list)
        assert len(names) > 0
        assert "code_archaeologist" in names
        assert "architect" in names

    def test_coding_persona_generate_backstory(self):
        """Test backstory generation from coding persona."""
        architect = get_persona("architect")

        backstory = architect.generate_backstory()

        assert "Solution Architect" in backstory
        assert "planner" in backstory
        assert len(backstory) > 50  # Should be substantial

    def test_coding_persona_traits_to_prompt_hints(self):
        """Test persona traits generating prompt hints."""
        archaeologist = get_persona("code_archaeologist")
        hints = archaeologist.traits.to_prompt_hints()

        assert len(hints) > 0
        # Should contain style-related hints
        assert any(
            keyword in hints.lower() for keyword in ["data", "evidence", "detail", "careful"]
        )

    def test_coding_persona_to_dict(self):
        """Test coding persona serialization."""
        persona = get_persona("craftsman")
        data = persona.to_dict()

        assert data["name"] == "Code Craftsman"
        assert data["role"] == "executor"
        assert "expertise" in data
        assert "backstory" in data

    def test_apply_persona_to_spec(self, mock_team_member_spec):
        """Test applying persona attributes to TeamMemberSpec."""
        spec = mock_team_member_spec

        # Apply persona
        enhanced = apply_persona_to_spec(spec, "code_archaeologist")

        # Should have expertise from persona
        assert len(enhanced.expertise) > 0
        assert "code_analysis" in enhanced.expertise

        # Should have generated backstory
        assert len(enhanced.backstory) > 0
        assert "Code Archaeologist" in enhanced.backstory

        # Should have personality
        assert len(enhanced.personality) > 0

    def test_apply_persona_preserves_existing(self, mock_team_member_spec):
        """Test that applying persona preserves existing attributes."""
        spec = mock_team_member_spec
        spec.expertise = ["existing_skill"]
        spec.backstory = "Existing backstory."

        enhanced = apply_persona_to_spec(spec, "debugger")

        # Should merge expertise
        assert "existing_skill" in enhanced.expertise
        # Should append to backstory, not replace
        assert "Existing backstory" in enhanced.backstory

    def test_apply_nonexistent_persona(self, mock_team_member_spec):
        """Test applying non-existent persona returns unchanged spec."""
        spec = mock_team_member_spec
        original_expertise = spec.expertise.copy()

        result = apply_persona_to_spec(spec, "fake_persona")

        # Should return unchanged
        assert result.expertise == original_expertise


# =============================================================================
# Test Class: Multi-Agent Team Composition
# =============================================================================


@pytest.mark.integration
class TestMultiAgentTeamComposition:
    """Integration tests for multi-agent team composition patterns."""

    def test_create_research_team(self):
        """Test creating a research-focused team."""
        template = TeamTemplate(
            name="Research Team",
            description="Investigate codebase patterns",
            topology=TeamTopology.MESH,
            assignment_strategy=TaskAssignmentStrategy.SKILL_MATCH,
            member_slots={"researcher": 2, "analyst": 1},
        )

        archaeologist = get_persona("code_archaeologist")
        auditor = get_persona("security_auditor")

        members = [
            TeamMember(
                persona=PersonaTraits(
                    name=archaeologist.name,
                    role=archaeologist.role,
                    description=archaeologist.generate_backstory()[:200],
                    strengths=archaeologist.strengths,
                ),
                role_in_team="researcher",
                tool_access=["code_search", "read", "grep"],
            ),
            TeamMember(
                persona=PersonaTraits(
                    name=auditor.name,
                    role=auditor.role,
                    description=auditor.generate_backstory()[:200],
                    strengths=auditor.strengths,
                ),
                role_in_team="researcher",
                tool_access=["security_scan", "read"],
            ),
            TeamMember(
                persona=PersonaTraits(
                    name="Pattern Analyst",
                    role="analyst",
                    description="Analyzes patterns found by researchers",
                ),
                role_in_team="analyst",
                is_leader=True,
            ),
        ]

        spec = TeamSpec(template=template, members=members)

        assert spec.name == "Research Team"
        assert spec.topology == TeamTopology.MESH
        assert len(spec.get_members_by_role("researcher")) == 2
        assert spec.leader.name == "Pattern Analyst"

    def test_create_pipeline_team(self):
        """Test creating a pipeline-style development team."""
        template = TeamTemplate(
            name="Development Pipeline",
            description="Sequential development workflow",
            topology=TeamTopology.PIPELINE,
            member_slots={"planner": 1, "executor": 1, "reviewer": 1},
            max_iterations=25,
        )

        architect = get_persona("architect")
        craftsman = get_persona("craftsman")
        guardian = get_persona("quality_guardian")

        members = [
            TeamMember(
                persona=PersonaTraits(
                    name=architect.name,
                    role="planner",
                    description="Plans implementation",
                ),
                role_in_team="planner",
            ),
            TeamMember(
                persona=PersonaTraits(
                    name=craftsman.name,
                    role="executor",
                    description="Implements code",
                ),
                role_in_team="executor",
            ),
            TeamMember(
                persona=PersonaTraits(
                    name=guardian.name,
                    role="reviewer",
                    description="Reviews implementation",
                ),
                role_in_team="reviewer",
                is_leader=True,
            ),
        ]

        spec = TeamSpec(template=template, members=members)

        # Verify pipeline structure
        assert spec.topology == TeamTopology.PIPELINE
        assert len(spec.members) == 3

        # Pipeline should have clear role sequence
        roles = [m.role_in_team for m in spec.members]
        assert "planner" in roles
        assert "executor" in roles
        assert "reviewer" in roles

    def test_hub_spoke_team_topology(self):
        """Test creating a hub-and-spoke team structure."""
        template = TeamTemplate(
            name="Hub-Spoke Team",
            description="Coordinator with specialized workers",
            topology=TeamTopology.HUB_SPOKE,
            assignment_strategy=TaskAssignmentStrategy.LOAD_BALANCED,
            member_slots={"coordinator": 1, "worker": 3},
        )

        coordinator = TeamMember(
            persona=PersonaTraits(
                name="Task Coordinator",
                role="coordinator",
                description="Coordinates all worker tasks",
            ),
            role_in_team="coordinator",
            is_leader=True,
            max_concurrent_tasks=5,
        )

        workers = [
            TeamMember(
                persona=PersonaTraits(
                    name=f"Worker {i}",
                    role="worker",
                    description=f"Specialized worker {i}",
                ),
                role_in_team="worker",
                max_concurrent_tasks=2,
            )
            for i in range(1, 4)
        ]

        spec = TeamSpec(template=template, members=[coordinator] + workers)

        assert spec.topology == TeamTopology.HUB_SPOKE
        assert spec.leader == coordinator
        assert spec.leader.max_concurrent_tasks == 5
        assert len(spec.get_members_by_role("worker")) == 3

    def test_team_with_expertise_matching(self):
        """Test creating team with expertise-based assignments."""
        # Find personas with specific expertise
        security_personas = get_persona_by_expertise(ExpertiseCategory.SECURITY)
        testing_personas = get_persona_by_expertise(ExpertiseCategory.TESTING)

        assert len(security_personas) >= 1
        assert len(testing_personas) >= 1

        # Build team from expertise-matched personas
        template = TeamTemplate(
            name="Quality Assurance Team",
            description="Security and testing specialists",
            topology=TeamTopology.HIERARCHY,
            assignment_strategy=TaskAssignmentStrategy.SKILL_MATCH,
        )

        security_member = TeamMember(
            persona=PersonaTraits(
                name=security_personas[0].name,
                role="security",
                description="Security specialist",
                strengths=security_personas[0].strengths,
            ),
            role_in_team="security_analyst",
        )

        testing_member = TeamMember(
            persona=PersonaTraits(
                name=testing_personas[0].name,
                role="testing",
                description="Testing specialist",
                strengths=testing_personas[0].strengths,
            ),
            role_in_team="test_lead",
            is_leader=True,
        )

        spec = TeamSpec(
            template=template,
            members=[security_member, testing_member],
        )

        assert spec.topology == TeamTopology.HIERARCHY
        assert spec.leader.role_in_team == "test_lead"


# =============================================================================
# Test Class: Cross-Component Integration
# =============================================================================


@pytest.mark.integration
class TestCrossComponentIntegration:
    """Tests verifying integration across persona and team components."""

    def test_persona_traits_in_team_member(self, sample_persona_traits):
        """Test PersonaTraits integration within TeamMember."""
        member = TeamMember(
            persona=sample_persona_traits,
            role_in_team="developer",
            tool_access=["edit", "git"],
        )

        # Access persona properties through member
        assert member.name == "Test Developer"
        assert member.expertise_level == ExpertiseLevel.EXPERT
        assert member.persona.communication_style == CommunicationStyle.TECHNICAL

    def test_team_spec_full_serialization(self, sample_team_template, sample_persona_traits):
        """Test full TeamSpec serialization with nested objects."""
        template = sample_team_template

        members = [
            TeamMember(
                persona=sample_persona_traits,
                role_in_team="developer",
                is_leader=True,
                max_concurrent_tasks=3,
                tool_access=["edit", "read", "git"],
            ),
            TeamMember(
                persona=PersonaTraits(
                    name="Junior Dev",
                    role="developer",
                    description="Learning developer",
                    expertise_level=ExpertiseLevel.NOVICE,
                ),
                role_in_team="developer",
                is_leader=False,
            ),
        ]

        spec = TeamSpec(template=template, members=members)
        data = spec.to_dict()

        # Verify nested structure
        assert data["template"]["name"] == "Code Review Team"
        assert len(data["members"]) == 2
        assert data["members"][0]["persona"]["name"] == "Test Developer"
        assert data["members"][0]["persona"]["expertise_level"] == "expert"
        assert data["members"][1]["persona"]["expertise_level"] == "novice"

    def test_coding_persona_with_framework_persona(self):
        """Test that coding personas can be converted to framework PersonaTraits."""
        coding_persona = get_persona("debugger")

        # Create framework PersonaTraits from coding persona
        framework_persona = PersonaTraits(
            name=coding_persona.name,
            role=coding_persona.role,
            description=coding_persona.generate_backstory(),
            communication_style=(
                CommunicationStyle.TECHNICAL
                if coding_persona.traits.communication_style == CodingCommunicationStyle.ANALYTICAL
                else CommunicationStyle.TECHNICAL
            ),
            strengths=coding_persona.strengths,
            risk_tolerance=coding_persona.traits.risk_tolerance,
            verbosity=coding_persona.traits.verbosity,
        )

        assert framework_persona.name == "Bug Hunter"
        assert framework_persona.role == "executor"
        assert len(framework_persona.description) > 50
