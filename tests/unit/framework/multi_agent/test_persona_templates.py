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

"""Tests for persona templates.

These tests verify the base persona template functions and the template
registry system.
"""

import pytest

from victor.framework.multi_agent.persona_templates import (
    PERSONA_TEMPLATES,
    get_executor_template,
    get_persona_template,
    get_planner_template,
    get_reviewer_template,
    get_researcher_template,
    list_persona_templates,
)
from victor.framework.multi_agent.personas import (
    PersonaTraits,
    CommunicationStyle,
    ExpertiseLevel,
)

# =============================================================================
# Researcher Template Tests
# =============================================================================


class TestResearcherTemplate:
    """Tests for get_researcher_template()."""

    def test_returns_persona_traits(self):
        """Should return PersonaTraits instance."""
        template = get_researcher_template()

        assert isinstance(template, PersonaTraits)

    def test_has_correct_name(self):
        """Should have 'researcher' as name."""
        template = get_researcher_template()

        assert template.name == "researcher"

    def test_has_correct_role(self):
        """Should have 'Research Specialist' as role."""
        template = get_researcher_template()

        assert template.role == "Research Specialist"

    def test_has_description(self):
        """Should have a description focused on research."""
        template = get_researcher_template()

        assert template.description is not None
        assert len(template.description) > 0
        # Description should mention information gathering or analysis
        desc_lower = template.description.lower()
        assert any(
            term in desc_lower
            for term in ["information", "gathering", "synthesis", "analysis", "sources"]
        )

    def test_has_research_strengths(self):
        """Should have research-related strengths."""
        template = get_researcher_template()

        assert "information_gathering" in template.strengths
        assert "source_verification" in template.strengths
        assert "synthesis" in template.strengths
        assert "analysis" in template.strengths

    def test_has_technical_communication_style(self):
        """Should have TECHNICAL communication style."""
        template = get_researcher_template()

        assert template.communication_style == CommunicationStyle.TECHNICAL

    def test_has_expert_expertise_level(self):
        """Should have EXPERT expertise level."""
        template = get_researcher_template()

        assert template.expertise_level == ExpertiseLevel.EXPERT

    def test_has_higher_verbosity(self):
        """Should have higher verbosity for detailed output."""
        template = get_researcher_template()

        assert template.verbosity == 0.7

    def test_has_research_traits_in_custom_traits(self):
        """Should have research-oriented traits in custom_traits."""
        template = get_researcher_template()

        assert "traits" in template.custom_traits
        traits = template.custom_traits["traits"]

        assert "thorough" in traits
        assert "systematic" in traits
        assert "evidence_based" in traits
        assert "objective" in traits

    def test_has_prompt_extensions(self):
        """Should have prompt_extensions in custom_traits."""
        template = get_researcher_template()

        assert "prompt_extensions" in template.custom_traits
        prompt_extensions = template.custom_traits["prompt_extensions"]

        assert prompt_extensions is not None
        assert isinstance(prompt_extensions, dict)

    def test_prompt_extensions_content(self):
        """Prompt extensions should guide research behavior."""
        template = get_researcher_template()

        extensions = template.custom_traits["prompt_extensions"]

        assert "focus" in extensions
        assert "approach" in extensions
        assert "output" in extensions

        # Check content makes sense for research
        focus_lower = extensions["focus"].lower()
        assert any(term in focus_lower for term in ["research", "comprehensive", "sources"])

    def test_is_immutable(self):
        """Template should be a new instance each time."""
        template1 = get_researcher_template()
        template2 = get_researcher_template()

        # Should be separate instances
        assert template1 is not template2

        # Modifying one should not affect the other
        template1.strengths.append("modified")
        assert "modified" not in template2.strengths


# =============================================================================
# Planner Template Tests
# =============================================================================


class TestPlannerTemplate:
    """Tests for get_planner_template()."""

    def test_returns_persona_traits(self):
        """Should return PersonaTraits instance."""
        template = get_planner_template()

        assert isinstance(template, PersonaTraits)

    def test_has_correct_name(self):
        """Should have 'planner' as name."""
        template = get_planner_template()

        assert template.name == "planner"

    def test_has_correct_role(self):
        """Should have 'Planning Specialist' as role."""
        template = get_planner_template()

        assert template.role == "Planning Specialist"

    def test_has_description(self):
        """Should have a description focused on planning."""
        template = get_planner_template()

        assert template.description is not None
        assert len(template.description) > 0
        assert "planning" in template.description.lower() or "plan" in template.description.lower()

    def test_has_planning_strengths(self):
        """Should have planning-related strengths."""
        template = get_planner_template()

        assert "task_breakdown" in template.strengths
        assert "architecture" in template.strengths
        assert "sequencing" in template.strengths
        assert "dependency_analysis" in template.strengths

    def test_has_technical_communication_style(self):
        """Should have TECHNICAL communication style."""
        template = get_planner_template()

        assert template.communication_style == CommunicationStyle.TECHNICAL

    def test_has_expert_expertise_level(self):
        """Should have EXPERT expertise level."""
        template = get_planner_template()

        assert template.expertise_level == ExpertiseLevel.EXPERT

    def test_has_moderate_verbosity(self):
        """Should have moderate verbosity."""
        template = get_planner_template()

        assert template.verbosity == 0.6

    def test_has_planning_traits_in_custom_traits(self):
        """Should have planning-oriented traits in custom_traits."""
        template = get_planner_template()

        assert "traits" in template.custom_traits
        traits = template.custom_traits["traits"]

        assert "strategic" in traits
        assert "methodical" in traits
        assert "forward_thinking" in traits
        assert "risk_aware" in traits

    def test_has_prompt_extensions(self):
        """Should have prompt_extensions in custom_traits."""
        template = get_planner_template()

        assert "prompt_extensions" in template.custom_traits
        prompt_extensions = template.custom_traits["prompt_extensions"]

        assert prompt_extensions is not None
        assert isinstance(prompt_extensions, dict)

    def test_prompt_extensions_content(self):
        """Prompt extensions should guide planning behavior."""
        template = get_planner_template()

        extensions = template.custom_traits["prompt_extensions"]

        assert "focus" in extensions
        assert "approach" in extensions
        assert "output" in extensions

        # Check content makes sense for planning
        focus_lower = extensions["focus"].lower()
        assert any(term in focus_lower for term in ["structured", "plan", "approach"])

    def test_is_immutable(self):
        """Template should be a new instance each time."""
        template1 = get_planner_template()
        template2 = get_planner_template()

        # Should be separate instances
        assert template1 is not template2

        # Modifying one should not affect the other
        template1.strengths.append("modified")
        assert "modified" not in template2.strengths


# =============================================================================
# Executor Template Tests
# =============================================================================


class TestExecutorTemplate:
    """Tests for get_executor_template()."""

    def test_returns_persona_traits(self):
        """Should return PersonaTraits instance."""
        template = get_executor_template()

        assert isinstance(template, PersonaTraits)

    def test_has_correct_name(self):
        """Should have 'executor' as name."""
        template = get_executor_template()

        assert template.name == "executor"

    def test_has_correct_role(self):
        """Should have 'Execution Specialist' as role."""
        template = get_executor_template()

        assert template.role == "Execution Specialist"

    def test_has_description(self):
        """Should have a description focused on execution."""
        template = get_executor_template()

        assert template.description is not None
        assert len(template.description) > 0
        # Description should mention implementation or efficiency
        desc_lower = template.description.lower()
        assert any(term in desc_lower for term in ["implement", "solution", "efficient", "correct"])

    def test_has_execution_strengths(self):
        """Should have execution-related strengths."""
        template = get_executor_template()

        assert "implementation" in template.strengths
        assert "debugging" in template.strengths
        assert "testing" in template.strengths
        assert "optimization" in template.strengths

    def test_has_concise_communication_style(self):
        """Should have CONCISE communication style."""
        template = get_executor_template()

        assert template.communication_style == CommunicationStyle.CONCISE

    def test_has_expert_expertise_level(self):
        """Should have EXPERT expertise level."""
        template = get_executor_template()

        assert template.expertise_level == ExpertiseLevel.EXPERT

    def test_has_lower_verbosity(self):
        """Should have lower verbosity for direct action."""
        template = get_executor_template()

        assert template.verbosity == 0.4

    def test_has_execution_traits_in_custom_traits(self):
        """Should have execution-oriented traits in custom_traits."""
        template = get_executor_template()

        assert "traits" in template.custom_traits
        traits = template.custom_traits["traits"]

        assert "focused" in traits
        assert "efficient" in traits
        assert "quality_conscious" in traits
        assert "pragmatic" in traits

    def test_has_prompt_extensions(self):
        """Should have prompt_extensions in custom_traits."""
        template = get_executor_template()

        assert "prompt_extensions" in template.custom_traits
        prompt_extensions = template.custom_traits["prompt_extensions"]

        assert prompt_extensions is not None
        assert isinstance(prompt_extensions, dict)

    def test_prompt_extensions_content(self):
        """Prompt extensions should guide execution behavior."""
        template = get_executor_template()

        extensions = template.custom_traits["prompt_extensions"]

        assert "focus" in extensions
        assert "approach" in extensions
        assert "output" in extensions

        # Check content makes sense for execution
        focus_lower = extensions["focus"].lower()
        assert any(term in focus_lower for term in ["implement", "correct", "efficient"])

    def test_is_immutable(self):
        """Template should be a new instance each time."""
        template1 = get_executor_template()
        template2 = get_executor_template()

        # Should be separate instances
        assert template1 is not template2

        # Modifying one should not affect the other
        template1.strengths.append("modified")
        assert "modified" not in template2.strengths


# =============================================================================
# Reviewer Template Tests
# =============================================================================


class TestReviewerTemplate:
    """Tests for get_reviewer_template()."""

    def test_returns_persona_traits(self):
        """Should return PersonaTraits instance."""
        template = get_reviewer_template()

        assert isinstance(template, PersonaTraits)

    def test_has_correct_name(self):
        """Should have 'reviewer' as name."""
        template = get_reviewer_template()

        assert template.name == "reviewer"

    def test_has_correct_role(self):
        """Should have 'Review Specialist' as role."""
        template = get_reviewer_template()

        assert template.role == "Review Specialist"

    def test_has_description(self):
        """Should have a description focused on review."""
        template = get_reviewer_template()

        assert template.description is not None
        assert len(template.description) > 0
        assert "review" in template.description.lower() or "evaluat" in template.description.lower()

    def test_has_review_strengths(self):
        """Should have review-related strengths."""
        template = get_reviewer_template()

        assert "code_review" in template.strengths
        assert "quality_assessment" in template.strengths
        assert "best_practices" in template.strengths
        assert "security" in template.strengths

    def test_has_formal_communication_style(self):
        """Should have FORMAL communication style."""
        template = get_reviewer_template()

        assert template.communication_style == CommunicationStyle.FORMAL

    def test_has_specialist_expertise_level(self):
        """Should have SPECIALIST expertise level."""
        template = get_reviewer_template()

        assert template.expertise_level == ExpertiseLevel.SPECIALIST

    def test_has_moderate_verbosity(self):
        """Should have moderate verbosity."""
        template = get_reviewer_template()

        assert template.verbosity == 0.6

    def test_has_review_traits_in_custom_traits(self):
        """Should have review-oriented traits in custom_traits."""
        template = get_reviewer_template()

        assert "traits" in template.custom_traits
        traits = template.custom_traits["traits"]

        assert "detail_oriented" in traits
        assert "critical_thinking" in traits
        assert "standards_driven" in traits
        assert "helpful" in traits

    def test_has_prompt_extensions(self):
        """Should have prompt_extensions in custom_traits."""
        template = get_reviewer_template()

        assert "prompt_extensions" in template.custom_traits
        prompt_extensions = template.custom_traits["prompt_extensions"]

        assert prompt_extensions is not None
        assert isinstance(prompt_extensions, dict)

    def test_prompt_extensions_content(self):
        """Prompt extensions should guide review behavior."""
        template = get_reviewer_template()

        extensions = template.custom_traits["prompt_extensions"]

        assert "focus" in extensions
        assert "approach" in extensions
        assert "output" in extensions

        # Check content makes sense for review
        focus_lower = extensions["focus"].lower()
        assert any(term in focus_lower for term in ["quality", "best", "practice"])

    def test_is_immutable(self):
        """Template should be a new instance each time."""
        template1 = get_reviewer_template()
        template2 = get_reviewer_template()

        # Should be separate instances
        assert template1 is not template2

        # Modifying one should not affect the other
        template1.strengths.append("modified")
        assert "modified" not in template2.strengths


# =============================================================================
# Template Registry Tests
# =============================================================================


class TestPersonaTemplateRegistry:
    """Tests for PERSONA_TEMPLATES registry."""

    def test_has_all_four_templates(self):
        """Registry should contain all four base templates."""
        assert "researcher" in PERSONA_TEMPLATES
        assert "planner" in PERSONA_TEMPLATES
        assert "executor" in PERSONA_TEMPLATES
        assert "reviewer" in PERSONA_TEMPLATES

    def test_all_templates_are_persona_traits(self):
        """All templates in registry should be PersonaTraits instances."""
        for name, template in PERSONA_TEMPLATES.items():
            assert isinstance(template, PersonaTraits), f"{name} is not a PersonaTraits instance"

    def test_researcher_template_matches(self):
        """Registry researcher template should match get_researcher_template()."""
        registry_template = PERSONA_TEMPLATES["researcher"]
        function_template = get_researcher_template()

        assert registry_template.name == function_template.name
        assert registry_template.role == function_template.role
        assert registry_template.strengths == function_template.strengths

    def test_planner_template_matches(self):
        """Registry planner template should match get_planner_template()."""
        registry_template = PERSONA_TEMPLATES["planner"]
        function_template = get_planner_template()

        assert registry_template.name == function_template.name
        assert registry_template.role == function_template.role
        assert registry_template.strengths == function_template.strengths

    def test_executor_template_matches(self):
        """Registry executor template should match get_executor_template()."""
        registry_template = PERSONA_TEMPLATES["executor"]
        function_template = get_executor_template()

        assert registry_template.name == function_template.name
        assert registry_template.role == function_template.role
        assert registry_template.strengths == function_template.strengths

    def test_reviewer_template_matches(self):
        """Registry reviewer template should match get_reviewer_template()."""
        registry_template = PERSONA_TEMPLATES["reviewer"]
        function_template = get_reviewer_template()

        assert registry_template.name == function_template.name
        assert registry_template.role == function_template.role
        assert registry_template.strengths == function_template.strengths


# =============================================================================
# get_persona_template() Tests
# =============================================================================


class TestGetPersonaTemplate:
    """Tests for get_persona_template() function."""

    def test_get_researcher_template(self):
        """Should retrieve researcher template by name."""
        template = get_persona_template("researcher")

        assert template is not None
        assert isinstance(template, PersonaTraits)
        assert template.name == "researcher"

    def test_get_planner_template(self):
        """Should retrieve planner template by name."""
        template = get_persona_template("planner")

        assert template is not None
        assert isinstance(template, PersonaTraits)
        assert template.name == "planner"

    def test_get_executor_template(self):
        """Should retrieve executor template by name."""
        template = get_persona_template("executor")

        assert template is not None
        assert isinstance(template, PersonaTraits)
        assert template.name == "executor"

    def test_get_reviewer_template(self):
        """Should retrieve reviewer template by name."""
        template = get_persona_template("reviewer")

        assert template is not None
        assert isinstance(template, PersonaTraits)
        assert template.name == "reviewer"

    def test_get_nonexistent_template(self):
        """Should return None for non-existent template."""
        template = get_persona_template("nonexistent")

        assert template is None

    def test_get_template_with_case_sensitive_name(self):
        """Template names should be case-sensitive."""
        template = get_persona_template("Researcher")

        assert template is None  # Case doesn't match


# =============================================================================
# list_persona_templates() Tests
# =============================================================================


class TestListPersonaTemplates:
    """Tests for list_persona_templates() function."""

    def test_returns_list(self):
        """Should return a list of template names."""
        templates = list_persona_templates()

        assert isinstance(templates, list)

    def test_contains_all_templates(self):
        """Should contain all four template names."""
        templates = list_persona_templates()

        assert "researcher" in templates
        assert "planner" in templates
        assert "executor" in templates
        assert "reviewer" in templates

    def test_length(self):
        """Should return exactly four templates."""
        templates = list_persona_templates()

        assert len(templates) == 4

    def test_returns_string_names(self):
        """All items in list should be strings."""
        templates = list_persona_templates()

        for template_name in templates:
            assert isinstance(template_name, str)


# =============================================================================
# Template Customization Tests
# =============================================================================


class TestTemplateCustomization:
    """Tests for customizing base templates."""

    def test_template_can_be_copied(self):
        """Templates should support copying for customization."""
        base = get_researcher_template()

        # Create a customized version using PersonaTraits constructor
        customized = PersonaTraits(
            name="Security Researcher",
            role="security_researcher",
            description="Specializes in vulnerability research and threat analysis",
            communication_style=base.communication_style,
            expertise_level=base.expertise_level,
            strengths=base.strengths + ["vulnerability_assessment", "threat_modeling"],
            verbosity=base.verbosity,
            custom_traits={
                **base.custom_traits,
                "traits": base.custom_traits.get("traits", []) + ["security_focused"],
            },
        )

        assert customized.name == "Security Researcher"
        assert customized.role == "security_researcher"
        assert "vulnerability_assessment" in customized.strengths
        assert "threat_modeling" in customized.strengths
        assert "information_gathering" in customized.strengths  # Inherited

    def test_researcher_customization(self):
        """Researcher template can be customized for specific domains."""
        base = get_researcher_template()

        # Customize for security research
        security_researcher = PersonaTraits(
            name="Security Researcher",
            role="security_researcher",
            description="Specializes in vulnerability research and threat analysis",
            communication_style=base.communication_style,
            expertise_level=base.expertise_level,
            strengths=base.strengths + ["vulnerability_assessment", "threat_modeling"],
            verbosity=base.verbosity,
            custom_traits={
                **base.custom_traits,
                "traits": base.custom_traits.get("traits", []) + ["security_focused"],
            },
        )

        assert security_researcher.name == "Security Researcher"
        assert security_researcher.role == "security_researcher"
        assert "vulnerability_assessment" in security_researcher.strengths
        assert "threat_modeling" in security_researcher.strengths
        assert "information_gathering" in security_researcher.strengths  # Inherited

    def test_planner_customization(self):
        """Planner template can be customized for specific domains."""
        base = get_planner_template()

        # Customize for architecture planning
        arch_planner = PersonaTraits(
            name="Architecture Planner",
            role="architect",
            description="Specializes in system architecture and design",
            communication_style=base.communication_style,
            expertise_level=base.expertise_level,
            strengths=base.strengths + ["system_design", "scalability_planning"],
            verbosity=base.verbosity,
            custom_traits=base.custom_traits,
        )

        assert arch_planner.name == "Architecture Planner"
        assert arch_planner.role == "architect"
        assert "system_design" in arch_planner.strengths
        assert "scalability_planning" in arch_planner.strengths
        assert "task_breakdown" in arch_planner.strengths  # Inherited

    def test_executor_customization(self):
        """Executor template can be customized for specific domains."""
        base = get_executor_template()

        # Customize for database operations
        db_executor = PersonaTraits(
            name="Database Executor",
            role="db_specialist",
            description="Specializes in database implementation and optimization",
            communication_style=base.communication_style,
            expertise_level=base.expertise_level,
            strengths=base.strengths + ["sql_optimization", "schema_design"],
            verbosity=base.verbosity,
            custom_traits=base.custom_traits,
        )

        assert db_executor.name == "Database Executor"
        assert db_executor.role == "db_specialist"
        assert "sql_optimization" in db_executor.strengths
        assert "schema_design" in db_executor.strengths
        assert "implementation" in db_executor.strengths  # Inherited

    def test_reviewer_customization(self):
        """Reviewer template can be customized for specific domains."""
        base = get_reviewer_template()

        # Customize for security review
        security_reviewer = PersonaTraits(
            name="Security Reviewer",
            role="security_auditor",
            description="Specializes in security vulnerability assessment",
            communication_style=base.communication_style,
            expertise_level=base.expertise_level,
            strengths=base.strengths + ["penetration_testing", "compliance"],
            verbosity=base.verbosity,
            custom_traits=base.custom_traits,
        )

        assert security_reviewer.name == "Security Reviewer"
        assert security_reviewer.role == "security_auditor"
        assert "penetration_testing" in security_reviewer.strengths
        assert "compliance" in security_reviewer.strengths
        assert "code_review" in security_reviewer.strengths  # Inherited


# =============================================================================
# Template Consistency Tests
# =============================================================================


class TestTemplateConsistency:
    """Tests for consistency across templates."""

    def test_all_templates_have_required_fields(self):
        """All templates should have required fields populated."""
        templates = {
            "researcher": get_researcher_template(),
            "planner": get_planner_template(),
            "executor": get_executor_template(),
            "reviewer": get_reviewer_template(),
        }

        for name, template in templates.items():
            assert template.name is not None, f"{name} missing name"
            assert template.role is not None, f"{name} missing role"
            assert template.description is not None, f"{name} missing description"
            assert len(template.strengths) > 0, f"{name} has empty strengths"

    def test_all_templates_have_unique_names(self):
        """Each template should have a unique name."""
        templates = [
            get_researcher_template(),
            get_planner_template(),
            get_executor_template(),
            get_reviewer_template(),
        ]

        names = [t.name for t in templates]
        assert len(names) == len(set(names)), "Template names are not unique"

    def test_all_templates_have_unique_roles(self):
        """Each template should have a unique role."""
        templates = [
            get_researcher_template(),
            get_planner_template(),
            get_executor_template(),
            get_reviewer_template(),
        ]

        roles = [t.role for t in templates]
        assert len(roles) == len(set(roles)), "Template roles are not unique"

    def test_all_templates_have_prompt_extensions(self):
        """All templates should have prompt_extensions in custom_traits."""
        templates = {
            "researcher": get_researcher_template(),
            "planner": get_planner_template(),
            "executor": get_executor_template(),
            "reviewer": get_reviewer_template(),
        }

        for name, template in templates.items():
            assert (
                "prompt_extensions" in template.custom_traits
            ), f"{name} missing prompt_extensions"
            assert isinstance(
                template.custom_traits["prompt_extensions"], dict
            ), f"{name} prompt_extensions not a dict"
            assert (
                len(template.custom_traits["prompt_extensions"]) > 0
            ), f"{name} has empty prompt_extensions"

    def test_all_prompt_extensions_have_consistent_keys(self):
        """All prompt_extensions should have consistent structure."""
        templates = {
            "researcher": get_researcher_template(),
            "planner": get_planner_template(),
            "executor": get_executor_template(),
            "reviewer": get_reviewer_template(),
        }

        expected_keys = {"focus", "approach", "output"}

        for name, template in templates.items():
            actual_keys = set(template.custom_traits["prompt_extensions"].keys())
            assert actual_keys == expected_keys, f"{name} has unexpected keys: {actual_keys}"

    def test_template_descriptions_are_meaningful(self):
        """All template descriptions should be meaningful."""
        templates = {
            "researcher": get_researcher_template(),
            "planner": get_planner_template(),
            "executor": get_executor_template(),
            "reviewer": get_reviewer_template(),
        }

        for name, template in templates.items():
            assert len(template.description) > 20, f"{name} description too short"
            # Should not be generic placeholder text
            assert "TODO" not in template.description, f"{name} has placeholder description"
            assert "Lorem ipsum" not in template.description, f"{name} has placeholder description"

    def test_all_templates_have_valid_communication_styles(self):
        """All templates should have valid CommunicationStyle enum values."""
        templates = {
            "researcher": get_researcher_template(),
            "planner": get_planner_template(),
            "executor": get_executor_template(),
            "reviewer": get_reviewer_template(),
        }

        for name, template in templates.items():
            assert isinstance(
                template.communication_style, CommunicationStyle
            ), f"{name} has invalid communication_style type"

    def test_all_templates_have_valid_expertise_levels(self):
        """All templates should have valid ExpertiseLevel enum values."""
        templates = {
            "researcher": get_researcher_template(),
            "planner": get_planner_template(),
            "executor": get_executor_template(),
            "reviewer": get_reviewer_template(),
        }

        for name, template in templates.items():
            assert isinstance(
                template.expertise_level, ExpertiseLevel
            ), f"{name} has invalid expertise_level type"


# =============================================================================
# Module Export Tests
# =============================================================================


class TestModuleExports:
    """Tests for module exports."""

    def test_all_exports_from_module(self):
        """All expected items should be exported from the module."""
        from victor.framework.multi_agent import persona_templates

        assert hasattr(persona_templates, "get_researcher_template")
        assert hasattr(persona_templates, "get_planner_template")
        assert hasattr(persona_templates, "get_executor_template")
        assert hasattr(persona_templates, "get_reviewer_template")
        assert hasattr(persona_templates, "get_persona_template")
        assert hasattr(persona_templates, "list_persona_templates")
        assert hasattr(persona_templates, "PERSONA_TEMPLATES")
