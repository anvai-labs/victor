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

"""Unit tests for TemplateLibrary and workflow templates.

Tests template matching, instantiation, and placeholder replacement.
"""

import pytest
from typing import Dict, Any

from victor.workflows.generation.templates import (
    TemplateLibrary,
    WorkflowTemplate,
    TemplateType,
)
from victor.workflows.generation.requirements import (
    WorkflowRequirements,
    FunctionalRequirements,
    StructuralRequirements,
    QualityRequirements,
    ContextRequirements,
    TaskRequirement,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def template_library():
    """Template library instance for testing."""
    return TemplateLibrary()


@pytest.fixture
def sample_requirements():
    """Sample workflow requirements."""
    return WorkflowRequirements(
        description="Research AI trends and summarize findings",
        functional=FunctionalRequirements(
            tasks=[
                TaskRequirement(
                    id="research",
                    description="Research AI trends",
                    task_type="agent",
                    role="researcher",
                ),
                TaskRequirement(
                    id="summarize",
                    description="Summarize findings",
                    task_type="agent",
                    role="writer",
                    dependencies=["research"],
                ),
            ],
            success_criteria=["Comprehensive summary"],
        ),
        structural=StructuralRequirements(execution_order="sequential"),
        quality=QualityRequirements(),
        context=ContextRequirements(vertical="research"),
    )


# =============================================================================
# Test: WorkflowTemplate Matching
# =============================================================================


class TestWorkflowTemplateMatching:
    """Tests for template matching logic."""

    def test_template_matches_perfectly(self):
        """Test perfect template match."""
        template = WorkflowTemplate(
            name="test_template",
            description="Test template",
            template_type=TemplateType.SEQUENTIAL,
            verticals=["research"],
            keywords=["research", "analyze"],
            execution_order="sequential",
            min_tasks=1,
            max_tasks=5,
        )

        requirements = WorkflowRequirements(
            description="Research this topic",
            functional=FunctionalRequirements(
                tasks=[
                    TaskRequirement(
                        id="t1",
                        description="Research",
                        task_type="agent",
                    )
                ]
            ),
            structural=StructuralRequirements(execution_order="sequential"),
            quality=QualityRequirements(),
            context=ContextRequirements(vertical="research"),
        )

        score = template.matches(requirements, vertical="research")

        # Should have high score (vertical + order + task count + keyword)
        assert score > 0.7

    def test_template_matches_wrong_vertical(self):
        """Test template match with wrong vertical has lower score."""
        template = WorkflowTemplate(
            name="coding_template",
            description="Coding template",
            template_type=TemplateType.SEQUENTIAL,
            verticals=["coding"],
            keywords=["code", "implement"],
            execution_order="sequential",
            min_tasks=1,
        )

        requirements = WorkflowRequirements(
            description="Research this topic",
            functional=FunctionalRequirements(
                tasks=[
                    TaskRequirement(
                        id="t1",
                        description="Research",
                        task_type="agent",
                    )
                ]
            ),
            structural=StructuralRequirements(execution_order="sequential"),
            quality=QualityRequirements(),
            context=ContextRequirements(vertical="research"),
        )

        score = template.matches(requirements, vertical="research")

        # Should have zero score (wrong vertical is hard requirement)
        assert score == 0.0

    def test_template_matches_wrong_execution_order(self):
        """Test template match with wrong execution order."""
        template = WorkflowTemplate(
            name="parallel_template",
            description="Parallel template",
            template_type=TemplateType.PARALLEL,
            verticals=["research"],
            keywords=["research"],
            execution_order="parallel",
            min_tasks=1,
        )

        requirements = WorkflowRequirements(
            description="Research this topic",
            functional=FunctionalRequirements(
                tasks=[
                    TaskRequirement(
                        id="t1",
                        description="Research",
                        task_type="agent",
                    )
                ]
            ),
            structural=StructuralRequirements(execution_order="sequential"),
            quality=QualityRequirements(),
            context=ContextRequirements(vertical="research"),
        )

        score = template.matches(requirements, vertical="research")

        # Should have lower score (wrong execution order)
        assert score < 0.8

    def test_template_matches_too_many_tasks(self):
        """Test template match fails when task count exceeds max."""
        template = WorkflowTemplate(
            name="small_template",
            description="Small template",
            template_type=TemplateType.SEQUENTIAL,
            verticals=["research"],
            keywords=["research"],
            execution_order="sequential",
            min_tasks=1,
            max_tasks=2,  # Max 2 tasks
        )

        requirements = WorkflowRequirements(
            description="Research with many tasks",
            functional=FunctionalRequirements(
                tasks=[
                    TaskRequirement(
                        id=f"t{i}",
                        description=f"Task {i}",
                        task_type="agent",
                    )
                    for i in range(5)  # 5 tasks
                ]
            ),
            structural=StructuralRequirements(execution_order="sequential"),
            quality=QualityRequirements(),
            context=ContextRequirements(vertical="research"),
        )

        score = template.matches(requirements, vertical="research")

        # Should have lower score (too many tasks)
        assert score < 0.8


# =============================================================================
# Test: TemplateLibrary
# =============================================================================


class TestTemplateLibrary:
    """Tests for TemplateLibrary functionality."""

    def test_library_initialization(self, template_library):
        """Test library initializes with templates."""
        templates = template_library.list_templates()

        assert len(templates) > 0
        assert all(isinstance(t, WorkflowTemplate) for t in templates)

    def test_list_templates_all(self, template_library):
        """Test listing all templates."""
        templates = template_library.list_templates()

        assert len(templates) > 0

    def test_list_templates_by_vertical(self, template_library):
        """Test filtering templates by vertical."""
        coding_templates = template_library.list_templates(vertical="coding")
        research_templates = template_library.list_templates(vertical="research")

        assert all("coding" in t.verticals for t in coding_templates)
        assert all("research" in t.verticals for t in research_templates)

    def test_list_templates_by_type(self, template_library):
        """Test filtering templates by type."""
        sequential_templates = template_library.list_templates(
            template_type=TemplateType.SEQUENTIAL
        )

        assert all(t.template_type == TemplateType.SEQUENTIAL for t in sequential_templates)

    def test_match_template_success(self, template_library, sample_requirements):
        """Test successful template matching."""
        template = template_library.match_template(sample_requirements, vertical="research")

        assert template is not None
        assert isinstance(template, WorkflowTemplate)
        assert "research" in template.verticals

    def test_match_template_no_match(self, template_library):
        """Test template matching when no template matches."""
        requirements = WorkflowRequirements(
            description="Very complex workflow with 100 tasks",
            functional=FunctionalRequirements(
                tasks=[
                    TaskRequirement(
                        id=f"t{i}",
                        description=f"Task {i}",
                        task_type="agent",
                    )
                    for i in range(100)
                ]
            ),
            structural=StructuralRequirements(execution_order="mixed"),
            quality=QualityRequirements(),
            context=ContextRequirements(vertical="benchmark"),
        )

        template = template_library.match_template(
            requirements, vertical="benchmark", min_score=0.9
        )

        # Should return None for high threshold
        assert template is None

    def test_instantiate_template_success(self, template_library, sample_requirements):
        """Test successful template instantiation."""
        template = template_library.match_template(sample_requirements, vertical="research")

        schema = template_library.instantiate_template(template, sample_requirements)

        assert "nodes" in schema
        assert "edges" in schema
        assert "entry_point" in schema
        assert isinstance(schema["nodes"], list)
        assert len(schema["nodes"]) > 0

    def test_instantiate_template_replaces_placeholders(
        self, template_library, sample_requirements
    ):
        """Test that placeholders are replaced in instantiated template."""
        template = template_library.match_template(sample_requirements, vertical="research")

        schema = template_library.instantiate_template(template, sample_requirements)

        # Check that common placeholders are replaced
        schema_str = str(schema)

        # Should not contain placeholder syntax
        assert "{workflow_name}" not in schema_str
        assert "{description}" not in schema_str

        # Should contain actual values
        assert sample_requirements.description in schema_str or any(
            task.description in schema_str for task in sample_requirements.functional.tasks
        )


# =============================================================================
# Test: Built-in Templates
# =============================================================================


class TestBuiltinTemplates:
    """Tests for built-in workflow templates."""

    def test_sequential_research_template_exists(self, template_library):
        """Test sequential research template exists."""
        templates = template_library.list_templates(vertical="research")
        research_templates = [t for t in templates if "sequential" in t.name.lower()]

        assert len(research_templates) > 0

    def test_conditional_research_template_exists(self, template_library):
        """Test conditional research template exists."""
        templates = template_library.list_templates(vertical="research")
        conditional_templates = [t for t in templates if "conditional" in t.name.lower()]

        assert len(conditional_templates) > 0

    def test_bug_fix_template_exists(self, template_library):
        """Test bug fix template exists."""
        templates = template_library.list_templates(vertical="coding")
        bug_templates = [t for t in templates if "bug" in t.name.lower()]

        assert len(bug_templates) > 0

    def test_deploy_template_exists(self, template_library):
        """Test deployment template exists."""
        templates = template_library.list_templates(vertical="devops")
        deploy_templates = [t for t in templates if "deploy" in t.name.lower()]

        assert len(deploy_templates) > 0

    def test_eda_template_exists(self, template_library):
        """Test EDA template exists."""
        templates = template_library.list_templates(vertical="dataanalysis")
        eda_templates = [t for t in templates if "eda" in t.name.lower()]

        assert len(eda_templates) > 0


# =============================================================================
# Test: Placeholder Replacement
# =============================================================================


class TestPlaceholderReplacement:
    """Tests for placeholder replacement in templates."""

    def test_replace_workflow_name(self, template_library, sample_requirements):
        """Test workflow name placeholder is replaced."""
        template = template_library.match_template(sample_requirements, vertical="research")

        schema = template_library.instantiate_template(template, sample_requirements)

        # Workflow name should be derived from description
        assert "workflow_name" in schema
        assert schema["workflow_name"] != "{workflow_name}"

    def test_replace_description(self, template_library, sample_requirements):
        """Test description placeholder is replaced."""
        template = template_library.match_template(sample_requirements, vertical="research")

        schema = template_library.instantiate_template(template, sample_requirements)

        # Description should match requirements
        assert schema["description"] == sample_requirements.description

    def test_replace_topic(self, template_library, sample_requirements):
        """Test topic placeholder is replaced."""
        template = template_library.match_template(sample_requirements, vertical="research")

        schema = template_library.instantiate_template(template, sample_requirements)

        # Topic should be extracted from first task or description
        # Check that no {topic} placeholder remains
        schema_str = str(schema)
        assert "{topic}" not in schema_str


# =============================================================================
# Test: Schema Validation
# =============================================================================


class TestSchemaValidation:
    """Tests for instantiated schema validation."""

    def test_instantiated_schema_has_valid_structure(self, template_library, sample_requirements):
        """Test instantiated schema has valid structure."""
        template = template_library.match_template(sample_requirements, vertical="research")

        schema = template_library.instantiate_template(template, sample_requirements)

        # Should have required top-level fields
        assert "nodes" in schema
        assert "edges" in schema
        assert "entry_point" in schema

        # Nodes should be a list
        assert isinstance(schema["nodes"], list)

        # Entry point should exist in nodes
        node_ids = {n["id"] for n in schema["nodes"]}
        assert schema["entry_point"] in node_ids

    def test_instantiated_schema_no_remaining_placeholders(
        self, template_library, sample_requirements
    ):
        """Test instantiated schema has no remaining placeholders."""
        template = template_library.match_template(sample_requirements, vertical="research")

        schema = template_library.instantiate_template(template, sample_requirements)

        # Convert to string and check for placeholders
        import json

        schema_str = json.dumps(schema)
        placeholders = [
            "{workflow_name}",
            "{description}",
            "{topic}",
        ]

        for placeholder in placeholders:
            assert placeholder not in schema_str


# =============================================================================
# Test: Template Coverage
# =============================================================================


class TestTemplateCoverage:
    """Tests for template coverage across verticals."""

    def test_all_verticals_have_templates(self, template_library):
        """Test that all major verticals have templates."""
        verticals = ["coding", "devops", "research", "dataanalysis"]

        for vertical in verticals:
            templates = template_library.list_templates(vertical=vertical)
            assert len(templates) > 0, f"No templates found for vertical: {vertical}"

    def test_all_template_types_represented(self, template_library):
        """Test that all template types are represented."""
        all_templates = template_library.list_templates()

        template_types = {t.template_type for t in all_templates}

        # Should have at least sequential and conditional
        assert TemplateType.SEQUENTIAL in template_types
        assert TemplateType.CONDITIONAL in template_types
