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

"""Unit tests for WorkflowGenerator.

Tests the core workflow generation functionality including:
- Multi-stage generation
- Single-stage generation
- Template-based generation
- Schema validation
- Refinement
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any

from victor.workflows.generation.generator import (
    WorkflowGenerator,
    GenerationStrategy,
    GenerationMetadata,
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
def mock_orchestrator():
    """Mock orchestrator for LLM calls."""
    orchestrator = MagicMock()
    orchestrator.current_model = "claude-sonnet-4-5"
    orchestrator.chat = AsyncMock()
    return orchestrator


@pytest.fixture
def sample_requirements():
    """Sample workflow requirements for testing."""
    return WorkflowRequirements(
        description="Analyze code and fix bugs",
        functional=FunctionalRequirements(
            tasks=[
                TaskRequirement(
                    id="analyze",
                    description="Analyze codebase",
                    task_type="agent",
                    role="researcher",
                ),
                TaskRequirement(
                    id="fix",
                    description="Fix bugs",
                    task_type="agent",
                    role="executor",
                    dependencies=["analyze"],
                ),
            ],
            tools={"analyze": ["read"], "fix": ["write"]},
            success_criteria=["All tests pass"],
        ),
        structural=StructuralRequirements(
            execution_order="sequential",
            dependencies={"fix": ["analyze"]},
        ),
        quality=QualityRequirements(max_duration_seconds=300),
        context=ContextRequirements(vertical="coding"),
    )


@pytest.fixture
def sample_workflow_schema():
    """Sample valid workflow schema."""
    return {
        "workflow_name": "bug_fix",
        "description": "Fix bugs in code",
        "nodes": [
            {
                "id": "analyze",
                "type": "agent",
                "role": "researcher",
                "goal": "Analyze codebase",
                "tool_budget": 10,
                "output_key": "analysis",
            },
            {
                "id": "fix",
                "type": "agent",
                "role": "executor",
                "goal": "Fix bugs",
                "tool_budget": 15,
                "output_key": "fixes",
            },
        ],
        "edges": [
            {"source": "analyze", "target": "fix", "type": "normal"},
            {"source": "fix", "target": "__end__", "type": "normal"},
        ],
        "entry_point": "analyze",
        "metadata": {"vertical": "coding", "max_iterations": 20},
    }


# =============================================================================
# Test: Generator Initialization
# =============================================================================


class TestWorkflowGeneratorInitialization:
    """Tests for WorkflowGenerator initialization."""

    def test_init_with_defaults(self, mock_orchestrator):
        """Test generator initialization with default parameters."""
        generator = WorkflowGenerator(
            mock_orchestrator,
            vertical="coding",
        )

        assert generator._orchestrator == mock_orchestrator
        assert generator._vertical == "coding"
        assert generator._strategy == GenerationStrategy.LLM_MULTI_STAGE
        assert generator._max_retries == 3
        assert generator._enable_templates is True

    def test_init_with_custom_strategy(self, mock_orchestrator):
        """Test generator initialization with custom strategy."""
        generator = WorkflowGenerator(
            mock_orchestrator,
            vertical="research",
            strategy=GenerationStrategy.LLM_SINGLE_STAGE,
            max_retries=5,
            enable_templates=False,
        )

        assert generator._strategy == GenerationStrategy.LLM_SINGLE_STAGE
        assert generator._max_retries == 5
        assert generator._enable_templates is False


# =============================================================================
# Test: Requirement Validation
# =============================================================================


class TestRequirementValidation:
    """Tests for requirement validation before generation."""

    def test_validate_requirements_success(self, mock_orchestrator, sample_requirements):
        """Test validation of valid requirements."""
        generator = WorkflowGenerator(mock_orchestrator, vertical="coding")

        # Should not raise
        generator._validate_requirements(sample_requirements)

    def test_validate_requirements_missing_description(self, mock_orchestrator):
        """Test validation fails when description is missing."""
        generator = WorkflowGenerator(mock_orchestrator, vertical="coding")

        requirements = WorkflowRequirements(
            description="",  # Empty description
            functional=FunctionalRequirements(tasks=[]),
            structural=StructuralRequirements(execution_order="sequential"),
            quality=QualityRequirements(),
            context=ContextRequirements(vertical="coding"),
        )

        with pytest.raises(ValueError, match="must have a description"):
            generator._validate_requirements(requirements)

    def test_validate_requirements_no_tasks(self, mock_orchestrator):
        """Test validation fails when no tasks are defined."""
        generator = WorkflowGenerator(mock_orchestrator, vertical="coding")

        requirements = WorkflowRequirements(
            description="Test workflow",
            functional=FunctionalRequirements(tasks=[]),  # No tasks
            structural=StructuralRequirements(execution_order="sequential"),
            quality=QualityRequirements(),
            context=ContextRequirements(vertical="coding"),
        )

        with pytest.raises(ValueError, match="must have at least one task"):
            generator._validate_requirements(requirements)


# =============================================================================
# Test: Schema Validation
# =============================================================================


class TestSchemaValidation:
    """Tests for schema structure validation."""

    def test_validate_schema_success(self, mock_orchestrator, sample_workflow_schema):
        """Test validation of valid schema."""
        generator = WorkflowGenerator(mock_orchestrator, vertical="coding")

        # Should not raise
        generator._validate_schema_structure(sample_workflow_schema)

    def test_validate_schema_missing_nodes(self, mock_orchestrator):
        """Test validation fails when nodes are missing."""
        generator = WorkflowGenerator(mock_orchestrator, vertical="coding")

        invalid_schema = {"entry_point": "start"}

        with pytest.raises(ValueError, match="missing required field.*nodes"):
            generator._validate_schema_structure(invalid_schema)

    def test_validate_schema_missing_entry_point(self, mock_orchestrator):
        """Test validation fails when entry_point is missing."""
        generator = WorkflowGenerator(mock_orchestrator, vertical="coding")

        invalid_schema = {
            "nodes": [{"id": "node1", "type": "agent"}],
            "edges": [],
        }

        with pytest.raises(ValueError, match="missing required field.*entry_point"):
            generator._validate_schema_structure(invalid_schema)

    def test_validate_schema_invalid_node_type(self, mock_orchestrator):
        """Test validation fails with invalid node type."""
        generator = WorkflowGenerator(mock_orchestrator, vertical="coding")

        invalid_schema = {
            "nodes": [
                {"id": "node1", "type": "invalid_type"},  # Invalid type
            ],
            "edges": [],
            "entry_point": "node1",
        }

        with pytest.raises(ValueError, match="has invalid type"):
            generator._validate_schema_structure(invalid_schema)

    def test_validate_schema_invalid_entry_point(self, mock_orchestrator):
        """Test validation fails when entry_point doesn't exist."""
        generator = WorkflowGenerator(mock_orchestrator, vertical="coding")

        invalid_schema = {
            "nodes": [{"id": "node1", "type": "agent"}],
            "edges": [],
            "entry_point": "nonexistent",  # Not in nodes
        }

        with pytest.raises(ValueError, match="Entry point.*not found"):
            generator._validate_schema_structure(invalid_schema)

    def test_validate_schema_invalid_edge_target(self, mock_orchestrator):
        """Test validation fails when edge target doesn't exist."""
        generator = WorkflowGenerator(mock_orchestrator, vertical="coding")

        invalid_schema = {
            "nodes": [{"id": "node1", "type": "agent"}],
            "edges": [
                {
                    "source": "node1",
                    "target": "nonexistent",  # Not in nodes
                    "type": "normal",
                }
            ],
            "entry_point": "node1",
        }

        with pytest.raises(ValueError, match="Edge target.*not found"):
            generator._validate_schema_structure(invalid_schema)


# =============================================================================
# Test: JSON Parsing
# =============================================================================


class TestJSONParsing:
    """Tests for JSON response parsing."""

    def test_parse_json_response_clean(self, mock_orchestrator):
        """Test parsing clean JSON response."""
        generator = WorkflowGenerator(mock_orchestrator, vertical="coding")

        response = '{"key": "value", "number": 123}'
        result = generator._parse_json_response(response)

        assert result == {"key": "value", "number": 123}

    def test_parse_json_response_with_markdown(self, mock_orchestrator):
        """Test parsing JSON wrapped in markdown code blocks."""
        generator = WorkflowGenerator(mock_orchestrator, vertical="coding")

        response = """```json
        {"key": "value"}
        ```"""
        result = generator._parse_json_response(response)

        assert result == {"key": "value"}

    def test_parse_json_response_with_plain_markdown(self, mock_orchestrator):
        """Test parsing JSON wrapped in plain markdown blocks."""
        generator = WorkflowGenerator(mock_orchestrator, vertical="coding")

        response = """```
        {"key": "value"}
        ```"""
        result = generator._parse_json_response(response)

        assert result == {"key": "value"}

    def test_parse_json_response_invalid(self, mock_orchestrator):
        """Test parsing invalid JSON raises error."""
        generator = WorkflowGenerator(mock_orchestrator, vertical="coding")

        response = "This is not valid JSON"

        with pytest.raises(ValueError, match="Invalid JSON"):
            generator._parse_json_response(response)


# =============================================================================
# Test: Multi-Stage Generation
# =============================================================================


class TestMultiStageGeneration:
    """Tests for multi-stage generation strategy."""

    @pytest.mark.asyncio
    async def test_generate_multi_stage_success(self, mock_orchestrator, sample_requirements):
        """Test successful multi-stage generation."""
        generator = WorkflowGenerator(
            mock_orchestrator,
            vertical="coding",
            strategy=GenerationStrategy.LLM_MULTI_STAGE,
        )

        # Mock LLM responses
        mock_orchestrator.chat.side_effect = [
            "Understanding: Analyze code and fix bugs sequentially",
            '{"nodes": [{"id": "analyze", "type": "agent"}], "flow": "analyze -> fix"}',
            '{"workflow_name": "test", "nodes": [{"id": "analyze", "type": "agent", "role": "executor", "goal": "Analyze"}], "edges": [{"source": "analyze", "target": "__end__", "type": "normal"}], "entry_point": "analyze"}',
        ]

        schema, metadata = await generator._generate_multi_stage(sample_requirements, attempt=1)

        assert metadata.strategy == GenerationStrategy.LLM_MULTI_STAGE
        assert metadata.iterations == 3
        assert schema is not None

    @pytest.mark.asyncio
    async def test_generate_multi_stage_llm_failure(self, mock_orchestrator, sample_requirements):
        """Test multi-stage generation handles LLM failures."""
        generator = WorkflowGenerator(
            mock_orchestrator,
            vertical="coding",
            strategy=GenerationStrategy.LLM_MULTI_STAGE,
        )

        # Mock LLM failure
        mock_orchestrator.chat.side_effect = Exception("LLM API error")

        with pytest.raises(Exception, match="LLM API error"):
            await generator._generate_multi_stage(sample_requirements, attempt=1)


# =============================================================================
# Test: Template-Based Generation
# =============================================================================


class TestTemplateGeneration:
    """Tests for template-based generation."""

    @pytest.mark.asyncio
    async def test_generate_from_template_success(self, mock_orchestrator, sample_requirements):
        """Test successful template-based generation."""
        generator = WorkflowGenerator(
            mock_orchestrator,
            vertical="coding",
        )

        schema, metadata = await generator._generate_from_template(sample_requirements)

        assert metadata.strategy == GenerationStrategy.TEMPLATE_BASED
        assert metadata.iterations == 0
        assert "nodes" in schema
        assert "edges" in schema
        assert "entry_point" in schema

    @pytest.mark.asyncio
    async def test_generate_from_template_no_match(self, mock_orchestrator, sample_requirements):
        """Test template generation fails when no template matches."""
        generator = WorkflowGenerator(
            mock_orchestrator,
            vertical="benchmark",  # Vertical with no templates
        )

        with pytest.raises(ValueError, match="No matching template found"):
            await generator._generate_from_template(sample_requirements)


# =============================================================================
# Test: Schema Refinement
# =============================================================================


class TestSchemaRefinement:
    """Tests for schema refinement."""

    @pytest.mark.asyncio
    async def test_refine_schema_success(self, mock_orchestrator, sample_workflow_schema):
        """Test successful schema refinement."""
        generator = WorkflowGenerator(mock_orchestrator, vertical="coding")

        feedback = "Add a validation step"
        mock_orchestrator.chat.return_value = """{"workflow_name": "refined", "nodes": [{"id": "validate", "type": "agent", "role": "reviewer", "goal": "Validate changes", "tool_budget": 5, "output_key": "validation"}], "edges": [{"source": "validate", "target": "__end__", "type": "normal"}], "entry_point": "validate"}"""

        refined = await generator.refine_schema(sample_workflow_schema, feedback)

        assert "nodes" in refined
        assert "edges" in refined

    @pytest.mark.asyncio
    async def test_refine_schema_with_validation_errors(
        self, mock_orchestrator, sample_workflow_schema
    ):
        """Test refinement with validation error context."""
        from victor.workflows.generation.types import (
            WorkflowValidationError,
            ErrorCategory,
        )

        generator = WorkflowGenerator(mock_orchestrator, vertical="coding")

        errors = [
            WorkflowValidationError(
                category=ErrorCategory.SCHEMA,
                severity="error",
                message="Missing field",
                location="nodes[0]",
            )
        ]

        mock_orchestrator.chat.return_value = """{"workflow_name": "refined", "nodes": [{"id": "start", "type": "agent", "role": "planner", "goal": "Start workflow", "tool_budget": 5, "output_key": "result"}], "edges": [{"source": "start", "target": "__end__", "type": "normal"}], "entry_point": "start"}"""

        refined = await generator.refine_schema(sample_workflow_schema, "Fix errors", errors)

        assert "nodes" in refined

    @pytest.mark.asyncio
    async def test_refine_schema_llm_failure(self, mock_orchestrator, sample_workflow_schema):
        """Test refinement handles LLM failures."""
        generator = WorkflowGenerator(mock_orchestrator, vertical="coding")

        mock_orchestrator.chat.side_effect = Exception("LLM error")

        with pytest.raises(RuntimeError, match="Failed to refine schema"):
            await generator.refine_schema(sample_workflow_schema, "Fix this")


# =============================================================================
# Test: Full Generation Pipeline
# =============================================================================


class TestFullGeneration:
    """Tests for full generation workflow."""

    @pytest.mark.asyncio
    async def test_generate_from_requirements_success(self, mock_orchestrator, sample_requirements):
        """Test successful generation from requirements."""
        generator = WorkflowGenerator(
            mock_orchestrator,
            vertical="coding",
            max_retries=1,
        )

        # Mock LLM response
        mock_orchestrator.chat.return_value = '{"workflow_name": "test", "nodes": [{"id": "n1", "type": "agent", "role": "executor", "goal": "Test"}], "edges": [{"source": "n1", "target": "__end__", "type": "normal"}], "entry_point": "n1"}'

        schema, metadata = await generator.generate_from_requirements(sample_requirements)

        assert schema is not None
        assert metadata is not None
        assert metadata.attempt_number == 1

    @pytest.mark.asyncio
    async def test_generate_from_requirements_with_validation_callback(
        self, mock_orchestrator, sample_requirements
    ):
        """Test generation with custom validation callback."""
        generator = WorkflowGenerator(
            mock_orchestrator,
            vertical="coding",
            max_retries=1,
        )

        # Mock LLM response
        mock_orchestrator.chat.return_value = '{"workflow_name": "test", "nodes": [{"id": "n1", "type": "agent", "role": "executor", "goal": "Test"}], "edges": [{"source": "n1", "target": "__end__", "type": "normal"}], "entry_point": "n1"}'

        # Mock validation callback
        validation_callback = MagicMock()
        validation_callback.return_value = MagicMock(is_valid=True)

        schema, metadata = await generator.generate_from_requirements(
            sample_requirements,
            validation_callback=validation_callback,
        )

        assert schema is not None
        validation_callback.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_from_requirements_validation_fails(
        self, mock_orchestrator, sample_requirements
    ):
        """Test generation when validation fails."""
        generator = WorkflowGenerator(
            mock_orchestrator,
            vertical="coding",
            max_retries=1,
            enable_templates=False,  # Disable template fallback
        )

        # Mock LLM response
        mock_orchestrator.chat.return_value = '{"workflow_name": "test", "nodes": [{"id": "n1", "type": "agent", "role": "executor", "goal": "Test"}], "edges": [{"source": "n1", "target": "__end__", "type": "normal"}], "entry_point": "n1"}'

        # Mock validation callback that fails
        validation_callback = MagicMock()
        validation_callback.return_value = MagicMock(
            is_valid=False, all_errors=["Validation failed"]
        )

        with pytest.raises(RuntimeError, match="Generated schema failed validation"):
            await generator.generate_from_requirements(
                sample_requirements,
                validation_callback=validation_callback,
            )


# =============================================================================
# Test: Refinement Prompt Building
# =============================================================================


class TestRefinementPromptBuilding:
    """Tests for refinement prompt construction."""

    def test_build_refinement_prompt_with_string_feedback(
        self, mock_orchestrator, sample_workflow_schema
    ):
        """Test building refinement prompt with string feedback."""
        generator = WorkflowGenerator(mock_orchestrator, vertical="coding")

        prompt = generator._build_refinement_prompt(sample_workflow_schema, "Add more validation")

        assert "Add more validation" in prompt
        assert "Current Schema" in prompt
        assert "Feedback" in prompt

    def test_build_refinement_prompt_with_list_feedback(
        self, mock_orchestrator, sample_workflow_schema
    ):
        """Test building refinement prompt with list feedback."""
        generator = WorkflowGenerator(mock_orchestrator, vertical="coding")

        prompt = generator._build_refinement_prompt(sample_workflow_schema, ["Error 1", "Error 2"])

        assert "- Error 1" in prompt
        assert "- Error 2" in prompt

    def test_build_refinement_prompt_with_validation_errors(
        self, mock_orchestrator, sample_workflow_schema
    ):
        """Test building refinement prompt with validation errors."""
        from victor.workflows.generation.types import (
            WorkflowValidationError,
            ErrorCategory,
        )

        generator = WorkflowGenerator(mock_orchestrator, vertical="coding")

        errors = [
            WorkflowValidationError(
                category=ErrorCategory.SCHEMA,
                severity="error",
                message="Missing field",
                location="nodes[0]",
            )
        ]

        prompt = generator._build_refinement_prompt(sample_workflow_schema, "Fix errors", errors)

        assert "Validation Errors" in prompt
        assert "[schema]" in prompt
