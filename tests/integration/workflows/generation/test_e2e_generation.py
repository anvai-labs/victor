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

"""Integration tests for end-to-end workflow generation.

Tests the complete pipeline from natural language to executable StateGraph.
These tests can use real LLM providers or mocks depending on configuration.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any

from victor.workflows.generation.workflow_pipeline import (
    WorkflowGenerationPipeline,
    PipelineMode,
    PipelineResult,
)
from victor.framework.graph import StateGraph

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_orchestrator():
    """Mock orchestrator for testing."""
    orchestrator = MagicMock()
    orchestrator.current_model = "claude-sonnet-4-5"
    orchestrator.chat = AsyncMock()
    return orchestrator


@pytest.fixture
def sample_llm_responses():
    """Sample LLM responses for mocking."""
    return {
        "extraction": """{
            "description": "Analyze codebase, find bugs, and fix them",
            "functional": {
                "tasks": [
                    {
                        "id": "analyze",
                        "description": "Analyze codebase",
                        "task_type": "agent",
                        "role": "researcher",
                        "goal": "Find bugs using static analysis"
                    },
                    {
                        "id": "fix",
                        "description": "Fix identified bugs",
                        "task_type": "agent",
                        "role": "executor",
                        "dependencies": ["analyze"]
                    }
                ],
                "tools": {
                    "analyze": ["read", "search"],
                    "fix": ["write", "edit"]
                },
                "success_criteria": ["All tests pass"]
            },
            "structural": {
                "execution_order": "sequential",
                "dependencies": {
                    "fix": ["analyze"]
                }
            },
            "quality": {
                "max_duration_seconds": 300,
                "max_cost_tier": "MEDIUM"
            },
            "context": {
                "vertical": "coding",
                "subdomain": "bug_fix"
            }
        }""",
        "generation": """{
            "workflow_name": "bug_fix_workflow",
            "description": "Fix bugs in codebase",
            "nodes": [
                {
                    "id": "analyze",
                    "type": "agent",
                    "role": "researcher",
                    "goal": "Analyze codebase for bugs",
                    "tool_budget": 10,
                    "allowed_tools": ["read", "search"],
                    "output_key": "analysis"
                },
                {
                    "id": "fix",
                    "type": "agent",
                    "role": "executor",
                    "goal": "Fix identified bugs",
                    "tool_budget": 15,
                    "allowed_tools": ["write", "edit"],
                    "output_key": "fixes"
                }
            ],
            "edges": [
                {
                    "source": "analyze",
                    "target": "fix",
                    "type": "normal"
                },
                {
                    "source": "fix",
                    "target": "__end__",
                    "type": "normal"
                }
            ],
            "entry_point": "analyze",
            "metadata": {
                "vertical": "coding",
                "max_iterations": 20,
                "timeout_seconds": 300
            }
        }""",
    }


# =============================================================================
# Test: End-to-End Generation (Auto Mode)
# =============================================================================


class TestE2EGenerationAutoMode:
    """Tests for end-to-end generation in auto mode."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_generate_simple_bug_fix_workflow(self, mock_orchestrator, sample_llm_responses):
        """Test generating a simple bug fix workflow."""
        pipeline = WorkflowGenerationPipeline(
            mock_orchestrator,
            vertical="coding",
            strategy="llm_single_stage",
        )

        # Mock LLM responses
        mock_orchestrator.chat.side_effect = [
            sample_llm_responses["extraction"],
            sample_llm_responses["generation"],
        ]

        # Mock validation to pass
        with patch(
            "victor.workflows.generation.workflow_pipeline.WorkflowGenerationPipeline._validate_schema"
        ) as mock_validate:
            mock_validate.return_value = MagicMock(is_valid=True)

            result = await pipeline.generate_workflow(
                "Analyze this codebase, find bugs, and fix them",
                mode="auto",
            )

        assert result.success is True
        assert result.graph is not None
        assert result.schema is not None
        assert result.requirements is not None
        assert result.metadata is not None
        assert result.duration_seconds > 0

        # Verify schema structure
        assert "nodes" in result.schema
        assert "edges" in result.schema
        assert "entry_point" in result.schema

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_generate_with_validation_callback(self, mock_orchestrator, sample_llm_responses):
        """Test generation with custom validation callback."""
        pipeline = WorkflowGenerationPipeline(
            mock_orchestrator,
            vertical="coding",
            strategy="llm_single_stage",
        )

        # Mock LLM responses
        mock_orchestrator.chat.side_effect = [
            sample_llm_responses["extraction"],
            sample_llm_responses["generation"],
        ]

        # Custom validation callback
        validation_callback = MagicMock()
        validation_callback.return_value = MagicMock(is_valid=True)

        with patch(
            "victor.workflows.generation.workflow_pipeline.WorkflowGenerationPipeline._validate_schema"
        ) as mock_validate:
            mock_validate.return_value = MagicMock(is_valid=True)

            result = await pipeline.generate_workflow(
                "Fix the authentication bug",
                mode="auto",
                validation_callback=validation_callback,
            )

        assert result.success is True
        validation_callback.assert_called_once()

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_generate_with_progress_callback(self, mock_orchestrator, sample_llm_responses):
        """Test generation with progress callback."""
        pipeline = WorkflowGenerationPipeline(
            mock_orchestrator,
            vertical="coding",
        )

        # Mock LLM responses
        mock_orchestrator.chat.side_effect = [
            sample_llm_responses["extraction"],
            sample_llm_responses["generation"],
        ]

        # Progress callback
        progress_callback = AsyncMock()
        progress_messages = []

        async def capture_progress(message):
            progress_messages.append(message)

        with patch(
            "victor.workflows.generation.workflow_pipeline.WorkflowGenerationPipeline._validate_schema"
        ) as mock_validate:
            mock_validate.return_value = MagicMock(is_valid=True)

            result = await pipeline.generate_workflow(
                "Analyze and fix bugs",
                mode="auto",
                progress_callback=capture_progress,
            )

        assert result.success is True
        assert len(progress_messages) > 0


# =============================================================================
# Test: End-to-End Generation (Interactive Mode)
# =============================================================================


class TestE2EGenerationInteractiveMode:
    """Tests for end-to-end generation in interactive mode."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_generate_with_user_approval(self, mock_orchestrator, sample_llm_responses):
        """Test generation with user approval at each stage."""
        pipeline = WorkflowGenerationPipeline(
            mock_orchestrator,
            vertical="coding",
        )

        # Mock LLM responses
        mock_orchestrator.chat.side_effect = [
            sample_llm_responses["extraction"],
            sample_llm_responses["generation"],
        ]

        # Mock user approval (auto-approve in non-TTY)
        with (
            patch(
                "victor.workflows.generation.workflow_pipeline.WorkflowGenerationPipeline._interactive_approval",
                return_value=True,
            ),
            patch(
                "victor.workflows.generation.workflow_pipeline.WorkflowGenerationPipeline._validate_schema",
                return_value=MagicMock(is_valid=True),
            ),
        ):
            result = await pipeline.generate_workflow(
                "Fix bugs in the code",
                mode="interactive",
            )

        assert result.success is True

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_generate_with_user_rejection(self, mock_orchestrator, sample_llm_responses):
        """Test generation when user rejects at requirements stage."""
        pipeline = WorkflowGenerationPipeline(
            mock_orchestrator,
            vertical="coding",
        )

        # Mock LLM response
        mock_orchestrator.chat.return_value = sample_llm_responses["extraction"]

        # Mock user rejection
        with patch(
            "victor.workflows.generation.workflow_pipeline.WorkflowGenerationPipeline._interactive_approval",
            return_value=False,
        ):
            result = await pipeline.generate_workflow(
                "Fix bugs",
                mode="interactive",
            )

        assert result.success is False
        assert "User rejected" in result.errors[0]


# =============================================================================
# Test: End-to-End Generation (Safe Mode)
# =============================================================================


class TestE2EGenerationSafeMode:
    """Tests for end-to-end generation in safe mode."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_safe_mode_stops_on_validation_errors(
        self, mock_orchestrator, sample_llm_responses
    ):
        """Test that safe mode stops when validation fails."""
        pipeline = WorkflowGenerationPipeline(
            mock_orchestrator,
            vertical="coding",
        )

        # Mock LLM responses
        mock_orchestrator.chat.side_effect = [
            sample_llm_responses["extraction"],
            sample_llm_responses["generation"],
        ]

        # Mock validation failure
        with patch(
            "victor.workflows.generation.workflow_pipeline.WorkflowGenerationPipeline._validate_schema"
        ) as mock_validate:
            mock_validate.return_value = MagicMock(
                is_valid=False,
                all_errors=[
                    MagicMock(message="Invalid node type"),
                    MagicMock(message="Missing edge target"),
                ],
            )

            result = await pipeline.generate_workflow(
                "Fix bugs",
                mode="safe",
            )

        assert result.success is False
        assert len(result.errors) > 0


# =============================================================================
# Test: Workflow Refinement
# =============================================================================


class TestE2EWorkflowRefinement:
    """Tests for end-to-end workflow refinement."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_refine_existing_workflow(self, mock_orchestrator, sample_llm_responses):
        """Test refining an existing workflow."""
        pipeline = WorkflowGenerationPipeline(
            mock_orchestrator,
            vertical="coding",
        )

        # Create a simple workflow graph
        original_schema = {
            "workflow_name": "simple_workflow",
            "nodes": [{"id": "node1", "type": "agent", "role": "executor", "goal": "Task 1"}],
            "edges": [{"source": "node1", "target": "__end__", "type": "normal"}],
            "entry_point": "node1",
        }

        # Mock graph (don't use spec=StateGraph since we need to_dict method)
        mock_graph = MagicMock()
        mock_graph.to_dict.return_value = original_schema

        # Mock refined schema
        mock_orchestrator.chat.return_value = """{
            "workflow_name": "refined_workflow",
            "nodes": [
                {
                    "id": "node1",
                    "type": "agent",
                    "role": "executor",
                    "goal": "Task 1 with improvements"
                },
                {
                    "id": "validate",
                    "type": "compute",
                    "handler": "validate_output"
                }
            ],
            "edges": [
                {"source": "node1", "target": "validate", "type": "normal"},
                {"source": "validate", "target": "__end__", "type": "normal"}
            ],
            "entry_point": "node1"
        }"""

        with patch(
            "victor.workflows.generation.workflow_pipeline.WorkflowGenerationPipeline._validate_schema"
        ) as mock_validate:
            mock_validate.return_value = MagicMock(is_valid=True)

            result = await pipeline.refine_workflow(
                mock_graph,
                "Add a validation step after task 1",
                mode="auto",
            )

        assert result.success is True
        assert result.schema is not None


# =============================================================================
# Test: Error Handling
# =============================================================================


class TestE2EErrorHandling:
    """Tests for error handling in end-to-end generation."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_handles_llm_failure_gracefully(self, mock_orchestrator):
        """Test that LLM failures are handled gracefully."""
        pipeline = WorkflowGenerationPipeline(
            mock_orchestrator,
            vertical="coding",
            max_refinement_iterations=1,
        )

        # Mock LLM failure
        mock_orchestrator.chat.side_effect = Exception("LLM API error")

        result = await pipeline.generate_workflow(
            "Fix bugs",
            mode="auto",
        )

        assert result.success is False
        assert len(result.errors) > 0
        assert "Pipeline failed" in result.errors[0]

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_handles_invalid_json_response(self, mock_orchestrator):
        """Test that invalid JSON responses are handled."""
        pipeline = WorkflowGenerationPipeline(
            mock_orchestrator,
            vertical="coding",
        )

        # Mock invalid JSON response
        mock_orchestrator.chat.return_value = "This is not valid JSON"

        result = await pipeline.generate_workflow(
            "Fix bugs",
            mode="auto",
        )

        assert result.success is False


# =============================================================================
# Test: Multiple Verticals
# =============================================================================


class TestE2EGenerationMultipleVerticals:
    """Tests for generation across different verticals."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    @pytest.mark.parametrize(
        "vertical,description",
        [
            ("coding", "Analyze code and fix bugs"),
            ("research", "Research AI trends and summarize"),
            ("devops", "Deploy to production"),
            ("dataanalysis", "Analyze dataset and create visualizations"),
        ],
    )
    async def test_generate_workflow_for_vertical(self, mock_orchestrator, vertical, description):
        """Test generation for different verticals."""
        pipeline = WorkflowGenerationPipeline(
            mock_orchestrator,
            vertical=vertical,
        )

        # Mock LLM responses (include required description field and at least one task)
        mock_orchestrator.chat.side_effect = [
            f'{{"description": "{description}", "functional": {{"tasks": [{{"id": "task1", "description": "Main task", "task_type": "agent", "role": "executor", "goal": "{description}"}}]}}, "structural": {{"execution_order": "sequential"}}, "quality": {{}}, "context": {{"vertical": "{vertical}"}}}}',
            '{{"workflow_name": "test", "nodes": [{{"id": "task1", "type": "agent", "role": "executor", "goal": "{description}"}}], "edges": [{{"source": "task1", "target": "__end__", "type": "normal"}}], "entry_point": "task1"}}'.replace(
                "{description}", description
            ),
        ]

        with patch(
            "victor.workflows.generation.workflow_pipeline.WorkflowGenerationPipeline._validate_schema"
        ) as mock_validate:
            mock_validate.return_value = MagicMock(is_valid=True)

            result = await pipeline.generate_workflow(description, mode="auto")

        assert result.success is True
