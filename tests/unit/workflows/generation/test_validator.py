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

"""Tests for workflow validator.

Tests the 4-layer validation system:
- Layer 1: Schema validation
- Layer 2: Graph structure validation
- Layer 3: Semantic validation
- Layer 4: Security validation
"""

import pytest

from victor.workflows.generation import (
    WorkflowValidator,
    SchemaValidator,
    GraphStructureValidator,
    SemanticValidator,
    SecurityValidator,
    ErrorSeverity,
    ErrorCategory,
)


class TestSchemaValidator:
    """Tests for SchemaValidator (Layer 1)."""

    def test_valid_workflow(self):
        """Test validation of valid workflow passes."""
        validator = SchemaValidator()

        workflow = {
            "name": "test_workflow",
            "nodes": [
                {
                    "id": "start",
                    "type": "agent",
                    "role": "executor",
                    "goal": "Start task",
                    "tool_budget": 15,
                }
            ],
            "edges": [{"source": "start", "target": "__end__"}],
            "entry_point": "start",
        }

        errors = validator.validate(workflow)
        assert len(errors) == 0

    def test_missing_required_fields(self):
        """Test detection of missing required fields."""
        validator = SchemaValidator()

        workflow = {"name": "incomplete"}

        errors = validator.validate(workflow)
        assert len(errors) > 0

        # Should have errors for missing nodes and entry_point
        error_messages = {e.message for e in errors}
        assert any("nodes" in msg for msg in error_messages)
        assert any("entry_point" in msg for msg in error_messages)

    def test_invalid_node_type(self):
        """Test detection of invalid node type."""
        validator = SchemaValidator()

        workflow = {
            "nodes": [{"id": "bad_node", "type": "invalid_type"}],
            "entry_point": "bad_node",
        }

        errors = validator.validate(workflow)
        assert len(errors) > 0

        error_messages = {e.message for e in errors}
        assert any("Invalid node type" in msg for msg in error_messages)

    def test_agent_node_validation(self):
        """Test agent node field validation."""
        validator = SchemaValidator()

        # Agent node without role
        workflow = {
            "nodes": [{"id": "agent1", "type": "agent", "goal": "Do something"}],
            "entry_point": "agent1",
        }

        errors = validator.validate(workflow)
        assert len(errors) > 0
        assert any("role" in e.message for e in errors)

    def test_tool_budget_range(self):
        """Test tool_budget range validation."""
        validator = SchemaValidator()

        workflow = {
            "nodes": [
                {
                    "id": "agent1",
                    "type": "agent",
                    "role": "executor",
                    "goal": "Task",
                    "tool_budget": 1000,  # Out of range
                }
            ],
            "entry_point": "agent1",
        }

        errors = validator.validate(workflow)
        assert len(errors) > 0
        assert any("tool_budget" in e.message and "range" in e.message for e in errors)

    def test_duplicate_node_ids(self):
        """Test detection of duplicate node IDs."""
        validator = SchemaValidator()

        workflow = {
            "nodes": [
                {"id": "dup", "type": "agent", "role": "executor", "goal": "Task 1"},
                {"id": "dup", "type": "agent", "role": "executor", "goal": "Task 2"},
            ],
            "entry_point": "dup",
        }

        errors = validator.validate(workflow)
        assert len(errors) > 0
        assert any("Duplicate" in e.message for e in errors)


class TestGraphStructureValidator:
    """Tests for GraphStructureValidator (Layer 2)."""

    def test_valid_graph(self):
        """Test validation of valid graph structure."""
        validator = GraphStructureValidator()

        workflow = {
            "nodes": [
                {"id": "start", "type": "agent", "role": "executor", "goal": "Start"},
                {"id": "end", "type": "agent", "role": "executor", "goal": "End"},
            ],
            "edges": [
                {"source": "start", "target": "end"},
                {"source": "end", "target": "__end__"},
            ],
            "entry_point": "start",
        }

        errors = validator.validate(workflow)
        assert len(errors) == 0

    def test_unreachable_node(self):
        """Test detection of unreachable nodes."""
        validator = GraphStructureValidator()

        workflow = {
            "nodes": [
                {"id": "start", "type": "agent", "role": "executor", "goal": "Start"},
                {"id": "orphan", "type": "agent", "role": "executor", "goal": "Orphan"},
            ],
            "edges": [{"source": "start", "target": "__end__"}],
            "entry_point": "start",
        }

        errors = validator.validate(workflow)
        assert len(errors) > 0
        assert any("not reachable" in e.message for e in errors)

    def test_invalid_edge_reference(self):
        """Test detection of invalid edge references."""
        validator = GraphStructureValidator()

        workflow = {
            "nodes": [{"id": "start", "type": "agent", "role": "executor", "goal": "Start"}],
            "edges": [{"source": "start", "target": "nonexistent"}],
            "entry_point": "start",
        }

        errors = validator.validate(workflow)
        assert len(errors) > 0
        assert any("not found" in e.message for e in errors)

    def test_unconditional_cycle(self):
        """Test detection of unconditional cycles."""
        validator = GraphStructureValidator()

        workflow = {
            "nodes": [
                {"id": "a", "type": "compute", "tools": []},
                {"id": "b", "type": "compute", "tools": []},
            ],
            "edges": [
                {"source": "a", "target": "b"},
                {"source": "b", "target": "a"},  # Cycle without condition
            ],
            "entry_point": "a",
        }

        errors = validator.validate(workflow)
        assert len(errors) > 0
        assert any("cycle" in e.message for e in errors)

    def test_conditional_cycle_allowed(self):
        """Test that conditional cycles are allowed."""
        validator = GraphStructureValidator()

        workflow = {
            "nodes": [
                {"id": "a", "type": "compute", "tools": []},
                {"id": "b", "type": "compute", "tools": []},
                {
                    "id": "check",
                    "type": "condition",
                    "branches": {"continue": "a", "done": "__end__"},
                },
            ],
            "edges": [
                {"source": "a", "target": "b"},
                {"source": "b", "target": "check"},
                {"source": "check", "target": {"continue": "a", "done": "__end__"}},
            ],
            "entry_point": "a",
        }

        errors = validator.validate(workflow)
        # Should not have unconditional cycle error
        cycle_errors = [e for e in errors if "cycle" in e.message.lower()]
        assert len(cycle_errors) == 0


class TestSemanticValidator:
    """Tests for SemanticValidator (Layer 3)."""

    def test_valid_semantics(self):
        """Test validation of valid semantics."""
        validator = SemanticValidator(tool_registry=None, strict_mode=False)

        workflow = {
            "nodes": [
                {
                    "id": "agent1",
                    "type": "agent",
                    "role": "executor",
                    "goal": "Execute task",
                    "tool_budget": 15,
                }
            ]
        }

        errors = validator.validate(workflow)
        assert len(errors) == 0

    def test_agent_tools_validation(self):
        """Test agent tools validation."""

        # Mock tool registry
        class MockToolRegistry:
            def get_tool(self, name):
                # Return None for invalid_tool, object for valid_tool
                if name == "valid_tool":
                    return {"name": "valid_tool"}
                return None

        validator = SemanticValidator(tool_registry=MockToolRegistry(), strict_mode=True)

        workflow = {
            "nodes": [
                {
                    "id": "agent1",
                    "type": "agent",
                    "role": "executor",
                    "goal": "Task",
                    "tools": ["valid_tool", "invalid_tool"],
                }
            ]
        }

        errors = validator.validate(workflow)
        assert len(errors) > 0
        assert any("invalid_tool" in e.message for e in errors)

    def test_condition_branches_type(self):
        """Test condition branches type validation."""
        validator = SemanticValidator()

        workflow = {
            "nodes": [
                {
                    "id": "cond1",
                    "type": "condition",
                    "branches": "not_a_dict",
                }  # Invalid type
            ]
        }

        errors = validator.validate(workflow)
        assert len(errors) > 0
        assert any("branches" in e.message for e in errors)


class TestSecurityValidator:
    """Tests for SecurityValidator (Layer 4)."""

    def test_valid_security(self):
        """Test validation of valid security constraints."""
        validator = SecurityValidator()

        workflow = {
            "nodes": [
                {
                    "id": "agent1",
                    "type": "agent",
                    "role": "executor",
                    "goal": "Task",
                    "tool_budget": 15,
                }
            ]
        }

        errors = validator.validate(workflow)
        assert len(errors) == 0

    def test_tool_budget_exceeded(self):
        """Test detection of excessive tool budget."""
        validator = SecurityValidator(max_tool_budget=100)

        workflow = {
            "nodes": [
                {
                    "id": "agent1",
                    "type": "agent",
                    "role": "executor",
                    "goal": "Task",
                    "tool_budget": 60,
                },
                {
                    "id": "agent2",
                    "type": "agent",
                    "role": "executor",
                    "goal": "Task",
                    "tool_budget": 60,
                },
            ]
        }

        errors = validator.validate(workflow)
        assert len(errors) > 0
        assert any("exceeds limit" in e.message for e in errors)

    def test_parallel_branches_limit(self):
        """Test detection of excessive parallel branches."""
        validator = SecurityValidator(max_parallel_branches=5)

        workflow = {
            "nodes": [
                {
                    "id": "parallel1",
                    "type": "parallel",
                    "parallel_nodes": ["a", "b", "c", "d", "e", "f"],  # 6 branches
                }
            ]
        }

        errors = validator.validate(workflow)
        assert len(errors) > 0
        assert any("exceeds limit" in e.message for e in errors)

    def test_airgapped_mode_network_tools(self):
        """Test detection of network tools in airgapped mode."""
        validator = SecurityValidator(airgapped_mode=True)

        workflow = {
            "nodes": [
                {
                    "id": "agent1",
                    "type": "agent",
                    "role": "executor",
                    "goal": "Task",
                    "tools": ["web_search", "http_request"],
                }
            ]
        }

        errors = validator.validate(workflow)
        assert len(errors) > 0
        assert any(
            "airgapped" in e.message.lower() or "network" in e.message.lower() for e in errors
        )


class TestWorkflowValidator:
    """Tests for WorkflowValidator (main facade)."""

    def test_valid_workflow(self):
        """Test complete validation of valid workflow."""
        validator = WorkflowValidator()

        workflow = {
            "name": "complete_test",
            "nodes": [
                {
                    "id": "start",
                    "type": "agent",
                    "role": "executor",
                    "goal": "Start the workflow",
                    "tool_budget": 15,
                },
                {
                    "id": "check",
                    "type": "condition",
                    "branches": {"continue": "start", "done": "__end__"},
                },
            ],
            "edges": [
                {"source": "start", "target": "check"},
                {"source": "check", "target": {"continue": "start", "done": "__end__"}},
            ],
            "entry_point": "start",
            "max_iterations": 10,
        }

        result = validator.validate(workflow)

        assert result.is_valid
        assert len(result.all_errors) == 0

    def test_invalid_workflow_aggregates_errors(self):
        """Test that invalid workflow aggregates errors from all layers."""
        validator = WorkflowValidator()

        workflow = {
            "name": "broken",
            "nodes": [{"id": "bad", "type": "invalid_type"}],
            "entry_point": "bad",
        }

        result = validator.validate(workflow)

        assert not result.is_valid
        assert len(result.all_errors) > 0

        # Should have schema errors
        assert len(result.schema_errors) > 0

    def test_error_categorization(self):
        """Test that errors are properly categorized."""
        validator = WorkflowValidator()

        workflow = {
            "nodes": [
                {
                    "id": "start",
                    "type": "agent",
                    # Missing role and goal - schema error
                }
            ],
            "entry_point": "start",
        }

        result = validator.validate(workflow)

        # Check categorization
        assert any(e.category == ErrorCategory.SCHEMA for e in result.all_errors)

    def test_severity_levels(self):
        """Test error severity levels."""
        validator = WorkflowValidator()

        workflow = {}  # Completely empty

        result = validator.validate(workflow)

        # Should have critical errors for missing required fields
        assert any(e.severity == ErrorSeverity.CRITICAL for e in result.all_errors)

    def test_single_layer_validation(self):
        """Test validation of single layer."""
        validator = WorkflowValidator()

        workflow = {"nodes": [{"id": "n1", "type": "agent"}], "entry_point": "n1"}

        # Validate only schema layer
        errors = validator.validate_layer(workflow, "schema")

        assert len(errors) > 0
        assert all(e.category == ErrorCategory.SCHEMA for e in errors)

    def test_invalid_layer_name(self):
        """Test that invalid layer name raises error."""
        validator = WorkflowValidator()

        workflow = {"nodes": [], "entry_point": "start"}

        with pytest.raises(ValueError, match="Invalid layer"):
            validator.validate_layer(workflow, "invalid_layer")

    def test_workflow_with_to_dict(self):
        """Test validation of workflow object with to_dict method."""
        validator = WorkflowValidator()

        class MockWorkflow:
            def to_dict(self):
                return {
                    "name": "mock",
                    "nodes": [
                        {
                            "id": "start",
                            "type": "agent",
                            "role": "executor",
                            "goal": "Task",
                            "tool_budget": 15,
                        }
                    ],
                    "edges": [{"source": "start", "target": "__end__"}],
                    "entry_point": "start",
                }

        workflow = MockWorkflow()
        result = validator.validate(workflow)

        assert result.is_valid

    def test_validation_result_properties(self):
        """Test ValidationResult convenience properties."""
        validator = WorkflowValidator()

        workflow = {
            "nodes": [{"id": "a", "type": "agent", "role": "executor", "goal": "Task"}],
            "entry_point": "a",
        }

        result = validator.validate(workflow, workflow_name="test_workflow")

        # Test summary
        summary = result.summary()
        assert isinstance(summary, str)

        # Test error count
        counts = result.error_count
        assert "critical" in counts
        assert "error" in counts
        assert "warning" in counts
        assert "info" in counts

        # Test to_dict
        result_dict = result.to_dict()
        assert "is_valid" in result_dict
        assert "summary" in result_dict
        assert "error_counts" in result_dict


class TestValidationIntegration:
    """Integration tests for complete validation scenarios."""

    def test_real_world_valid_workflow(self):
        """Test validation of realistic valid workflow."""
        validator = WorkflowValidator(strict_mode=False)

        workflow = {
            "name": "data_analysis_pipeline",
            "description": "Analyze customer data",
            "nodes": [
                {
                    "id": "fetch_data",
                    "type": "agent",
                    "role": "researcher",
                    "goal": "Fetch customer data from database",
                    "tool_budget": 20,
                },
                {
                    "id": "analyze",
                    "type": "agent",
                    "role": "analyst",
                    "goal": "Analyze data for patterns",
                    "tool_budget": 30,
                },
                {
                    "id": "check_quality",
                    "type": "condition",
                    "branches": {
                        "good_quality": "generate_report",
                        "needs_work": "fetch_data",
                    },
                },
                {
                    "id": "generate_report",
                    "type": "agent",
                    "role": "writer",
                    "goal": "Generate analysis report",
                    "tool_budget": 25,
                },
            ],
            "edges": [
                {"source": "fetch_data", "target": "analyze"},
                {"source": "analyze", "target": "check_quality"},
                {
                    "source": "check_quality",
                    "target": {
                        "good_quality": "generate_report",
                        "needs_work": "fetch_data",
                    },
                },
                {"source": "generate_report", "target": "__end__"},
            ],
            "entry_point": "fetch_data",
            "max_iterations": 10,
            "max_timeout_seconds": 600,
        }

        result = validator.validate(workflow, workflow_name="data_analysis_pipeline")

        assert result.is_valid
        assert len(result.all_errors) == 0

    def test_workflow_with_multiple_issues(self):
        """Test workflow with multiple validation issues."""
        validator = WorkflowValidator()

        workflow = {
            "name": "problematic",
            "nodes": [
                {
                    "id": "a",
                    "type": "agent",
                    # Missing role and goal
                },
                {"id": "b", "type": "invalid_type"},
                {
                    "id": "c",
                    "type": "agent",
                    "role": "executor",
                    "goal": "Task",
                    "tool_budget": 1000,  # Out of range
                },
            ],
            "edges": [{"source": "a", "target": "nonexistent"}],  # Invalid target
            "entry_point": "missing",  # Invalid entry point
        }

        result = validator.validate(workflow)

        assert not result.is_valid
        assert len(result.all_errors) > 5  # Should have many errors

        # Check error distribution across categories
        assert len(result.schema_errors) > 0
        assert len(result.structure_errors) > 0
