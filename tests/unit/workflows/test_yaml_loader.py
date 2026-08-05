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

"""Tests for YAML workflow loader."""

import tempfile
from pathlib import Path

import pytest

from victor.workflows import (
    AgentNode,
    ConditionNode,
    ParallelNode,
    TransformNode,
)
from victor.workflows.yaml_loader import (
    YAMLWorkflowConfig,
    YAMLWorkflowError,
    YAMLWorkflowProvider,
    load_workflow_from_dict,
    load_workflow_from_file,
    load_workflow_from_yaml,
    load_workflows_from_directory,
    _create_simple_condition,
    _parse_constraints,
    _parse_value,
)


class TestParseValue:
    """Tests for value parsing."""

    def test_parse_string_double_quotes(self):
        assert _parse_value('"hello"') == "hello"

    def test_parse_string_single_quotes(self):
        assert _parse_value("'hello'") == "hello"

    def test_parse_int(self):
        assert _parse_value("42") == 42

    def test_parse_float(self):
        assert _parse_value("3.14") == 3.14

    def test_parse_bool_true(self):
        assert _parse_value("true") is True
        assert _parse_value("True") is True

    def test_parse_bool_false(self):
        assert _parse_value("false") is False
        assert _parse_value("False") is False

    def test_parse_none(self):
        assert _parse_value("none") is None
        assert _parse_value("None") is None

    def test_parse_unquoted_string(self):
        assert _parse_value("hello") == "hello"


class TestSimpleCondition:
    """Tests for simple condition parsing."""

    def test_truthy_check(self):
        condition = _create_simple_condition("has_errors")
        assert condition({"has_errors": True}) == "true"
        assert condition({"has_errors": False}) == "false"
        assert condition({}) == "false"

    def test_equality(self):
        condition = _create_simple_condition("status == 'active'")
        assert condition({"status": "active"}) == "true"
        assert condition({"status": "inactive"}) == "false"

    def test_inequality(self):
        condition = _create_simple_condition("count != 0")
        assert condition({"count": 1}) == "true"
        assert condition({"count": 0}) == "false"

    def test_greater_than(self):
        condition = _create_simple_condition("score > 50")
        assert condition({"score": 60}) == "true"
        assert condition({"score": 40}) == "false"

    def test_less_than(self):
        condition = _create_simple_condition("errors < 5")
        assert condition({"errors": 3}) == "true"
        assert condition({"errors": 10}) == "false"

    def test_in_operator(self):
        condition = _create_simple_condition("status in [active, pending, review]")
        assert condition({"status": "active"}) == "true"
        assert condition({"status": "completed"}) == "false"


class TestParseConstraints:
    """Tests for constraint parsing."""

    def test_none_string_allows_all(self):
        """constraints: none allows everything."""
        constraints = _parse_constraints("none")
        assert constraints.llm_allowed is True
        assert constraints.network_allowed is True
        assert constraints.write_allowed is True

    def test_none_in_list_allows_all(self):
        """constraints: [none] allows everything."""
        constraints = _parse_constraints(["none"])
        assert constraints.llm_allowed is True
        assert constraints.network_allowed is True
        assert constraints.write_allowed is True

    def test_none_cannot_combine_with_others(self):
        """constraints: [none, llm] is invalid."""
        with pytest.raises(YAMLWorkflowError) as exc_info:
            _parse_constraints(["none", "llm"])
        assert "cannot be combined" in str(exc_info.value)

    def test_list_blocks_specified(self):
        """constraints: [llm, network] blocks those items."""
        constraints = _parse_constraints(["llm", "network"])
        assert constraints.llm_allowed is False
        assert constraints.network_allowed is False
        assert constraints.write_allowed is True  # Not in list

    def test_list_case_insensitive(self):
        """constraints: [LLM, NETWORK] works."""
        constraints = _parse_constraints(["LLM", "NETWORK"])
        assert constraints.llm_allowed is False
        assert constraints.network_allowed is False

    def test_empty_list_allows_all(self):
        """constraints: [] allows everything."""
        constraints = _parse_constraints([])
        assert constraints.llm_allowed is True
        assert constraints.network_allowed is True
        assert constraints.write_allowed is True

    def test_missing_constraints_uses_defaults(self):
        """No constraints = TaskConstraints defaults."""
        constraints = _parse_constraints(None)
        assert constraints.llm_allowed is False  # Default
        assert constraints.network_allowed is True  # Default
        assert constraints.write_allowed is False  # Default

    def test_dict_format_legacy_support(self):
        """Dict format still works for backwards compatibility."""
        constraints = _parse_constraints(
            {
                "llm_allowed": True,
                "network_allowed": False,
                "write_allowed": True,
            }
        )
        assert constraints.llm_allowed is True
        assert constraints.network_allowed is False
        assert constraints.write_allowed is True

    def test_timeout_passthrough(self):
        """Timeout is passed through."""
        constraints = _parse_constraints(["llm"], timeout=120.0)
        assert constraints.timeout == 120.0

    def test_invalid_string_raises_error(self):
        """constraints: 'invalid' raises error."""
        with pytest.raises(YAMLWorkflowError) as exc_info:
            _parse_constraints("invalid")
        # Error message includes valid values from enum
        assert "Invalid constraint string" in str(exc_info.value)
        assert "Valid values" in str(exc_info.value)


class TestLoadWorkflowFromDict:
    """Tests for loading workflows from dictionaries."""

    def test_load_simple_workflow(self):
        data = {
            "description": "Test workflow",
            "nodes": [
                {
                    "id": "start",
                    "type": "agent",
                    "role": "researcher",
                    "goal": "Research the topic",
                    "tool_budget": 20,
                }
            ],
        }
        workflow = load_workflow_from_dict(data, "test_workflow")
        assert workflow.name == "test_workflow"
        assert workflow.description == "Test workflow"
        assert len(workflow.nodes) == 1
        assert "start" in workflow.nodes
        assert isinstance(workflow.nodes["start"], AgentNode)
        assert workflow.nodes["start"].role == "researcher"

    def test_load_workflow_with_multiple_nodes(self):
        data = {
            "description": "Multi-node workflow",
            "nodes": [
                {
                    "id": "analyze",
                    "type": "agent",
                    "role": "analyst",
                    "goal": "Analyze data",
                    "next": ["decide"],
                },
                {
                    "id": "decide",
                    "type": "condition",
                    "condition": "has_issues",
                    "branches": {"true": "fix", "false": "done"},
                },
                {
                    "id": "fix",
                    "type": "agent",
                    "role": "executor",
                    "goal": "Fix issues",
                    "next": ["done"],
                },
                {
                    "id": "done",
                    "type": "agent",
                    "role": "writer",
                    "goal": "Write report",
                },
            ],
        }
        workflow = load_workflow_from_dict(data, "test")
        assert len(workflow.nodes) == 4
        assert isinstance(workflow.nodes["decide"], ConditionNode)
        assert workflow.nodes["decide"].branches == {"true": "fix", "false": "done"}

    def test_load_parallel_node(self):
        data = {
            "nodes": [
                {
                    "id": "parallel_step",
                    "type": "parallel",
                    "parallel_nodes": ["task_a", "task_b"],
                    "join_strategy": "all",
                    "next": ["merge"],
                },
                {"id": "task_a", "type": "agent", "role": "executor", "goal": "Task A"},
                {"id": "task_b", "type": "agent", "role": "executor", "goal": "Task B"},
                {
                    "id": "merge",
                    "type": "agent",
                    "role": "writer",
                    "goal": "Merge results",
                },
            ],
        }
        workflow = load_workflow_from_dict(data, "parallel_test")
        assert isinstance(workflow.nodes["parallel_step"], ParallelNode)
        assert workflow.nodes["parallel_step"].parallel_nodes == ["task_a", "task_b"]

    def test_load_transform_node(self):
        data = {
            "nodes": [
                {
                    "id": "transform_step",
                    "type": "transform",
                    "transform": "result = ctx.input",
                    "next": ["next_step"],
                },
                {
                    "id": "next_step",
                    "type": "agent",
                    "role": "executor",
                    "goal": "Use result",
                },
            ],
        }
        workflow = load_workflow_from_dict(data, "transform_test")
        assert isinstance(workflow.nodes["transform_step"], TransformNode)

    def test_load_with_metadata(self):
        data = {
            "description": "With metadata",
            "metadata": {"version": "1.0", "author": "test"},
            "nodes": [{"id": "start", "type": "agent", "role": "executor", "goal": "Do work"}],
        }
        workflow = load_workflow_from_dict(data, "metadata_test")
        assert workflow.metadata["version"] == "1.0"
        assert workflow.metadata["author"] == "test"

    def test_missing_node_id_raises_error(self):
        data = {"nodes": [{"type": "agent", "role": "executor", "goal": "No ID"}]}
        with pytest.raises(YAMLWorkflowError, match="missing required 'id' field"):
            load_workflow_from_dict(data, "bad_workflow")


class TestLoadWorkflowFromYAML:
    """Tests for loading workflows from YAML strings."""

    def test_load_single_workflow(self):
        yaml_content = """
workflows:
  code_review:
    description: "Review code quality"
    nodes:
      - id: analyze
        type: agent
        role: researcher
        goal: "Find code patterns"
        tool_budget: 20
      - id: report
        type: agent
        role: writer
        goal: "Summarize findings"
"""
        workflow = load_workflow_from_yaml(yaml_content, "code_review")
        assert workflow.name == "code_review"
        assert workflow.description == "Review code quality"
        assert len(workflow.nodes) == 2

    def test_load_all_workflows(self):
        yaml_content = """
workflows:
  workflow_a:
    nodes:
      - id: step1
        type: agent
        role: executor
        goal: "Task A"
  workflow_b:
    nodes:
      - id: step1
        type: agent
        role: executor
        goal: "Task B"
"""
        workflows = load_workflow_from_yaml(yaml_content)
        assert isinstance(workflows, dict)
        assert "workflow_a" in workflows
        assert "workflow_b" in workflows

    def test_invalid_yaml_raises_error(self):
        yaml_content = "not: valid: yaml: content:"
        with pytest.raises(YAMLWorkflowError, match="Invalid YAML"):
            load_workflow_from_yaml(yaml_content)

    def test_missing_workflow_raises_error(self):
        yaml_content = """
workflows:
  existing:
    nodes:
      - id: step
        type: agent
        role: executor
        goal: "Task"
"""
        with pytest.raises(YAMLWorkflowError, match="not found"):
            load_workflow_from_yaml(yaml_content, "nonexistent")


class TestLoadWorkflowFromFile:
    """Tests for loading workflows from files."""

    def test_load_from_file(self):
        yaml_content = """
workflows:
  test:
    description: "File test"
    nodes:
      - id: start
        type: agent
        role: executor
        goal: "From file"
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()

            workflow = load_workflow_from_file(f.name, "test")
            assert workflow.name == "test"
            assert workflow.description == "File test"

            Path(f.name).unlink()

    def test_file_not_found_raises_error(self):
        with pytest.raises(YAMLWorkflowError, match="File not found"):
            load_workflow_from_file("/nonexistent/path/workflow.yaml")


class TestLoadWorkflowsFromDirectory:
    """Tests for loading workflows from a directory."""

    def test_load_from_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create two workflow files
            for i, name in enumerate(["workflow_a", "workflow_b"]):
                content = f"""
workflows:
  {name}:
    description: "Workflow {i}"
    nodes:
      - id: step
        type: agent
        role: executor
        goal: "Task {i}"
"""
                Path(tmpdir, f"{name}.yaml").write_text(content)

            workflows = load_workflows_from_directory(tmpdir)
            assert "workflow_a" in workflows
            assert "workflow_b" in workflows


class TestYAMLWorkflowProvider:
    """Tests for the YAML workflow provider."""

    def test_provider_from_file(self):
        yaml_content = """
workflows:
  feature:
    description: "Feature workflow"
    nodes:
      - id: implement
        type: agent
        role: executor
        goal: "Implement feature"
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()

            provider = YAMLWorkflowProvider.from_file(f.name)
            assert "feature" in provider.list_workflows()
            workflow = provider.get_workflow("feature")
            assert workflow is not None
            assert workflow.description == "Feature workflow"

            Path(f.name).unlink()

    def test_provider_get_workflows(self):
        yaml_content = """
workflows:
  test:
    nodes:
      - id: step
        type: agent
        role: executor
        goal: "Task"
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()

            provider = YAMLWorkflowProvider.from_file(f.name)
            workflows = provider.get_workflows()
            assert "test" in workflows

            Path(f.name).unlink()


class TestCustomConditionRegistry:
    """Tests for custom condition functions."""

    def test_custom_condition(self):
        def custom_logic(ctx):
            return "proceed" if ctx.get("custom_flag") else "skip"

        config = YAMLWorkflowConfig(condition_registry={"custom_check": custom_logic})

        data = {
            "nodes": [
                {"id": "start", "type": "agent", "role": "executor", "goal": "Start"},
                {
                    "id": "check",
                    "type": "condition",
                    "condition": "custom_check",
                    "branches": {"proceed": "action", "skip": "end"},
                },
                {"id": "action", "type": "agent", "role": "executor", "goal": "Action"},
                {"id": "end", "type": "agent", "role": "executor", "goal": "End"},
            ],
        }
        workflow = load_workflow_from_dict(data, "custom_test", config)
        condition_node = workflow.nodes["check"]
        assert condition_node.evaluate({"custom_flag": True}) == "action"
        assert condition_node.evaluate({"custom_flag": False}) == "end"


class TestHITLNodes:
    """Tests for HITL node loading."""

    def test_load_hitl_approval(self):
        data = {
            "nodes": [
                {
                    "id": "approve",
                    "type": "hitl",
                    "hitl_type": "approval",
                    "prompt": "Approve changes?",
                    "timeout": 300,
                    "fallback": "abort",
                }
            ],
        }
        workflow = load_workflow_from_dict(data, "hitl_test")
        from victor.workflows.hitl import HITLNode

        assert isinstance(workflow.nodes["approve"], HITLNode)

    def test_load_hitl_choice(self):
        data = {
            "nodes": [
                {
                    "id": "select",
                    "type": "hitl",
                    "hitl_type": "choice",
                    "prompt": "Select option",
                    "choices": ["option_a", "option_b", "option_c"],
                    "default_value": "option_a",
                }
            ],
        }
        workflow = load_workflow_from_dict(data, "choice_test")
        from victor.workflows.hitl import HITLNode

        hitl_node = workflow.nodes["select"]
        assert isinstance(hitl_node, HITLNode)
        assert hitl_node.choices == ["option_a", "option_b", "option_c"]
