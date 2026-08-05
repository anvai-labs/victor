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

"""End-to-end integration tests for workflow execution.

Tests the complete workflow execution pipeline from definition to result,
including:
- WorkflowBuilder creating workflows
- WorkflowExecutor executing workflows
- Node execution (agent, condition, parallel, transform)
- YAML-defined workflow execution
- Vertical workflow provider integration
"""

import asyncio
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from victor.core.verticals.import_resolver import import_module_with_fallback
from victor.workflows import (
    WorkflowBuilder,
    WorkflowDefinition,
    WorkflowExecutor,
    WorkflowResult,
    ExecutorNodeStatus,
    load_workflow_from_yaml,
)


def _load_vertical_attr(module_path: str, attr_name: str):
    """Resolve a vertical attribute using external-first import fallbacks."""
    module, _resolved = import_module_with_fallback(module_path)
    if module is None:
        pytest.skip(f"Vertical module not available: {module_path}")
    if not hasattr(module, attr_name):
        pytest.skip(f"Missing attribute '{attr_name}' in module '{module_path}'")
    return getattr(module, attr_name)


@pytest.fixture
def mock_orchestrator():
    """Create a mock orchestrator for testing."""
    orchestrator = MagicMock()
    orchestrator.settings = MagicMock()
    orchestrator.provider = MagicMock()
    orchestrator.provider.name = "mock_provider"
    return orchestrator


@pytest.fixture
def mock_subagent_result():
    """Create a mock subagent result."""
    result = MagicMock()
    result.success = True
    result.output = "Mock agent output"
    result.tool_calls_made = 5
    return result


class TestWorkflowExecution:
    """Integration tests for workflow execution."""

    @pytest.mark.asyncio
    async def test_simple_linear_workflow(self, mock_orchestrator, mock_subagent_result):
        """Test execution of a simple linear workflow."""
        from victor.workflows import NodeResult

        # Create a simple workflow
        workflow = (
            WorkflowBuilder("test_linear", "Linear workflow test")
            .add_agent("step1", "researcher", "Research the topic", tool_budget=10)
            .add_agent("step2", "executor", "Execute the plan", tool_budget=15)
            .add_agent("step3", "writer", "Write the report", tool_budget=5)
            .build()
        )

        executor = WorkflowExecutor(mock_orchestrator)

        # Mock the subagent execution (signature: node, context, start_time) -> NodeResult
        async def mock_execute(node, context, start_time):
            return NodeResult(
                node_id=node.id,
                status=ExecutorNodeStatus.COMPLETED,
                output=mock_subagent_result.output,
                tool_calls_used=5,
            )

        with patch.object(executor, "_execute_agent_node", side_effect=mock_execute):
            result = await executor.execute(workflow, {"input": "test data"})

        assert result.success
        assert result.workflow_name == "test_linear"
        assert len(result.context.node_results) == 3
        assert all(
            r.status == ExecutorNodeStatus.COMPLETED for r in result.context.node_results.values()
        )

    @pytest.mark.asyncio
    async def test_workflow_with_condition(self, mock_orchestrator, mock_subagent_result):
        """Test workflow with conditional branching."""
        from victor.workflows import NodeResult

        def check_issues(ctx: Dict[str, Any]) -> str:
            return "fix" if ctx.get("has_issues") else "skip"

        workflow = (
            WorkflowBuilder("test_condition", "Conditional workflow")
            .add_agent("analyze", "analyst", "Analyze for issues", output_key="analysis")
            .add_condition("decide", check_issues, {"fix": "fix_step", "skip": "report"})
            .add_agent("fix_step", "executor", "Fix issues", next_nodes=["report"])
            .add_agent("report", "writer", "Generate report")
            .build()
        )

        executor = WorkflowExecutor(mock_orchestrator)

        # Test with issues - should go through fix_step
        async def mock_execute(node, context, start_time):
            return NodeResult(
                node_id=node.id,
                status=ExecutorNodeStatus.COMPLETED,
                output="Fixed!",
                tool_calls_used=3,
            )

        with patch.object(executor, "_execute_agent_node", side_effect=mock_execute):
            result = await executor.execute(workflow, {"has_issues": True})

        assert result.success
        # Should have executed: analyze, decide, fix_step, report
        assert "analyze" in result.context.node_results
        assert "decide" in result.context.node_results
        assert "fix_step" in result.context.node_results
        assert "report" in result.context.node_results

    @pytest.mark.asyncio
    async def test_workflow_skip_branch(self, mock_orchestrator, mock_subagent_result):
        """Test workflow takes skip branch when no issues."""
        from victor.workflows import NodeResult

        def check_issues(ctx: Dict[str, Any]) -> str:
            return "fix" if ctx.get("has_issues") else "skip"

        workflow = (
            WorkflowBuilder("test_skip", "Skip branch test")
            .add_agent("analyze", "analyst", "Analyze")
            .add_condition("decide", check_issues, {"fix": "fix_step", "skip": "report"})
            .add_agent("fix_step", "executor", "Fix", next_nodes=["report"])
            .add_agent("report", "writer", "Report")
            .build()
        )

        executor = WorkflowExecutor(mock_orchestrator)

        async def mock_execute(node, context, start_time):
            return NodeResult(
                node_id=node.id,
                status=ExecutorNodeStatus.COMPLETED,
                output="Done",
                tool_calls_used=2,
            )

        with patch.object(executor, "_execute_agent_node", side_effect=mock_execute):
            result = await executor.execute(workflow, {"has_issues": False})

        assert result.success
        # Should skip fix_step
        assert "analyze" in result.context.node_results
        assert "decide" in result.context.node_results
        assert "report" in result.context.node_results
        assert "fix_step" not in result.context.node_results

    @pytest.mark.asyncio
    async def test_workflow_with_transform(self, mock_orchestrator, mock_subagent_result):
        """Test workflow with transform node."""
        from victor.workflows import NodeResult

        workflow = (
            WorkflowBuilder("test_transform", "Transform workflow")
            .add_agent("fetch", "researcher", "Fetch data")
            .add_transform(
                "transform",
                lambda ctx: {**ctx, "transformed": True, "data": "processed"},
            )
            .add_agent("process", "executor", "Process data")
            .build()
        )

        executor = WorkflowExecutor(mock_orchestrator)

        async def mock_execute(node, context, start_time):
            return NodeResult(
                node_id=node.id,
                status=ExecutorNodeStatus.COMPLETED,
                output="Result",
                tool_calls_used=3,
            )

        with patch.object(executor, "_execute_agent_node", side_effect=mock_execute):
            result = await executor.execute(workflow, {"initial": "data"})

        assert result.success
        assert result.context.data.get("transformed") is True
        assert result.context.data.get("data") == "processed"

    @pytest.mark.asyncio
    async def test_workflow_timeout(self, mock_orchestrator):
        """Test workflow timeout handling."""
        from victor.workflows import NodeResult

        workflow = (
            WorkflowBuilder("test_timeout", "Timeout test")
            .add_agent("slow", "executor", "Slow task")
            .build()
        )

        executor = WorkflowExecutor(mock_orchestrator)

        # Mock a slow execution (signature: node, context, start_time)
        async def slow_execute(node, context, start_time):
            await asyncio.sleep(5)
            return NodeResult(
                node_id=node.id,
                status=ExecutorNodeStatus.COMPLETED,
                output="Done",
                tool_calls_used=1,
            )

        with patch.object(executor, "_execute_agent_node", side_effect=slow_execute):
            result = await executor.execute(workflow, timeout=0.1)

        assert not result.success
        assert "timed out" in result.error.lower()

    @pytest.mark.asyncio
    async def test_workflow_context_passing(self, mock_orchestrator):
        """Test that context is passed between nodes."""
        from victor.workflows import NodeResult

        workflow = (
            WorkflowBuilder("test_context", "Context passing test")
            .add_agent("step1", "researcher", "Research", output_key="research")
            .add_agent("step2", "executor", "Use research", output_key="result")
            .build()
        )

        executor = WorkflowExecutor(mock_orchestrator)

        call_count = [0]

        async def mock_execute(node, context, start_time):
            call_count[0] += 1
            if node.id == "step1":
                context.set("research", "Research findings")
                return NodeResult(
                    node_id=node.id,
                    status=ExecutorNodeStatus.COMPLETED,
                    output="Research done",
                    tool_calls_used=5,
                )
            elif node.id == "step2":
                # Verify we can access previous output
                research = context.get("research")
                assert research == "Research findings"
                return NodeResult(
                    node_id=node.id,
                    status=ExecutorNodeStatus.COMPLETED,
                    output=f"Used: {research}",
                    tool_calls_used=3,
                )

        with patch.object(executor, "_execute_agent_node", side_effect=mock_execute):
            result = await executor.execute(workflow)

        assert result.success
        assert call_count[0] == 2

    @pytest.mark.asyncio
    async def test_workflow_from_yaml(self, mock_orchestrator, mock_subagent_result):
        """Test executing a workflow loaded from YAML."""
        from victor.workflows import NodeResult

        yaml_content = """
workflows:
  yaml_test:
    description: "YAML-defined workflow"
    nodes:
      - id: research
        type: agent
        role: researcher
        goal: "Research the topic"
        tool_budget: 15
      - id: write
        type: agent
        role: writer
        goal: "Write summary"
"""
        workflow = load_workflow_from_yaml(yaml_content, "yaml_test")
        executor = WorkflowExecutor(mock_orchestrator)

        async def mock_execute(node, context, start_time):
            return NodeResult(
                node_id=node.id,
                status=ExecutorNodeStatus.COMPLETED,
                output="YAML workflow done",
                tool_calls_used=5,
            )

        with patch.object(executor, "_execute_agent_node", side_effect=mock_execute):
            result = await executor.execute(workflow)

        assert result.success
        assert result.workflow_name == "yaml_test"

    @pytest.mark.asyncio
    async def test_workflow_result_aggregation(self, mock_orchestrator):
        """Test that workflow aggregates results from all nodes."""
        from victor.workflows import NodeResult

        workflow = (
            WorkflowBuilder("test_aggregate", "Aggregation test")
            .add_agent("a", "researcher", "Task A", output_key="output_a")
            .add_agent("b", "executor", "Task B", output_key="output_b")
            .add_agent("c", "writer", "Task C", output_key="output_c")
            .build()
        )

        executor = WorkflowExecutor(mock_orchestrator)

        async def mock_execute(node, context, start_time):
            output = f"Output from {node.id}"
            context.set(node.output_key, output)
            return NodeResult(
                node_id=node.id,
                status=ExecutorNodeStatus.COMPLETED,
                output=output,
                tool_calls_used=2,
            )

        with patch.object(executor, "_execute_agent_node", side_effect=mock_execute):
            result = await executor.execute(workflow)

        assert result.success
        outputs = result.context.get_outputs()
        assert "a" in outputs
        assert "b" in outputs
        assert "c" in outputs


class TestVerticalWorkflowIntegration:
    """Integration tests for vertical workflow providers."""

    def test_coding_workflow_provider(self):
        """Test Coding vertical workflow provider."""
        CodingAssistant = _load_vertical_attr("victor.coding", "CodingAssistant")

        provider = CodingAssistant.get_workflow_provider()
        if provider is None:
            pytest.skip("CodingAssistant workflow provider not implemented (external package)")

        workflows = provider.get_workflows()
        assert len(workflows) > 0
        assert "code_review" in workflows or "feature_implementation" in workflows

    def test_data_analysis_workflow_provider(self):
        """Test Data Analysis vertical workflow provider."""
        DataAnalysisAssistant = _load_vertical_attr(
            "victor.dataanalysis",
            "DataAnalysisAssistant",
        )

        provider = DataAnalysisAssistant.get_workflow_provider()
        if provider is None:
            pytest.skip(
                "DataAnalysisAssistant workflow provider not implemented (external package)"
            )

        workflows = provider.get_workflows()
        assert len(workflows) > 0

    def test_research_workflow_provider(self):
        """Test Research vertical workflow provider."""
        ResearchAssistant = _load_vertical_attr("victor.research", "ResearchAssistant")

        provider = ResearchAssistant.get_workflow_provider()
        if provider is None:
            pytest.skip("ResearchAssistant workflow provider not implemented (external package)")

        workflows = provider.get_workflows()
        assert len(workflows) > 0
        assert "deep_research" in workflows

    def test_devops_workflow_provider(self):
        """Test DevOps vertical workflow provider."""
        DevOpsAssistant = _load_vertical_attr("victor.devops", "DevOpsAssistant")

        provider = DevOpsAssistant.get_workflow_provider()
        if provider is None:
            pytest.skip("DevOpsAssistant workflow provider not implemented (external package)")

        workflows = provider.get_workflows()
        assert len(workflows) > 0

    def test_workflow_definition_validity(self):
        """Test that all vertical workflows pass validation."""
        CodingAssistant = _load_vertical_attr("victor.coding", "CodingAssistant")
        DataAnalysisAssistant = _load_vertical_attr(
            "victor.dataanalysis",
            "DataAnalysisAssistant",
        )
        DevOpsAssistant = _load_vertical_attr("victor.devops", "DevOpsAssistant")
        ResearchAssistant = _load_vertical_attr("victor.research", "ResearchAssistant")

        verticals = [
            CodingAssistant,
            DataAnalysisAssistant,
            DevOpsAssistant,
            ResearchAssistant,
        ]

        for vertical in verticals:
            provider = vertical.get_workflow_provider()
            if provider:
                workflows = provider.get_workflows()
                for name in workflows.keys():
                    workflow = provider.get_workflow(name)
                    assert workflow is not None
                    errors = workflow.validate()
                    assert not errors, f"{vertical.name}/{name}: {errors}"


class TestWorkflowBuilderIntegration:
    """Integration tests for WorkflowBuilder."""

    def test_complex_workflow_building(self):
        """Test building a complex workflow with all node types."""

        def categorize(ctx):
            score = ctx.get("score", 0)
            if score > 80:
                return "excellent"
            elif score > 50:
                return "good"
            else:
                return "needs_work"

        workflow = (
            WorkflowBuilder("complex_test", "Complex workflow")
            .add_agent("analyze", "analyst", "Analyze data", output_key="analysis")
            .add_transform("score", lambda ctx: {**ctx, "score": 75})
            .add_condition(
                "categorize",
                categorize,
                {"excellent": "celebrate", "good": "report", "needs_work": "improve"},
            )
            .add_agent("celebrate", "writer", "Write celebration", next_nodes=["finish"])
            .add_agent("report", "writer", "Write report", next_nodes=["finish"])
            .add_agent("improve", "executor", "Improve work", next_nodes=["finish"])
            .add_agent("finish", "planner", "Final steps")
            .build()
        )

        assert workflow.name == "complex_test"
        assert len(workflow.nodes) == 7
        assert workflow.start_node == "analyze"

        # Validate structure
        errors = workflow.validate()
        assert not errors

    def test_workflow_with_hitl_nodes(self):
        """Test building workflow with HITL nodes."""
        workflow = (
            WorkflowBuilder("hitl_test", "HITL workflow")
            .add_agent("propose", "planner", "Propose changes")
            .add_hitl_approval(
                "approve",
                prompt="Do you approve these changes?",
                context_keys=["proposed_changes"],
                timeout=300.0,
                fallback="abort",
            )
            .add_agent("apply", "executor", "Apply changes")
            .build()
        )

        assert len(workflow.nodes) == 3
        assert "approve" in workflow.nodes

        from victor.workflows.hitl import HITLNode

        hitl_node = workflow.nodes["approve"]
        assert isinstance(hitl_node, HITLNode)
        assert hitl_node.prompt == "Do you approve these changes?"

    def test_auto_chaining(self):
        """Test that nodes are auto-chained when no explicit next is provided."""
        workflow = (
            WorkflowBuilder("autochain_test")
            .add_agent("a", "executor", "Task A")
            .add_agent("b", "executor", "Task B")
            .add_agent("c", "executor", "Task C")
            .build()
        )

        # Verify chain: a -> b -> c
        assert "b" in workflow.nodes["a"].next_nodes
        assert "c" in workflow.nodes["b"].next_nodes
        assert len(workflow.nodes["c"].next_nodes) == 0  # Terminal node


class TestYAMLWorkflowIntegration:
    """Integration tests for YAML workflow loading."""

    def test_complex_yaml_workflow(self):
        """Test loading and validating a complex YAML workflow."""
        yaml_content = """
workflows:
  complete_analysis:
    description: "Complete data analysis pipeline"
    metadata:
      version: "1.0"
      category: "data_analysis"
    nodes:
      - id: load_data
        type: agent
        role: researcher
        goal: "Load and validate dataset"
        tool_budget: 10
        tools: [read, shell]
        output: raw_data

      - id: check_quality
        type: condition
        condition: "data_quality > 0.8"
        branches:
          "true": analyze
          "false": clean

      - id: clean
        type: agent
        role: executor
        goal: "Clean and preprocess data"
        tool_budget: 20
        next: [analyze]

      - id: analyze
        type: agent
        role: analyst
        goal: "Perform statistical analysis"
        tool_budget: 25
        output: analysis_results

      - id: visualize
        type: agent
        role: executor
        goal: "Create visualizations"
        tool_budget: 15

      - id: report
        type: agent
        role: writer
        goal: "Write analysis report"
        tool_budget: 10
"""
        workflows = load_workflow_from_yaml(yaml_content)
        assert "complete_analysis" in workflows

        workflow = workflows["complete_analysis"]
        assert workflow.description == "Complete data analysis pipeline"
        assert workflow.metadata.get("version") == "1.0"
        assert len(workflow.nodes) == 6

        # Validate
        errors = workflow.validate()
        assert not errors

    def test_yaml_with_all_node_types(self):
        """Test YAML with agent, condition, parallel, and transform nodes."""
        yaml_content = """
workflows:
  multi_type:
    nodes:
      - id: start
        type: agent
        role: researcher
        goal: "Start work"

      - id: prepare
        type: transform
        transform: "status = 'prepared'"

      - id: branch
        type: condition
        condition: ready
        branches:
          "true": process
          "false": retry

      - id: retry
        type: agent
        role: executor
        goal: "Retry preparation"
        next: [branch]

      - id: process
        type: agent
        role: executor
        goal: "Process data"
"""
        workflow = load_workflow_from_yaml(yaml_content, "multi_type")
        assert len(workflow.nodes) == 5

        from victor.workflows import ConditionNode, TransformNode, AgentNode

        assert isinstance(workflow.nodes["start"], AgentNode)
        assert isinstance(workflow.nodes["prepare"], TransformNode)
        assert isinstance(workflow.nodes["branch"], ConditionNode)
