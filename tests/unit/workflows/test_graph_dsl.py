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

"""Tests for the WorkflowGraph DSL module."""

from __future__ import annotations

import pytest
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from unittest.mock import patch

from victor.core.async_utils import run_sync as shared_run_sync, run_sync_in_thread
from victor.workflows.graph_dsl import (
    State,
    WorkflowGraph,
    GraphNode,
    GraphNodeType,
    create_graph,
    compile_graph,
)
from victor.workflows.definition import (
    AgentNode,
    ConditionNode,
    ParallelNode,
    TransformNode,
    WorkflowDefinition,
)


# Test State Classes
@dataclass
class SimpleState(State):
    """Simple state for basic tests."""

    value: int = 0
    message: str = ""


@dataclass
class CodeReviewState(State):
    """State for code review workflow tests."""

    files: List[str] = field(default_factory=list)
    analysis: Optional[str] = None
    issues: List[str] = field(default_factory=list)
    has_issues: bool = False
    report: Optional[str] = None


@dataclass
class ParallelState(State):
    """State for parallel execution tests."""

    inputs: List[str] = field(default_factory=list)
    results: Dict[str, Any] = field(default_factory=dict)


# Test Node Functions
def increment(state: SimpleState) -> SimpleState:
    """Increment value."""
    state.value += 1
    return state


def double(state: SimpleState) -> SimpleState:
    """Double value."""
    state.value *= 2
    return state


def set_message(state: SimpleState) -> SimpleState:
    """Set a message."""
    state.message = f"Value is {state.value}"
    return state


async def async_increment(state: SimpleState) -> SimpleState:
    """Async increment value."""
    state.value += 1
    return state


def analyze_code(state: CodeReviewState) -> CodeReviewState:
    """Analyze code files."""
    state.analysis = f"Analyzed {len(state.files)} files"
    return state


def find_issues(state: CodeReviewState) -> CodeReviewState:
    """Find issues in code."""
    if "buggy.py" in state.files:
        state.issues = ["Bug in line 42", "Missing docstring"]
        state.has_issues = True
    return state


def generate_report(state: CodeReviewState) -> CodeReviewState:
    """Generate report."""
    state.report = f"Analysis: {state.analysis}. Issues: {len(state.issues)}"
    return state


def route_by_issues(state: CodeReviewState) -> str:
    """Route based on issues found."""
    if state.has_issues:
        return "has_issues"
    return "no_issues"


class TestState:
    """Tests for State base class."""

    def test_to_dict(self):
        """Test state serialization."""
        state = SimpleState(value=42, message="hello")
        data = state.to_dict()

        assert data == {"value": 42, "message": "hello"}

    def test_from_dict(self):
        """Test state deserialization."""
        data = {"value": 100, "message": "test"}
        state = SimpleState.from_dict(data)

        assert state.value == 100
        assert state.message == "test"

    def test_from_dict_ignores_extra_keys(self):
        """Test that from_dict ignores unknown keys."""
        data = {"value": 5, "message": "x", "extra": "ignored"}
        state = SimpleState.from_dict(data)

        assert state.value == 5
        assert not hasattr(state, "extra")

    def test_copy(self):
        """Test state copy."""
        original = SimpleState(value=10)
        copy = original.copy()

        copy.value = 20
        assert original.value == 10
        assert copy.value == 20

    def test_merge(self):
        """Test state merge."""
        state = SimpleState(value=5, message="original")
        merged = state.merge({"message": "updated"})

        assert merged.value == 5
        assert merged.message == "updated"
        assert state.message == "original"  # Original unchanged


class TestWorkflowGraph:
    """Tests for WorkflowGraph class."""

    def test_create_empty_graph(self):
        """Test creating an empty graph."""
        graph = WorkflowGraph(SimpleState, name="test")

        assert graph.name == "test"
        assert graph.state_type is SimpleState
        assert len(graph._nodes) == 0

    def test_add_node(self):
        """Test adding a node."""
        graph = WorkflowGraph(SimpleState)
        graph.add_node("process", increment)

        assert "process" in graph._nodes
        assert graph._nodes["process"].func is increment

    def test_add_node_duplicate_raises(self):
        """Test that duplicate node names raise."""
        graph = WorkflowGraph(SimpleState)
        graph.add_node("process", increment)

        with pytest.raises(ValueError, match="already exists"):
            graph.add_node("process", double)

    def test_add_agent_node(self):
        """Test adding an agent node."""
        graph = WorkflowGraph(CodeReviewState)
        graph.add_agent_node(
            "analyze",
            role="researcher",
            goal="Analyze the codebase",
            tool_budget=20,
        )

        node = graph._nodes["analyze"]
        assert node.node_type == GraphNodeType.AGENT
        assert node.agent_role == "researcher"
        assert node.agent_goal == "Analyze the codebase"
        assert node.tool_budget == 20

    def test_add_edge(self):
        """Test adding an edge."""
        graph = WorkflowGraph(SimpleState)
        graph.add_node("a", increment)
        graph.add_node("b", double)
        graph.add_edge("a", "b")

        assert "b" in graph._edges["a"]

    def test_add_edge_invalid_source(self):
        """Test adding edge with invalid source."""
        graph = WorkflowGraph(SimpleState)
        graph.add_node("b", double)

        with pytest.raises(ValueError, match="not found"):
            graph.add_edge("a", "b")

    def test_add_edge_invalid_target(self):
        """Test adding edge with invalid target."""
        graph = WorkflowGraph(SimpleState)
        graph.add_node("a", increment)

        with pytest.raises(ValueError, match="not found"):
            graph.add_edge("a", "nonexistent")

    def test_chain_method(self):
        """Test chaining nodes."""
        graph = WorkflowGraph(SimpleState)
        graph.add_node("a", increment)
        graph.add_node("b", double)
        graph.add_node("c", set_message)
        graph.chain("a", "b", "c")

        assert "b" in graph._edges["a"]
        assert "c" in graph._edges["b"]

    def test_operator_chaining(self):
        """Test >> operator chaining."""
        graph = WorkflowGraph(SimpleState)
        graph.add_node("a", increment)
        graph.add_node("b", double)
        graph.add_node("c", set_message)

        # Use >> operator
        graph.node("a") >> graph.node("b") >> graph.node("c")

        assert "b" in graph._edges["a"]
        assert "c" in graph._edges["b"]

    def test_set_entry_point(self):
        """Test setting entry point."""
        graph = WorkflowGraph(SimpleState)
        graph.add_node("start", increment)
        graph.set_entry_point("start")

        assert graph._entry_point == "start"
        assert "start" in graph._edges.get(WorkflowGraph.START, [])

    def test_set_finish_point(self):
        """Test setting finish point."""
        graph = WorkflowGraph(SimpleState)
        graph.add_node("end", set_message)
        graph.set_finish_point("end")

        assert "end" in graph._finish_points
        assert WorkflowGraph.END in graph._edges.get("end", [])

    def test_add_conditional_edges(self):
        """Test adding conditional edges."""
        graph = WorkflowGraph(CodeReviewState)
        graph.add_node("analyze", analyze_code)
        graph.add_node("fix", find_issues)
        graph.add_node("report", generate_report)

        graph.add_conditional_edges(
            "analyze",
            route_by_issues,
            {"has_issues": "fix", "no_issues": "report"},
        )

        assert "analyze" in graph._conditional_edges
        router, routes = graph._conditional_edges["analyze"]
        assert routes["has_issues"] == "fix"
        assert routes["no_issues"] == "report"

    def test_branch_method(self):
        """Test branch helper method."""
        graph = WorkflowGraph(SimpleState)
        graph.add_node("split", increment)
        graph.add_node("a", double)
        graph.add_node("b", set_message)

        graph.branch("split", "a", "b")

        assert "a" in graph._edges["split"]
        assert "b" in graph._edges["split"]

    def test_merge_method(self):
        """Test merge helper method."""
        graph = WorkflowGraph(SimpleState)
        graph.add_node("a", increment)
        graph.add_node("b", double)
        graph.add_node("join", set_message)

        graph.merge("join", "a", "b")

        assert "join" in graph._edges["a"]
        assert "join" in graph._edges["b"]


class TestValidation:
    """Tests for graph validation."""

    def test_validate_empty_graph(self):
        """Test validation of empty graph."""
        graph = WorkflowGraph(SimpleState)
        errors = graph.validate()

        assert "Graph has no nodes" in errors

    def test_validate_no_entry_point(self):
        """Test validation without entry point."""
        graph = WorkflowGraph(SimpleState)
        graph.add_node("a", increment)
        errors = graph.validate()

        assert any("entry point" in e.lower() for e in errors)

    def test_validate_no_finish_points(self):
        """Test validation without finish points."""
        graph = WorkflowGraph(SimpleState)
        graph.add_node("a", increment)
        graph.set_entry_point("a")
        errors = graph.validate()

        # Should warn about terminal nodes
        assert any("finish" in e.lower() or "terminal" in e.lower() for e in errors)

    def test_validate_unreachable_node(self):
        """Test validation with unreachable node."""
        graph = WorkflowGraph(SimpleState)
        graph.add_node("a", increment)
        graph.add_node("b", double)  # Not connected
        graph.set_entry_point("a")
        graph.set_finish_point("a")

        errors = graph.validate()
        assert any("unreachable" in e.lower() for e in errors)

    def test_validate_valid_graph(self):
        """Test validation of valid graph."""
        graph = WorkflowGraph(SimpleState)
        graph.add_node("a", increment)
        graph.add_node("b", double)
        graph.add_edge("a", "b")
        graph.set_entry_point("a")
        graph.set_finish_point("b")

        errors = graph.validate()
        assert len(errors) == 0


class TestCompilation:
    """Tests for graph compilation."""

    def test_compile_simple_graph(self):
        """Test compiling a simple linear graph."""
        graph = WorkflowGraph(SimpleState, name="simple_test")
        graph.add_node("a", increment)
        graph.add_node("b", double)
        graph.add_edge("a", "b")
        graph.set_entry_point("a")
        graph.set_finish_point("b")

        workflow = graph.compile()

        assert isinstance(workflow, WorkflowDefinition)
        assert workflow.name == "simple_test"
        assert workflow.start_node == "a"
        assert len(workflow.nodes) == 2

    def test_compile_with_agent_node(self):
        """Test compiling graph with agent node."""
        graph = WorkflowGraph(CodeReviewState, name="review")
        graph.add_agent_node(
            "analyze",
            role="researcher",
            goal="Analyze code",
            tool_budget=15,
        )
        graph.add_node("report", generate_report)
        graph.add_edge("analyze", "report")
        graph.set_entry_point("analyze")
        graph.set_finish_point("report")

        workflow = graph.compile()

        assert isinstance(workflow.nodes["analyze"], AgentNode)
        agent_node = workflow.nodes["analyze"]
        assert agent_node.role == "researcher"
        assert agent_node.tool_budget == 15

    def test_compile_with_transform_node(self):
        """Test that function nodes become TransformNodes."""
        graph = WorkflowGraph(SimpleState)
        graph.add_node("process", increment)
        graph.set_entry_point("process")
        graph.set_finish_point("process")

        workflow = graph.compile()

        assert isinstance(workflow.nodes["process"], TransformNode)

    def test_compile_with_async_transform_node_uses_shared_sync_bridge(self):
        """Async function nodes should use the shared sync bridge when compiled."""
        graph = WorkflowGraph(SimpleState)
        graph.add_node("process", async_increment)
        graph.set_entry_point("process")
        graph.set_finish_point("process")

        workflow = graph.compile()

        with patch(
            "victor.workflows.graph_dsl.run_sync",
            side_effect=lambda coro: shared_run_sync(coro),
        ) as mock_run_sync:
            result = workflow.nodes["process"].transform({"value": 1})

        assert result["value"] == 2
        mock_run_sync.assert_called_once()

    @pytest.mark.asyncio
    async def test_compile_with_async_transform_node_uses_thread_bridge_in_async_context(
        self,
    ):
        """Compiled async function nodes should avoid nested loops in async contexts."""
        graph = WorkflowGraph(SimpleState)
        graph.add_node("process", async_increment)
        graph.set_entry_point("process")
        graph.set_finish_point("process")

        workflow = graph.compile()

        with patch(
            "victor.workflows.graph_dsl.run_sync_in_thread",
            side_effect=lambda coro: run_sync_in_thread(coro),
        ) as mock_run_sync_in_thread:
            result = workflow.nodes["process"].transform({"value": 2})

        assert result["value"] == 3
        mock_run_sync_in_thread.assert_called_once()

    def test_compile_preserves_edges(self):
        """Test that edges are preserved in compilation."""
        graph = WorkflowGraph(SimpleState)
        graph.add_node("a", increment)
        graph.add_node("b", double)
        graph.add_node("c", set_message)
        graph.chain("a", "b", "c")
        graph.set_entry_point("a")
        graph.set_finish_point("c")

        workflow = graph.compile()

        assert "b" in workflow.nodes["a"].next_nodes
        assert "c" in workflow.nodes["b"].next_nodes

    def test_compile_invalid_graph_raises(self):
        """Test that compiling invalid graph raises."""
        graph = WorkflowGraph(SimpleState)  # Empty graph

        with pytest.raises(ValueError, match="validation failed"):
            graph.compile()

    def test_compile_with_metadata(self):
        """Test that metadata is preserved."""
        graph = WorkflowGraph(SimpleState, description="Test workflow")
        graph.add_node("a", increment)
        graph.set_entry_point("a")
        graph.set_finish_point("a")
        graph.set_metadata("version", "1.0")

        workflow = graph.compile()

        assert workflow.description == "Test workflow"
        assert workflow.metadata.get("version") == "1.0"
        assert workflow.metadata.get("compiled_from") == "WorkflowGraph"


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_create_graph(self):
        """Test create_graph function."""
        graph = create_graph(SimpleState, "test", "A test workflow")

        assert isinstance(graph, WorkflowGraph)
        assert graph.name == "test"
        assert graph.description == "A test workflow"

    def test_compile_graph(self):
        """Test compile_graph function."""
        graph = create_graph(SimpleState, "test")
        graph.add_node("a", increment)
        graph.set_entry_point("a")
        graph.set_finish_point("a")

        workflow = compile_graph(graph)

        assert isinstance(workflow, WorkflowDefinition)


class TestIntegration:
    """Integration tests for complete workflows."""

    def test_code_review_workflow(self):
        """Test a complete code review workflow."""
        graph = WorkflowGraph(CodeReviewState, name="code_review")

        # Add nodes
        graph.add_node("analyze", analyze_code)
        graph.add_node("find_issues", find_issues)
        graph.add_node("report", generate_report)

        # Chain nodes
        graph.chain("analyze", "find_issues", "report")

        # Set entry and exit
        graph.set_entry_point("analyze")
        graph.set_finish_point("report")

        # Compile
        workflow = graph.compile()

        assert workflow.name == "code_review"
        assert len(workflow.nodes) == 3
        assert workflow.start_node == "analyze"

    def test_workflow_with_conditional_edges(self):
        """Test workflow with conditional branching."""
        graph = WorkflowGraph(CodeReviewState, name="conditional_review")

        # Add nodes
        graph.add_node("analyze", analyze_code)
        graph.add_node("fix", find_issues)
        graph.add_node("report", generate_report)

        # Add conditional edges
        graph.add_conditional_edges(
            "analyze",
            route_by_issues,
            {"has_issues": "fix", "no_issues": "report"},
        )
        graph.add_edge("fix", "report")

        graph.set_entry_point("analyze")
        graph.set_finish_point("report")

        workflow = graph.compile()

        # Analyze node should be converted to ConditionNode
        assert isinstance(workflow.nodes["analyze"], ConditionNode)
        condition_node = workflow.nodes["analyze"]
        assert "has_issues" in condition_node.branches
        assert condition_node.branches["has_issues"] == "fix"

    def test_mixed_agent_and_function_nodes(self):
        """Test workflow mixing agent and function nodes."""
        graph = WorkflowGraph(CodeReviewState, name="mixed_workflow")

        # Agent node
        graph.add_agent_node(
            "research",
            role="researcher",
            goal="Research the codebase",
            tool_budget=25,
            allowed_tools=["read", "grep", "code_search"],
        )

        # Function nodes
        graph.add_node("process", find_issues)
        graph.add_node("report", generate_report)

        # Chain
        graph.chain("research", "process", "report")
        graph.set_entry_point("research")
        graph.set_finish_point("report")

        workflow = graph.compile()

        assert isinstance(workflow.nodes["research"], AgentNode)
        assert isinstance(workflow.nodes["process"], TransformNode)
        assert isinstance(workflow.nodes["report"], TransformNode)

        agent = workflow.nodes["research"]
        assert agent.allowed_tools == ["read", "grep", "code_search"]
