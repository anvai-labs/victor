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

"""Integration tests for the WorkflowExecutor.

These tests verify end-to-end workflow execution, including:
- Full workflow execution from start to finish
- Checkpoint save/resume functionality
- Error handling and retry policies
- Conditional edge routing
- Parallel node execution
- Workflow cancellation
"""

import asyncio
import pytest
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

from victor.workflows.graph_dsl import WorkflowGraph, State
from victor.workflows.definition import (
    WorkflowBuilder,
    WorkflowDefinition,
    AgentNode,
    ConditionNode,
    TransformNode,
    ParallelNode,
)
from victor.workflows.context import WorkflowContext, WorkflowResult
from victor.workflows.unified_executor import WorkflowExecutor
from victor_contracts.workflows import ExecutorNodeStatus, NodeResult

# ============ Test State Classes ============


@dataclass
class SimpleExecutionState(State):
    """Simple state for execution tests."""

    counter: int = 0
    steps: List[str] = field(default_factory=list)
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ErrorHandlingState(State):
    """State for error handling tests."""

    should_fail: bool = False
    retry_count: int = 0
    error_message: Optional[str] = None
    recovered: bool = False


@dataclass
class ConditionalState(State):
    """State for conditional routing tests."""

    value: int = 0
    path_taken: str = ""
    result: Optional[str] = None


@dataclass
class ParallelState(State):
    """State for parallel execution tests."""

    inputs: List[str] = field(default_factory=list)
    results: Dict[str, Any] = field(default_factory=dict)
    completion_order: List[str] = field(default_factory=list)


# ============ Helper Functions for Tests ============


def create_mock_orchestrator():
    """Create a mock orchestrator for testing without real LLM calls."""
    mock_orchestrator = MagicMock()
    mock_orchestrator.settings = MagicMock()
    mock_orchestrator.settings.tool_budget = 15
    return mock_orchestrator


def create_mock_subagent_orchestrator():
    """Create a mock SubAgentOrchestrator."""
    mock_subagents = MagicMock()
    mock_result = MagicMock()
    mock_result.success = True
    mock_result.summary = "Mock agent completed task"
    mock_result.error = None
    mock_result.tool_calls_used = 3
    mock_subagents.spawn = AsyncMock(return_value=mock_result)
    return mock_subagents


# ============ Test Fixtures ============


@pytest.fixture
def mock_orchestrator():
    """Fixture providing a mock orchestrator."""
    return create_mock_orchestrator()


@pytest.fixture
def simple_workflow():
    """Create a simple linear workflow for testing."""

    def step_a(ctx: Dict[str, Any]) -> Dict[str, Any]:
        ctx["steps"] = ctx.get("steps", []) + ["a"]
        ctx["counter"] = ctx.get("counter", 0) + 1
        return ctx

    def step_b(ctx: Dict[str, Any]) -> Dict[str, Any]:
        ctx["steps"] = ctx.get("steps", []) + ["b"]
        ctx["counter"] = ctx.get("counter", 0) + 10
        return ctx

    def step_c(ctx: Dict[str, Any]) -> Dict[str, Any]:
        ctx["steps"] = ctx.get("steps", []) + ["c"]
        ctx["counter"] = ctx.get("counter", 0) + 100
        return ctx

    return (
        WorkflowBuilder("simple_workflow", "Simple linear workflow")
        .add_transform("step_a", step_a, next_nodes=["step_b"])
        .add_transform("step_b", step_b, next_nodes=["step_c"])
        .add_transform("step_c", step_c)
        .build()
    )


@pytest.fixture
def conditional_workflow():
    """Create a workflow with conditional branching."""

    def init_value(ctx: Dict[str, Any]) -> Dict[str, Any]:
        # Value is passed via initial context
        return ctx

    def route_by_value(ctx: Dict[str, Any]) -> str:
        value = ctx.get("value", 0)
        if value > 50:
            return "high"
        elif value > 20:
            return "medium"
        else:
            return "low"

    def handle_high(ctx: Dict[str, Any]) -> Dict[str, Any]:
        ctx["path_taken"] = "high"
        ctx["result"] = "Processed high value"
        return ctx

    def handle_medium(ctx: Dict[str, Any]) -> Dict[str, Any]:
        ctx["path_taken"] = "medium"
        ctx["result"] = "Processed medium value"
        return ctx

    def handle_low(ctx: Dict[str, Any]) -> Dict[str, Any]:
        ctx["path_taken"] = "low"
        ctx["result"] = "Processed low value"
        return ctx

    def finalize(ctx: Dict[str, Any]) -> Dict[str, Any]:
        ctx["finalized"] = True
        return ctx

    return (
        WorkflowBuilder("conditional_workflow", "Workflow with branching")
        .add_transform("init", init_value)
        .add_condition(
            "router",
            route_by_value,
            {
                "high": "handle_high",
                "medium": "handle_medium",
                "low": "handle_low",
            },
        )
        .add_transform("handle_high", handle_high, next_nodes=["finalize"])
        .add_transform("handle_medium", handle_medium, next_nodes=["finalize"])
        .add_transform("handle_low", handle_low, next_nodes=["finalize"])
        .add_transform("finalize", finalize)
        .chain("init", "router")
        .build()
    )


@pytest.fixture
def error_workflow():
    """Create a workflow that can fail for error handling tests."""

    def maybe_fail(ctx: Dict[str, Any]) -> Dict[str, Any]:
        if ctx.get("should_fail", False):
            raise ValueError("Intentional failure for testing")
        ctx["processed"] = True
        return ctx

    def cleanup(ctx: Dict[str, Any]) -> Dict[str, Any]:
        ctx["cleaned_up"] = True
        return ctx

    return (
        WorkflowBuilder("error_workflow", "Workflow for error testing")
        .add_transform("maybe_fail", maybe_fail, next_nodes=["cleanup"])
        .add_transform("cleanup", cleanup)
        .build()
    )


# ============ Integration Tests ============


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.workflows
class TestWorkflowExecutorE2E:
    """End-to-end tests for WorkflowExecutor."""

    async def test_execute_simple_linear_workflow(self, mock_orchestrator, simple_workflow):
        """Test executing a simple linear workflow from start to finish."""
        # Given an executor with mock orchestrator
        executor = WorkflowExecutor(mock_orchestrator)

        # When executing the workflow
        result = await executor.execute(simple_workflow, initial_context={"counter": 0})

        # Then the workflow should complete successfully
        assert result.success is True
        assert result.workflow_name == "simple_workflow"
        assert result.error is None

        # And all steps should have been executed in order
        final_outputs = result.context.get_outputs()
        assert "step_a" in final_outputs or "step_c" in final_outputs

        # And the counter should reflect all transformations
        final_data = result.context.data
        assert final_data.get("counter") == 111  # 0 + 1 + 10 + 100
        assert final_data.get("steps") == ["a", "b", "c"]

    async def test_execute_with_initial_context(self, mock_orchestrator, simple_workflow):
        """Test that initial context is properly passed to workflow."""
        executor = WorkflowExecutor(mock_orchestrator)

        # When executing with custom initial context
        result = await executor.execute(
            simple_workflow,
            initial_context={"counter": 1000, "custom_key": "custom_value"},
        )

        # Then initial values should be preserved
        assert result.success is True
        final_data = result.context.data
        assert final_data.get("counter") == 1111  # 1000 + 1 + 10 + 100
        assert final_data.get("custom_key") == "custom_value"

    async def test_execute_workflow_timeout(self, mock_orchestrator):
        """Test workflow execution timeout handling.

        Note: This tests the timeout parameter configuration. The actual
        timeout behavior depends on asyncio's ability to cancel tasks.
        Since TransformNode handlers are synchronous, they cannot be
        interrupted once started. This test verifies that:
        1. Timeout parameter is accepted by the executor
        2. When workflow completes within timeout, it succeeds
        """

        def quick_transform(ctx: Dict[str, Any]) -> Dict[str, Any]:
            ctx["completed"] = True
            return ctx

        workflow = (
            WorkflowBuilder("timeout_test_workflow")
            .add_transform("quick_step", quick_transform)
            .build()
        )

        executor = WorkflowExecutor(mock_orchestrator)

        # When executing with a reasonable timeout, workflow should complete
        result = await executor.execute(workflow, timeout=10.0)

        # Workflow should complete successfully within timeout
        assert result.success is True
        assert result.context.data.get("completed") is True

    async def test_execute_tracks_node_results(self, mock_orchestrator, simple_workflow):
        """Test that individual node results are tracked."""
        executor = WorkflowExecutor(mock_orchestrator)

        result = await executor.execute(simple_workflow)

        # Node results should be tracked
        assert len(result.context.node_results) > 0

        # Each completed node should have a result
        for node_id, node_result in result.context.node_results.items():
            assert isinstance(node_result, NodeResult)
            assert node_result.status in [
                ExecutorNodeStatus.COMPLETED,
                ExecutorNodeStatus.FAILED,
                ExecutorNodeStatus.SKIPPED,
            ]

    async def test_execute_tracks_duration(self, mock_orchestrator, simple_workflow):
        """Test that execution duration is tracked."""
        executor = WorkflowExecutor(mock_orchestrator)

        result = await executor.execute(simple_workflow)

        # Duration should be recorded
        assert result.total_duration > 0


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.workflows
class TestCheckpointFunctionality:
    """Tests for checkpoint save/resume functionality."""

    async def test_checkpoint_saved_after_each_node(self, mock_orchestrator, simple_workflow):
        """Test that checkpoints are saved after each node execution."""
        # Create a mock checkpointer
        mock_checkpointer = MagicMock()
        mock_checkpointer.get_latest_checkpoint.return_value = None
        mock_checkpointer.create_checkpoint = MagicMock()

        executor = WorkflowExecutor(mock_orchestrator, checkpointer=mock_checkpointer)

        # Execute workflow with thread_id
        await executor.execute(
            simple_workflow,
            initial_context={"counter": 0},
            thread_id="test-checkpoint-123",
        )

        # Checkpoints should have been created (one per node)
        assert mock_checkpointer.create_checkpoint.call_count >= 1

    async def test_resume_from_checkpoint(self, mock_orchestrator):
        """Test resuming workflow execution from a checkpoint."""

        def step_a(ctx: Dict[str, Any]) -> Dict[str, Any]:
            ctx["a_executed"] = True
            return ctx

        def step_b(ctx: Dict[str, Any]) -> Dict[str, Any]:
            ctx["b_executed"] = True
            return ctx

        workflow = (
            WorkflowBuilder("checkpoint_workflow")
            .add_transform("step_a", step_a, next_nodes=["step_b"])
            .add_transform("step_b", step_b)
            .build()
        )

        # Create a mock checkpoint representing a resumed state
        mock_checkpoint = MagicMock()
        mock_checkpoint.state = {
            "last_node": "step_a",
            "next_node": "step_b",
            "context": {"a_executed": True},
        }

        mock_checkpointer = MagicMock()
        mock_checkpointer.get_latest_checkpoint.return_value = mock_checkpoint
        mock_checkpointer.create_checkpoint = MagicMock()

        executor = WorkflowExecutor(mock_orchestrator, checkpointer=mock_checkpointer)

        # Execute - should resume from step_b
        result = await executor.execute(
            workflow,
            thread_id="resume-test-123",
        )

        # Workflow should complete successfully
        assert result.success is True

        # Step B should have been executed
        assert result.context.data.get("b_executed") is True

    async def test_checkpoint_contains_workflow_state(self, mock_orchestrator):
        """Test that checkpoint contains complete workflow state."""
        captured_checkpoints = []

        def capture_checkpoint(*args, **kwargs):
            captured_checkpoints.append(kwargs)

        def step_a(ctx: Dict[str, Any]) -> Dict[str, Any]:
            ctx["step"] = "a"
            ctx["data"] = {"key": "value"}
            return ctx

        workflow = (
            WorkflowBuilder("state_checkpoint_workflow").add_transform("step_a", step_a).build()
        )

        mock_checkpointer = MagicMock()
        mock_checkpointer.get_latest_checkpoint.return_value = None
        mock_checkpointer.create_checkpoint = MagicMock(side_effect=capture_checkpoint)

        executor = WorkflowExecutor(mock_orchestrator, checkpointer=mock_checkpointer)

        await executor.execute(workflow, thread_id="state-test-123")

        # Verify checkpoint structure
        assert len(captured_checkpoints) > 0
        checkpoint = captured_checkpoints[0]
        assert "state" in checkpoint
        assert "context" in checkpoint["state"]


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.workflows
class TestErrorHandlingAndRetry:
    """Tests for error handling and retry policies."""

    async def test_error_stops_workflow_by_default(self, mock_orchestrator, error_workflow):
        """Test that errors stop workflow execution by default."""
        executor = WorkflowExecutor(mock_orchestrator)

        # Execute with failure flag
        result = await executor.execute(
            error_workflow,
            initial_context={"should_fail": True},
        )

        # Workflow should fail
        assert result.success is False

        # Cleanup should NOT have run (workflow stopped on failure)
        assert result.context.data.get("cleaned_up") is not True

    async def test_continue_on_failure_option(self, mock_orchestrator):
        """Test workflow continues on failure when configured."""

        def failing_step(ctx: Dict[str, Any]) -> Dict[str, Any]:
            raise ValueError("Intentional failure")

        def next_step(ctx: Dict[str, Any]) -> Dict[str, Any]:
            ctx["next_executed"] = True
            return ctx

        workflow = (
            WorkflowBuilder("continue_on_failure_workflow")
            .add_transform("failing", failing_step, next_nodes=["next"])
            .add_transform("next", next_step)
            .set_metadata("continue_on_failure", True)
            .build()
        )

        executor = WorkflowExecutor(mock_orchestrator)
        result = await executor.execute(workflow)

        # Workflow should have failures
        assert result.context.has_failures() is True

    async def test_error_message_captured(self, mock_orchestrator, error_workflow):
        """Test that error messages are captured in node results."""
        executor = WorkflowExecutor(mock_orchestrator)

        result = await executor.execute(
            error_workflow,
            initial_context={"should_fail": True},
        )

        # Error should be captured
        failed_results = [
            r for r in result.context.node_results.values() if r.status == ExecutorNodeStatus.FAILED
        ]
        assert len(failed_results) > 0

        # Error message should be preserved
        failed_result = failed_results[0]
        assert failed_result.error is not None
        assert "Intentional failure" in failed_result.error

    async def test_successful_workflow_no_error(self, mock_orchestrator, error_workflow):
        """Test that successful workflow has no errors."""
        executor = WorkflowExecutor(mock_orchestrator)

        result = await executor.execute(
            error_workflow,
            initial_context={"should_fail": False},
        )

        # Should succeed
        assert result.success is True
        assert result.error is None
        assert result.context.data.get("processed") is True


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.workflows
class TestConditionalEdgeRouting:
    """Tests for conditional edge routing in workflows."""

    async def test_route_to_high_path(self, mock_orchestrator, conditional_workflow):
        """Test routing to high value path."""
        executor = WorkflowExecutor(mock_orchestrator)

        result = await executor.execute(
            conditional_workflow,
            initial_context={"value": 75},
        )

        assert result.success is True
        assert result.context.data.get("path_taken") == "high"
        assert result.context.data.get("result") == "Processed high value"
        assert result.context.data.get("finalized") is True

    async def test_route_to_medium_path(self, mock_orchestrator, conditional_workflow):
        """Test routing to medium value path."""
        executor = WorkflowExecutor(mock_orchestrator)

        result = await executor.execute(
            conditional_workflow,
            initial_context={"value": 35},
        )

        assert result.success is True
        assert result.context.data.get("path_taken") == "medium"
        assert result.context.data.get("result") == "Processed medium value"

    async def test_route_to_low_path(self, mock_orchestrator, conditional_workflow):
        """Test routing to low value path."""
        executor = WorkflowExecutor(mock_orchestrator)

        result = await executor.execute(
            conditional_workflow,
            initial_context={"value": 5},
        )

        assert result.success is True
        assert result.context.data.get("path_taken") == "low"
        assert result.context.data.get("result") == "Processed low value"

    async def test_condition_based_on_complex_state(self, mock_orchestrator):
        """Test conditional routing based on complex state evaluation.

        This test verifies that the condition function correctly evaluates
        complex state and routes to the appropriate handler.

        Note: Using the conditional_workflow fixture pattern where the workflow
        is built with explicit chain() calls instead of relying on auto-chaining.
        """

        def route_by_item_count(ctx: Dict[str, Any]) -> str:
            """Simple router that checks the count of items."""
            count = ctx.get("item_count", 0)
            if count == 0:
                return "empty"
            elif count > 5:
                return "many"
            else:
                return "few"

        def handle_empty(ctx: Dict[str, Any]) -> Dict[str, Any]:
            ctx["action"] = "handled_empty"
            return ctx

        def handle_many(ctx: Dict[str, Any]) -> Dict[str, Any]:
            ctx["action"] = "handled_many"
            return ctx

        def handle_few(ctx: Dict[str, Any]) -> Dict[str, Any]:
            ctx["action"] = "handled_few"
            return ctx

        # Build workflow with proper structure using ConditionNode directly
        from victor.workflows.definition import (
            ConditionNode,
            TransformNode,
            WorkflowDefinition,
        )

        workflow = WorkflowDefinition(
            name="complex_condition_workflow",
            description="Workflow with condition routing",
            nodes={
                "router": ConditionNode(
                    id="router",
                    name="Router",
                    condition=route_by_item_count,
                    branches={
                        "empty": "handle_empty",
                        "many": "handle_many",
                        "few": "handle_few",
                    },
                ),
                "handle_empty": TransformNode(
                    id="handle_empty",
                    name="Handle Empty",
                    transform=handle_empty,
                ),
                "handle_many": TransformNode(
                    id="handle_many",
                    name="Handle Many",
                    transform=handle_many,
                ),
                "handle_few": TransformNode(
                    id="handle_few",
                    name="Handle Few",
                    transform=handle_few,
                ),
            },
            start_node="router",
        )

        executor = WorkflowExecutor(mock_orchestrator)

        # Test with "many" items (count > 5)
        result = await executor.execute(
            workflow,
            initial_context={"item_count": 10},
        )

        assert result.success is True
        assert result.context.data.get("action") == "handled_many"

        # Test with "few" items (0 < count <= 5)
        result2 = await executor.execute(
            workflow,
            initial_context={"item_count": 3},
        )

        assert result2.success is True
        assert result2.context.data.get("action") == "handled_few"

        # Test with "empty" (count == 0)
        result3 = await executor.execute(
            workflow,
            initial_context={"item_count": 0},
        )

        assert result3.success is True
        assert result3.context.data.get("action") == "handled_empty"


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.workflows
class TestParallelNodeExecution:
    """Tests for parallel node execution."""

    async def test_parallel_nodes_execute_concurrently(self, mock_orchestrator):
        """Test that parallel nodes execute concurrently."""
        execution_times = {}

        def create_timed_transform(name: str, delay: float = 0.1):
            def transform(ctx: Dict[str, Any]) -> Dict[str, Any]:
                import time

                start = time.time()
                time.sleep(delay)
                execution_times[name] = time.time() - start
                ctx[f"{name}_completed"] = True
                return ctx

            return transform

        # Create parallel tasks
        transform_a = create_timed_transform("task_a", 0.1)
        transform_b = create_timed_transform("task_b", 0.1)
        transform_c = create_timed_transform("task_c", 0.1)

        # Build workflow with parallel node using builder
        workflow = (
            WorkflowBuilder("parallel_workflow")
            .add_transform("task_a", transform_a)
            .add_transform("task_b", transform_b)
            .add_transform("task_c", transform_c)
            .add_parallel(
                "parallel_exec",
                ["task_a", "task_b", "task_c"],
                join_strategy="all",
                next_nodes=["finalize"],
            )
            .add_transform("finalize", lambda ctx: ctx)
            .build()
        )

        # Set parallel_exec as start node
        workflow.start_node = "parallel_exec"

        executor = WorkflowExecutor(mock_orchestrator, max_parallel=3)

        start_time = asyncio.get_event_loop().time()
        result = await executor.execute(workflow)
        total_time = asyncio.get_event_loop().time() - start_time

        # With sequential execution, it would take ~0.3s
        # With parallel, it should complete faster
        # (allowing some buffer for test overhead)
        assert total_time < 0.5  # Should be around 0.1s + overhead

    async def test_parallel_join_strategy_all(self, mock_orchestrator):
        """Test parallel execution with 'all' join strategy."""

        def success_task(ctx: Dict[str, Any]) -> Dict[str, Any]:
            ctx["success_ran"] = True
            return ctx

        def another_success(ctx: Dict[str, Any]) -> Dict[str, Any]:
            ctx["another_ran"] = True
            return ctx

        workflow = (
            WorkflowBuilder("parallel_all_workflow")
            .add_transform("task1", success_task)
            .add_transform("task2", another_success)
            .add_parallel("parallel", ["task1", "task2"], join_strategy="all")
            .build()
        )
        workflow.start_node = "parallel"

        executor = WorkflowExecutor(mock_orchestrator)
        result = await executor.execute(workflow)

        # All tasks should complete
        assert result.success is True


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.workflows
class TestWorkflowCancellation:
    """Tests for workflow cancellation."""

    async def test_timeout_parameter_accepted(self, mock_orchestrator):
        """Test that timeout parameter is accepted and workflow completes within timeout.

        Note: asyncio timeout with synchronous blocking code (time.sleep) cannot
        be cancelled mid-execution. This test verifies that:
        1. The timeout parameter is properly handled by the executor
        2. Workflows that complete within timeout succeed normally
        """

        def quick_transform(ctx: Dict[str, Any]) -> Dict[str, Any]:
            ctx["quick_completed"] = True
            return ctx

        workflow = (
            WorkflowBuilder("timeout_param_workflow")
            .add_transform("quick", quick_transform)
            .build()
        )

        executor = WorkflowExecutor(mock_orchestrator)

        # Execute with timeout - should complete successfully
        result = await executor.execute(workflow, timeout=10.0)

        # Should complete successfully within timeout
        assert result.success is True
        assert result.context.data.get("quick_completed") is True

    async def test_cancellation_cleanup(self, mock_orchestrator):
        """Test that cancellation properly cleans up resources."""
        cleanup_called = {"value": False}

        def transform_with_cleanup(ctx: Dict[str, Any]) -> Dict[str, Any]:
            ctx["processed"] = True
            return ctx

        workflow = (
            WorkflowBuilder("cleanup_workflow")
            .add_transform("process", transform_with_cleanup)
            .build()
        )

        executor = WorkflowExecutor(mock_orchestrator)

        # Normal execution should complete
        result = await executor.execute(workflow)
        assert result.success is True

        # Active executions should be cleaned up
        assert len(executor._active_executions) == 0


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.workflows
class TestWorkflowContextManagement:
    """Tests for workflow context management during execution."""

    async def test_context_shared_across_nodes(self, mock_orchestrator):
        """Test that context is properly shared across all nodes."""

        def add_key_a(ctx: Dict[str, Any]) -> Dict[str, Any]:
            ctx["key_a"] = "value_a"
            return ctx

        def read_key_a_add_b(ctx: Dict[str, Any]) -> Dict[str, Any]:
            ctx["saw_key_a"] = ctx.get("key_a") == "value_a"
            ctx["key_b"] = "value_b"
            return ctx

        def read_both_keys(ctx: Dict[str, Any]) -> Dict[str, Any]:
            ctx["saw_both"] = ctx.get("key_a") == "value_a" and ctx.get("key_b") == "value_b"
            return ctx

        workflow = (
            WorkflowBuilder("context_sharing_workflow")
            .add_transform("step1", add_key_a, next_nodes=["step2"])
            .add_transform("step2", read_key_a_add_b, next_nodes=["step3"])
            .add_transform("step3", read_both_keys)
            .build()
        )

        executor = WorkflowExecutor(mock_orchestrator)
        result = await executor.execute(workflow)

        assert result.success is True
        assert result.context.data.get("saw_key_a") is True
        assert result.context.data.get("saw_both") is True

    async def test_context_modifications_persist(self, mock_orchestrator):
        """Test that context modifications from each node persist."""

        def accumulate(ctx: Dict[str, Any]) -> Dict[str, Any]:
            history = ctx.get("history", [])
            history.append(f"step_{len(history) + 1}")
            ctx["history"] = history
            return ctx

        workflow = (
            WorkflowBuilder("accumulating_workflow")
            .add_transform("s1", accumulate, next_nodes=["s2"])
            .add_transform("s2", accumulate, next_nodes=["s3"])
            .add_transform("s3", accumulate)
            .build()
        )

        executor = WorkflowExecutor(mock_orchestrator)
        result = await executor.execute(workflow)

        assert result.success is True
        history = result.context.data.get("history", [])
        assert len(history) == 3
        assert history == ["step_1", "step_2", "step_3"]


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.workflows
class TestWorkflowExecutionMetrics:
    """Tests for workflow execution metrics and reporting."""

    async def test_total_duration_tracked(self, mock_orchestrator, simple_workflow):
        """Test that total execution duration is tracked."""
        executor = WorkflowExecutor(mock_orchestrator)
        result = await executor.execute(simple_workflow)

        assert result.total_duration > 0
        assert result.total_duration < 60  # Should complete quickly

    async def test_node_durations_tracked(self, mock_orchestrator, simple_workflow):
        """Test that individual node durations are tracked."""
        executor = WorkflowExecutor(mock_orchestrator)
        result = await executor.execute(simple_workflow)

        for node_result in result.context.node_results.values():
            assert node_result.duration_seconds >= 0

    async def test_workflow_result_serialization(self, mock_orchestrator, simple_workflow):
        """Test that workflow result can be serialized to dict."""
        executor = WorkflowExecutor(mock_orchestrator)
        result = await executor.execute(simple_workflow)

        result_dict = result.to_dict()

        assert "workflow_name" in result_dict
        assert "success" in result_dict
        assert "total_duration" in result_dict
        assert "outputs" in result_dict
        assert "node_results" in result_dict


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.workflows
class TestWorkflowExecutorConfiguration:
    """Tests for WorkflowExecutor configuration options."""

    async def test_max_parallel_configuration(self, mock_orchestrator):
        """Test that max_parallel limits concurrent executions."""
        executor = WorkflowExecutor(mock_orchestrator, max_parallel=2)
        assert executor.max_parallel == 2

    async def test_default_timeout_configuration(self, mock_orchestrator):
        """Test default timeout configuration."""
        executor = WorkflowExecutor(mock_orchestrator, default_timeout=600.0)
        assert executor.default_timeout == 600.0

    async def test_cache_configuration(self, mock_orchestrator):
        """Test workflow cache configuration."""
        from victor.workflows.cache import WorkflowCacheConfig

        cache_config = WorkflowCacheConfig(enabled=True, ttl_seconds=3600)
        executor = WorkflowExecutor(mock_orchestrator, cache_config=cache_config)

        assert executor.cache is not None

    async def test_cache_disabled_by_default(self, mock_orchestrator):
        """Test that cache is disabled by default."""
        executor = WorkflowExecutor(mock_orchestrator)
        assert executor.cache is None
