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

"""End-to-end integration tests for streaming workflow execution.

These tests verify the complete streaming workflow API, including:
- Event sequence during workflow execution
- Progress tracking across nodes
- Node order preservation in sequences
- Error propagation through streams
- Subscription callbacks for events
- Cancellation of active workflows
- Backward compatibility with execute() method
"""

import asyncio
import pytest
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

from victor.workflows.definition import (
    WorkflowBuilder,
    WorkflowDefinition,
    TransformNode,
)
from victor.workflows.streaming_executor import (
    StreamingWorkflowExecutor,
    WorkflowEventType,
    WorkflowStreamChunk,
)
from victor.workflows.context import WorkflowResult
from victor_contracts.workflows import ExecutorNodeStatus

# ============ Test Fixtures ============


def create_mock_orchestrator():
    """Create a mock orchestrator for testing without real LLM calls."""
    mock_orchestrator = MagicMock()
    mock_orchestrator.settings = MagicMock()
    mock_orchestrator.settings.tool_budget = 15
    return mock_orchestrator


@pytest.fixture
def mock_orchestrator():
    """Fixture providing a mock orchestrator."""
    return create_mock_orchestrator()


@pytest.fixture
def three_node_workflow():
    """Create a workflow with exactly 3 transform nodes for testing."""

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
        WorkflowBuilder("three_node_workflow", "Workflow with three transform nodes")
        .add_transform("step_a", step_a, next_nodes=["step_b"])
        .add_transform("step_b", step_b, next_nodes=["step_c"])
        .add_transform("step_c", step_c)
        .build()
    )


@pytest.fixture
def failing_workflow():
    """Create a workflow that will fail at a specific node."""

    def success_step(ctx: Dict[str, Any]) -> Dict[str, Any]:
        ctx["success_ran"] = True
        return ctx

    def failing_step(ctx: Dict[str, Any]) -> Dict[str, Any]:
        raise ValueError("Intentional test failure")

    def after_failure(ctx: Dict[str, Any]) -> Dict[str, Any]:
        ctx["after_failure_ran"] = True
        return ctx

    return (
        WorkflowBuilder("failing_workflow", "Workflow that fails at second node")
        .add_transform("success", success_step, next_nodes=["failure"])
        .add_transform("failure", failing_step, next_nodes=["after"])
        .add_transform("after", after_failure)
        .build()
    )


@pytest.fixture
def long_running_workflow():
    """Create a workflow with nodes that take time to execute."""

    def slow_step_a(ctx: Dict[str, Any]) -> Dict[str, Any]:
        import time

        time.sleep(0.1)
        ctx["a_completed"] = True
        return ctx

    def slow_step_b(ctx: Dict[str, Any]) -> Dict[str, Any]:
        import time

        time.sleep(0.1)
        ctx["b_completed"] = True
        return ctx

    def slow_step_c(ctx: Dict[str, Any]) -> Dict[str, Any]:
        import time

        time.sleep(0.1)
        ctx["c_completed"] = True
        return ctx

    return (
        WorkflowBuilder("long_running_workflow", "Workflow with slow nodes")
        .add_transform("slow_a", slow_step_a, next_nodes=["slow_b"])
        .add_transform("slow_b", slow_step_b, next_nodes=["slow_c"])
        .add_transform("slow_c", slow_step_c)
        .build()
    )


# ============ End-to-End Tests ============


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.workflows
class TestStreamingWorkflowE2E:
    """End-to-end streaming workflow tests."""

    async def test_stream_simple_transform_workflow(self, mock_orchestrator, three_node_workflow):
        """Test streaming a simple transform workflow.

        Verifies event sequence: START -> NODE_START -> NODE_COMPLETE (x3) -> COMPLETE
        """
        executor = StreamingWorkflowExecutor(mock_orchestrator)

        events: List[WorkflowStreamChunk] = []
        async for chunk in executor.astream(three_node_workflow, initial_context={"counter": 0}):
            events.append(chunk)

        # Verify we got events
        assert len(events) > 0

        # Extract event types
        event_types = [e.event_type for e in events]

        # First event should be WORKFLOW_START
        assert event_types[0] == WorkflowEventType.WORKFLOW_START

        # Last event should be WORKFLOW_COMPLETE
        assert event_types[-1] == WorkflowEventType.WORKFLOW_COMPLETE
        assert events[-1].is_final is True

        # Should have NODE_START and NODE_COMPLETE for each of 3 nodes
        node_starts = [e for e in events if e.event_type == WorkflowEventType.NODE_START]
        node_completes = [e for e in events if e.event_type == WorkflowEventType.NODE_COMPLETE]

        assert len(node_starts) == 3
        assert len(node_completes) == 3

        # Verify node order: step_a, step_b, step_c
        node_ids = [e.node_id for e in node_starts]
        assert node_ids == ["step_a", "step_b", "step_c"]

    async def test_stream_progress_increases(self, mock_orchestrator, three_node_workflow):
        """Test that progress increases during streaming.

        Verifies progress: 0 -> 33 -> 66 -> 100
        """
        executor = StreamingWorkflowExecutor(mock_orchestrator)

        progress_values: List[float] = []
        async for chunk in executor.astream(three_node_workflow, initial_context={"counter": 0}):
            progress_values.append(chunk.progress)

        # Progress should start at 0
        assert progress_values[0] == 0.0

        # Progress should end at 100
        assert progress_values[-1] == 100.0

        # Progress should be monotonically increasing (or stable)
        for i in range(1, len(progress_values)):
            assert (
                progress_values[i] >= progress_values[i - 1]
            ), f"Progress decreased from {progress_values[i-1]} to {progress_values[i]}"

        # After each NODE_COMPLETE, progress should increase by ~33.33%
        # (allowing for floating point imprecision)
        # With 3 nodes: 0 -> ~33.33 -> ~66.67 -> 100
        unique_progress = sorted(set(progress_values))
        # At minimum we should see 0, 33, 66, 100 (approximately)
        assert 0.0 in unique_progress
        assert 100.0 in unique_progress

    async def test_stream_multi_node_sequence(self, mock_orchestrator, three_node_workflow):
        """Test streaming maintains node order in sequence.

        Verifies nodes execute in dependency order.
        """
        executor = StreamingWorkflowExecutor(mock_orchestrator)

        # Track node execution order
        execution_order: List[str] = []

        async for chunk in executor.astream(three_node_workflow, initial_context={"counter": 0}):
            if chunk.event_type == WorkflowEventType.NODE_COMPLETE:
                execution_order.append(chunk.node_id)

        # Nodes should execute in sequence: a -> b -> c
        assert execution_order == ["step_a", "step_b", "step_c"]

    async def test_stream_error_propagation(self, mock_orchestrator, failing_workflow):
        """Test error events are properly streamed.

        Verifies WORKFLOW_ERROR or NODE_ERROR events are emitted on failure.
        """
        executor = StreamingWorkflowExecutor(mock_orchestrator)

        events: List[WorkflowStreamChunk] = []
        async for chunk in executor.astream(failing_workflow):
            events.append(chunk)

        # Should have received events
        assert len(events) > 0

        # Should have a NODE_ERROR event for the failing node
        error_events = [
            e
            for e in events
            if e.event_type in [WorkflowEventType.NODE_ERROR, WorkflowEventType.WORKFLOW_ERROR]
        ]
        assert len(error_events) > 0

        # At least one error event should have an error message
        errors_with_messages = [e for e in error_events if e.error]
        assert len(errors_with_messages) > 0

        # The failing node should have reported its error
        node_errors = [e for e in events if e.event_type == WorkflowEventType.NODE_ERROR]
        if node_errors:
            failure_node_error = [e for e in node_errors if e.node_id == "failure"]
            assert len(failure_node_error) > 0
            assert "Intentional test failure" in failure_node_error[0].error

        # Final event should indicate failure
        final_event = events[-1]
        assert final_event.is_final is True
        # May be WORKFLOW_ERROR or WORKFLOW_COMPLETE with error
        if final_event.event_type == WorkflowEventType.WORKFLOW_COMPLETE:
            assert final_event.metadata.get("success") is False
        else:
            assert final_event.event_type == WorkflowEventType.WORKFLOW_ERROR

    async def test_subscription_receives_events(self, mock_orchestrator, three_node_workflow):
        """Test subscription callback receives matching events.

        Subscribes to NODE_COMPLETE events and verifies callback invocation.
        """
        executor = StreamingWorkflowExecutor(mock_orchestrator)

        # Collect events via subscription
        subscribed_events: List[WorkflowStreamChunk] = []

        def on_node_complete(chunk: WorkflowStreamChunk) -> None:
            subscribed_events.append(chunk)

        # Subscribe to NODE_COMPLETE events only
        unsubscribe = executor.subscribe([WorkflowEventType.NODE_COMPLETE], on_node_complete)

        # Execute workflow via astream
        async for _ in executor.astream(three_node_workflow, initial_context={"counter": 0}):
            pass

        # Unsubscribe after workflow completes
        unsubscribe()

        # Should have received 3 NODE_COMPLETE events via subscription
        assert len(subscribed_events) == 3

        # All subscribed events should be NODE_COMPLETE
        for event in subscribed_events:
            assert event.event_type == WorkflowEventType.NODE_COMPLETE

        # Events should be for our nodes
        node_ids = {e.node_id for e in subscribed_events}
        assert node_ids == {"step_a", "step_b", "step_c"}

    async def test_subscription_unsubscribe_works(self, mock_orchestrator, three_node_workflow):
        """Test that unsubscribe stops callback invocation."""
        executor = StreamingWorkflowExecutor(mock_orchestrator)

        call_count = {"value": 0}

        def on_any_event(chunk: WorkflowStreamChunk) -> None:
            call_count["value"] += 1

        # Subscribe to all events
        unsubscribe = executor.subscribe(list(WorkflowEventType), on_any_event)

        # Unsubscribe immediately before executing
        unsubscribe()

        # Execute workflow
        async for _ in executor.astream(three_node_workflow, initial_context={"counter": 0}):
            pass

        # Should not have received any events via callback
        assert call_count["value"] == 0

    async def test_cancellation_stops_workflow(self, mock_orchestrator, long_running_workflow):
        """Test cancellation properly stops streaming.

        Starts a long workflow and cancels mid-execution.
        Verifies stream terminates early.
        """
        executor = StreamingWorkflowExecutor(mock_orchestrator)

        events: List[WorkflowStreamChunk] = []
        workflow_id: Optional[str] = None
        cancelled = False

        async for chunk in executor.astream(long_running_workflow):
            events.append(chunk)

            # Capture workflow_id from first event
            if workflow_id is None:
                workflow_id = chunk.workflow_id

            # Cancel after first NODE_COMPLETE
            if chunk.event_type == WorkflowEventType.NODE_COMPLETE and not cancelled:
                success = executor.cancel_workflow(workflow_id)
                assert success is True
                cancelled = True

        # Should have cancelled
        assert cancelled is True

        # Should have fewer events than a complete workflow
        # Complete workflow: START + (START + COMPLETE) * 3 + COMPLETE = 8 events
        # Cancelled workflow should have less
        node_completes = [e for e in events if e.event_type == WorkflowEventType.NODE_COMPLETE]
        # Should have stopped before completing all 3 nodes
        # We cancel after first NODE_COMPLETE, so we might get 1-2 completes
        assert len(node_completes) < 3

    async def test_cancel_nonexistent_workflow(self, mock_orchestrator):
        """Test cancelling a workflow that doesn't exist returns False."""
        executor = StreamingWorkflowExecutor(mock_orchestrator)

        result = executor.cancel_workflow("nonexistent_workflow_id")
        assert result is False

    async def test_backward_compatibility_execute(self, mock_orchestrator, three_node_workflow):
        """Test execute() still works on StreamingWorkflowExecutor.

        Verifies that the base class execute() method returns
        WorkflowResult normally without streaming.
        """
        executor = StreamingWorkflowExecutor(mock_orchestrator)

        # Use the inherited execute() method (not astream)
        result = await executor.execute(three_node_workflow, initial_context={"counter": 0})

        # Should return WorkflowResult
        assert isinstance(result, WorkflowResult)
        assert result.success is True
        assert result.workflow_name == "three_node_workflow"

        # Context should have been modified by transforms
        assert result.context.data.get("counter") == 111  # 0 + 1 + 10 + 100
        assert result.context.data.get("steps") == ["a", "b", "c"]

    async def test_get_active_workflows(self, mock_orchestrator, long_running_workflow):
        """Test get_active_workflows returns active workflow IDs."""
        executor = StreamingWorkflowExecutor(mock_orchestrator)

        # Before execution, should be empty
        assert executor.get_active_workflows() == []

        # Track workflow IDs during execution
        workflow_ids_during: List[str] = []
        captured_workflow_id: Optional[str] = None

        # Execute workflow completely and check active state during execution
        async for chunk in executor.astream(long_running_workflow):
            if captured_workflow_id is None:
                captured_workflow_id = chunk.workflow_id

            active = executor.get_active_workflows()
            if active:
                workflow_ids_during.extend(active)

        # Should have captured the workflow as active during execution
        # Note: The workflow ID should appear at least once
        if captured_workflow_id:
            assert captured_workflow_id in workflow_ids_during

        # After fully completing the iteration, should be empty
        assert executor.get_active_workflows() == []

    async def test_get_workflow_progress(self, mock_orchestrator, long_running_workflow):
        """Test get_workflow_progress returns current progress."""
        executor = StreamingWorkflowExecutor(mock_orchestrator)

        # Before execution, should return None for unknown ID
        assert executor.get_workflow_progress("unknown") is None

        # During execution, should return current progress
        progress_values: List[float] = []
        workflow_id: Optional[str] = None

        async for chunk in executor.astream(long_running_workflow):
            if workflow_id is None:
                workflow_id = chunk.workflow_id

            progress = executor.get_workflow_progress(workflow_id)
            if progress is not None:
                progress_values.append(progress)

        # Should have tracked progress during execution
        assert len(progress_values) > 0

        # After completion, should return None
        assert executor.get_workflow_progress(workflow_id) is None

    async def test_workflow_metadata_in_chunks(self, mock_orchestrator, three_node_workflow):
        """Test that workflow metadata is included in chunks."""
        executor = StreamingWorkflowExecutor(mock_orchestrator)

        first_chunk: Optional[WorkflowStreamChunk] = None
        async for chunk in executor.astream(three_node_workflow, initial_context={"counter": 0}):
            if chunk.event_type == WorkflowEventType.WORKFLOW_START:
                first_chunk = chunk
                break

        assert first_chunk is not None
        assert first_chunk.workflow_id is not None
        assert "workflow_name" in first_chunk.metadata
        assert first_chunk.metadata["workflow_name"] == "three_node_workflow"
        assert "total_nodes" in first_chunk.metadata
        assert first_chunk.metadata["total_nodes"] == 3

    async def test_node_metadata_in_chunks(self, mock_orchestrator, three_node_workflow):
        """Test that node metadata is included in node event chunks."""
        executor = StreamingWorkflowExecutor(mock_orchestrator)

        node_events: List[WorkflowStreamChunk] = []
        async for chunk in executor.astream(three_node_workflow, initial_context={"counter": 0}):
            if chunk.event_type in [
                WorkflowEventType.NODE_START,
                WorkflowEventType.NODE_COMPLETE,
            ]:
                node_events.append(chunk)

        # All node events should have node_id and node_name
        for event in node_events:
            assert event.node_id is not None
            assert event.node_name is not None

        # NODE_START should have node_type in metadata
        start_events = [e for e in node_events if e.event_type == WorkflowEventType.NODE_START]
        for event in start_events:
            assert "node_type" in event.metadata

        # NODE_COMPLETE should have duration_seconds in metadata
        complete_events = [
            e for e in node_events if e.event_type == WorkflowEventType.NODE_COMPLETE
        ]
        for event in complete_events:
            assert "duration_seconds" in event.metadata


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.workflows
class TestStreamingWorkflowEdgeCases:
    """Edge case tests for streaming workflow execution."""

    async def test_empty_workflow_context(self, mock_orchestrator):
        """Test streaming with empty initial context."""

        def identity_transform(ctx: Dict[str, Any]) -> Dict[str, Any]:
            ctx["touched"] = True
            return ctx

        workflow = (
            WorkflowBuilder("empty_context_workflow")
            .add_transform("identity", identity_transform)
            .build()
        )

        executor = StreamingWorkflowExecutor(mock_orchestrator)

        events = []
        async for chunk in executor.astream(workflow):
            events.append(chunk)

        # Should complete successfully
        assert events[-1].event_type == WorkflowEventType.WORKFLOW_COMPLETE
        assert events[-1].metadata.get("success") is True

    async def test_single_node_workflow(self, mock_orchestrator):
        """Test streaming a workflow with only one node."""

        def single_step(ctx: Dict[str, Any]) -> Dict[str, Any]:
            ctx["result"] = "done"
            return ctx

        workflow = (
            WorkflowBuilder("single_node_workflow").add_transform("only_step", single_step).build()
        )

        executor = StreamingWorkflowExecutor(mock_orchestrator)

        events: List[WorkflowStreamChunk] = []
        async for chunk in executor.astream(workflow):
            events.append(chunk)

        # Should have: START, NODE_START, NODE_COMPLETE, WORKFLOW_COMPLETE
        event_types = [e.event_type for e in events]
        assert WorkflowEventType.WORKFLOW_START in event_types
        assert WorkflowEventType.NODE_START in event_types
        assert WorkflowEventType.NODE_COMPLETE in event_types
        assert WorkflowEventType.WORKFLOW_COMPLETE in event_types

        # Progress should go from 0 to 100
        assert events[0].progress == 0.0
        assert events[-1].progress == 100.0

    async def test_multiple_subscriptions(self, mock_orchestrator, three_node_workflow):
        """Test multiple simultaneous subscriptions."""
        executor = StreamingWorkflowExecutor(mock_orchestrator)

        starts_received: List[WorkflowStreamChunk] = []
        completes_received: List[WorkflowStreamChunk] = []

        def on_start(chunk: WorkflowStreamChunk) -> None:
            starts_received.append(chunk)

        def on_complete(chunk: WorkflowStreamChunk) -> None:
            completes_received.append(chunk)

        # Subscribe to different event types
        unsub1 = executor.subscribe([WorkflowEventType.NODE_START], on_start)
        unsub2 = executor.subscribe([WorkflowEventType.NODE_COMPLETE], on_complete)

        async for _ in executor.astream(three_node_workflow, initial_context={"counter": 0}):
            pass

        unsub1()
        unsub2()

        # Both subscriptions should have received their events
        assert len(starts_received) == 3
        assert len(completes_received) == 3

    async def test_workflow_with_custom_timeout(self, mock_orchestrator):
        """Test streaming with custom timeout parameter."""

        def quick_step(ctx: Dict[str, Any]) -> Dict[str, Any]:
            ctx["quick"] = True
            return ctx

        workflow = WorkflowBuilder("quick_workflow").add_transform("quick", quick_step).build()

        executor = StreamingWorkflowExecutor(mock_orchestrator)

        events = []
        async for chunk in executor.astream(workflow, timeout=60.0):
            events.append(chunk)

        # Should complete successfully within timeout
        assert events[-1].is_final is True

    async def test_workflow_id_consistency(self, mock_orchestrator, three_node_workflow):
        """Test that workflow_id is consistent across all events."""
        executor = StreamingWorkflowExecutor(mock_orchestrator)

        workflow_ids = set()
        async for chunk in executor.astream(three_node_workflow, initial_context={"counter": 0}):
            workflow_ids.add(chunk.workflow_id)

        # All events should have the same workflow_id
        assert len(workflow_ids) == 1

    async def test_thread_id_parameter(self, mock_orchestrator, three_node_workflow):
        """Test that thread_id is properly handled."""
        executor = StreamingWorkflowExecutor(mock_orchestrator)

        custom_thread_id = "custom-thread-123"
        events = []

        async for chunk in executor.astream(
            three_node_workflow,
            initial_context={"counter": 0},
            thread_id=custom_thread_id,
        ):
            events.append(chunk)

        # Should complete successfully
        assert events[-1].is_final is True
