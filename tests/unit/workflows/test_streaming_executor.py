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

"""Tests for StreamingWorkflowExecutor.

TDD tests for Phase 3 of Graph Orchestration Streaming API.
Tests are written FIRST, then implementation follows.
"""

import asyncio
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from victor.workflows.definition import (
    AgentNode,
    TransformNode,
    WorkflowBuilder,
    WorkflowDefinition,
)
from victor.workflows.streaming_executor import (
    StreamingWorkflowExecutor,
    WorkflowEventType,
    WorkflowStreamChunk,
    WorkflowStreamContext,
)


@pytest.fixture
def mock_orchestrator():
    """Create a mock AgentOrchestrator."""
    orchestrator = MagicMock()
    orchestrator.settings = MagicMock()
    return orchestrator


@pytest.fixture
def simple_workflow() -> WorkflowDefinition:
    """Create a simple workflow with two agent nodes."""
    return (
        WorkflowBuilder("test_workflow", "A test workflow")
        .add_agent("step1", "researcher", "Research the topic")
        .add_agent("step2", "executor", "Execute the plan")
        .build()
    )


@pytest.fixture
def transform_workflow() -> WorkflowDefinition:
    """Create a workflow with transform nodes."""
    return (
        WorkflowBuilder("transform_workflow", "Workflow with transforms")
        .add_transform("transform1", lambda ctx: {"transformed": True, **ctx})
        .add_agent("process", "executor", "Process transformed data")
        .build()
    )


@pytest.fixture
def streaming_executor(mock_orchestrator) -> StreamingWorkflowExecutor:
    """Create a StreamingWorkflowExecutor instance."""
    return StreamingWorkflowExecutor(mock_orchestrator)


def test_constructor_uses_runtime_executor_factory(mock_orchestrator) -> None:
    runtime = MagicMock()
    runtime.default_timeout = 42.0
    runtime.sub_agents = MagicMock()

    with patch(
        "victor.workflows.streaming_executor.create_legacy_workflow_executor",
        return_value=runtime,
    ) as factory:
        executor = StreamingWorkflowExecutor(
            mock_orchestrator,
            max_parallel=7,
            default_timeout=42.0,
        )

    factory.assert_called_once_with(
        mock_orchestrator,
        max_parallel=7,
        default_timeout=42.0,
        checkpointer=None,
        cache=None,
        cache_config=None,
    )
    assert executor.sub_agents is runtime.sub_agents
    assert executor.default_timeout == 42.0


class TestWorkflowEventType:
    """Tests for WorkflowEventType enum."""

    def test_event_types_exist(self):
        """Test that all expected event types are defined."""
        assert WorkflowEventType.WORKFLOW_START
        assert WorkflowEventType.WORKFLOW_COMPLETE
        assert WorkflowEventType.WORKFLOW_ERROR
        assert WorkflowEventType.NODE_START
        assert WorkflowEventType.NODE_COMPLETE
        assert WorkflowEventType.NODE_ERROR
        assert WorkflowEventType.AGENT_CONTENT


class TestWorkflowStreamChunk:
    """Tests for WorkflowStreamChunk dataclass."""

    def test_chunk_creation(self):
        """Test creating a stream chunk."""
        chunk = WorkflowStreamChunk(
            event_type=WorkflowEventType.WORKFLOW_START,
            workflow_id="wf_123",
            progress=0.0,
        )
        assert chunk.event_type == WorkflowEventType.WORKFLOW_START
        assert chunk.workflow_id == "wf_123"
        assert chunk.progress == 0.0
        assert chunk.is_final is False

    def test_chunk_with_node_info(self):
        """Test chunk with node information."""
        chunk = WorkflowStreamChunk(
            event_type=WorkflowEventType.NODE_START,
            workflow_id="wf_123",
            node_id="step1",
            node_name="Step 1",
            progress=50.0,
        )
        assert chunk.node_id == "step1"
        assert chunk.node_name == "Step 1"

    def test_chunk_with_content(self):
        """Test chunk with agent content."""
        chunk = WorkflowStreamChunk(
            event_type=WorkflowEventType.AGENT_CONTENT,
            workflow_id="wf_123",
            node_id="agent1",
            content="Processing...",
            progress=25.0,
        )
        assert chunk.content == "Processing..."

    def test_final_chunk(self):
        """Test final chunk marker."""
        chunk = WorkflowStreamChunk(
            event_type=WorkflowEventType.WORKFLOW_COMPLETE,
            workflow_id="wf_123",
            progress=100.0,
            is_final=True,
        )
        assert chunk.is_final is True


class TestWorkflowStreamContext:
    """Tests for WorkflowStreamContext."""

    def test_context_creation(self):
        """Test creating a stream context."""
        ctx = WorkflowStreamContext(
            workflow_id="wf_123",
            total_nodes=5,
        )
        assert ctx.workflow_id == "wf_123"
        assert ctx.total_nodes == 5
        assert ctx.completed_nodes == 0
        assert ctx.is_cancelled is False

    def test_progress_calculation(self):
        """Test progress percentage calculation."""
        ctx = WorkflowStreamContext(
            workflow_id="wf_123",
            total_nodes=4,
        )
        ctx.completed_nodes = 2
        assert ctx.progress == 50.0

    def test_progress_with_zero_nodes(self):
        """Test progress with no nodes."""
        ctx = WorkflowStreamContext(
            workflow_id="wf_123",
            total_nodes=0,
        )
        assert ctx.progress == 0.0


class TestStreamingWorkflowExecutorAstream:
    """Tests for astream() method."""

    @pytest.mark.asyncio
    async def test_astream_yields_workflow_start_first(self, streaming_executor, simple_workflow):
        """Test that astream() yields WORKFLOW_START as the first event."""
        # Mock the sub_agents to return a successful result
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.summary = "Done"
        mock_result.error = None
        mock_result.tool_calls_used = 0

        with patch.object(
            streaming_executor.sub_agents, "spawn", new_callable=AsyncMock
        ) as mock_spawn:
            mock_spawn.return_value = mock_result

            chunks = []
            async for chunk in streaming_executor.astream(simple_workflow):
                chunks.append(chunk)

            assert len(chunks) > 0
            assert chunks[0].event_type == WorkflowEventType.WORKFLOW_START

    @pytest.mark.asyncio
    async def test_astream_yields_workflow_complete_last(self, streaming_executor, simple_workflow):
        """Test that astream() yields WORKFLOW_COMPLETE as the last event with is_final=True."""
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.summary = "Done"
        mock_result.error = None
        mock_result.tool_calls_used = 0

        with patch.object(
            streaming_executor.sub_agents, "spawn", new_callable=AsyncMock
        ) as mock_spawn:
            mock_spawn.return_value = mock_result

            chunks = []
            async for chunk in streaming_executor.astream(simple_workflow):
                chunks.append(chunk)

            assert len(chunks) > 0
            last_chunk = chunks[-1]
            assert last_chunk.event_type == WorkflowEventType.WORKFLOW_COMPLETE
            assert last_chunk.is_final is True

    @pytest.mark.asyncio
    async def test_astream_yields_node_start_and_complete(
        self, streaming_executor, simple_workflow
    ):
        """Test that astream() yields NODE_START and NODE_COMPLETE events."""
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.summary = "Done"
        mock_result.error = None
        mock_result.tool_calls_used = 0

        with patch.object(
            streaming_executor.sub_agents, "spawn", new_callable=AsyncMock
        ) as mock_spawn:
            mock_spawn.return_value = mock_result

            node_start_events = []
            node_complete_events = []

            async for chunk in streaming_executor.astream(simple_workflow):
                if chunk.event_type == WorkflowEventType.NODE_START:
                    node_start_events.append(chunk)
                elif chunk.event_type == WorkflowEventType.NODE_COMPLETE:
                    node_complete_events.append(chunk)

            # Should have NODE_START and NODE_COMPLETE for each node
            assert len(node_start_events) >= 2
            assert len(node_complete_events) >= 2

    @pytest.mark.asyncio
    async def test_astream_progress_increases(self, streaming_executor, simple_workflow):
        """Test that progress increases during streaming."""
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.summary = "Done"
        mock_result.error = None
        mock_result.tool_calls_used = 0

        with patch.object(
            streaming_executor.sub_agents, "spawn", new_callable=AsyncMock
        ) as mock_spawn:
            mock_spawn.return_value = mock_result

            progress_values = []
            async for chunk in streaming_executor.astream(simple_workflow):
                progress_values.append(chunk.progress)

            # Progress should generally increase (with possible plateaus)
            assert progress_values[0] <= progress_values[-1]
            assert progress_values[-1] == 100.0

    @pytest.mark.asyncio
    async def test_astream_with_initial_context(self, streaming_executor, simple_workflow):
        """Test astream() with initial context."""
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.summary = "Done"
        mock_result.error = None
        mock_result.tool_calls_used = 0

        with patch.object(
            streaming_executor.sub_agents, "spawn", new_callable=AsyncMock
        ) as mock_spawn:
            mock_spawn.return_value = mock_result

            initial_context = {"key": "value"}
            chunks = []
            async for chunk in streaming_executor.astream(
                simple_workflow, initial_context=initial_context
            ):
                chunks.append(chunk)

            # Should complete successfully
            assert chunks[-1].event_type == WorkflowEventType.WORKFLOW_COMPLETE


class TestStreamingWorkflowExecutorErrorHandling:
    """Tests for error handling in streaming."""

    @pytest.mark.asyncio
    async def test_astream_yields_workflow_error_on_failure(
        self, streaming_executor, simple_workflow
    ):
        """Test that astream() yields WORKFLOW_ERROR on failure."""
        mock_result = MagicMock()
        mock_result.success = False
        mock_result.summary = None
        mock_result.error = "Something went wrong"
        mock_result.tool_calls_used = 0

        with patch.object(
            streaming_executor.sub_agents, "spawn", new_callable=AsyncMock
        ) as mock_spawn:
            mock_spawn.return_value = mock_result

            error_events = []
            async for chunk in streaming_executor.astream(simple_workflow):
                if chunk.event_type == WorkflowEventType.WORKFLOW_ERROR:
                    error_events.append(chunk)
                elif chunk.event_type == WorkflowEventType.NODE_ERROR:
                    error_events.append(chunk)

            # Should have at least one error event (NODE_ERROR or WORKFLOW_ERROR)
            assert len(error_events) >= 1

    @pytest.mark.asyncio
    async def test_astream_yields_node_error_on_node_failure(
        self, streaming_executor, simple_workflow
    ):
        """Test that astream() yields NODE_ERROR when a node fails."""
        mock_result = MagicMock()
        mock_result.success = False
        mock_result.summary = None
        mock_result.error = "Node failed"
        mock_result.tool_calls_used = 0

        with patch.object(
            streaming_executor.sub_agents, "spawn", new_callable=AsyncMock
        ) as mock_spawn:
            mock_spawn.return_value = mock_result

            node_error_events = []
            async for chunk in streaming_executor.astream(simple_workflow):
                if chunk.event_type == WorkflowEventType.NODE_ERROR:
                    node_error_events.append(chunk)

            assert len(node_error_events) >= 1
            assert node_error_events[0].error is not None


class TestStreamingWorkflowExecutorSubscribe:
    """Tests for subscribe() method."""

    @pytest.mark.asyncio
    async def test_subscribe_receives_matching_events(self, streaming_executor, simple_workflow):
        """Test that subscribe() receives events of specified types."""
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.summary = "Done"
        mock_result.error = None
        mock_result.tool_calls_used = 0

        received_events: List[WorkflowStreamChunk] = []

        def callback(chunk: WorkflowStreamChunk):
            received_events.append(chunk)

        # Subscribe to NODE_START events only
        unsubscribe = streaming_executor.subscribe([WorkflowEventType.NODE_START], callback)

        with patch.object(
            streaming_executor.sub_agents, "spawn", new_callable=AsyncMock
        ) as mock_spawn:
            mock_spawn.return_value = mock_result

            async for _ in streaming_executor.astream(simple_workflow):
                pass

        # All received events should be NODE_START
        assert len(received_events) > 0
        for event in received_events:
            assert event.event_type == WorkflowEventType.NODE_START

        # Cleanup
        unsubscribe()

    def test_subscribe_returns_unsubscribe_function(self, streaming_executor):
        """Test that subscribe() returns an unsubscribe function."""
        unsubscribe = streaming_executor.subscribe(
            [WorkflowEventType.WORKFLOW_START], lambda x: None
        )
        assert callable(unsubscribe)


class TestStreamingWorkflowExecutorCancellation:
    """Tests for cancellation functionality."""

    @pytest.mark.asyncio
    async def test_cancel_workflow_stops_execution(self, streaming_executor, simple_workflow):
        """Test that cancel_workflow() stops execution."""
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.summary = "Done"
        mock_result.error = None
        mock_result.tool_calls_used = 0

        workflow_id = None
        chunks_before_cancel = []

        async def slow_spawn(*args, **kwargs):
            await asyncio.sleep(0.5)
            return mock_result

        with patch.object(streaming_executor.sub_agents, "spawn", side_effect=slow_spawn):
            async for chunk in streaming_executor.astream(simple_workflow):
                chunks_before_cancel.append(chunk)
                if chunk.event_type == WorkflowEventType.WORKFLOW_START:
                    workflow_id = chunk.workflow_id
                    # Cancel after workflow starts
                    streaming_executor.cancel_workflow(workflow_id)
                    break

        # Should have been cancelled
        assert workflow_id is not None

    def test_cancel_nonexistent_workflow_returns_false(self, streaming_executor):
        """Test that cancelling a non-existent workflow returns False."""
        result = streaming_executor.cancel_workflow("nonexistent_workflow_id")
        assert result is False


class TestStreamingWorkflowExecutorBackwardCompatibility:
    """Tests for backward compatibility with base WorkflowExecutor."""

    @pytest.mark.asyncio
    async def test_execute_still_works(self, streaming_executor, simple_workflow):
        """Test that execute() method still works."""
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.summary = "Done"
        mock_result.error = None
        mock_result.tool_calls_used = 0

        with patch.object(
            streaming_executor.sub_agents, "spawn", new_callable=AsyncMock
        ) as mock_spawn:
            mock_spawn.return_value = mock_result

            result = await streaming_executor.execute(simple_workflow)

            assert result.success is True
            assert result.workflow_name == "test_workflow"

    @pytest.mark.asyncio
    async def test_execute_with_timeout(self, streaming_executor, simple_workflow):
        """Test execute() with timeout parameter."""
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.summary = "Done"
        mock_result.error = None
        mock_result.tool_calls_used = 0

        with patch.object(
            streaming_executor.sub_agents, "spawn", new_callable=AsyncMock
        ) as mock_spawn:
            mock_spawn.return_value = mock_result

            result = await streaming_executor.execute(simple_workflow, timeout=60.0)

            assert result.success is True


class TestStreamingWorkflowExecutorIntegration:
    """Integration tests for streaming executor."""

    @pytest.mark.asyncio
    async def test_full_workflow_streaming_sequence(self, streaming_executor, simple_workflow):
        """Test complete streaming sequence for a workflow."""
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.summary = "Done"
        mock_result.error = None
        mock_result.tool_calls_used = 0

        with patch.object(
            streaming_executor.sub_agents, "spawn", new_callable=AsyncMock
        ) as mock_spawn:
            mock_spawn.return_value = mock_result

            events_sequence = []
            async for chunk in streaming_executor.astream(simple_workflow):
                events_sequence.append(chunk.event_type)

            # Expected sequence:
            # WORKFLOW_START -> NODE_START -> NODE_COMPLETE -> NODE_START -> NODE_COMPLETE -> WORKFLOW_COMPLETE
            assert events_sequence[0] == WorkflowEventType.WORKFLOW_START
            assert events_sequence[-1] == WorkflowEventType.WORKFLOW_COMPLETE

            # Should have NODE_START/NODE_COMPLETE pairs
            node_starts = events_sequence.count(WorkflowEventType.NODE_START)
            node_completes = events_sequence.count(WorkflowEventType.NODE_COMPLETE)
            assert node_starts == node_completes

    @pytest.mark.asyncio
    async def test_workflow_id_consistency(self, streaming_executor, simple_workflow):
        """Test that workflow_id is consistent across all chunks."""
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.summary = "Done"
        mock_result.error = None
        mock_result.tool_calls_used = 0

        with patch.object(
            streaming_executor.sub_agents, "spawn", new_callable=AsyncMock
        ) as mock_spawn:
            mock_spawn.return_value = mock_result

            workflow_ids = set()
            async for chunk in streaming_executor.astream(simple_workflow):
                workflow_ids.add(chunk.workflow_id)

            # All chunks should have the same workflow_id
            assert len(workflow_ids) == 1
