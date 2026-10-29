"""Tests for workflow streaming data models and protocols.

These tests follow TDD approach - written before implementation.
They verify the streaming data models for graph orchestration.
"""

from datetime import datetime, timezone


class TestWorkflowEventType:
    """Tests for WorkflowEventType enum."""

    def test_has_workflow_start_event(self):
        """WorkflowEventType should have WORKFLOW_START."""
        from victor.workflows.streaming import WorkflowEventType

        assert WorkflowEventType.WORKFLOW_START is not None
        assert WorkflowEventType.WORKFLOW_START.value == "workflow_start"

    def test_has_workflow_complete_event(self):
        """WorkflowEventType should have WORKFLOW_COMPLETE."""
        from victor.workflows.streaming import WorkflowEventType

        assert WorkflowEventType.WORKFLOW_COMPLETE is not None
        assert WorkflowEventType.WORKFLOW_COMPLETE.value == "workflow_complete"

    def test_has_workflow_error_event(self):
        """WorkflowEventType should have WORKFLOW_ERROR."""
        from victor.workflows.streaming import WorkflowEventType

        assert WorkflowEventType.WORKFLOW_ERROR is not None
        assert WorkflowEventType.WORKFLOW_ERROR.value == "workflow_error"

    def test_has_node_start_event(self):
        """WorkflowEventType should have NODE_START."""
        from victor.workflows.streaming import WorkflowEventType

        assert WorkflowEventType.NODE_START is not None
        assert WorkflowEventType.NODE_START.value == "node_start"

    def test_has_node_complete_event(self):
        """WorkflowEventType should have NODE_COMPLETE."""
        from victor.workflows.streaming import WorkflowEventType

        assert WorkflowEventType.NODE_COMPLETE is not None
        assert WorkflowEventType.NODE_COMPLETE.value == "node_complete"

    def test_has_node_error_event(self):
        """WorkflowEventType should have NODE_ERROR."""
        from victor.workflows.streaming import WorkflowEventType

        assert WorkflowEventType.NODE_ERROR is not None
        assert WorkflowEventType.NODE_ERROR.value == "node_error"

    def test_has_agent_content_event(self):
        """WorkflowEventType should have AGENT_CONTENT."""
        from victor.workflows.streaming import WorkflowEventType

        assert WorkflowEventType.AGENT_CONTENT is not None
        assert WorkflowEventType.AGENT_CONTENT.value == "agent_content"

    def test_has_agent_tool_call_event(self):
        """WorkflowEventType should have AGENT_TOOL_CALL."""
        from victor.workflows.streaming import WorkflowEventType

        assert WorkflowEventType.AGENT_TOOL_CALL is not None
        assert WorkflowEventType.AGENT_TOOL_CALL.value == "agent_tool_call"

    def test_has_agent_tool_result_event(self):
        """WorkflowEventType should have AGENT_TOOL_RESULT."""
        from victor.workflows.streaming import WorkflowEventType

        assert WorkflowEventType.AGENT_TOOL_RESULT is not None
        assert WorkflowEventType.AGENT_TOOL_RESULT.value == "agent_tool_result"

    def test_has_progress_update_event(self):
        """WorkflowEventType should have PROGRESS_UPDATE."""
        from victor.workflows.streaming import WorkflowEventType

        assert WorkflowEventType.PROGRESS_UPDATE is not None
        assert WorkflowEventType.PROGRESS_UPDATE.value == "progress_update"

    def test_has_checkpoint_saved_event(self):
        """WorkflowEventType should have CHECKPOINT_SAVED."""
        from victor.workflows.streaming import WorkflowEventType

        assert WorkflowEventType.CHECKPOINT_SAVED is not None
        assert WorkflowEventType.CHECKPOINT_SAVED.value == "checkpoint_saved"

    def test_event_type_is_str_enum(self):
        """WorkflowEventType should be a str enum for JSON serialization."""
        from victor.workflows.streaming import WorkflowEventType

        # str enum values should be usable as strings directly
        event_type = WorkflowEventType.WORKFLOW_START
        assert isinstance(event_type.value, str)
        # Should be directly JSON serializable
        import json

        json_str = json.dumps({"event": event_type.value})
        assert "workflow_start" in json_str


class TestWorkflowStreamChunk:
    """Tests for WorkflowStreamChunk dataclass."""

    def test_creation_with_required_fields(self):
        """WorkflowStreamChunk should be creatable with minimal fields."""
        from victor.workflows.streaming import WorkflowEventType, WorkflowStreamChunk

        chunk = WorkflowStreamChunk(
            event_type=WorkflowEventType.NODE_START,
            workflow_id="wf_123",
        )
        assert chunk.event_type == WorkflowEventType.NODE_START
        assert chunk.workflow_id == "wf_123"

    def test_default_values(self):
        """WorkflowStreamChunk should have sensible defaults."""
        from victor.workflows.streaming import WorkflowEventType, WorkflowStreamChunk

        chunk = WorkflowStreamChunk(
            event_type=WorkflowEventType.AGENT_CONTENT,
            workflow_id="wf_123",
        )
        assert chunk.node_id is None
        assert chunk.content == ""
        assert chunk.tool_calls is None
        assert chunk.metadata == {}
        assert chunk.is_final is False
        assert chunk.progress is None
        assert chunk.error is None
        assert chunk.timestamp is not None  # Should have auto timestamp

    def test_creation_with_all_fields(self):
        """WorkflowStreamChunk should accept all fields."""
        from victor.workflows.streaming import WorkflowEventType, WorkflowStreamChunk

        timestamp = datetime.now(timezone.utc)
        chunk = WorkflowStreamChunk(
            event_type=WorkflowEventType.AGENT_CONTENT,
            workflow_id="wf_123",
            node_id="node_1",
            content="Hello, world!",
            tool_calls=[{"name": "search", "args": {}}],
            metadata={"key": "value"},
            timestamp=timestamp,
            is_final=True,
            progress=0.75,
            error="Something went wrong",
        )
        assert chunk.event_type == WorkflowEventType.AGENT_CONTENT
        assert chunk.workflow_id == "wf_123"
        assert chunk.node_id == "node_1"
        assert chunk.content == "Hello, world!"
        assert chunk.tool_calls == [{"name": "search", "args": {}}]
        assert chunk.metadata == {"key": "value"}
        assert chunk.timestamp == timestamp
        assert chunk.is_final is True
        assert chunk.progress == 0.75
        assert chunk.error == "Something went wrong"

    def test_to_stream_chunk_conversion(self):
        """WorkflowStreamChunk should convert to base StreamChunk."""
        from victor.workflows.streaming import WorkflowEventType, WorkflowStreamChunk
        from victor.providers.base import StreamChunk

        workflow_chunk = WorkflowStreamChunk(
            event_type=WorkflowEventType.AGENT_CONTENT,
            workflow_id="wf_123",
            node_id="node_1",
            content="Hello, world!",
            tool_calls=[{"name": "search", "args": {"query": "test"}}],
            is_final=True,
            metadata={"source": "workflow"},
        )

        stream_chunk = workflow_chunk.to_stream_chunk()

        assert isinstance(stream_chunk, StreamChunk)
        assert stream_chunk.content == "Hello, world!"
        assert stream_chunk.tool_calls == [{"name": "search", "args": {"query": "test"}}]
        assert stream_chunk.is_final is True
        # Metadata should include workflow info
        assert stream_chunk.metadata is not None
        assert stream_chunk.metadata.get("workflow_id") == "wf_123"
        assert stream_chunk.metadata.get("node_id") == "node_1"
        assert stream_chunk.metadata.get("event_type") == "agent_content"

    def test_to_stream_chunk_minimal(self):
        """to_stream_chunk should work with minimal fields."""
        from victor.workflows.streaming import WorkflowEventType, WorkflowStreamChunk
        from victor.providers.base import StreamChunk

        workflow_chunk = WorkflowStreamChunk(
            event_type=WorkflowEventType.WORKFLOW_START,
            workflow_id="wf_123",
        )

        stream_chunk = workflow_chunk.to_stream_chunk()

        assert isinstance(stream_chunk, StreamChunk)
        assert stream_chunk.content == ""
        assert stream_chunk.is_final is False

    def test_timestamp_auto_generated(self):
        """WorkflowStreamChunk timestamp should be auto-generated if not provided."""
        from victor.workflows.streaming import WorkflowEventType, WorkflowStreamChunk

        before = datetime.now(timezone.utc)
        chunk = WorkflowStreamChunk(
            event_type=WorkflowEventType.NODE_START,
            workflow_id="wf_123",
        )
        after = datetime.now(timezone.utc)

        assert chunk.timestamp is not None
        assert before <= chunk.timestamp <= after


class TestWorkflowStreamContext:
    """Tests for WorkflowStreamContext dataclass."""

    def test_creation_with_required_fields(self):
        """WorkflowStreamContext should be creatable with required fields."""
        from victor.workflows.streaming import WorkflowStreamContext

        context = WorkflowStreamContext(
            workflow_id="wf_123",
            workflow_name="my_workflow",
        )
        assert context.workflow_id == "wf_123"
        assert context.workflow_name == "my_workflow"

    def test_default_values(self):
        """WorkflowStreamContext should have sensible defaults."""
        from victor.workflows.streaming import WorkflowStreamContext

        context = WorkflowStreamContext(
            workflow_id="wf_123",
            workflow_name="my_workflow",
        )
        assert context.start_time is not None
        assert context.total_nodes == 0
        assert context.completed_nodes == 0
        assert context.current_node_id is None
        assert context.is_cancelled is False
        assert context.thread_id is None
        assert context.chunk_queue is not None

    def test_progress_calculation(self):
        """WorkflowStreamContext should calculate progress correctly."""
        from victor.workflows.streaming import WorkflowStreamContext

        context = WorkflowStreamContext(
            workflow_id="wf_123",
            workflow_name="my_workflow",
            total_nodes=10,
            completed_nodes=3,
        )
        progress = context.get_progress()
        assert progress == 0.3

    def test_progress_calculation_zero_nodes(self):
        """WorkflowStreamContext should handle zero total nodes."""
        from victor.workflows.streaming import WorkflowStreamContext

        context = WorkflowStreamContext(
            workflow_id="wf_123",
            workflow_name="my_workflow",
            total_nodes=0,
            completed_nodes=0,
        )
        progress = context.get_progress()
        assert progress == 0.0

    def test_progress_calculation_all_complete(self):
        """WorkflowStreamContext should return 1.0 when all nodes complete."""
        from victor.workflows.streaming import WorkflowStreamContext

        context = WorkflowStreamContext(
            workflow_id="wf_123",
            workflow_name="my_workflow",
            total_nodes=5,
            completed_nodes=5,
        )
        progress = context.get_progress()
        assert progress == 1.0

    def test_is_complete_property(self):
        """WorkflowStreamContext should report completion status."""
        from victor.workflows.streaming import WorkflowStreamContext

        context = WorkflowStreamContext(
            workflow_id="wf_123",
            workflow_name="my_workflow",
            total_nodes=3,
            completed_nodes=2,
        )
        assert context.is_complete is False

        context.completed_nodes = 3
        assert context.is_complete is True

    def test_is_complete_with_zero_nodes(self):
        """WorkflowStreamContext should handle zero nodes for completion check."""
        from victor.workflows.streaming import WorkflowStreamContext

        context = WorkflowStreamContext(
            workflow_id="wf_123",
            workflow_name="my_workflow",
            total_nodes=0,
            completed_nodes=0,
        )
        # With zero nodes, we consider it complete (no work to do)
        assert context.is_complete is True

    def test_cancellation_flag(self):
        """WorkflowStreamContext should support cancellation."""
        from victor.workflows.streaming import WorkflowStreamContext

        context = WorkflowStreamContext(
            workflow_id="wf_123",
            workflow_name="my_workflow",
        )
        assert context.is_cancelled is False

        context.is_cancelled = True
        assert context.is_cancelled is True

    def test_creation_with_all_fields(self):
        """WorkflowStreamContext should accept all fields."""
        import asyncio
        from victor.workflows.streaming import WorkflowStreamContext

        start_time = datetime.now(timezone.utc)
        queue: asyncio.Queue = asyncio.Queue()

        context = WorkflowStreamContext(
            workflow_id="wf_123",
            workflow_name="my_workflow",
            start_time=start_time,
            total_nodes=5,
            completed_nodes=2,
            current_node_id="node_2",
            is_cancelled=False,
            thread_id="thread_456",
            chunk_queue=queue,
        )

        assert context.workflow_id == "wf_123"
        assert context.workflow_name == "my_workflow"
        assert context.start_time == start_time
        assert context.total_nodes == 5
        assert context.completed_nodes == 2
        assert context.current_node_id == "node_2"
        assert context.is_cancelled is False
        assert context.thread_id == "thread_456"
        assert context.chunk_queue is queue


class TestIStreamingWorkflowExecutor:
    """Tests for IStreamingWorkflowExecutor protocol."""

    def test_protocol_is_runtime_checkable(self):
        """IStreamingWorkflowExecutor should be runtime checkable."""
        from victor.workflows.protocols import IStreamingWorkflowExecutor

        # Protocol should be importable and usable for isinstance checks
        assert hasattr(IStreamingWorkflowExecutor, "__protocol_attrs__") or isinstance(
            IStreamingWorkflowExecutor, type
        )

    def test_compliant_class_is_instance(self):
        """A compliant class should pass isinstance check."""
        from victor.workflows.protocols import (
            IStreamingWorkflowExecutor,
            IWorkflowGraph,
            ICheckpointStore,
        )
        from victor.workflows.streaming import WorkflowStreamChunk, WorkflowEventType
        from typing import AsyncIterator, Callable, Optional, Dict, Any

        class MockStreamingExecutor:
            async def astream(
                self,
                graph: IWorkflowGraph,
                initial_state: Dict[str, Any],
                checkpoint_store: Optional[ICheckpointStore] = None,
            ) -> AsyncIterator[WorkflowStreamChunk]:
                yield WorkflowStreamChunk(
                    event_type=WorkflowEventType.WORKFLOW_START,
                    workflow_id="wf_123",
                )

            def subscribe(
                self,
                callback: Callable[[WorkflowStreamChunk], None],
            ) -> Callable[[], None]:
                def unsubscribe():
                    pass

                return unsubscribe

            def cancel_workflow(self, workflow_id: str) -> bool:
                return True

        executor = MockStreamingExecutor()
        assert isinstance(executor, IStreamingWorkflowExecutor)

    def test_non_compliant_class_is_not_instance(self):
        """A non-compliant class should fail isinstance check."""
        from victor.workflows.protocols import IStreamingWorkflowExecutor

        class IncompleteExecutor:
            # Missing all required methods
            pass

        executor = IncompleteExecutor()
        assert not isinstance(executor, IStreamingWorkflowExecutor)
