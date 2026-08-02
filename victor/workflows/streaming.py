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

"""Streaming data models for workflow execution.

This module provides data models for streaming workflow execution events,
enabling real-time visibility into graph-based workflow progress.

The streaming API follows the observer pattern, allowing multiple consumers
to receive workflow events as they occur.

Example:
    from victor.workflows.streaming import (
        WorkflowEventType,
        WorkflowStreamChunk,
        WorkflowStreamContext,
    )

    # Create a stream chunk for node start
    chunk = WorkflowStreamChunk(
        event_type=WorkflowEventType.NODE_START,
        workflow_id="wf_123",
        node_id="analyze_code",
    )

    # Convert to base StreamChunk for provider compatibility
    stream_chunk = chunk.to_stream_chunk()
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from victor.providers.base import StreamChunk


class WorkflowEventType(str, Enum):
    """Event types for workflow streaming.

    This enum uses str as a base class to enable direct JSON serialization
    of event type values without requiring custom serializers.

    Attributes:
        WORKFLOW_START: Workflow execution has begun.
        WORKFLOW_COMPLETE: Workflow finished successfully.
        WORKFLOW_ERROR: Workflow terminated due to an error.
        NODE_START: A node has started executing.
        NODE_COMPLETE: A node finished successfully.
        NODE_ERROR: A node failed with an error.
        AGENT_CONTENT: Streaming content from an agent node.
        AGENT_TOOL_CALL: Agent is making a tool call.
        AGENT_TOOL_RESULT: Tool execution result received.
        PROGRESS_UPDATE: Progress update event.
        CHECKPOINT_SAVED: Checkpoint was saved.
    """

    WORKFLOW_START = "workflow_start"
    WORKFLOW_COMPLETE = "workflow_complete"
    WORKFLOW_ERROR = "workflow_error"
    NODE_START = "node_start"
    NODE_COMPLETE = "node_complete"
    NODE_ERROR = "node_error"
    AGENT_CONTENT = "agent_content"
    AGENT_TOOL_CALL = "agent_tool_call"
    AGENT_TOOL_RESULT = "agent_tool_result"
    PROGRESS_UPDATE = "progress_update"
    CHECKPOINT_SAVED = "checkpoint_saved"


# Register domain event mapping with canonical EventType taxonomy
def _register_workflow_event_taxonomy() -> None:
    try:
        from victor.core.events.taxonomy import EventTaxonomyRegistry
        from victor.framework.events import EventType

        EventTaxonomyRegistry.register_domain(
            "workflow",
            WorkflowEventType,
            {
                WorkflowEventType.WORKFLOW_START: EventType.STREAM_START,
                WorkflowEventType.WORKFLOW_COMPLETE: EventType.STREAM_END,
                WorkflowEventType.WORKFLOW_ERROR: EventType.ERROR,
                WorkflowEventType.NODE_START: EventType.PROGRESS,
                WorkflowEventType.NODE_COMPLETE: EventType.MILESTONE,
                WorkflowEventType.NODE_ERROR: EventType.ERROR,
                WorkflowEventType.AGENT_CONTENT: EventType.CONTENT,
                WorkflowEventType.AGENT_TOOL_CALL: EventType.TOOL_CALL,
                WorkflowEventType.AGENT_TOOL_RESULT: EventType.TOOL_RESULT,
                WorkflowEventType.PROGRESS_UPDATE: EventType.PROGRESS,
                WorkflowEventType.CHECKPOINT_SAVED: EventType.MILESTONE,
            },
        )
    except ImportError:
        pass  # Events module not available during early import


_register_workflow_event_taxonomy()


def _now_utc() -> datetime:
    """Return current UTC timestamp."""
    return datetime.now(timezone.utc)


@dataclass
class WorkflowStreamChunk:
    """A chunk of streaming data from workflow execution.

    This dataclass represents a single event in the workflow execution stream.
    It can be converted to the base StreamChunk for compatibility with
    existing streaming infrastructure.

    This is the CANONICAL streaming chunk type. Use this for all workflow
    streaming operations. The streaming_executor module imports from here.

    Attributes:
        event_type: The type of event this chunk represents.
        workflow_id: Unique identifier for the workflow instance.
        node_id: ID of the node that generated this event (if applicable).
        node_name: Human-readable name of the node (if applicable).
        content: Text content (for AGENT_CONTENT events).
        tool_calls: Tool call information (for AGENT_TOOL_CALL events).
        metadata: Additional event metadata.
        timestamp: When the event occurred.
        is_final: Whether this is the final chunk of the stream.
        progress: Workflow progress (0.0 to 1.0 as fraction, or 0.0 to 100.0 as percentage).
        error: Error message (for error events).

    Example:
        chunk = WorkflowStreamChunk(
            event_type=WorkflowEventType.AGENT_CONTENT,
            workflow_id="wf_123",
            node_id="summarize",
            node_name="Code Summarizer",
            content="The analysis shows...",
        )
    """

    event_type: WorkflowEventType
    workflow_id: str
    node_id: Optional[str] = None
    node_name: Optional[str] = None
    content: str = ""
    tool_calls: Optional[List[Dict[str, Any]]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=_now_utc)
    is_final: bool = False
    progress: Optional[float] = None
    error: Optional[str] = None

    def to_stream_chunk(self) -> StreamChunk:
        """Convert to base StreamChunk for provider compatibility.

        Returns:
            A StreamChunk instance with workflow metadata embedded.

        Example:
            workflow_chunk = WorkflowStreamChunk(...)
            stream_chunk = workflow_chunk.to_stream_chunk()
            # stream_chunk can be yielded in provider streams
        """
        # Build metadata including workflow-specific information
        chunk_metadata = dict(self.metadata)
        chunk_metadata["workflow_id"] = self.workflow_id
        chunk_metadata["event_type"] = self.event_type.value

        if self.node_id is not None:
            chunk_metadata["node_id"] = self.node_id

        if self.progress is not None:
            chunk_metadata["progress"] = self.progress

        if self.error is not None:
            chunk_metadata["error"] = self.error

        chunk_metadata["timestamp"] = self.timestamp.isoformat()

        return StreamChunk(
            content=self.content,
            tool_calls=self.tool_calls,
            is_final=self.is_final,
            metadata=chunk_metadata,
        )


def _create_queue() -> asyncio.Queue:
    """Create a new asyncio Queue for chunk streaming."""
    return asyncio.Queue()


@dataclass
class WorkflowStreamContext:
    """Context for managing workflow streaming state.

    This is the CANONICAL streaming context type. Use this for all workflow
    streaming operations. Provides both fraction-based (get_progress) and
    percentage-based (progress property) progress tracking for compatibility.

    Attributes:
        workflow_id: Unique identifier for the workflow instance.
        workflow_name: Human-readable workflow name (optional, defaults to "").
        start_time: When the workflow started.
        total_nodes: Total number of nodes in the workflow.
        completed_nodes: Number of nodes that have completed.
        current_node_id: ID of the currently executing node.
        is_cancelled: Whether the workflow has been cancelled.
        thread_id: Optional thread identifier for multi-threaded execution.
        chunk_queue: Queue for streaming chunks to consumers.

    Example:
        context = WorkflowStreamContext(
            workflow_id="wf_123",
            workflow_name="code_review",
            total_nodes=5,
        )

        # Update progress as nodes complete
        context.completed_nodes += 1
        fraction = context.get_progress()  # Returns 0.2
        percentage = context.progress  # Returns 20.0
    """

    workflow_id: str
    workflow_name: str = ""
    start_time: datetime = field(default_factory=_now_utc)
    total_nodes: int = 0
    completed_nodes: int = 0
    current_node_id: Optional[str] = None
    is_cancelled: bool = False
    thread_id: Optional[str] = None
    chunk_queue: asyncio.Queue = field(default_factory=_create_queue)

    def get_progress(self) -> float:
        """Calculate workflow progress as a fraction.

        Returns:
            Progress value between 0.0 and 1.0.
            Returns 0.0 if there are no nodes.

        Example:
            context = WorkflowStreamContext(
                workflow_id="wf_123",
                workflow_name="test",
                total_nodes=10,
                completed_nodes=3,
            )
            progress = context.get_progress()  # Returns 0.3
        """
        if self.total_nodes == 0:
            return 0.0
        return self.completed_nodes / self.total_nodes

    @property
    def progress(self) -> float:
        """Calculate workflow progress as a percentage.

        Returns:
            Progress value between 0.0 and 100.0.
            Returns 0.0 if there are no nodes.

        This property provides backward compatibility with code that
        expects percentage-based progress.

        Example:
            context = WorkflowStreamContext(
                workflow_id="wf_123",
                total_nodes=10,
                completed_nodes=3,
            )
            pct = context.progress  # Returns 30.0
        """
        return self.get_progress() * 100.0

    @property
    def is_complete(self) -> bool:
        """Check if the workflow has completed all nodes.

        Returns:
            True if all nodes have completed or there are no nodes.

        Example:
            if context.is_complete:
                print("Workflow finished!")
        """
        if self.total_nodes == 0:
            return True
        return self.completed_nodes >= self.total_nodes
