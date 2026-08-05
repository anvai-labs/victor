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

"""Unified observability infrastructure for workflow execution.

Phase 5 of workflow architecture consolidation: Unified Streaming Observability.

This module provides:
- StreamingObserver: Protocol for subscribing to workflow events
- ObservabilityEmitter: Emits events to registered observers
- Event types unified across workflow and agent execution

The observability system enables:
- Real-time progress tracking
- Metrics collection
- Tracing integration
- Debugging and logging

Example:
    from victor.workflows.observability import (
        StreamingObserver,
        ObservabilityEmitter,
        create_emitter,
    )

    # Create an emitter
    emitter = create_emitter()

    # Register observers
    def log_observer(chunk: WorkflowStreamChunk) -> None:
        print(f"[{chunk.event_type}] {chunk.node_id}")

    emitter.add_observer(log_observer)

    # Emit events during execution
    emitter.emit_node_start("analyze", "Analyzer")
    emitter.emit_node_complete("analyze", duration=1.5)
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Set,
    Tuple,
    runtime_checkable,
)

from victor.workflows.streaming import (
    WorkflowEventType,
    WorkflowStreamChunk,
    WorkflowStreamContext,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Observer Protocol
# =============================================================================


@runtime_checkable
class StreamingObserver(Protocol):
    """Protocol for observing workflow streaming events.

    Observers receive WorkflowStreamChunk events during workflow execution.
    This enables decoupled event processing for logging, metrics, tracing, etc.

    Example:
        class MetricsObserver:
            def __init__(self):
                self.node_durations: Dict[str, float] = {}

            def on_event(self, chunk: WorkflowStreamChunk) -> None:
                if chunk.event_type == WorkflowEventType.NODE_COMPLETE:
                    duration = chunk.metadata.get("duration_seconds", 0)
                    self.node_durations[chunk.node_id] = duration

            def get_filter(self) -> Optional[Set[WorkflowEventType]]:
                return {WorkflowEventType.NODE_COMPLETE}
    """

    def on_event(self, chunk: WorkflowStreamChunk) -> None:
        """Handle a streaming event.

        Args:
            chunk: The workflow stream chunk to process.
        """
        ...

    def get_filter(self) -> Optional[Set[WorkflowEventType]]:
        """Get the set of event types this observer cares about.

        Returns:
            Set of event types to receive, or None to receive all events.
        """
        ...


# =============================================================================
# Function-based Observer Wrapper
# =============================================================================


@dataclass
class FunctionObserver:
    """Wrapper to make a callback function conform to StreamingObserver.

    Allows using simple functions as observers without implementing
    the full protocol.

    Example:
        def log_all(chunk):
            print(chunk)

        observer = FunctionObserver(callback=log_all)
    """

    callback: Callable[[WorkflowStreamChunk], None]
    event_filter: Optional[Set[WorkflowEventType]] = None

    def on_event(self, chunk: WorkflowStreamChunk) -> None:
        """Call the wrapped function."""
        self.callback(chunk)

    def get_filter(self) -> Optional[Set[WorkflowEventType]]:
        """Return the configured filter."""
        return self.event_filter


# =============================================================================
# Async Observer Protocol
# =============================================================================


@runtime_checkable
class AsyncStreamingObserver(Protocol):
    """Protocol for async observers.

    Similar to StreamingObserver but for async event handlers.

    Example:
        class AsyncMetricsObserver:
            async def on_event(self, chunk: WorkflowStreamChunk) -> None:
                await self.metrics_client.record(chunk)

            def get_filter(self) -> Optional[Set[WorkflowEventType]]:
                return None  # Receive all events
    """

    async def on_event(self, chunk: WorkflowStreamChunk) -> None:
        """Handle a streaming event asynchronously."""
        ...

    def get_filter(self) -> Optional[Set[WorkflowEventType]]:
        """Get the set of event types this observer cares about."""
        ...


# =============================================================================
# Observability Emitter
# =============================================================================


@dataclass
class ObservabilityEmitter:
    """Emitter for workflow observability events.

    Manages a set of observers and emits events to them during
    workflow execution. Supports both sync and async observers.

    Attributes:
        workflow_id: Unique identifier for the workflow instance.
        workflow_name: Human-readable name of the workflow.
        total_nodes: Total number of nodes in the workflow.
        completed_nodes: Number of completed nodes.

    Example:
        emitter = ObservabilityEmitter(
            workflow_id="wf_123",
            workflow_name="code_review",
            total_nodes=5,
        )

        # Add observers
        emitter.add_observer(log_observer)
        emitter.add_async_observer(metrics_observer)

        # Emit events
        emitter.emit_workflow_start()
        emitter.emit_node_start("analyze")
        await emitter.emit_async(chunk)
    """

    workflow_id: str
    workflow_name: str = ""
    total_nodes: int = 0
    completed_nodes: int = 0
    _observers: List[StreamingObserver] = field(default_factory=list)
    _async_observers: List[AsyncStreamingObserver] = field(default_factory=list)
    _start_time: float = field(default_factory=time.time)

    def add_observer(
        self,
        observer: Callable[[WorkflowStreamChunk], None] | StreamingObserver,
        event_filter: Optional[Set[WorkflowEventType]] = None,
    ) -> Callable[[], None]:
        """Add an observer for workflow events.

        Args:
            observer: Either a StreamingObserver instance or a callback function.
            event_filter: Optional filter for specific event types (only for functions).

        Returns:
            Unsubscribe function to remove the observer.

        Example:
            def on_progress(chunk):
                print(f"Progress: {chunk.progress}%")

            unsubscribe = emitter.add_observer(on_progress, {
                WorkflowEventType.NODE_COMPLETE
            })
        """
        if callable(observer) and not isinstance(observer, StreamingObserver):
            # Wrap function in FunctionObserver
            wrapped = FunctionObserver(callback=observer, event_filter=event_filter)
            self._observers.append(wrapped)
            actual_observer = wrapped
        else:
            self._observers.append(observer)
            actual_observer = observer

        def unsubscribe() -> None:
            if actual_observer in self._observers:
                self._observers.remove(actual_observer)

        return unsubscribe

    def add_async_observer(
        self,
        observer: AsyncStreamingObserver,
    ) -> Callable[[], None]:
        """Add an async observer for workflow events.

        Args:
            observer: An AsyncStreamingObserver instance.

        Returns:
            Unsubscribe function to remove the observer.
        """
        self._async_observers.append(observer)

        def unsubscribe() -> None:
            if observer in self._async_observers:
                self._async_observers.remove(observer)

        return unsubscribe

    @property
    def progress(self) -> float:
        """Calculate progress as percentage (0-100)."""
        if self.total_nodes == 0:
            return 0.0
        return (self.completed_nodes / self.total_nodes) * 100.0

    def _should_notify(
        self,
        observer: StreamingObserver | AsyncStreamingObserver,
        event_type: WorkflowEventType,
    ) -> bool:
        """Check if an observer should be notified for an event type."""
        filter_set = observer.get_filter()
        return filter_set is None or event_type in filter_set

    def emit(self, chunk: WorkflowStreamChunk) -> None:
        """Emit an event to all sync observers.

        Args:
            chunk: The event to emit.
        """
        for observer in self._observers:
            if self._should_notify(observer, chunk.event_type):
                try:
                    observer.on_event(chunk)
                except Exception as e:
                    logger.warning(f"Observer error: {e}")

    async def emit_async(self, chunk: WorkflowStreamChunk) -> None:
        """Emit an event to all async observers.

        Args:
            chunk: The event to emit.
        """
        # Emit to sync observers first
        self.emit(chunk)

        # Then emit to async observers
        for observer in self._async_observers:
            if self._should_notify(observer, chunk.event_type):
                try:
                    await observer.on_event(chunk)
                except Exception as e:
                    logger.warning(f"Async observer error: {e}")

    def emit_workflow_start(self, metadata: Optional[Dict[str, Any]] = None) -> WorkflowStreamChunk:
        """Emit a WORKFLOW_START event.

        Args:
            metadata: Additional metadata for the event.

        Returns:
            The emitted chunk.
        """
        chunk = WorkflowStreamChunk(
            event_type=WorkflowEventType.WORKFLOW_START,
            workflow_id=self.workflow_id,
            progress=0.0,
            metadata={
                "workflow_name": self.workflow_name,
                "total_nodes": self.total_nodes,
                **(metadata or {}),
            },
        )
        self.emit(chunk)
        return chunk

    def emit_workflow_complete(
        self,
        success: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> WorkflowStreamChunk:
        """Emit a WORKFLOW_COMPLETE event.

        Args:
            success: Whether the workflow completed successfully.
            metadata: Additional metadata for the event.

        Returns:
            The emitted chunk.
        """
        chunk = WorkflowStreamChunk(
            event_type=WorkflowEventType.WORKFLOW_COMPLETE,
            workflow_id=self.workflow_id,
            progress=100.0,
            is_final=True,
            metadata={
                "workflow_name": self.workflow_name,
                "total_duration": time.time() - self._start_time,
                "success": success,
                **(metadata or {}),
            },
        )
        self.emit(chunk)
        return chunk

    def emit_workflow_error(
        self,
        error: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> WorkflowStreamChunk:
        """Emit a WORKFLOW_ERROR event.

        Args:
            error: The error message.
            metadata: Additional metadata for the event.

        Returns:
            The emitted chunk.
        """
        chunk = WorkflowStreamChunk(
            event_type=WorkflowEventType.WORKFLOW_ERROR,
            workflow_id=self.workflow_id,
            progress=self.progress,
            error=error,
            is_final=True,
            metadata={
                "workflow_name": self.workflow_name,
                "total_duration": time.time() - self._start_time,
                **(metadata or {}),
            },
        )
        self.emit(chunk)
        return chunk

    def emit_node_start(
        self,
        node_id: str,
        node_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> WorkflowStreamChunk:
        """Emit a NODE_START event.

        Args:
            node_id: The node's unique identifier.
            node_name: Human-readable node name.
            metadata: Additional metadata for the event.

        Returns:
            The emitted chunk.
        """
        chunk = WorkflowStreamChunk(
            event_type=WorkflowEventType.NODE_START,
            workflow_id=self.workflow_id,
            node_id=node_id,
            node_name=node_name,
            progress=self.progress,
            metadata=metadata or {},
        )
        self.emit(chunk)
        return chunk

    def emit_node_complete(
        self,
        node_id: str,
        node_name: Optional[str] = None,
        duration: float = 0.0,
        output: Any = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> WorkflowStreamChunk:
        """Emit a NODE_COMPLETE event.

        Args:
            node_id: The node's unique identifier.
            node_name: Human-readable node name.
            duration: Execution duration in seconds.
            output: Node output (will be added to metadata).
            metadata: Additional metadata for the event.

        Returns:
            The emitted chunk.
        """
        self.completed_nodes += 1
        chunk = WorkflowStreamChunk(
            event_type=WorkflowEventType.NODE_COMPLETE,
            workflow_id=self.workflow_id,
            node_id=node_id,
            node_name=node_name,
            progress=self.progress,
            metadata={
                "duration_seconds": duration,
                "output": output,
                **(metadata or {}),
            },
        )
        self.emit(chunk)
        return chunk

    def emit_node_error(
        self,
        node_id: str,
        error: str,
        node_name: Optional[str] = None,
        duration: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> WorkflowStreamChunk:
        """Emit a NODE_ERROR event.

        Args:
            node_id: The node's unique identifier.
            error: The error message.
            node_name: Human-readable node name.
            duration: Execution duration in seconds.
            metadata: Additional metadata for the event.

        Returns:
            The emitted chunk.
        """
        chunk = WorkflowStreamChunk(
            event_type=WorkflowEventType.NODE_ERROR,
            workflow_id=self.workflow_id,
            node_id=node_id,
            node_name=node_name,
            progress=self.progress,
            error=error,
            metadata={
                "duration_seconds": duration,
                **(metadata or {}),
            },
        )
        self.emit(chunk)
        return chunk

    def emit_agent_content(
        self,
        node_id: str,
        content: str,
        node_name: Optional[str] = None,
    ) -> WorkflowStreamChunk:
        """Emit an AGENT_CONTENT event for streaming text.

        Args:
            node_id: The node's unique identifier.
            content: The streamed content.
            node_name: Human-readable node name.

        Returns:
            The emitted chunk.
        """
        chunk = WorkflowStreamChunk(
            event_type=WorkflowEventType.AGENT_CONTENT,
            workflow_id=self.workflow_id,
            node_id=node_id,
            node_name=node_name,
            content=content,
            progress=self.progress,
        )
        self.emit(chunk)
        return chunk

    def emit_tool_call(
        self,
        node_id: str,
        tool_calls: List[Dict[str, Any]],
        node_name: Optional[str] = None,
    ) -> WorkflowStreamChunk:
        """Emit an AGENT_TOOL_CALL event.

        Args:
            node_id: The node's unique identifier.
            tool_calls: List of tool call information.
            node_name: Human-readable node name.

        Returns:
            The emitted chunk.
        """
        chunk = WorkflowStreamChunk(
            event_type=WorkflowEventType.AGENT_TOOL_CALL,
            workflow_id=self.workflow_id,
            node_id=node_id,
            node_name=node_name,
            tool_calls=tool_calls,
            progress=self.progress,
        )
        self.emit(chunk)
        return chunk

    def emit_progress_update(
        self,
        node_id: Optional[str] = None,
        message: str = "",
    ) -> WorkflowStreamChunk:
        """Emit a PROGRESS_UPDATE event.

        Args:
            node_id: Optional node identifier.
            message: Progress message.

        Returns:
            The emitted chunk.
        """
        chunk = WorkflowStreamChunk(
            event_type=WorkflowEventType.PROGRESS_UPDATE,
            workflow_id=self.workflow_id,
            node_id=node_id,
            content=message,
            progress=self.progress,
        )
        self.emit(chunk)
        return chunk

    async def astream(self) -> AsyncIterator[WorkflowStreamChunk]:
        """Create an async iterator that yields events as they are emitted.

        This method sets up a queue-based observer that collects events
        and yields them asynchronously.

        Yields:
            WorkflowStreamChunk events as they occur.

        Example:
            async for chunk in emitter.astream():
                print(chunk)
        """
        queue: asyncio.Queue[Optional[WorkflowStreamChunk]] = asyncio.Queue()

        def queue_observer(chunk: WorkflowStreamChunk) -> None:
            queue.put_nowait(chunk)

        unsubscribe = self.add_observer(queue_observer)

        try:
            while True:
                chunk = await queue.get()
                if chunk is None:
                    break
                yield chunk
                if chunk.is_final:
                    break
        finally:
            unsubscribe()


# =============================================================================
# Factory Functions
# =============================================================================


def create_emitter(
    workflow_id: Optional[str] = None,
    workflow_name: str = "",
    total_nodes: int = 0,
) -> ObservabilityEmitter:
    """Create an ObservabilityEmitter with a generated ID if not provided.

    Args:
        workflow_id: Optional workflow ID (generated if not provided).
        workflow_name: Human-readable workflow name.
        total_nodes: Total number of nodes in the workflow.

    Returns:
        Configured ObservabilityEmitter instance.

    Example:
        emitter = create_emitter(
            workflow_name="code_review",
            total_nodes=5,
        )
    """
    return ObservabilityEmitter(
        workflow_id=workflow_id or uuid.uuid4().hex[:8],
        workflow_name=workflow_name,
        total_nodes=total_nodes,
    )


def create_logging_observer(
    level: int = logging.INFO,
    logger_name: str = "victor.workflows.observability",
) -> FunctionObserver:
    """Create an observer that logs all events.

    Args:
        level: Logging level to use.
        logger_name: Name of the logger to use.

    Returns:
        FunctionObserver that logs events.

    Example:
        observer = create_logging_observer(level=logging.DEBUG)
        emitter.add_observer(observer)
    """
    obs_logger = logging.getLogger(logger_name)

    def log_event(chunk: WorkflowStreamChunk) -> None:
        obs_logger.log(
            level,
            "[%s] %s node=%s progress=%.1f%%",
            chunk.workflow_id,
            chunk.event_type.value,
            chunk.node_id or "-",
            chunk.progress or 0.0,
        )

    return FunctionObserver(callback=log_event)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Protocols
    "StreamingObserver",
    "AsyncStreamingObserver",
    # Observer implementations
    "FunctionObserver",
    # Emitter
    "ObservabilityEmitter",
    # Factory functions
    "create_emitter",
    "create_logging_observer",
]
