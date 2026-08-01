"""Events - Stream of observable agent actions.

This module defines the canonical event format for observing agent execution
in real-time. Events provide visibility into thinking, tool calls, content
generation, state changes, and errors.

Example:
    async for event in agent.stream("Analyze this code"):
        match event.type:
            case EventType.CONTENT:
                print(event.content, end="")
            case EventType.TOOL_CALL:
                print(f"Calling {event.tool_name}")
            case EventType.THINKING:
                print(f"Thinking: {event.content[:50]}...")
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

# Monotonically increasing counter for event ordering across concurrent operations
_event_counter = itertools.count(1)


class EventType(str, Enum):
    """Types of events emitted during agent execution.

    Events are categorized by their purpose:
    - Content events: CONTENT, THINKING
    - Tool events: TOOL_CALL, TOOL_RESULT, TOOL_ERROR, TOOL_PROGRESS
    - State events: STAGE_CHANGE
    - Lifecycle events: STREAM_START, STREAM_END
    - Error events: ERROR, RECOVERY
    - Progress events: PROGRESS, MILESTONE
    - Custom events: CUSTOM
    """

    # Content events
    CONTENT = "content"
    """Text content from the model's response."""

    THINKING = "thinking"
    """Extended thinking content (Claude's reasoning mode)."""

    # Tool events
    TOOL_CALL = "tool_call"
    """Tool is being invoked with arguments."""

    TOOL_RESULT = "tool_result"
    """Tool has returned a result."""

    TOOL_ERROR = "tool_error"
    """Tool execution failed."""

    TOOL_PROGRESS = "tool_progress"
    """Partial/streamed stdout/stderr while a tool is actively running.

    Emitted incrementally for long-running tools (graph, code_search, shell)
    so the UI can render a live, updating terminal block instead of waiting
    for the final result.
    """

    # State events
    STAGE_CHANGE = "stage_change"
    """Conversation stage has changed (e.g., PLANNING -> EXECUTION)."""

    # Lifecycle events
    STREAM_START = "stream_start"
    """Streaming response has started."""

    STREAM_END = "stream_end"
    """Streaming response has completed."""

    # Error events
    ERROR = "error"
    """An error occurred during execution."""

    RECOVERY = "recovery"
    """Recovery attempt is being made."""

    # Progress events
    PROGRESS = "progress"
    """Progress update (0.0 to 1.0)."""

    MILESTONE = "milestone"
    """Task milestone has been reached."""

    # Durable pause (FEP-0029)
    AWAITING_APPROVAL = "awaiting_approval"
    """The run durably paused on a policy ASK; metadata carries run_id + approval_request."""

    # Custom events
    CUSTOM = "custom"
    """User-defined custom event. Use metadata['custom_type'] for sub-typing."""


class EventSeverity(str, Enum):
    """Severity levels for events and validation issues.

    Severity levels indicate the importance or impact of events and issues:
    - Debug: Detailed diagnostic information for development
    - Info: General informational messages
    - Warning: Something that warrants attention but isn't critical
    - Error: An error that prevents normal operation
    - Critical: A serious error that requires immediate attention
    """

    DEBUG = "debug"
    """Detailed diagnostic information for development."""

    INFO = "info"
    """General informational messages."""

    WARNING = "warning"
    """Something that warrants attention but isn't critical."""

    ERROR = "error"
    """An error that prevents normal operation."""

    CRITICAL = "critical"
    """A serious error that requires immediate attention."""


@dataclass
class AgentExecutionEvent:
    """An observable event from agent execution.

    Events provide real-time visibility into what the agent is doing.
    Use the `type` field to determine what kind of event this is and
    which fields are relevant.

    Attributes:
        type: The type of event (see EventType enum)
        content: Text content for CONTENT/THINKING events
        tool_name: Tool name for tool-related events
        tool_id: Unique identifier for tool calls
        arguments: Arguments passed to the tool
        result: Result returned by the tool
        success: Whether the operation succeeded
        old_stage: Previous stage for STAGE_CHANGE events
        new_stage: New stage for STAGE_CHANGE events
        error: Error message for error events
        recoverable: Whether the error can be recovered from
        progress: Progress value (0.0 to 1.0) for PROGRESS events
        milestone: Milestone name for MILESTONE events
        metadata: Additional context-specific data
        timestamp: Unix timestamp when event was created

    Example:
        async for event in agent.stream("Refactor this"):
            if event.is_content_event:
                print(event.content, end="")
            elif event.is_tool_event:
                print(f"{event.tool_name}: {event.result[:100] if event.result else 'running...'}")
            elif event.is_error_event:
                print(f"Error: {event.error}")
    """

    type: EventType

    # Content fields
    content: str = ""

    # Tool fields
    tool_name: Optional[str] = None
    tool_id: Optional[str] = None
    arguments: Dict[str, Any] = field(default_factory=dict)
    result: Optional[str] = None
    success: bool = True

    # State fields
    old_stage: Optional[str] = None
    new_stage: Optional[str] = None

    # Error fields
    error: Optional[str] = None
    recoverable: bool = True

    # Progress fields
    progress: float = 0.0
    milestone: Optional[str] = None

    # Sequencing — monotonic ordering and causality tracking
    sequence_id: int = field(default_factory=lambda: next(_event_counter))
    parent_event_id: Optional[str] = None

    # Member identity — set for team member lifecycle events so consumers can
    # render per-member lanes (ADR-023 / FEP-0028). None for single-agent turns.
    member_id: Optional[str] = None

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    @property
    def is_tool_event(self) -> bool:
        """Check if this is a tool-related event."""
        return self.type in (
            EventType.TOOL_CALL,
            EventType.TOOL_RESULT,
            EventType.TOOL_ERROR,
            EventType.TOOL_PROGRESS,
        )

    @property
    def is_content_event(self) -> bool:
        """Check if this is a content event."""
        return self.type in (EventType.CONTENT, EventType.THINKING)

    @property
    def is_error_event(self) -> bool:
        """Check if this is an error event."""
        return self.type in (EventType.ERROR, EventType.TOOL_ERROR)

    @property
    def is_lifecycle_event(self) -> bool:
        """Check if this is a lifecycle event."""
        return self.type in (EventType.STREAM_START, EventType.STREAM_END)

    @property
    def is_custom_event(self) -> bool:
        """Check if this is a custom user-defined event."""
        return self.type == EventType.CUSTOM

    @property
    def event_type(self) -> str:
        """String alias for callers that use event_type naming."""
        return self.type.value

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "type": self.type.value,
            "content": self.content,
            "tool_name": self.tool_name,
            "tool_id": self.tool_id,
            "arguments": self.arguments,
            "result": self.result,
            "success": self.success,
            "old_stage": self.old_stage,
            "new_stage": self.new_stage,
            "error": self.error,
            "recoverable": self.recoverable,
            "progress": self.progress,
            "milestone": self.milestone,
            "sequence_id": self.sequence_id,
            "parent_event_id": self.parent_event_id,
            "member_id": self.member_id,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }


# =============================================================================
# Convenience Constructors
# =============================================================================


def content_event(content: str, **kwargs: Any) -> AgentExecutionEvent:
    """Create a content event.

    Args:
        content: Text content from the model
        **kwargs: Additional event attributes

    Returns:
        AgentExecutionEvent with type=CONTENT
    """
    return AgentExecutionEvent(type=EventType.CONTENT, content=content, **kwargs)


def thinking_event(content: str, **kwargs: Any) -> AgentExecutionEvent:
    """Create a thinking event.

    Args:
        content: Thinking/reasoning content
        **kwargs: Additional event attributes

    Returns:
        AgentExecutionEvent with type=THINKING
    """
    return AgentExecutionEvent(type=EventType.THINKING, content=content, **kwargs)


def tool_call_event(
    tool_name: str,
    arguments: Dict[str, Any],
    tool_id: Optional[str] = None,
    **kwargs: Any,
) -> AgentExecutionEvent:
    """Create a tool call event.

    Args:
        tool_name: Name of the tool being called
        arguments: Arguments passed to the tool
        tool_id: Optional unique identifier for this call
        **kwargs: Additional event attributes

    Returns:
        AgentExecutionEvent with type=TOOL_CALL
    """
    return AgentExecutionEvent(
        type=EventType.TOOL_CALL,
        tool_name=tool_name,
        tool_id=tool_id,
        arguments=arguments,
        **kwargs,
    )


def tool_result_event(
    tool_name: str,
    result: str,
    success: bool = True,
    tool_id: Optional[str] = None,
    **kwargs: Any,
) -> AgentExecutionEvent:
    """Create a tool result event.

    Args:
        tool_name: Name of the tool that returned
        result: Result from the tool
        success: Whether the tool succeeded
        tool_id: Optional unique identifier for this call
        **kwargs: Additional event attributes

    Returns:
        AgentExecutionEvent with type=TOOL_RESULT
    """
    return AgentExecutionEvent(
        type=EventType.TOOL_RESULT,
        tool_name=tool_name,
        tool_id=tool_id,
        result=result,
        success=success,
        **kwargs,
    )


def tool_progress_event(
    tool_name: str,
    stdout: str = "",
    stderr: str = "",
    progress: float = 0.0,
    tool_id: Optional[str] = None,
    is_final: bool = False,
    **kwargs: Any,
) -> AgentExecutionEvent:
    """Create a tool progress event carrying partial streamed tool output.

    Args:
        tool_name: Name of the tool that is running.
        stdout: Partial stdout chunk produced since the last progress event.
        stderr: Partial stderr chunk produced since the last progress event.
        progress: Optional progress estimate in the range 0.0 to 1.0.
        tool_id: Optional unique identifier matching the originating tool call.
        is_final: Whether this is the last progress chunk before the result.
        **kwargs: Additional event attributes.

    Returns:
        AgentExecutionEvent with type=TOOL_PROGRESS. The partial output is
        exposed both via ``content`` (stdout) and ``metadata["tool_progress"]``.
    """
    metadata = kwargs.pop("metadata", {})
    metadata["tool_progress"] = {
        "name": tool_name,
        "stdout": stdout,
        "stderr": stderr,
        "progress": progress,
        "is_final": is_final,
    }
    return AgentExecutionEvent(
        type=EventType.TOOL_PROGRESS,
        tool_name=tool_name,
        tool_id=tool_id,
        content=stdout,
        progress=progress,
        metadata=metadata,
        **kwargs,
    )


def tool_error_event(
    tool_name: str,
    error: str,
    tool_id: Optional[str] = None,
    **kwargs: Any,
) -> AgentExecutionEvent:
    """Create a tool error event.

    Args:
        tool_name: Name of the tool that failed
        error: Error message
        tool_id: Optional unique identifier for this call
        **kwargs: Additional event attributes

    Returns:
        AgentExecutionEvent with type=TOOL_ERROR
    """
    return AgentExecutionEvent(
        type=EventType.TOOL_ERROR,
        tool_name=tool_name,
        tool_id=tool_id,
        error=error,
        success=False,
        **kwargs,
    )


def stage_change_event(
    old_stage: str,
    new_stage: str,
    **kwargs: Any,
) -> AgentExecutionEvent:
    """Create a stage change event.

    Args:
        old_stage: Previous conversation stage
        new_stage: New conversation stage
        **kwargs: Additional event attributes

    Returns:
        AgentExecutionEvent with type=STAGE_CHANGE
    """
    return AgentExecutionEvent(
        type=EventType.STAGE_CHANGE,
        old_stage=old_stage,
        new_stage=new_stage,
        **kwargs,
    )


def error_event(
    error: str,
    recoverable: bool = True,
    **kwargs: Any,
) -> AgentExecutionEvent:
    """Create an error event.

    Args:
        error: Error message
        recoverable: Whether the error can be recovered from
        **kwargs: Additional event attributes

    Returns:
        AgentExecutionEvent with type=ERROR
    """
    return AgentExecutionEvent(
        type=EventType.ERROR,
        error=error,
        recoverable=recoverable,
        success=False,
        **kwargs,
    )


def stream_start_event(**kwargs: Any) -> AgentExecutionEvent:
    """Create a stream start event.

    Returns:
        AgentExecutionEvent with type=STREAM_START
    """
    return AgentExecutionEvent(type=EventType.STREAM_START, **kwargs)


def stream_end_event(
    success: bool = True, error: Optional[str] = None, **kwargs: Any
) -> AgentExecutionEvent:
    """Create a stream end event.

    Args:
        success: Whether streaming completed successfully
        error: Error message if failed

    Returns:
        AgentExecutionEvent with type=STREAM_END
    """
    return AgentExecutionEvent(type=EventType.STREAM_END, success=success, error=error, **kwargs)


def progress_event(progress: float, **kwargs: Any) -> AgentExecutionEvent:
    """Create a progress event.

    Args:
        progress: Progress value from 0.0 to 1.0
        **kwargs: Additional event attributes

    Returns:
        AgentExecutionEvent with type=PROGRESS
    """
    return AgentExecutionEvent(type=EventType.PROGRESS, progress=progress, **kwargs)


def milestone_event(milestone: str, **kwargs: Any) -> AgentExecutionEvent:
    """Create a milestone event.

    Args:
        milestone: Name/description of the milestone reached
        **kwargs: Additional event attributes

    Returns:
        AgentExecutionEvent with type=MILESTONE
    """
    return AgentExecutionEvent(type=EventType.MILESTONE, milestone=milestone, **kwargs)


def custom_event(
    custom_type: str,
    content: str = "",
    **kwargs: Any,
) -> AgentExecutionEvent:
    """Create a custom user-defined event.

    Args:
        custom_type: Sub-type identifier stored in metadata['custom_type']
        content: Optional content payload
        **kwargs: Additional event attributes

    Returns:
        AgentExecutionEvent with type=CUSTOM
    """
    metadata = kwargs.pop("metadata", {})
    metadata["custom_type"] = custom_type
    return AgentExecutionEvent(
        type=EventType.CUSTOM,
        content=content,
        metadata=metadata,
        **kwargs,
    )


def member_event(
    kind: str,
    member_id: str,
    *,
    content: str = "",
    success: bool = True,
    index: Optional[int] = None,
    formation: Optional[str] = None,
    **kwargs: Any,
) -> AgentExecutionEvent:
    """Create a team member lifecycle event (ADR-023 / FEP-0028 pillar 3).

    A CUSTOM event whose ``metadata['custom_type']`` is the member lifecycle kind
    (``member_start`` / ``member_completed`` / ``member_error``), tagged with
    ``member_id`` so downstream consumers can render per-member lanes. Reuses
    :func:`custom_event` semantics so no new ``EventType`` members leak into callers'
    ``match`` statements.

    Args:
        kind: The lifecycle kind (a ``member_*`` custom sub-type).
        member_id: The team member's id.
        content: Optional short payload (e.g. an error message).
        success: Whether the member step succeeded.
        index: The member's index in the formation, when known.
        formation: The active formation name, for context.
        **kwargs: Additional event attributes.

    Returns:
        AgentExecutionEvent with type=CUSTOM and member_id set.
    """
    metadata = kwargs.pop("metadata", {})
    metadata["custom_type"] = kind
    if index is not None:
        metadata["member_index"] = index
    if formation is not None:
        metadata["formation"] = formation
    return AgentExecutionEvent(
        type=EventType.CUSTOM,
        content=content,
        success=success,
        member_id=member_id,
        metadata=metadata,
        **kwargs,
    )
