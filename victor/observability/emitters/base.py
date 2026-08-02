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

"""Base emitter interface for observability events.

This module defines the Protocol-based interfaces for event emitters,
following the Interface Segregation Principle (ISP) from SOLID.
Each emitter has a focused, single-responsibility interface.

NOTE: These protocols define the high-level emitter API. The underlying
event system uses the canonical Event class with topic-based routing,
but these protocols provide type-safe domain-specific methods.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable, Any, Dict, Optional
from contextlib import contextmanager

from victor.core.events import MessagingEvent


@runtime_checkable
class IEventEmitter(Protocol):
    """Base protocol for event emitters.

    SOLID: Interface Segregation Principle (ISP)
    - Focused interface for event emission
    - All emitters implement this protocol
    - Substitutable via Liskov Substitution Principle (LSP)

    Implementations must:
    - Be thread-safe
    - Handle ObservabilityBus unavailability gracefully
    - Support context managers for scoped events
    """

    def emit(self, event: MessagingEvent) -> None:
        """Emit a single event.

        Args:
            event: The event to emit
        """
        ...

    def emit_safe(self, event: MessagingEvent) -> bool:
        """Safely emit an event, catching any exceptions.

        Args:
            event: The event to emit

        Returns:
            True if emission succeeded, False otherwise
        """
        ...


@runtime_checkable
class IToolEventEmitter(IEventEmitter, Protocol):
    """Protocol for tool execution event emission.

    SOLID: Single Responsibility Principle (SRP)
    - Focused solely on tool execution events
    - Emits start, end, success, failure events

    Lifecycle:
        tool_start() → [tool execution] → tool_end() / tool_failure()
    """

    def tool_start(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        **metadata: Any,
    ) -> None:
        """Emit tool execution start event.

        Args:
            tool_name: Name of the tool being executed
            arguments: Tool arguments
            **metadata: Additional metadata (agent_id, session_id, etc.)
        """
        ...

    def tool_end(
        self,
        tool_name: str,
        duration_ms: float,
        result: Optional[Any] = None,
        **metadata: Any,
    ) -> None:
        """Emit tool execution end event (success).

        Args:
            tool_name: Name of the tool that completed
            duration_ms: Execution duration in milliseconds
            result: Tool result (will be truncated if large)
            **metadata: Additional metadata
        """
        ...

    def tool_failure(
        self,
        tool_name: str,
        duration_ms: float,
        error: Exception,
        **metadata: Any,
    ) -> None:
        """Emit tool execution failure event.

        Args:
            tool_name: Name of the tool that failed
            duration_ms: Execution duration before failure
            error: The exception that occurred
            **metadata: Additional metadata
        """
        ...

    @contextmanager
    def track_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        **metadata: Any,
    ):
        """Context manager for tracking tool execution.

        Automatically emits start/end events.

        Args:
            tool_name: Name of the tool
            arguments: Tool arguments
            **metadata: Additional metadata

        Yields:
            None

        Example:
            >>> with emitter.track_tool("read_file", {"path": "file.txt"}):
            ...     result = await tool(**arguments)
        """
        ...


@runtime_checkable
class IModelEventEmitter(IEventEmitter, Protocol):
    """Protocol for LLM model interaction event emission.

    SOLID: Single Responsibility Principle (SRP)
    - Focused solely on LLM call events
    - Emits request, response, streaming events

    Lifecycle:
        model_request() → [LLM processing] → model_response() / model_error()
    """

    def model_request(
        self,
        provider: str,
        model: str,
        prompt_tokens: int,
        **metadata: Any,
    ) -> None:
        """Emit LLM request event.

        Args:
            provider: Model provider (anthropic, openai, etc.)
            model: Model name
            prompt_tokens: Number of tokens in prompt
            **metadata: Additional metadata
        """
        ...

    def model_response(
        self,
        provider: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: float,
        **metadata: Any,
    ) -> None:
        """Emit LLM response event.

        Args:
            provider: Model provider
            model: Model name
            prompt_tokens: Number of tokens in prompt
            completion_tokens: Number of tokens in completion
            latency_ms: Request latency in milliseconds
            **metadata: Additional metadata
        """
        ...

    def model_streaming_delta(
        self,
        provider: str,
        model: str,
        delta: str,
        **metadata: Any,
    ) -> None:
        """Emit streaming delta event.

        Args:
            provider: Model provider
            model: Model name
            delta: Text delta from streaming response
            **metadata: Additional metadata
        """
        ...

    def model_error(
        self,
        provider: str,
        model: str,
        error: Exception,
        **metadata: Any,
    ) -> None:
        """Emit LLM error event.

        Args:
            provider: Model provider
            model: Model name
            error: The exception that occurred
            **metadata: Additional metadata
        """
        ...


@runtime_checkable
class IStateEventEmitter(IEventEmitter, Protocol):
    """Protocol for state transition event emission.

    SOLID: Single Responsibility Principle (SRP)
    - Focused solely on state machine transitions
    - Emits state change events

    Lifecycle:
        state_transition() → [new state]
    """

    def state_transition(
        self,
        old_stage: str,
        new_stage: str,
        confidence: float,
        **metadata: Any,
    ) -> None:
        """Emit state transition event.

        Args:
            old_stage: Previous state/stage
            new_stage: New state/stage
            confidence: Transition confidence (0.0 to 1.0)
            **metadata: Additional metadata
        """
        ...


@runtime_checkable
class ILifecycleEventEmitter(IEventEmitter, Protocol):
    """Protocol for lifecycle event emission.

    SOLID: Single Responsibility Principle (SRP)
    - Focused solely on lifecycle events
    - Emits session start/end events

    Lifecycle:
        session_start() → [session] → session_end()
    """

    def session_start(
        self,
        session_id: str,
        **metadata: Any,
    ) -> None:
        """Emit session start event.

        Args:
            session_id: Unique session identifier
            **metadata: Additional metadata
        """
        ...

    def session_end(
        self,
        session_id: str,
        duration_ms: float,
        **metadata: Any,
    ) -> None:
        """Emit session end event.

        Args:
            session_id: Unique session identifier
            duration_ms: Session duration in milliseconds
            **metadata: Additional metadata
        """
        ...


@runtime_checkable
class IErrorEventEmitter(IEventEmitter, Protocol):
    """Protocol for error event emission.

    SOLID: Single Responsibility Principle (SRP)
    - Focused solely on error events
    - Emits error events with context
    """

    def error(
        self,
        error: Exception,
        recoverable: bool,
        context: Optional[Dict[str, Any]] = None,
        **metadata: Any,
    ) -> None:
        """Emit error event.

        Args:
            error: The exception that occurred
            recoverable: Whether the error is recoverable
            context: Additional error context
            **metadata: Additional metadata
        """
        ...
