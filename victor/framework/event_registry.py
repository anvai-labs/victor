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

"""Event Registry - Registry-based bidirectional event conversion.

This module implements a centralized event registry that replaces the
scattered conversion functions in cqrs_bridge.py with a unified,
extensible architecture.

Design Patterns:
- **Registry Pattern**: Central repository for event converters
- **Strategy Pattern**: Pluggable conversion strategies per event type
- **Protocol-First**: Type-safe interfaces via Python protocols
- **Single Source of Truth**: One place defines all event mappings

Architecture:
    ┌─────────────────────────────────────────────────────────────────┐
    │                     EventRegistry                                │
    │  ┌─────────────────────────────────────────────────────────┐    │
    │  │              Converter Strategies                        │    │
    │  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐        │    │
    │  │  │ CONTENT │ │TOOL_CALL│ │ STAGE   │ │ ERROR   │ ...    │    │
    │  │  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘        │    │
    │  └───────┴───────────┴───────────┴───────────┴──────────────┘    │
    │                          │                                       │
    │            ┌─────────────┴─────────────┐                        │
    │            ▼                           ▼                        │
    │     to_external()                from_external()                │
    │   (Event → Dict)               (Dict → Event)                   │
    └─────────────────────────────────────────────────────────────────┘

Benefits:
- Replace ~400 lines of conversion code with ~200 lines
- Eliminates duplicate event handling logic
- Easy to add new event types
- Clear separation of concerns
- Testable individual converters

Example:
    from victor.framework.event_registry import (
        get_event_registry,
        EventTarget,
    )

    registry = get_event_registry()

    # Convert to CQRS
    cqrs_data = registry.to_external(event, EventTarget.CQRS)

    # Convert from CQRS
    event = registry.from_external(cqrs_data, EventTarget.CQRS)

    # Convert to Observability
    obs_data = registry.to_external(event, EventTarget.OBSERVABILITY)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Protocol,
    TypedDict,
    Type,
    TypeVar,
    Union,
    runtime_checkable,
)

from victor.framework.events import (
    AgentExecutionEvent,
    EventType,
    content_event,
    error_event,
    milestone_event,
    progress_event,
    stage_change_event,
    stream_end_event,
    stream_start_event,
    thinking_event,
    tool_call_event,
    tool_error_event,
    tool_result_event,
)

if TYPE_CHECKING:
    from victor.core.event_sourcing import DomainEvent as CQRSEvent

logger = logging.getLogger(__name__)


# =============================================================================
# Type Definitions
# =============================================================================


class EventErrorDetails(TypedDict, total=False):
    """Typed structure for event error details.

    Provides consistent structure for error information across event
    converters. All fields are optional except error.
    """

    error: str
    error_type: Optional[str]
    error_category: Optional[str]
    tool_name: Optional[str]
    tool_id: Optional[str]


# =============================================================================
# Target System Enum
# =============================================================================


class EventTarget(str, Enum):
    """Target systems for event conversion."""

    CQRS = "cqrs"
    """CQRS event sourcing system."""

    OBSERVABILITY = "observability"
    """Observability EventBus system."""

    STREAM_CHUNK = "stream_chunk"
    """Protocol StreamChunk format."""


# =============================================================================
# Converter Protocol
# =============================================================================


@runtime_checkable
class EventConverterProtocol(Protocol):
    """Protocol for event type converters.

    Each converter handles bidirectional conversion for a specific
    framework EventType.
    """

    @property
    def event_type(self) -> EventType:
        """The framework EventType this converter handles."""
        ...

    @property
    def external_type_names(self) -> Dict[EventTarget, List[str]]:
        """External type names this converter can parse, per target."""
        ...

    def to_external(self, event: AgentExecutionEvent, target: EventTarget) -> Dict[str, Any]:
        """Convert framework event to external format.

        Args:
            event: Framework Event instance.
            target: Target system for conversion.

        Returns:
            Dictionary suitable for the target system.
        """
        ...

    def from_external(
        self,
        data: Dict[str, Any],
        target: EventTarget,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AgentExecutionEvent:
        """Convert external data to framework event.

        Args:
            data: External event data.
            target: Source system of the data.
            metadata: Additional metadata from the external event.

        Returns:
            Framework Event instance.
        """
        ...


# =============================================================================
# Base Converter Implementation
# =============================================================================


class BaseEventConverter(ABC):
    """Abstract base class for event converters.

    Provides common functionality and defines the converter interface.
    Subclasses implement specific conversion logic per event type.
    """

    @property
    @abstractmethod
    def event_type(self) -> EventType:
        """The framework EventType this converter handles."""
        pass

    @property
    @abstractmethod
    def external_type_names(self) -> Dict[EventTarget, List[str]]:
        """External type names this converter can parse, per target.

        Returns:
            Dictionary mapping EventTarget to list of recognizable type names.
        """
        pass

    @abstractmethod
    def to_external(self, event: AgentExecutionEvent, target: EventTarget) -> Dict[str, Any]:
        """Convert framework event to external format."""
        pass

    @abstractmethod
    def from_external(
        self,
        data: Dict[str, Any],
        target: EventTarget,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AgentExecutionEvent:
        """Convert external data to framework event."""
        pass

    def _base_data(self, event: AgentExecutionEvent) -> Dict[str, Any]:
        """Extract common base data from event."""
        return {
            "timestamp": event.timestamp,
            "metadata": event.metadata.copy() if event.metadata else {},
        }


# =============================================================================
# Concrete Converters
# =============================================================================


class ContentEventConverter(BaseEventConverter):
    """Converter for CONTENT events."""

    @property
    def event_type(self) -> EventType:
        return EventType.CONTENT

    @property
    def external_type_names(self) -> Dict[EventTarget, List[str]]:
        return {
            EventTarget.CQRS: ["content_generated", "ContentGeneratedEvent"],
            EventTarget.OBSERVABILITY: ["content"],
            EventTarget.STREAM_CHUNK: ["content", "text"],
        }

    def to_external(self, event: AgentExecutionEvent, target: EventTarget) -> Dict[str, Any]:
        base = self._base_data(event)

        if target == EventTarget.CQRS:
            return {
                **base,
                "event_type": "content_generated",
                "content": event.content,
            }
        elif target == EventTarget.OBSERVABILITY:
            return {
                "category": "model",
                "name": "content",
                "data": {"content": event.content, **base["metadata"]},
                "priority": "normal",
            }
        return base

    def from_external(
        self,
        data: Dict[str, Any],
        target: EventTarget,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AgentExecutionEvent:
        return content_event(
            content=data.get("content", ""),
            metadata=metadata or data.get("metadata", {}),
        )


class ThinkingEventConverter(BaseEventConverter):
    """Converter for THINKING events."""

    @property
    def event_type(self) -> EventType:
        return EventType.THINKING

    @property
    def external_type_names(self) -> Dict[EventTarget, List[str]]:
        return {
            EventTarget.CQRS: ["thinking_generated", "ThinkingGeneratedEvent"],
            EventTarget.OBSERVABILITY: ["thinking"],
            EventTarget.STREAM_CHUNK: ["reasoning_content", "thinking"],
        }

    def to_external(self, event: AgentExecutionEvent, target: EventTarget) -> Dict[str, Any]:
        base = self._base_data(event)

        if target == EventTarget.CQRS:
            return {
                **base,
                "event_type": "thinking_generated",
                "reasoning_content": event.content,
            }
        elif target == EventTarget.OBSERVABILITY:
            return {
                "category": "model",
                "name": "thinking",
                "data": {"reasoning_content": event.content, **base["metadata"]},
                "priority": "normal",
            }
        return base

    def from_external(
        self,
        data: Dict[str, Any],
        target: EventTarget,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AgentExecutionEvent:
        return thinking_event(
            content=data.get("reasoning_content", data.get("content", "")),
            metadata=metadata or data.get("metadata", {}),
        )


class ToolCallEventConverter(BaseEventConverter):
    """Converter for TOOL_CALL events."""

    @property
    def event_type(self) -> EventType:
        return EventType.TOOL_CALL

    @property
    def external_type_names(self) -> Dict[EventTarget, List[str]]:
        return {
            EventTarget.CQRS: ["tool_called", "ToolCalledEvent"],
            EventTarget.OBSERVABILITY: ["tool.start"],
            EventTarget.STREAM_CHUNK: ["tool_start", "tool_call"],
        }

    def to_external(self, event: AgentExecutionEvent, target: EventTarget) -> Dict[str, Any]:
        base = self._base_data(event)

        if target == EventTarget.CQRS:
            return {
                **base,
                "event_type": "tool_called",
                "tool_name": event.tool_name,
                "tool_id": event.tool_id,
                "arguments": event.arguments,
            }
        elif target == EventTarget.OBSERVABILITY:
            return {
                "category": "tool",
                "name": f"{event.tool_name}.start",
                "data": {
                    "tool_name": event.tool_name,
                    "tool_id": event.tool_id,
                    "arguments": event.arguments,
                },
                "priority": "normal",
            }
        return base

    def from_external(
        self,
        data: Dict[str, Any],
        target: EventTarget,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AgentExecutionEvent:
        meta = metadata or data.get("metadata", {})
        return tool_call_event(
            tool_name=data.get("tool_name", "unknown"),
            tool_id=data.get("tool_id") or meta.get("tool_id"),
            arguments=data.get("arguments", {}),
            metadata=meta,
        )


class ToolResultEventConverter(BaseEventConverter):
    """Converter for TOOL_RESULT events."""

    @property
    def event_type(self) -> EventType:
        return EventType.TOOL_RESULT

    @property
    def external_type_names(self) -> Dict[EventTarget, List[str]]:
        return {
            EventTarget.CQRS: ["tool_result", "ToolResultEvent"],
            EventTarget.OBSERVABILITY: ["tool.end"],
            EventTarget.STREAM_CHUNK: ["tool_result"],
        }

    def to_external(self, event: AgentExecutionEvent, target: EventTarget) -> Dict[str, Any]:
        base = self._base_data(event)

        if target == EventTarget.CQRS:
            return {
                **base,
                "event_type": "tool_result",
                "tool_name": event.tool_name,
                "tool_id": event.tool_id,
                "result": event.result,
                "success": event.success,
            }
        elif target == EventTarget.OBSERVABILITY:
            return {
                "category": "tool",
                "name": f"{event.tool_name}.end",
                "data": {
                    "tool_name": event.tool_name,
                    "tool_id": event.tool_id,
                    "result": event.result,
                    "success": event.success,
                },
                "priority": "normal",
            }
        return base

    def from_external(
        self,
        data: Dict[str, Any],
        target: EventTarget,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AgentExecutionEvent:
        meta = metadata or data.get("metadata", {})
        return tool_result_event(
            tool_name=data.get("tool_name", "unknown"),
            tool_id=data.get("tool_id") or meta.get("tool_id"),
            result=str(data.get("result", "")),
            success=data.get("success", True),
            metadata=meta,
        )


class ToolErrorEventConverter(BaseEventConverter):
    """Converter for TOOL_ERROR events."""

    @property
    def event_type(self) -> EventType:
        return EventType.TOOL_ERROR

    @property
    def external_type_names(self) -> Dict[EventTarget, List[str]]:
        return {
            EventTarget.CQRS: ["tool_error", "ToolErrorEvent"],
            EventTarget.OBSERVABILITY: ["tool.error"],
        }

    def to_external(self, event: AgentExecutionEvent, target: EventTarget) -> Dict[str, Any]:
        base = self._base_data(event)

        if target == EventTarget.CQRS:
            return {
                **base,
                "event_type": "tool_error",
                "tool_name": event.tool_name,
                "tool_id": event.tool_id,
                "error": event.error,
            }
        elif target == EventTarget.OBSERVABILITY:
            return {
                "category": "error",
                "name": f"tool.{event.tool_name}.error",
                "data": {
                    "tool_name": event.tool_name,
                    "tool_id": event.tool_id,
                    "error": event.error,
                },
                "priority": "high",
            }
        return base

    def from_external(
        self,
        data: Dict[str, Any],
        target: EventTarget,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AgentExecutionEvent:
        meta = metadata or data.get("metadata", {})
        return tool_error_event(
            tool_name=data.get("tool_name", "unknown"),
            tool_id=data.get("tool_id") or meta.get("tool_id"),
            error=data.get("error", "Unknown error"),
            metadata=meta,
        )


class StageChangeEventConverter(BaseEventConverter):
    """Converter for STAGE_CHANGE events."""

    @property
    def event_type(self) -> EventType:
        return EventType.STAGE_CHANGE

    @property
    def external_type_names(self) -> Dict[EventTarget, List[str]]:
        return {
            EventTarget.CQRS: ["stage_changed", "StateChangedEvent"],
            EventTarget.OBSERVABILITY: ["stage_transition", "state.transition"],
        }

    def to_external(self, event: AgentExecutionEvent, target: EventTarget) -> Dict[str, Any]:
        base = self._base_data(event)

        if target == EventTarget.CQRS:
            return {
                **base,
                "event_type": "stage_changed",
                "old_stage": event.old_stage,
                "new_stage": event.new_stage,
            }
        elif target == EventTarget.OBSERVABILITY:
            return {
                "category": "state",
                "name": "stage_transition",
                "data": {
                    "old_stage": event.old_stage,
                    "new_stage": event.new_stage,
                },
                "priority": "normal",
            }
        return base

    def from_external(
        self,
        data: Dict[str, Any],
        target: EventTarget,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AgentExecutionEvent:
        return stage_change_event(
            old_stage=data.get("old_stage", data.get("from_state", "unknown")),
            new_stage=data.get("new_stage", data.get("to_state", "unknown")),
            metadata=metadata or data.get("metadata", {}),
        )


class StreamStartEventConverter(BaseEventConverter):
    """Converter for STREAM_START events."""

    @property
    def event_type(self) -> EventType:
        return EventType.STREAM_START

    @property
    def external_type_names(self) -> Dict[EventTarget, List[str]]:
        return {
            EventTarget.CQRS: ["stream_started", "TaskStartedEvent"],
            EventTarget.OBSERVABILITY: ["stream.start", "lifecycle.start"],
        }

    def to_external(self, event: AgentExecutionEvent, target: EventTarget) -> Dict[str, Any]:
        base = self._base_data(event)

        if target == EventTarget.CQRS:
            return {
                **base,
                "event_type": "stream_started",
            }
        elif target == EventTarget.OBSERVABILITY:
            return {
                "category": "lifecycle",
                "name": "stream.start",
                "data": base["metadata"],
                "priority": "normal",
            }
        return base

    def from_external(
        self,
        data: Dict[str, Any],
        target: EventTarget,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AgentExecutionEvent:
        meta = metadata or data.get("metadata", {})
        # Include task_id, provider, model if present
        for key in ["task_id", "provider", "model"]:
            if key in data and key not in meta:
                meta[key] = data[key]
        return stream_start_event(metadata=meta)


class StreamEndEventConverter(BaseEventConverter):
    """Converter for STREAM_END events."""

    @property
    def event_type(self) -> EventType:
        return EventType.STREAM_END

    @property
    def external_type_names(self) -> Dict[EventTarget, List[str]]:
        return {
            EventTarget.CQRS: [
                "stream_ended",
                "TaskCompletedEvent",
                "TaskFailedEvent",
            ],
            EventTarget.OBSERVABILITY: ["stream.end", "lifecycle.end"],
        }

    def to_external(self, event: AgentExecutionEvent, target: EventTarget) -> Dict[str, Any]:
        base = self._base_data(event)

        if target == EventTarget.CQRS:
            return {
                **base,
                "event_type": "stream_ended",
                "success": event.success,
                "error": event.error,
            }
        elif target == EventTarget.OBSERVABILITY:
            return {
                "category": "lifecycle",
                "name": "stream.end",
                "data": {
                    "success": event.success,
                    "error": event.error,
                },
                "priority": "normal",
            }
        return base

    def from_external(
        self,
        data: Dict[str, Any],
        target: EventTarget,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AgentExecutionEvent:
        # Handle TaskCompletedEvent vs TaskFailedEvent
        success = data.get("success", True)
        error = data.get("error")

        # Check for TaskFailedEvent pattern
        if "error_type" in data or (error and not success):
            success = False

        return stream_end_event(
            success=success,
            error=error,
            metadata=metadata or data.get("metadata", {}),
        )


class ErrorEventConverter(BaseEventConverter):
    """Converter for ERROR events."""

    @property
    def event_type(self) -> EventType:
        return EventType.ERROR

    @property
    def external_type_names(self) -> Dict[EventTarget, List[str]]:
        return {
            EventTarget.CQRS: ["error_occurred", "ErrorEvent"],
            EventTarget.OBSERVABILITY: ["error", "framework_error"],
        }

    def to_external(self, event: AgentExecutionEvent, target: EventTarget) -> Dict[str, Any]:
        base = self._base_data(event)

        if target == EventTarget.CQRS:
            return {
                **base,
                "event_type": "error_occurred",
                "error": event.error,
                "recoverable": event.recoverable,
            }
        elif target == EventTarget.OBSERVABILITY:
            return {
                "category": "error",
                "name": "framework_error",
                "data": {
                    "message": event.error,
                    "recoverable": event.recoverable,
                },
                "priority": "high" if not event.recoverable else "normal",
            }
        return base

    def from_external(
        self,
        data: Dict[str, Any],
        target: EventTarget,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AgentExecutionEvent:
        return error_event(
            error=data.get("error", data.get("message", "Unknown error")),
            recoverable=data.get("recoverable", True),
            metadata=metadata or data.get("metadata", {}),
        )


class ProgressEventConverter(BaseEventConverter):
    """Converter for PROGRESS events."""

    @property
    def event_type(self) -> EventType:
        return EventType.PROGRESS

    @property
    def external_type_names(self) -> Dict[EventTarget, List[str]]:
        return {
            EventTarget.CQRS: ["progress_updated", "ProgressEvent"],
            EventTarget.OBSERVABILITY: ["progress"],
        }

    def to_external(self, event: AgentExecutionEvent, target: EventTarget) -> Dict[str, Any]:
        base = self._base_data(event)

        if target == EventTarget.CQRS:
            return {
                **base,
                "event_type": "progress_updated",
                "progress": event.progress,
            }
        elif target == EventTarget.OBSERVABILITY:
            return {
                "category": "lifecycle",
                "name": "progress",
                "data": {"progress": event.progress},
                "priority": "low",
            }
        return base

    def from_external(
        self,
        data: Dict[str, Any],
        target: EventTarget,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AgentExecutionEvent:
        return progress_event(
            progress=data.get("progress", 0.0),
            metadata=metadata or data.get("metadata", {}),
        )


class MilestoneEventConverter(BaseEventConverter):
    """Converter for MILESTONE events."""

    @property
    def event_type(self) -> EventType:
        return EventType.MILESTONE

    @property
    def external_type_names(self) -> Dict[EventTarget, List[str]]:
        return {
            EventTarget.CQRS: ["milestone_reached", "MilestoneEvent"],
            EventTarget.OBSERVABILITY: ["milestone"],
        }

    def to_external(self, event: AgentExecutionEvent, target: EventTarget) -> Dict[str, Any]:
        base = self._base_data(event)

        if target == EventTarget.CQRS:
            return {
                **base,
                "event_type": "milestone_reached",
                "milestone": event.milestone,
            }
        elif target == EventTarget.OBSERVABILITY:
            return {
                "category": "lifecycle",
                "name": "milestone",
                "data": {"milestone": event.milestone},
                "priority": "normal",
            }
        return base

    def from_external(
        self,
        data: Dict[str, Any],
        target: EventTarget,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AgentExecutionEvent:
        return milestone_event(
            milestone=data.get("milestone", ""),
            metadata=metadata or data.get("metadata", {}),
        )


# =============================================================================
# Event Registry
# =============================================================================


class EventRegistry:
    """Central registry for event type converters.

    Provides unified bidirectional conversion between framework events
    and external systems (CQRS, Observability).

    Design:
    - Singleton pattern for global access
    - Extensible via register_converter()
    - Type-safe via protocol checking

    Example:
        registry = EventRegistry()

        # Convert framework event to CQRS
        cqrs_data = registry.to_external(event, EventTarget.CQRS)

        # Convert CQRS data back to framework event
        event = registry.from_external(cqrs_data, "tool_called", EventTarget.CQRS)
    """

    _instance: Optional["EventRegistry"] = None

    def __init__(self) -> None:
        """Initialize the registry with built-in converters."""
        # Map EventType -> Converter
        self._converters: Dict[EventType, BaseEventConverter] = {}

        # Map (target, external_type_name) -> Converter for reverse lookup
        self._reverse_map: Dict[tuple[EventTarget, str], BaseEventConverter] = {}

        # Register built-in converters
        self._register_builtin_converters()

    @classmethod
    def get_instance(cls) -> "EventRegistry":
        """Get the singleton instance.

        Returns:
            The global EventRegistry instance.
        """
        if cls._instance is None:
            cls._instance = EventRegistry()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance (for testing)."""
        cls._instance = None

    def _register_builtin_converters(self) -> None:
        """Register all built-in event converters."""
        converters: List[BaseEventConverter] = [
            ContentEventConverter(),
            ThinkingEventConverter(),
            ToolCallEventConverter(),
            ToolResultEventConverter(),
            ToolErrorEventConverter(),
            StageChangeEventConverter(),
            StreamStartEventConverter(),
            StreamEndEventConverter(),
            ErrorEventConverter(),
            ProgressEventConverter(),
            MilestoneEventConverter(),
        ]

        for converter in converters:
            self.register_converter(converter)

    def register_converter(self, converter: BaseEventConverter) -> None:
        """Register a converter for an event type.

        Args:
            converter: The converter to register.

        Raises:
            ValueError: If converter doesn't implement the protocol.
        """
        if not isinstance(converter, EventConverterProtocol):
            raise ValueError(f"Converter must implement EventConverterProtocol: {type(converter)}")

        # Register forward mapping
        self._converters[converter.event_type] = converter

        # Register reverse mappings
        for target, type_names in converter.external_type_names.items():
            for type_name in type_names:
                self._reverse_map[(target, type_name)] = converter

    def to_external(self, event: AgentExecutionEvent, target: EventTarget) -> Dict[str, Any]:
        """Convert a framework event to external format.

        Args:
            event: Framework Event to convert.
            target: Target system for conversion.

        Returns:
            Dictionary suitable for the target system.

        Raises:
            ValueError: If no converter exists for the event type.
        """
        converter = self._converters.get(event.type)

        if converter is None:
            # Fallback: generic conversion
            logger.warning(f"No converter for event type: {event.type}, using fallback")
            return self._fallback_to_external(event, target)

        return converter.to_external(event, target)

    def from_external(
        self,
        data: Dict[str, Any],
        external_type: str,
        target: EventTarget,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AgentExecutionEvent:
        """Convert external data to a framework event.

        Args:
            data: External event data.
            external_type: The type name from the external system.
            target: Source system of the data.
            metadata: Additional metadata from the external event.

        Returns:
            Framework Event instance.

        Raises:
            ValueError: If no converter exists for the external type.
        """
        converter = self._reverse_map.get((target, external_type))

        if converter is None:
            # Try to find a partial match (e.g., "ToolCalledEvent" matches "tool_called")
            converter = self._find_converter_fuzzy(external_type, target)

        if converter is None:
            # Fallback: generic conversion
            logger.warning(
                f"No converter for external type: {external_type} "
                f"(target={target}), using fallback"
            )
            return self._fallback_from_external(data, external_type, metadata)

        return converter.from_external(data, target, metadata)

    def _find_converter_fuzzy(
        self, external_type: str, target: EventTarget
    ) -> Optional[BaseEventConverter]:
        """Find a converter using fuzzy matching.

        Handles cases where the external type name doesn't exactly match
        (e.g., class names vs string identifiers).
        """
        normalized = external_type.lower().replace("_", "").replace("event", "")

        # Empty string matches everything - that's not a valid fuzzy match
        if not normalized:
            return None

        for (t, name), converter in self._reverse_map.items():
            if t != target:
                continue
            name_normalized = name.lower().replace("_", "").replace("event", "")
            if normalized == name_normalized or normalized in name_normalized:
                return converter

        return None

    def _fallback_to_external(
        self, event: AgentExecutionEvent, target: EventTarget
    ) -> Dict[str, Any]:
        """Fallback conversion for unknown event types."""
        base = {
            "timestamp": event.timestamp,
            "metadata": event.metadata.copy() if event.metadata else {},
        }

        if target == EventTarget.CQRS:
            return {
                **base,
                "event_type": event.type.value,
                "content": event.content,
            }
        elif target == EventTarget.OBSERVABILITY:
            return {
                "category": "custom",
                "name": event.type.value,
                "data": {"content": event.content, **base["metadata"]},
                "priority": "normal",
            }
        return base

    def _fallback_from_external(
        self,
        data: Dict[str, Any],
        external_type: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AgentExecutionEvent:
        """Fallback conversion for unknown external types."""
        return content_event(
            content=str(data.get("content", data)),
            metadata={"original_type": external_type, **(metadata or {})},
        )

    def get_converter(self, event_type: EventType) -> Optional[BaseEventConverter]:
        """Get the converter for a specific event type.

        Args:
            event_type: The framework EventType.

        Returns:
            The registered converter or None.
        """
        return self._converters.get(event_type)

    def list_supported_types(self) -> List[EventType]:
        """List all supported framework event types.

        Returns:
            List of EventType values with registered converters.
        """
        return list(self._converters.keys())

    def list_external_types(self, target: EventTarget) -> List[str]:
        """List all external type names for a target.

        Args:
            target: The target system.

        Returns:
            List of recognizable external type names.
        """
        return [name for (t, name) in self._reverse_map.keys() if t == target]


# =============================================================================
# Module-Level Functions
# =============================================================================


def get_event_registry() -> EventRegistry:
    """Get the global event registry instance.

    Returns:
        The singleton EventRegistry.
    """
    return EventRegistry.get_instance()


def convert_to_cqrs(event: AgentExecutionEvent) -> Dict[str, Any]:
    """Convenience function to convert event to CQRS format.

    Args:
        event: Framework AgentExecutionEvent.

    Returns:
        CQRS-compatible dictionary.
    """
    return get_event_registry().to_external(event, EventTarget.CQRS)


def convert_from_cqrs(
    data: Dict[str, Any],
    external_type: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> AgentExecutionEvent:
    """Convenience function to convert CQRS data to event.

    Args:
        data: CQRS event data.
        external_type: The CQRS event type name.
        metadata: Additional metadata.

    Returns:
        Framework AgentExecutionEvent.
    """
    return get_event_registry().from_external(data, external_type, EventTarget.CQRS, metadata)


def convert_to_observability(event: AgentExecutionEvent) -> Dict[str, Any]:
    """Convenience function to convert event to observability format.

    Args:
        event: Framework AgentExecutionEvent.

    Returns:
        Observability-compatible dictionary.
    """
    return get_event_registry().to_external(event, EventTarget.OBSERVABILITY)


def convert_from_observability(
    data: Dict[str, Any],
    external_type: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> AgentExecutionEvent:
    """Convenience function to convert observability data to event.

    Args:
        data: Observability event data.
        external_type: The observability event name.
        metadata: Additional metadata.

    Returns:
        Framework AgentExecutionEvent.
    """
    return get_event_registry().from_external(
        data, external_type, EventTarget.OBSERVABILITY, metadata
    )
