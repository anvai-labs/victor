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

"""Tests for EventRegistry - registry-based event conversion.

Tests cover:
1. Registry singleton pattern
2. Forward conversion (framework -> external)
3. Reverse conversion (external -> framework)
4. All event types
5. Fallback handling
6. Custom converter registration
"""

import pytest

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
from victor.framework.event_registry import (
    BaseEventConverter,
    ContentEventConverter,
    ErrorEventConverter,
    EventRegistry,
    EventTarget,
    MilestoneEventConverter,
    ProgressEventConverter,
    StageChangeEventConverter,
    StreamEndEventConverter,
    StreamStartEventConverter,
    ThinkingEventConverter,
    ToolCallEventConverter,
    ToolErrorEventConverter,
    ToolResultEventConverter,
    convert_from_cqrs,
    convert_from_observability,
    convert_to_cqrs,
    convert_to_observability,
    get_event_registry,
)


class TestEventRegistrySingleton:
    """Tests for EventRegistry singleton pattern."""

    @pytest.fixture(autouse=True)
    def reset_registry(self):
        """Reset registry singleton before each test."""
        EventRegistry.reset_instance()
        yield
        EventRegistry.reset_instance()

    def test_get_instance_returns_same_instance(self):
        """get_instance should return the same instance."""
        registry1 = EventRegistry.get_instance()
        registry2 = EventRegistry.get_instance()
        assert registry1 is registry2

    def test_reset_instance_creates_new_instance(self):
        """reset_instance should clear the singleton."""
        registry1 = EventRegistry.get_instance()
        EventRegistry.reset_instance()
        registry2 = EventRegistry.get_instance()
        assert registry1 is not registry2

    def test_get_event_registry_returns_singleton(self):
        """get_event_registry should return the singleton."""
        registry1 = get_event_registry()
        registry2 = get_event_registry()
        assert registry1 is registry2


class TestEventRegistryConverters:
    """Tests for built-in converters."""

    @pytest.fixture
    def registry(self):
        """Create fresh registry for testing."""
        EventRegistry.reset_instance()
        return EventRegistry.get_instance()

    def test_all_event_types_have_converters(self, registry):
        """All EventTypes should have registered converters."""
        # These are the event types we support
        supported_types = [
            EventType.CONTENT,
            EventType.THINKING,
            EventType.TOOL_CALL,
            EventType.TOOL_RESULT,
            EventType.TOOL_ERROR,
            EventType.STAGE_CHANGE,
            EventType.STREAM_START,
            EventType.STREAM_END,
            EventType.ERROR,
            EventType.PROGRESS,
            EventType.MILESTONE,
        ]

        for event_type in supported_types:
            converter = registry.get_converter(event_type)
            assert converter is not None, f"No converter for {event_type}"

    def test_list_supported_types(self, registry):
        """list_supported_types should return all registered types."""
        supported = registry.list_supported_types()
        assert EventType.CONTENT in supported
        assert EventType.TOOL_CALL in supported
        assert EventType.ERROR in supported

    def test_list_external_types_cqrs(self, registry):
        """list_external_types should return CQRS type names."""
        external = registry.list_external_types(EventTarget.CQRS)
        assert "content_generated" in external
        assert "tool_called" in external
        assert "error_occurred" in external

    def test_list_external_types_observability(self, registry):
        """list_external_types should return observability type names."""
        external = registry.list_external_types(EventTarget.OBSERVABILITY)
        assert "content" in external
        assert "stage_transition" in external


class TestContentEventConversion:
    """Tests for CONTENT event conversion."""

    @pytest.fixture
    def registry(self):
        EventRegistry.reset_instance()
        return EventRegistry.get_instance()

    def test_to_cqrs(self, registry):
        """CONTENT event should convert to CQRS format."""
        event = content_event("Hello world")
        result = registry.to_external(event, EventTarget.CQRS)

        assert result["event_type"] == "content_generated"
        assert result["content"] == "Hello world"
        assert "timestamp" in result

    def test_to_observability(self, registry):
        """CONTENT event should convert to observability format."""
        event = content_event("Hello world")
        result = registry.to_external(event, EventTarget.OBSERVABILITY)

        assert result["category"] == "model"
        assert result["name"] == "content"
        assert result["data"]["content"] == "Hello world"

    def test_from_cqrs(self, registry):
        """CQRS content_generated should convert to CONTENT event."""
        data = {"content": "Hello world"}
        event = registry.from_external(data, "content_generated", EventTarget.CQRS)

        assert event.type == EventType.CONTENT
        assert event.content == "Hello world"

    def test_from_observability(self, registry):
        """Observability content event should convert to CONTENT event."""
        data = {"content": "Hello world"}
        event = registry.from_external(data, "content", EventTarget.OBSERVABILITY)

        assert event.type == EventType.CONTENT
        assert event.content == "Hello world"


class TestThinkingEventConversion:
    """Tests for THINKING event conversion."""

    @pytest.fixture
    def registry(self):
        EventRegistry.reset_instance()
        return EventRegistry.get_instance()

    def test_to_cqrs(self, registry):
        """THINKING event should convert to CQRS format."""
        event = thinking_event("Reasoning about the problem...")
        result = registry.to_external(event, EventTarget.CQRS)

        assert result["event_type"] == "thinking_generated"
        assert result["reasoning_content"] == "Reasoning about the problem..."

    def test_from_cqrs(self, registry):
        """CQRS thinking_generated should convert to THINKING event."""
        data = {"reasoning_content": "Reasoning about the problem..."}
        event = registry.from_external(data, "thinking_generated", EventTarget.CQRS)

        assert event.type == EventType.THINKING
        assert event.content == "Reasoning about the problem..."


class TestToolCallEventConversion:
    """Tests for TOOL_CALL event conversion."""

    @pytest.fixture
    def registry(self):
        EventRegistry.reset_instance()
        return EventRegistry.get_instance()

    def test_to_cqrs(self, registry):
        """TOOL_CALL event should convert to CQRS format."""
        event = tool_call_event(
            tool_name="read",
            tool_id="abc123",
            arguments={"path": "/tmp/test.txt"},
        )
        result = registry.to_external(event, EventTarget.CQRS)

        assert result["event_type"] == "tool_called"
        assert result["tool_name"] == "read"
        assert result["tool_id"] == "abc123"
        assert result["arguments"]["path"] == "/tmp/test.txt"

    def test_to_observability(self, registry):
        """TOOL_CALL event should convert to observability format."""
        event = tool_call_event(
            tool_name="read",
            tool_id="abc123",
            arguments={"path": "/tmp/test.txt"},
        )
        result = registry.to_external(event, EventTarget.OBSERVABILITY)

        assert result["category"] == "tool"
        assert result["name"] == "read.start"
        assert result["data"]["tool_name"] == "read"

    def test_from_cqrs(self, registry):
        """CQRS tool_called should convert to TOOL_CALL event."""
        data = {
            "tool_name": "read",
            "tool_id": "abc123",
            "arguments": {"path": "/tmp/test.txt"},
        }
        event = registry.from_external(data, "tool_called", EventTarget.CQRS)

        assert event.type == EventType.TOOL_CALL
        assert event.tool_name == "read"
        assert event.tool_id == "abc123"
        assert event.arguments["path"] == "/tmp/test.txt"

    def test_from_cqrs_class_name(self, registry):
        """CQRS ToolCalledEvent (class name) should convert to TOOL_CALL."""
        data = {
            "tool_name": "write",
            "arguments": {"path": "/tmp/out.txt", "content": "data"},
        }
        event = registry.from_external(data, "ToolCalledEvent", EventTarget.CQRS)

        assert event.type == EventType.TOOL_CALL
        assert event.tool_name == "write"


class TestToolResultEventConversion:
    """Tests for TOOL_RESULT event conversion."""

    @pytest.fixture
    def registry(self):
        EventRegistry.reset_instance()
        return EventRegistry.get_instance()

    def test_to_cqrs(self, registry):
        """TOOL_RESULT event should convert to CQRS format."""
        event = tool_result_event(
            tool_name="read",
            tool_id="abc123",
            result="file contents here",
            success=True,
        )
        result = registry.to_external(event, EventTarget.CQRS)

        assert result["event_type"] == "tool_result"
        assert result["tool_name"] == "read"
        assert result["result"] == "file contents here"
        assert result["success"] is True

    def test_to_observability(self, registry):
        """TOOL_RESULT event should convert to observability format."""
        event = tool_result_event(
            tool_name="read",
            tool_id="abc123",
            result="file contents",
            success=True,
        )
        result = registry.to_external(event, EventTarget.OBSERVABILITY)

        assert result["category"] == "tool"
        assert result["name"] == "read.end"
        assert result["data"]["success"] is True

    def test_from_cqrs(self, registry):
        """CQRS tool_result should convert to TOOL_RESULT event."""
        data = {
            "tool_name": "read",
            "tool_id": "abc123",
            "result": "file contents",
            "success": True,
        }
        event = registry.from_external(data, "tool_result", EventTarget.CQRS)

        assert event.type == EventType.TOOL_RESULT
        assert event.tool_name == "read"
        assert event.result == "file contents"
        assert event.success is True


class TestToolErrorEventConversion:
    """Tests for TOOL_ERROR event conversion."""

    @pytest.fixture
    def registry(self):
        EventRegistry.reset_instance()
        return EventRegistry.get_instance()

    def test_to_cqrs(self, registry):
        """TOOL_ERROR event should convert to CQRS format."""
        event = tool_error_event(
            tool_name="read",
            tool_id="abc123",
            error="File not found",
        )
        result = registry.to_external(event, EventTarget.CQRS)

        assert result["event_type"] == "tool_error"
        assert result["tool_name"] == "read"
        assert result["error"] == "File not found"

    def test_to_observability(self, registry):
        """TOOL_ERROR event should convert to observability format."""
        event = tool_error_event(
            tool_name="read",
            tool_id="abc123",
            error="File not found",
        )
        result = registry.to_external(event, EventTarget.OBSERVABILITY)

        assert result["category"] == "error"
        assert result["priority"] == "high"


class TestStageChangeEventConversion:
    """Tests for STAGE_CHANGE event conversion."""

    @pytest.fixture
    def registry(self):
        EventRegistry.reset_instance()
        return EventRegistry.get_instance()

    def test_to_cqrs(self, registry):
        """STAGE_CHANGE event should convert to CQRS format."""
        event = stage_change_event(old_stage="initial", new_stage="planning")
        result = registry.to_external(event, EventTarget.CQRS)

        assert result["event_type"] == "stage_changed"
        assert result["old_stage"] == "initial"
        assert result["new_stage"] == "planning"

    def test_to_observability(self, registry):
        """STAGE_CHANGE event should convert to observability format."""
        event = stage_change_event(old_stage="initial", new_stage="planning")
        result = registry.to_external(event, EventTarget.OBSERVABILITY)

        assert result["category"] == "state"
        assert result["name"] == "stage_transition"

    def test_from_cqrs_with_from_to_state(self, registry):
        """CQRS StateChangedEvent with from_state/to_state should convert."""
        data = {"from_state": "initial", "to_state": "planning"}
        event = registry.from_external(data, "StateChangedEvent", EventTarget.CQRS)

        assert event.type == EventType.STAGE_CHANGE
        assert event.old_stage == "initial"
        assert event.new_stage == "planning"


class TestStreamStartEventConversion:
    """Tests for STREAM_START event conversion."""

    @pytest.fixture
    def registry(self):
        EventRegistry.reset_instance()
        return EventRegistry.get_instance()

    def test_to_cqrs(self, registry):
        """STREAM_START event should convert to CQRS format."""
        event = stream_start_event(metadata={"provider": "anthropic"})
        result = registry.to_external(event, EventTarget.CQRS)

        assert result["event_type"] == "stream_started"

    def test_to_observability(self, registry):
        """STREAM_START event should convert to observability format."""
        event = stream_start_event()
        result = registry.to_external(event, EventTarget.OBSERVABILITY)

        assert result["category"] == "lifecycle"
        assert result["name"] == "stream.start"


class TestStreamEndEventConversion:
    """Tests for STREAM_END event conversion."""

    @pytest.fixture
    def registry(self):
        EventRegistry.reset_instance()
        return EventRegistry.get_instance()

    def test_to_cqrs_success(self, registry):
        """STREAM_END (success) should convert to CQRS format."""
        event = stream_end_event(success=True)
        result = registry.to_external(event, EventTarget.CQRS)

        assert result["event_type"] == "stream_ended"
        assert result["success"] is True

    def test_to_cqrs_failure(self, registry):
        """STREAM_END (failure) should convert to CQRS format."""
        event = stream_end_event(success=False, error="Timeout")
        result = registry.to_external(event, EventTarget.CQRS)

        assert result["event_type"] == "stream_ended"
        assert result["success"] is False
        assert result["error"] == "Timeout"

    def test_from_cqrs_task_completed(self, registry):
        """CQRS TaskCompletedEvent should convert to STREAM_END."""
        data = {"task_id": "123", "result": "done", "duration_ms": 1000}
        event = registry.from_external(data, "TaskCompletedEvent", EventTarget.CQRS)

        assert event.type == EventType.STREAM_END
        assert event.success is True

    def test_from_cqrs_task_failed(self, registry):
        """CQRS TaskFailedEvent should convert to STREAM_END (failure)."""
        data = {"task_id": "123", "error": "Timeout", "error_type": "TimeoutError"}
        event = registry.from_external(data, "TaskFailedEvent", EventTarget.CQRS)

        assert event.type == EventType.STREAM_END
        assert event.success is False


class TestErrorEventConversion:
    """Tests for ERROR event conversion."""

    @pytest.fixture
    def registry(self):
        EventRegistry.reset_instance()
        return EventRegistry.get_instance()

    def test_to_cqrs(self, registry):
        """ERROR event should convert to CQRS format."""
        event = error_event(error="Something went wrong", recoverable=False)
        result = registry.to_external(event, EventTarget.CQRS)

        assert result["event_type"] == "error_occurred"
        assert result["error"] == "Something went wrong"
        assert result["recoverable"] is False

    def test_to_observability_non_recoverable(self, registry):
        """Non-recoverable ERROR should have high priority."""
        event = error_event(error="Critical failure", recoverable=False)
        result = registry.to_external(event, EventTarget.OBSERVABILITY)

        assert result["category"] == "error"
        assert result["priority"] == "high"

    def test_to_observability_recoverable(self, registry):
        """Recoverable ERROR should have normal priority."""
        event = error_event(error="Minor issue", recoverable=True)
        result = registry.to_external(event, EventTarget.OBSERVABILITY)

        assert result["priority"] == "normal"


class TestProgressEventConversion:
    """Tests for PROGRESS event conversion."""

    @pytest.fixture
    def registry(self):
        EventRegistry.reset_instance()
        return EventRegistry.get_instance()

    def test_to_cqrs(self, registry):
        """PROGRESS event should convert to CQRS format."""
        event = progress_event(progress=0.5)
        result = registry.to_external(event, EventTarget.CQRS)

        assert result["event_type"] == "progress_updated"
        assert result["progress"] == 0.5

    def test_from_cqrs(self, registry):
        """CQRS progress_updated should convert to PROGRESS event."""
        data = {"progress": 0.75}
        event = registry.from_external(data, "progress_updated", EventTarget.CQRS)

        assert event.type == EventType.PROGRESS
        assert event.progress == 0.75


class TestMilestoneEventConversion:
    """Tests for MILESTONE event conversion."""

    @pytest.fixture
    def registry(self):
        EventRegistry.reset_instance()
        return EventRegistry.get_instance()

    def test_to_cqrs(self, registry):
        """MILESTONE event should convert to CQRS format."""
        event = milestone_event(milestone="Phase 1 complete")
        result = registry.to_external(event, EventTarget.CQRS)

        assert result["event_type"] == "milestone_reached"
        assert result["milestone"] == "Phase 1 complete"

    def test_from_cqrs(self, registry):
        """CQRS milestone_reached should convert to MILESTONE event."""
        data = {"milestone": "Tests passing"}
        event = registry.from_external(data, "milestone_reached", EventTarget.CQRS)

        assert event.type == EventType.MILESTONE
        assert event.milestone == "Tests passing"


class TestFallbackConversion:
    """Tests for fallback handling of unknown types."""

    @pytest.fixture
    def registry(self):
        EventRegistry.reset_instance()
        return EventRegistry.get_instance()

    def test_unknown_external_type_uses_fallback(self, registry):
        """Unknown external type should use fallback."""
        data = {"content": "some data", "custom_field": "value"}
        event = registry.from_external(data, "unknown_type", EventTarget.CQRS)

        # Should fall back to CONTENT event
        assert event.type == EventType.CONTENT
        assert "original_type" in event.metadata

    def test_fuzzy_matching_finds_converter(self, registry):
        """Fuzzy matching should find converters with similar names."""
        data = {"content": "Hello"}
        # "ContentGeneratedEvent" should fuzzy match "content_generated"
        event = registry.from_external(data, "ContentGeneratedEvent", EventTarget.CQRS)

        assert event.type == EventType.CONTENT


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    @pytest.fixture(autouse=True)
    def reset_registry(self):
        EventRegistry.reset_instance()
        yield
        EventRegistry.reset_instance()

    def test_convert_to_cqrs(self):
        """convert_to_cqrs should use registry."""
        event = content_event("Hello")
        result = convert_to_cqrs(event)

        assert result["event_type"] == "content_generated"
        assert result["content"] == "Hello"

    def test_convert_from_cqrs(self):
        """convert_from_cqrs should use registry."""
        data = {"content": "Hello"}
        event = convert_from_cqrs(data, "content_generated")

        assert event.type == EventType.CONTENT
        assert event.content == "Hello"

    def test_convert_to_observability(self):
        """convert_to_observability should use registry."""
        event = content_event("Hello")
        result = convert_to_observability(event)

        assert result["category"] == "model"
        assert result["data"]["content"] == "Hello"

    def test_convert_from_observability(self):
        """convert_from_observability should use registry."""
        data = {"content": "Hello"}
        event = convert_from_observability(data, "content")

        assert event.type == EventType.CONTENT


class TestRoundTripConversion:
    """Tests for round-trip conversion (framework -> external -> framework)."""

    @pytest.fixture
    def registry(self):
        EventRegistry.reset_instance()
        return EventRegistry.get_instance()

    def test_content_roundtrip_cqrs(self, registry):
        """CONTENT event should survive CQRS roundtrip."""
        original = content_event("Hello world", metadata={"key": "value"})
        cqrs_data = registry.to_external(original, EventTarget.CQRS)
        restored = registry.from_external(cqrs_data, cqrs_data["event_type"], EventTarget.CQRS)

        assert restored.type == original.type
        assert restored.content == original.content

    def test_tool_call_roundtrip_cqrs(self, registry):
        """TOOL_CALL event should survive CQRS roundtrip."""
        original = tool_call_event(
            tool_name="read",
            tool_id="abc123",
            arguments={"path": "/tmp/test.txt"},
        )
        cqrs_data = registry.to_external(original, EventTarget.CQRS)
        restored = registry.from_external(cqrs_data, cqrs_data["event_type"], EventTarget.CQRS)

        assert restored.type == original.type
        assert restored.tool_name == original.tool_name
        assert restored.tool_id == original.tool_id
        assert restored.arguments == original.arguments

    def test_stage_change_roundtrip_cqrs(self, registry):
        """STAGE_CHANGE event should survive CQRS roundtrip."""
        original = stage_change_event(old_stage="initial", new_stage="planning")
        cqrs_data = registry.to_external(original, EventTarget.CQRS)
        restored = registry.from_external(cqrs_data, cqrs_data["event_type"], EventTarget.CQRS)

        assert restored.type == original.type
        assert restored.old_stage == original.old_stage
        assert restored.new_stage == original.new_stage

    def test_error_roundtrip_cqrs(self, registry):
        """ERROR event should survive CQRS roundtrip."""
        original = error_event(error="Something broke", recoverable=False)
        cqrs_data = registry.to_external(original, EventTarget.CQRS)
        restored = registry.from_external(cqrs_data, cqrs_data["event_type"], EventTarget.CQRS)

        assert restored.type == original.type
        assert restored.error == original.error
        assert restored.recoverable == original.recoverable


class TestCustomConverterRegistration:
    """Tests for custom converter registration."""

    @pytest.fixture
    def registry(self):
        EventRegistry.reset_instance()
        return EventRegistry()

    def test_register_custom_converter(self, registry):
        """Custom converters should be registrable."""

        class CustomConverter(BaseEventConverter):
            @property
            def event_type(self) -> EventType:
                return EventType.CONTENT

            @property
            def external_type_names(self):
                return {EventTarget.CQRS: ["custom_content"]}

            def to_external(self, event, target):
                return {"event_type": "custom_content", "data": event.content}

            def from_external(self, data, target, metadata=None):
                return content_event(data.get("data", ""))

        # Override the default content converter
        registry.register_converter(CustomConverter())

        event = content_event("test")
        result = registry.to_external(event, EventTarget.CQRS)

        assert result["event_type"] == "custom_content"
        assert result["data"] == "test"
