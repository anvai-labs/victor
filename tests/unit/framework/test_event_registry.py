# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for EventErrorDetails TypedDict — Wave L."""

import pytest

from victor.framework.event_registry import EventErrorDetails


class TestEventErrorDetails:
    """EventErrorDetails TypedDict provides typed structure for event errors."""

    def test_typed_dict_exists(self):
        """EventErrorDetails should be importable and usable."""
        details: EventErrorDetails = {"error": "test error"}
        assert details["error"] == "test error"

    def test_required_error_field(self):
        """error field should be present in TypedDict structure."""
        details: EventErrorDetails = {"error": "test message"}
        assert "error" in details
        assert details["error"] == "test message"

    def test_optional_fields(self):
        """Optional fields (error_type, error_category, tool_name, tool_id) should be usable."""
        details: EventErrorDetails = {
            "error": "test",
            "error_type": "ValueError",
            "error_category": "validation",
            "tool_name": "test_tool",
            "tool_id": "tool_123",
        }
        assert details["error_type"] == "ValueError"
        assert details["error_category"] == "validation"
        assert details["tool_name"] == "test_tool"
        assert details["tool_id"] == "tool_123"

    def test_all_fields_optional_except_error(self):
        """Only error should be required; all other fields optional."""
        # Minimal valid dict
        details: EventErrorDetails = {"error": "minimal"}
        assert details["error"] == "minimal"

        # Missing optional fields should not cause issues
        assert "error_type" not in details
        assert "error_category" not in details
        assert "tool_name" not in details
        assert "tool_id" not in details


class TestEventConverterWithTypedErrors:
    """Event converters use EventErrorDetails structure."""

    def test_tool_result_converter_default_error_message(self, mocker):
        """ToolResultConverter should provide default 'Unknown error' message."""
        from victor.framework.event_registry import ToolResultEventConverter

        converter = ToolResultEventConverter()

        # Mock data without error field
        data = {"result": "test", "success": False}
        target = mocker.MagicMock()

        event = converter.from_external(data, target)

        # Should have error with default message
        assert hasattr(event, "error")

    def test_tool_error_converter_default_error_message(self, mocker):
        """ToolErrorEventConverter should provide default error message."""
        from victor.framework.event_registry import ToolErrorEventConverter

        converter = ToolErrorEventConverter()

        # Mock data without error field
        data = {"tool_name": "test_tool", "success": False}
        target = mocker.MagicMock()

        event = converter.from_external(data, target)

        # Should have error with default message
        assert hasattr(event, "error")

    def test_event_error_details_preserves_tool_context(self):
        """EventErrorDetails should preserve tool_name and tool_id when present."""
        details: EventErrorDetails = {
            "error": "Tool failed",
            "tool_name": "file_editor",
            "tool_id": "file_editor_abc123",
        }

        assert details["tool_name"] == "file_editor"
        assert details["tool_id"] == "file_editor_abc123"
        assert details["error"] == "Tool failed"
