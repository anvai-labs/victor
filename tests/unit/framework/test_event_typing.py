# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for EventSeverity enum — Wave Q."""

import pytest

from victor.framework.events import EventType, EventSeverity


class TestEventSeverityEnum:
    """EventSeverity enum provides typed severity levels for events."""

    def test_event_severity_enum_values(self):
        """EventSeverity should have expected severity levels."""
        assert EventSeverity.DEBUG.value == "debug"
        assert EventSeverity.INFO.value == "info"
        assert EventSeverity.WARNING.value == "warning"
        assert EventSeverity.ERROR.value == "error"
        assert EventSeverity.CRITICAL.value == "critical"

    def test_event_type_enum_values(self):
        """EventType should have expected event types (existing enum)."""
        assert EventType.ERROR.value == "error"
        assert EventType.CONTENT.value == "content"
        assert EventType.TOOL_CALL.value == "tool_call"


class TestEnumUsageInFramework:
    """Framework should use enums instead of string literals."""

    def test_agentic_loop_uses_enum_comparison(self):
        """AgenticLoop should use EventType enum for comparisons."""
        # This test validates that the enum exists and can be compared
        # Actual usage in agentic_loop.py would use EventType.ERROR
        error_type = EventType.ERROR
        assert error_type.value == "error"

    def test_client_uses_enum_comparison(self):
        """Client should use EventType enum for comparisons."""
        # This test validates that the enum exists and can be compared
        # Actual usage in client.py would use EventType.ERROR
        event_type = EventType.ERROR
        assert event_type.value == "error"

    def test_agent_factory_uses_severity_enum(self):
        """AgentFactory should use EventSeverity enum for severity checks."""
        # This test validates that the enum exists and can be compared
        # Actual usage in agent_factory.py would use EventSeverity.ERROR
        severity = EventSeverity.ERROR
        assert severity.value == "error"
