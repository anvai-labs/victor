# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for ASI trace enrichment middleware."""

import pytest

from victor.observability.analytics.trace_enrichment import (
    TraceEnricher,
    create_trace_enricher,
)


class MockLogger:
    """Minimal mock for UsageLogger."""

    def __init__(self):
        self.session_id = "test-session"
        self.events = []

    def log_event(self, event_type, data):
        self.events.append({"event_type": event_type, "data": data})


class TestTraceEnricher:
    def test_passthrough_when_disabled(self):
        inner = MockLogger()
        enricher = TraceEnricher(inner, enabled=False)
        enricher.log_event("tool_call", {"tool_name": "read"})
        assert len(inner.events) == 1
        assert "reasoning_before_call" not in inner.events[0]["data"]

    def test_enriches_tool_call_with_reasoning(self):
        inner = MockLogger()
        enricher = TraceEnricher(inner, enabled=True)
        enricher.set_reasoning_context("I need to check the file...")
        enricher.log_event(
            "tool_call",
            {"tool_name": "read", "tool_args": {"path": "foo.py"}},
        )
        data = inner.events[0]["data"]
        assert data["reasoning_before_call"] == "I need to check the file..."
        assert "arguments_sanitized" in data

    def test_reasoning_consumed_after_use(self):
        inner = MockLogger()
        enricher = TraceEnricher(inner, enabled=True)
        enricher.set_reasoning_context("First reasoning")
        enricher.log_event("tool_call", {"tool_name": "read"})
        enricher.log_event("tool_call", {"tool_name": "edit"})
        # Second call should NOT have reasoning (it was consumed)
        assert "reasoning_before_call" in inner.events[0]["data"]
        assert "reasoning_before_call" not in inner.events[1]["data"]

    def test_enriches_tool_result_with_duration(self):
        inner = MockLogger()
        enricher = TraceEnricher(inner, enabled=True)
        enricher.set_duration_context(45.2)
        enricher.log_event(
            "tool_result",
            {"tool_name": "read", "success": True, "result": "file contents"},
        )
        data = inner.events[0]["data"]
        assert data["duration_ms"] == 45.2
        assert "result_summary" in data

    def test_enriches_tool_result_error_detail(self):
        inner = MockLogger()
        enricher = TraceEnricher(inner, enabled=True)
        enricher.log_event(
            "tool_result",
            {
                "tool_name": "edit",
                "success": False,
                "error": "old_str not found in file",
            },
        )
        data = inner.events[0]["data"]
        assert data["error_detail"] == "old_str not found in file"

    def test_enriches_llm_response_with_thinking(self):
        inner = MockLogger()
        enricher = TraceEnricher(inner, enabled=True)
        enricher.log_event(
            "llm_response",
            {"content": "<think>Let me analyze...</think>Here is my answer."},
        )
        data = inner.events[0]["data"]
        assert "thinking_content" in data
        assert "analyze" in data["thinking_content"]
        assert "response_summary" in data
        assert "Here is my answer" in data["response_summary"]

    def test_truncates_long_output(self):
        inner = MockLogger()
        enricher = TraceEnricher(inner, enabled=True, max_output_chars=50)
        enricher.log_event(
            "tool_result",
            {"tool_name": "read", "success": True, "result": "x" * 200},
        )
        data = inner.events[0]["data"]
        assert len(data["result_summary"]) <= 53  # 50 + "..."

    def test_sanitizes_args(self):
        inner = MockLogger()
        enricher = TraceEnricher(inner, enabled=True)
        enricher.log_event(
            "tool_call",
            {"tool_name": "edit", "tool_args": {"old_str": "a" * 600}},
        )
        data = inner.events[0]["data"]
        sanitized = data["arguments_sanitized"]
        assert len(sanitized["old_str"]) <= 503  # 500 + "..."

    def test_session_id_proxy(self):
        inner = MockLogger()
        enricher = TraceEnricher(inner, enabled=True)
        assert enricher.session_id == "test-session"

    def test_getattr_proxy(self):
        inner = MockLogger()
        inner.custom_attr = "custom_value"
        enricher = TraceEnricher(inner, enabled=True)
        assert enricher.custom_attr == "custom_value"


class TestCreateTraceEnricher:
    def test_returns_inner_when_no_config(self):
        inner = MockLogger()
        result = create_trace_enricher(inner, gepa_settings=None)
        assert result is inner

    def test_returns_inner_when_disabled(self):
        from victor.config.gepa_settings import GEPASettings

        inner = MockLogger()
        config = GEPASettings(enabled=False)
        result = create_trace_enricher(inner, gepa_settings=config)
        assert result is inner

    def test_returns_enricher_when_enabled(self):
        from victor.config.gepa_settings import GEPASettings

        inner = MockLogger()
        config = GEPASettings(enabled=True, capture_reasoning=True)
        result = create_trace_enricher(inner, gepa_settings=config)
        assert isinstance(result, TraceEnricher)
