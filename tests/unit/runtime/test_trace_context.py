"""Tests for trace context propagation (OBS-2)."""

import pytest

from victor.runtime.trace_context import (
    TraceContext,
    current_trace,
    get_correlation_id,
)


class TestTraceContextBasics:
    def test_start_creates_trace(self):
        with TraceContext.start("test.operation") as trace:
            assert trace.trace_id is not None
            assert len(trace.trace_id) == 16

    def test_custom_trace_id(self):
        with TraceContext.start("test", trace_id="custom123") as trace:
            assert trace.trace_id == "custom123"

    def test_span_created_on_start(self):
        with TraceContext.start("test.op") as trace:
            spans = trace.get_spans()
            assert len(spans) == 1
            assert spans[0].name == "test.op"

    def test_span_has_timing(self):
        with TraceContext.start("test.op") as trace:
            pass
        span = trace.get_spans()[0]
        assert span.end_time is not None
        assert span.duration_ms is not None
        assert span.duration_ms >= 0

    def test_span_attributes(self):
        with TraceContext.start("load", vertical="coding", version="1.0") as trace:
            pass
        span = trace.get_spans()[0]
        assert span.attributes["vertical"] == "coding"
        assert span.attributes["version"] == "1.0"


class TestTraceEvents:
    def test_add_event(self):
        with TraceContext.start("test") as trace:
            trace.add_event("loaded", status="ok", count=5)
        span = trace.get_spans()[0]
        assert len(span.events) == 1
        assert span.events[0].name == "loaded"
        assert span.events[0].attributes["status"] == "ok"

    def test_multiple_events(self):
        with TraceContext.start("test") as trace:
            trace.add_event("step1")
            trace.add_event("step2")
            trace.add_event("step3")
        span = trace.get_spans()[0]
        assert len(span.events) == 3


class TestNestedTraces:
    def test_nested_spans(self):
        with TraceContext.start("parent") as trace:
            with TraceContext.start("child"):
                pass
        spans = trace.get_spans()
        assert len(spans) == 2
        assert spans[0].name == "parent"
        assert spans[1].name == "child"
        assert spans[1].parent_span_id == spans[0].span_id

    def test_nested_shares_trace_id(self):
        with TraceContext.start("parent") as parent_trace:
            with TraceContext.start("child") as child_trace:
                assert child_trace.trace_id == parent_trace.trace_id


class TestCurrentTrace:
    def test_current_trace_inside_context(self):
        with TraceContext.start("test"):
            trace = current_trace()
            assert trace is not None

    def test_current_trace_outside_context(self):
        assert current_trace() is None

    def test_correlation_id_inside_context(self):
        with TraceContext.start("test", trace_id="abc123"):
            assert get_correlation_id() == "abc123"

    def test_correlation_id_outside_context(self):
        assert get_correlation_id() is None

    def test_context_restored_after_exit(self):
        with TraceContext.start("outer"):
            outer_id = get_correlation_id()
        assert current_trace() is None


class TestErrorTracing:
    def test_error_sets_span_status(self):
        trace = None
        with pytest.raises(ValueError):
            with TraceContext.start("failing") as trace:
                raise ValueError("test error")
        span = trace.get_spans()[0]
        assert span.status == "error"
        assert span.error == "test error"

    def test_successful_span_status(self):
        with TraceContext.start("ok") as trace:
            pass
        assert trace.get_spans()[0].status == "ok"


class TestSerialization:
    def test_to_dict(self):
        with TraceContext.start("test", key="value") as trace:
            trace.add_event("event1", data=42)
        d = trace.to_dict()
        assert d["trace_id"] == trace.trace_id
        assert d["span_count"] == 1
        assert len(d["spans"]) == 1
        assert d["spans"][0]["name"] == "test"
        assert len(d["spans"][0]["events"]) == 1

    def test_span_to_dict(self):
        with TraceContext.start("test") as trace:
            pass
        span_dict = trace.get_spans()[0].to_dict()
        assert "trace_id" in span_dict
        assert "span_id" in span_dict
        assert "duration_ms" in span_dict
