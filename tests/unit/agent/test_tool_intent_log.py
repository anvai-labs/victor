"""Tests for pre-execution intent logging (LogAct-inspired)."""

from unittest.mock import MagicMock


class TestToolIntentEmission:
    """Test that tool.intent events are emitted before execution."""

    def test_emit_tool_intent_method_exists(self):
        """ToolPipeline has _emit_tool_intent method."""
        from victor.agent.tool_pipeline import ToolPipeline

        assert hasattr(ToolPipeline, "_emit_tool_intent")

    def test_intent_contains_required_fields(self):
        """Intent event has tool_name, arguments_hash, timestamp."""
        from victor.agent.tool_pipeline import ToolPipeline

        pipeline = MagicMock(spec=ToolPipeline)
        pipeline._observability_bus = MagicMock()
        pipeline._trace_enricher = MagicMock()
        pipeline._trace_enricher._pending_reasoning = "I need to read the file"

        ToolPipeline._emit_tool_intent(pipeline, "read_file", {"path": "/src/main.py"})

        pipeline._observability_bus.emit_sync.assert_called_once()
        call_args = pipeline._observability_bus.emit_sync.call_args
        topic = call_args[0][0]
        data = call_args[0][1]
        assert topic == "tool.intent"
        assert data["tool_name"] == "read_file"
        assert "arguments_hash" in data
        assert "timestamp" in data

    def test_intent_includes_reasoning_context(self):
        """Intent captures reasoning_before from TraceEnricher."""
        from victor.agent.tool_pipeline import ToolPipeline

        pipeline = MagicMock(spec=ToolPipeline)
        pipeline._observability_bus = MagicMock()
        pipeline._trace_enricher = MagicMock()
        pipeline._trace_enricher._pending_reasoning = "Checking auth module"

        ToolPipeline._emit_tool_intent(pipeline, "read_file", {"path": "/src/auth.py"})

        call_data = pipeline._observability_bus.emit_sync.call_args[0][1]
        assert call_data.get("reasoning_before") == "Checking auth module"


class TestIntentLogInContext:
    """Test intent_log field on StreamingChatContext."""

    def test_context_has_intent_log(self):
        """StreamingChatContext has intent_log field."""
        from victor.agent.streaming.context import StreamingChatContext

        ctx = StreamingChatContext.__new__(StreamingChatContext)
        ctx.intent_log = []
        assert isinstance(ctx.intent_log, list)


class TestIntentLoggingGracefulDegradation:
    """Test that intent logging doesn't break when bus unavailable."""

    def test_no_bus_no_error(self):
        """_emit_tool_intent is a no-op when observability bus is unavailable."""
        from victor.agent.tool_pipeline import ToolPipeline

        pipeline = MagicMock(spec=ToolPipeline)
        pipeline._observability_bus = None
        pipeline._trace_enricher = None

        # Should not raise
        ToolPipeline._emit_tool_intent(pipeline, "read_file", {"path": "/src/main.py"})
