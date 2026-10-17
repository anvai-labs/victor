"""Tests for victor/framework/_internal.py Phase 7.6 refactoring.

This module tests the internal helpers that bridge the framework API
to the orchestrator, with focus on:
- configure_tools() using ToolConfigurator
- setup_observability_integration()
- stream_with_events()
- Utility functions
"""

import logging

import httpx
from types import SimpleNamespace

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from victor.framework._internal import (
    apply_system_prompt,
    apply_vertical_to_orchestrator,
    collect_tool_calls,
    configure_tools,
    format_context_message,
    setup_observability_integration,
    stream_with_events,
)
from victor.framework.events import AgentExecutionEvent, EventType
from victor.framework.tools import ToolSet


class TestConfigureToolsWithToolConfigurator:
    """Test configure_tools using ToolConfigurator from Phase 7.5."""

    def test_configure_with_toolset(self):
        """Test configuring tools from a ToolSet."""
        mock_orchestrator = MagicMock()
        mock_orchestrator.tools = {"read": MagicMock(), "write": MagicMock()}

        # Create a ToolSet
        toolset = ToolSet.minimal()

        with patch("victor.framework.tool_config.get_tool_configurator") as mock_get_configurator:
            mock_configurator = MagicMock()
            mock_get_configurator.return_value = mock_configurator

            configure_tools(mock_orchestrator, toolset)

            # Should use configure_from_toolset for ToolSet
            mock_configurator.configure_from_toolset.assert_called_once_with(
                mock_orchestrator, toolset
            )

    def test_configure_with_list(self):
        """Test configuring tools from a list of tool names."""
        mock_orchestrator = MagicMock()
        mock_orchestrator.tools = {"read": MagicMock(), "write": MagicMock()}

        tool_list = ["read", "write", "edit"]

        with patch("victor.framework.tool_config.get_tool_configurator") as mock_get_configurator:
            mock_configurator = MagicMock()
            mock_get_configurator.return_value = mock_configurator

            configure_tools(mock_orchestrator, tool_list)

            # Should use configure with REPLACE mode
            mock_configurator.configure.assert_called_once()
            call_args = mock_configurator.configure.call_args
            assert call_args[0][0] == mock_orchestrator
            assert call_args[0][1] == {"read", "write", "edit"}

    def test_configure_with_airgapped_filter(self):
        """Test that airgapped mode adds AirgappedFilter."""
        mock_orchestrator = MagicMock()
        mock_orchestrator.tools = {"read": MagicMock()}

        tool_list = ["read", "write"]

        with patch("victor.framework.tool_config.get_tool_configurator") as mock_get_configurator:
            mock_configurator = MagicMock()
            mock_get_configurator.return_value = mock_configurator

            configure_tools(mock_orchestrator, tool_list, airgapped=True)

            # Should add airgapped filter
            mock_configurator.add_filter.assert_called_once()
            filter_arg = mock_configurator.add_filter.call_args[0][0]
            assert filter_arg.__class__.__name__ == "AirgappedFilter"


class TestSetupObservabilityIntegration:
    """Test observability integration setup."""

    def test_setup_creates_integration_uses_set_observability(self):
        """Test setup wires and stores integration via public setter port."""
        mock_orchestrator = MagicMock()

        with patch("victor.observability.integration.ObservabilityIntegration") as MockIntegration:
            mock_integration = MagicMock()
            MockIntegration.return_value = mock_integration

            result = setup_observability_integration(mock_orchestrator)

            MockIntegration.assert_called_once_with(session_id=None)
            mock_integration.wire_orchestrator.assert_called_once_with(mock_orchestrator)
            mock_orchestrator.set_observability.assert_called_once_with(mock_integration)
            assert result == mock_integration

    def test_setup_falls_back_to_observability_property(self):
        """Test setup falls back to observability property when setter is unavailable."""

        class PropertyOnlyOrchestrator:
            def __init__(self):
                self._observability = None

            @property
            def observability(self):
                return self._observability

            @observability.setter
            def observability(self, value):
                self._observability = value

        orchestrator = PropertyOnlyOrchestrator()

        with patch("victor.observability.integration.ObservabilityIntegration") as MockIntegration:
            mock_integration = MagicMock()
            MockIntegration.return_value = mock_integration

            result = setup_observability_integration(orchestrator)

            assert orchestrator.observability == mock_integration
            assert result == mock_integration

    def test_setup_with_session_id(self):
        """Test setup with custom session ID."""
        mock_orchestrator = MagicMock()

        with patch("victor.observability.integration.ObservabilityIntegration") as MockIntegration:
            mock_integration = MagicMock()
            MockIntegration.return_value = mock_integration

            setup_observability_integration(mock_orchestrator, session_id="test-session-123")

            MockIntegration.assert_called_once_with(session_id="test-session-123")


class TestVerticalApplicationDelegation:
    """Tests for shared vertical application service usage."""

    def test_apply_vertical_delegates_to_framework_service(self):
        """apply_vertical_to_orchestrator should use central framework service."""
        mock_orchestrator = MagicMock()
        mock_result = MagicMock(
            success=True,
            vertical_name="coding",
            tools_applied=set(),
            middleware_count=0,
            safety_patterns_count=0,
            errors=[],
            warnings=[],
        )

        with patch("victor.framework.vertical_service.apply_vertical_configuration") as mock_apply:
            mock_apply.return_value = mock_result

            apply_vertical_to_orchestrator(mock_orchestrator, "coding")

            mock_apply.assert_called_once_with(mock_orchestrator, "coding", source="sdk")


class TestApplySystemPrompt:
    """Test system prompt application.

    SOLID Compliance (DIP): These tests verify that apply_system_prompt
    only uses public methods and never writes to private attributes.
    """

    def test_apply_with_set_custom_prompt(self):
        """Test applying prompt via orchestrator's set_custom_prompt method.

        The capability-based approach calls orchestrator.invoke_capability
        when the object implements CapabilityRegistryProtocol.
        """
        from victor.framework.protocols import CapabilityRegistryProtocol

        # Mock that implements capability registry
        mock_orchestrator = MagicMock(spec=CapabilityRegistryProtocol)
        mock_orchestrator.has_capability.return_value = True
        mock_orchestrator.invoke_capability = MagicMock(return_value=True)

        apply_system_prompt(mock_orchestrator, "Custom prompt text")

        # Should invoke via capability registry
        mock_orchestrator.invoke_capability.assert_called_once()
        call_args = mock_orchestrator.invoke_capability.call_args
        assert call_args[0][0] == "custom_prompt"
        assert call_args[0][1] == "Custom prompt text"

    def test_apply_with_prompt_builder_public_method(self):
        """Test applying prompt via prompt_builder.set_custom_prompt.

        When orchestrator doesn't have set_custom_prompt, falls back to
        prompt_builder.set_custom_prompt (public method only).
        """
        # Create mock without set_custom_prompt so it falls through to prompt_builder
        mock_orchestrator = MagicMock(spec=["prompt_builder"])
        mock_orchestrator.prompt_builder = MagicMock()
        mock_orchestrator.prompt_builder.set_custom_prompt = MagicMock()

        apply_system_prompt(mock_orchestrator, "Another prompt")

        # Should call public method on prompt_builder
        mock_orchestrator.prompt_builder.set_custom_prompt.assert_called_once_with("Another prompt")

    def test_apply_without_public_method_logs_warning(self):
        """Test that missing public method logs warning (DIP compliance)."""
        import logging

        # Create mock without any public set_custom_prompt methods
        mock_orchestrator = MagicMock(spec=["prompt_builder"])
        mock_orchestrator.prompt_builder = MagicMock(spec=[])  # No public methods

        with patch.object(logging, "getLogger") as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger

            apply_system_prompt(mock_orchestrator, "Orphan prompt")

            # Should log warning about missing capability
            mock_logger.warning.assert_called()


class TestStreamWithEvents:
    """Test streaming with event conversion."""

    @pytest.mark.asyncio
    async def test_stream_emits_start_and_end(self):
        """Test that stream emits start and end events."""
        mock_orchestrator = MagicMock()

        async def mock_stream_chat(prompt):
            return
            yield  # Make it an async generator

        mock_orchestrator.stream_chat = mock_stream_chat

        events = []
        async for event in stream_with_events(mock_orchestrator, "test"):
            events.append(event)

        assert len(events) >= 2
        assert events[0].type == EventType.STREAM_START
        assert events[-1].type == EventType.STREAM_END
        assert events[-1].success is True

    @pytest.mark.asyncio
    async def test_stream_content_events(self):
        """Test that content chunks become content events."""
        mock_orchestrator = MagicMock()

        async def mock_stream_chat(prompt):
            chunk = MagicMock()
            chunk.content = "Hello"
            chunk.metadata = None
            chunk.tool_calls = None
            yield chunk

            chunk2 = MagicMock()
            chunk2.content = " World"
            chunk2.metadata = None
            chunk2.tool_calls = None
            yield chunk2

        mock_orchestrator.stream_chat = mock_stream_chat

        events = []
        async for event in stream_with_events(mock_orchestrator, "test"):
            events.append(event)

        content_events = [e for e in events if e.type == EventType.CONTENT]
        assert len(content_events) == 2
        assert content_events[0].content == "Hello"
        assert content_events[1].content == " World"

    @pytest.mark.asyncio
    async def test_stream_thinking_events(self):
        """Test that thinking content becomes thinking events."""
        mock_orchestrator = MagicMock()

        async def mock_stream_chat(prompt):
            chunk = MagicMock()
            chunk.content = ""
            chunk.metadata = {"reasoning_content": "Let me think..."}
            chunk.tool_calls = None
            yield chunk

        mock_orchestrator.stream_chat = mock_stream_chat

        events = []
        async for event in stream_with_events(mock_orchestrator, "test"):
            events.append(event)

        thinking_events = [e for e in events if e.type == EventType.THINKING]
        assert len(thinking_events) == 1
        assert thinking_events[0].content == "Let me think..."

    @pytest.mark.asyncio
    async def test_stream_status_metadata_becomes_milestone_event(self):
        """Status-bearing chunks should surface as milestone events for framework consumers."""
        mock_orchestrator = MagicMock()

        async def mock_stream_chat(prompt):
            chunk = MagicMock()
            chunk.content = ""
            chunk.metadata = {"status": "Waiting for provider..."}
            chunk.tool_calls = None
            yield chunk

        mock_orchestrator.stream_chat = mock_stream_chat

        events = []
        async for event in stream_with_events(mock_orchestrator, "test"):
            events.append(event)

        milestone_events = [e for e in events if e.type == EventType.MILESTONE]
        assert len(milestone_events) == 1
        assert milestone_events[0].milestone == "Waiting for provider..."

    @pytest.mark.asyncio
    async def test_stream_tool_call_events(self):
        """Test that tool calls in chunks become tool_call events."""
        mock_orchestrator = MagicMock()

        async def mock_stream_chat(prompt):
            chunk = MagicMock()
            chunk.content = ""
            chunk.metadata = None
            chunk.tool_calls = [{"name": "read", "id": "call_1", "arguments": {"path": "/test"}}]
            yield chunk

        mock_orchestrator.stream_chat = mock_stream_chat

        events = []
        async for event in stream_with_events(mock_orchestrator, "test"):
            events.append(event)

        tool_events = [e for e in events if e.type == EventType.TOOL_CALL]
        assert len(tool_events) == 1
        assert tool_events[0].tool_name == "read"
        assert tool_events[0].tool_id == "call_1"
        assert tool_events[0].arguments == {"path": "/test"}

    @pytest.mark.asyncio
    async def test_stream_uses_event_registry_for_chunk_conversion(self):
        """Chunk events should be converted through EventRegistry."""
        from victor.framework.event_registry import EventRegistry, EventTarget

        mock_orchestrator = MagicMock()

        async def mock_stream_chat(prompt):
            chunk = MagicMock()
            chunk.content = "hello"
            chunk.metadata = {
                "reasoning_content": "thinking...",
                "tool_start": {
                    "name": "read",
                    "id": "call_start",
                    "arguments": {"path": "/tmp/a.py"},
                },
                "tool_result": {
                    "name": "read",
                    "id": "call_start",
                    "result": "ok",
                    "success": True,
                },
            }
            chunk.tool_calls = [{"name": "grep", "id": "call_2", "arguments": {"pattern": "TODO"}}]
            yield chunk

        mock_orchestrator.stream_chat = mock_stream_chat

        real_registry = EventRegistry.get_instance()
        proxy_registry = MagicMock()
        proxy_registry.from_external.side_effect = (
            lambda data, external_type, target, metadata=None: real_registry.from_external(
                data,
                external_type,
                target,
                metadata=metadata,
            )
        )

        events = []
        with patch("victor.framework._internal.get_event_registry", return_value=proxy_registry):
            async for event in stream_with_events(mock_orchestrator, "test"):
                events.append(event)

        # 5 chunk-derived conversions: thinking, content, tool_start, tool_result, tool_call
        assert proxy_registry.from_external.call_count == 5
        assert [call.args[1] for call in proxy_registry.from_external.call_args_list] == [
            "reasoning_content",
            "content",
            "tool_start",
            "tool_result",
            "tool_call",
        ]
        assert all(
            call.args[2] == EventTarget.STREAM_CHUNK
            for call in proxy_registry.from_external.call_args_list
        )

        event_types = [event.type for event in events]
        assert EventType.THINKING in event_types
        assert EventType.CONTENT in event_types
        assert EventType.TOOL_CALL in event_types
        assert EventType.TOOL_RESULT in event_types

    @pytest.mark.asyncio
    async def test_stream_error_handling(self):
        """Test that errors produce error events."""
        mock_orchestrator = MagicMock()

        async def mock_stream_chat(prompt):
            # Yield a chunk first
            chunk = MagicMock()
            chunk.content = "partial"
            chunk.metadata = None
            chunk.tool_calls = None
            yield chunk
            # Then raise error
            raise ValueError("Test error")

        mock_orchestrator.stream_chat = mock_stream_chat

        events = []
        async for event in stream_with_events(mock_orchestrator, "test"):
            events.append(event)

        error_events = [e for e in events if e.type == EventType.ERROR]
        assert len(error_events) == 1
        assert "Test error" in error_events[0].error

        # Should still have stream_end with success=False
        end_events = [e for e in events if e.type == EventType.STREAM_END]
        assert len(end_events) == 1
        assert end_events[0].success is False

    @pytest.mark.asyncio
    async def test_stream_error_with_empty_exception_message_is_visible(self, caplog):
        """Errors with empty str(exc) should still produce actionable text."""
        mock_orchestrator = MagicMock()

        async def mock_stream_chat(prompt):
            raise RuntimeError()
            yield  # pragma: no cover

        mock_orchestrator.stream_chat = mock_stream_chat

        with caplog.at_level(logging.ERROR, logger="victor.framework._internal"):
            events = []
            async for event in stream_with_events(mock_orchestrator, "test"):
                events.append(event)

        error_events = [e for e in events if e.type == EventType.ERROR]
        assert len(error_events) == 1
        assert error_events[0].error == "RuntimeError: RuntimeError()"

        end_events = [e for e in events if e.type == EventType.STREAM_END]
        assert len(end_events) == 1
        assert end_events[0].success is False
        assert end_events[0].error == "RuntimeError: RuntimeError()"
        assert any("stream_with_events failed" in record.message for record in caplog.records)

    @pytest.mark.asyncio
    async def test_stream_error_sanitizes_html_auth_page(self):
        """HTML auth payloads should not be streamed into the chat transcript."""
        mock_orchestrator = MagicMock()
        request = httpx.Request("POST", "https://example.com/chat/completions")
        response = httpx.Response(
            401,
            request=request,
            text="<html><body><svg>large login page</svg></body></html>",
        )

        async def mock_stream_chat(prompt):
            raise httpx.HTTPStatusError("401 Unauthorized", request=request, response=response)
            yield  # pragma: no cover

        mock_orchestrator.stream_chat = mock_stream_chat

        events = []
        async for event in stream_with_events(mock_orchestrator, "test"):
            events.append(event)

        error_events = [e for e in events if e.type == EventType.ERROR]
        assert len(error_events) == 1
        assert "HTML authentication page" in error_events[0].error
        assert "<html>" not in error_events[0].error

    @pytest.mark.asyncio
    async def test_stream_buffers_and_normalizes_exact_response_output(self):
        """Exact-response prompts should be normalized by the framework bridge."""
        mock_orchestrator = MagicMock()

        async def mock_stream_chat(prompt):
            first = MagicMock()
            first.content = "The requested literal is "
            first.metadata = None
            first.tool_calls = None
            yield first

            second = MagicMock()
            second.content = "READY"
            second.metadata = {"source": "model"}
            second.tool_calls = None
            yield second

        mock_orchestrator.stream_chat = mock_stream_chat

        events = []
        async for event in stream_with_events(
            mock_orchestrator,
            "Reply with exactly READY",
        ):
            events.append(event)

        content_events = [e for e in events if e.type == EventType.CONTENT]
        assert [event.content for event in content_events] == ["READY"]
        assert content_events[0].metadata == {"source": "model"}

    @pytest.mark.asyncio
    async def test_stream_uses_completion_summary_fallback_when_no_content_event_was_emitted(
        self,
    ):
        """A swallowed stream should recover the persisted completion summary."""
        mock_orchestrator = MagicMock()
        mock_orchestrator._task_completion_detector = MagicMock(
            _state=MagicMock(last_summary="Final architecture findings")
        )

        async def mock_stream_chat(prompt):
            chunk = MagicMock()
            chunk.content = ""
            chunk.metadata = None
            chunk.tool_calls = None
            yield chunk

        mock_orchestrator.stream_chat = mock_stream_chat

        events = []
        async for event in stream_with_events(mock_orchestrator, "Summarize the graph design"):
            events.append(event)

        content_events = [e for e in events if e.type == EventType.CONTENT]
        assert [event.content for event in content_events] == ["Final architecture findings"]
        assert content_events[0].metadata["source"] == "completion_summary_fallback"

    @pytest.mark.parametrize("success", [True, False])
    async def test_explicit_loop_outcome_prevents_summary_fallback(self, success):
        from victor.providers.base import StreamChunk

        orchestrator = MagicMock()
        orchestrator._task_completion_detector._state.last_summary = "ungoverned prior summary"
        metadata = {"agentic_loop_success": success, "agentic_loop_iterations": 2}
        if not success:
            metadata["agentic_loop_error"] = "verification failed"

        async def stream(prompt):
            yield StreamChunk(is_final=True, metadata=metadata)

        orchestrator.stream_chat = stream
        events = [event async for event in stream_with_events(orchestrator, "test")]
        assert not [e for e in events if e.type == EventType.CONTENT]
        ends = [e for e in events if e.type == EventType.STREAM_END]
        errors = [e for e in events if e.type == EventType.ERROR]
        assert len(ends) == 1
        assert ends[0].success is success
        assert ends[0].metadata == metadata
        assert len(errors) == (0 if success else 1)
        if errors:
            assert errors[0].error == "verification failed"
            assert errors[0].recoverable is False

    async def test_invalid_loop_verdict_is_not_truthy_success(self):
        from victor.providers.base import StreamChunk

        async def stream(prompt):
            yield StreamChunk(is_final=True, metadata={"agentic_loop_success": "false"})

        events = [
            event async for event in stream_with_events(SimpleNamespace(stream_chat=stream), "q")
        ]
        assert events[-1].type == EventType.STREAM_END
        assert events[-1].success is False


class TestFormatContextMessage:
    """Test context formatting utility."""

    def test_format_empty_context(self):
        """Test formatting empty context returns None."""
        assert format_context_message({}) is None
        assert format_context_message(None) is None

    def test_format_with_file(self):
        """Test formatting with file."""
        result = format_context_message({"file": "test.py"})
        assert "File: test.py" in result

    def test_format_with_files(self):
        """Test formatting with multiple files."""
        result = format_context_message({"files": ["a.py", "b.py"]})
        assert "Files: a.py, b.py" in result

    def test_format_with_error(self):
        """Test formatting with error."""
        result = format_context_message({"error": "Something went wrong"})
        assert "Error: Something went wrong" in result

    def test_format_with_code(self):
        """Test formatting with code block."""
        result = format_context_message({"code": "print('hello')"})
        assert "```\nprint('hello')\n```" in result

    def test_format_with_custom_keys(self):
        """Test formatting with custom key-value pairs."""
        result = format_context_message({"custom_key": "custom_value"})
        assert "custom_key: custom_value" in result


class TestCollectToolCalls:
    """Test tool call collection utility."""

    def test_collect_empty_list(self):
        """Test collecting from empty list."""
        assert collect_tool_calls([]) == []

    def test_collect_ignores_non_tool_events(self):
        """Test that non-tool events are ignored."""
        events = [
            AgentExecutionEvent(type=EventType.CONTENT, content="hello"),
            AgentExecutionEvent(type=EventType.STREAM_START),
        ]
        assert collect_tool_calls(events) == []

    def test_collect_tool_results(self):
        """Test collecting tool result events."""
        events = [
            AgentExecutionEvent(
                type=EventType.TOOL_RESULT,
                tool_name="read",
                tool_id="call_1",
                arguments={"path": "/test"},
                result="file contents",
                success=True,
            ),
            AgentExecutionEvent(type=EventType.CONTENT, content="Done"),
            AgentExecutionEvent(
                type=EventType.TOOL_RESULT,
                tool_name="write",
                tool_id="call_2",
                arguments={"path": "/out"},
                result="written",
                success=True,
            ),
        ]

        result = collect_tool_calls(events)

        assert len(result) == 2
        assert result[0]["tool"] == "read"
        assert result[0]["tool_id"] == "call_1"
        assert result[0]["success"] is True
        assert result[1]["tool"] == "write"


class TestSanitizeStreamErrorMessage:
    """_sanitize_stream_error_message prefers structured, user-facing errors."""

    def test_structured_victor_error_uses_message_and_hint(self):
        from victor.core.errors import ModelNotFoundError
        from victor.framework._internal import _sanitize_stream_error_message

        err = ModelNotFoundError(
            provider="ollama",
            model="gpt-oss:latest",
            available_models=["gpt-oss:20b"],
            status_code=404,
        )

        msg = _sanitize_stream_error_message(err)

        # Clean message (no correlation-id prefix from str(error)) + recovery hint.
        assert msg.startswith("Model not found: gpt-oss:latest")
        assert "💡" in msg
        assert "gpt-oss:20b" in msg

    def test_plain_exception_falls_back_to_str(self):
        from victor.framework._internal import _sanitize_stream_error_message

        msg = _sanitize_stream_error_message(ValueError("boom"))

        assert "boom" in msg
