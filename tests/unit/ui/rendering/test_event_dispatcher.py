"""Unit tests for EventDispatcher.

Tests verify that the event dispatcher correctly routes each event type
to the appropriate renderer method, matching the behavior of the original
if-elif chain in handler.py.
"""

from __future__ import annotations

from typing import Optional, Any
from unittest.mock import MagicMock

import pytest

from victor.framework.events import AgentExecutionEvent, EventType
from victor.ui.rendering.event_dispatcher import EventDispatcher


class _RecorderRenderer:
    """Test double that records all renderer calls for verification."""

    def __init__(self) -> None:
        self.tool_calls: list[tuple[str, dict[str, Any]]] = []
        self.tool_results: list[dict[str, Any]] = []
        self.content: list[str] = []
        self.statuses: list[str] = []
        self.thinking_starts: int = 0
        self.thinking_ends: int = 0
        self.thinking_contents: list[str] = []
        self.file_previews: list[tuple[str, str]] = []
        self.edit_previews: list[tuple[str, str]] = []
        self.tool_progresses: list[dict[str, Any]] = []

    def start(self) -> None:
        pass

    def pause(self) -> None:
        pass

    def resume(self) -> None:
        pass

    def on_tool_start(self, name: str, arguments: dict[str, Any], **kwargs: Any) -> None:
        self.tool_calls.append((name, arguments))

    def on_tool_result(
        self,
        name: str,
        success: bool,
        elapsed: float,
        arguments: dict[str, Any],
        error: str | None = None,
        follow_up_suggestions: list[dict[str, Any]] | None = None,
        was_pruned: bool = False,
        original_result: Any = None,
        result: Any = None,
        **kwargs: Any,
    ) -> None:
        self.tool_results.append(
            {
                "name": name,
                "success": success,
                "elapsed": elapsed,
                "arguments": arguments,
                "error": error,
                "follow_up_suggestions": follow_up_suggestions,
                "was_pruned": was_pruned,
                "original_result": original_result,
                "result": result,
            }
        )

    def on_tool_progress(
        self,
        name: str,
        stdout: str = "",
        stderr: str = "",
        progress: float = 0.0,
        is_final: bool = False,
    ) -> None:
        self.tool_progresses.append(
            {
                "name": name,
                "stdout": stdout,
                "stderr": stderr,
                "progress": progress,
                "is_final": is_final,
            }
        )

    def on_status(self, message: str, severity: Optional["Severity"] = None) -> None:
        self.statuses.append(message)

    def on_file_preview(self, path: str, content: str) -> None:
        self.file_previews.append((path, content))

    def on_edit_preview(self, path: str, diff: str) -> None:
        self.edit_previews.append((path, diff))

    def on_content(self, text: str) -> None:
        self.content.append(text)

    def on_thinking_content(self, text: str) -> None:
        self.thinking_contents.append(text)

    def on_thinking_start(self) -> None:
        self.thinking_starts += 1

    def on_thinking_end(self) -> None:
        self.thinking_ends += 1

    def had_tool_calls(self) -> bool:
        return bool(self.tool_calls)

    def finalize(self) -> str:
        return "".join(self.content)

    def cleanup(self) -> None:
        pass


@pytest.fixture
def renderer():
    """Provide a fresh RecorderRenderer for each test."""
    return _RecorderRenderer()


@pytest.fixture
def content_filter():
    """Provide a mock StreamingContentFilter.

    By default, process_chunk passes content through as-is (non-thinking).
    Tests that need different behavior can override via ``monkeypatch``.
    """
    mock = MagicMock()
    mock.is_thinking = False

    def _process_chunk(content: str):
        result = MagicMock()
        result.content = content
        result.is_thinking = False
        result.entering_thinking = False
        result.exiting_thinking = False
        return result

    mock.process_chunk.side_effect = _process_chunk
    mock.flush.return_value = _process_chunk("")
    mock.should_abort.return_value = False
    return mock


@pytest.fixture
def normalizer():
    """Provide a mock StreamDeltaNormalizer that passes content through."""
    mock = MagicMock()
    mock.consume.side_effect = lambda x: x
    mock.reset = MagicMock()
    return mock


@pytest.fixture
def dispatcher(renderer, content_filter, normalizer):
    """Provide a fresh EventDispatcher for each test."""
    return EventDispatcher(
        renderer=renderer,
        content_filter=content_filter,
        content_normalizer=normalizer,
        reasoning_normalizer=normalizer,
        suppress_thinking=False,
    )


class TestEventDispatcherToolEvents:
    """Tests for tool event routing."""

    def test_dispatches_tool_call_via_metadata(self, dispatcher, renderer):
        """tool_start metadata should route to on_tool_start."""
        event = AgentExecutionEvent(
            type=EventType.TOOL_CALL,
            tool_name="graph",
            metadata={
                "tool_start": {
                    "name": "graph",
                    "arguments": {"mode": "find", "query": "test"},
                }
            },
        )
        dispatcher.dispatch(event)
        assert renderer.tool_calls == [("graph", {"mode": "find", "query": "test"})]

    def test_dispatches_tool_call_via_type(self, dispatcher, renderer):
        """TOOL_CALL typed event should route to on_tool_start."""
        event = AgentExecutionEvent(
            type=EventType.TOOL_CALL,
            tool_name="read",
            arguments={"path": "file.py"},
        )
        dispatcher.dispatch(event)
        assert renderer.tool_calls == [("read", {"path": "file.py"})]

    def test_dispatches_tool_result_via_metadata(self, dispatcher, renderer):
        """tool_result metadata should route to on_tool_result."""
        event = AgentExecutionEvent(
            type=EventType.TOOL_RESULT,
            tool_name="graph",
            metadata={
                "tool_result": {
                    "name": "graph",
                    "success": True,
                    "elapsed": 0.25,
                    "arguments": {"mode": "find"},
                    "result": "preview output",
                    "original_result": "full output",
                    "was_pruned": True,
                }
            },
        )
        dispatcher.dispatch(event)
        assert len(renderer.tool_results) == 1
        assert renderer.tool_results[0]["name"] == "graph"
        assert renderer.tool_results[0]["success"] is True
        assert renderer.tool_results[0]["elapsed"] == 0.25
        assert renderer.tool_results[0]["was_pruned"] is True

    def test_dispatches_tool_result_via_type(self, dispatcher, renderer):
        """TOOL_RESULT typed event should route to on_tool_result."""
        event = AgentExecutionEvent(
            type=EventType.TOOL_RESULT,
            tool_name="read",
            result="file content",
            success=True,
        )
        dispatcher.dispatch(event)
        assert len(renderer.tool_results) == 1
        assert renderer.tool_results[0]["name"] == "read"

    def test_dispatches_tool_progress(self, dispatcher, renderer):
        """tool_progress metadata should route to on_tool_progress."""
        event = AgentExecutionEvent(
            type=EventType.TOOL_PROGRESS,
            tool_name="shell",
            metadata={
                "tool_progress": {
                    "name": "shell",
                    "stdout": "building...",
                    "stderr": "",
                    "progress": 0.5,
                    "is_final": False,
                }
            },
        )
        dispatcher.dispatch(event)
        assert len(renderer.tool_progresses) == 1
        assert renderer.tool_progresses[0]["name"] == "shell"
        assert renderer.tool_progresses[0]["stdout"] == "building..."
        assert renderer.tool_progresses[0]["progress"] == 0.5


class TestEventDispatcherContentEvents:
    """Tests for content event routing."""

    def test_dispatches_content(self, dispatcher, renderer):
        """Content events should route to on_content."""
        event = AgentExecutionEvent(
            type=EventType.CONTENT,
            content="Hello, world!",
        )
        dispatcher.dispatch(event)
        assert renderer.content == ["Hello, world!"]

    def test_dispatches_file_preview(self, dispatcher, renderer):
        """file_preview metadata should route to on_file_preview."""
        event = AgentExecutionEvent(
            type=EventType.CONTENT,
            metadata={
                "file_preview": "file content here",
                "path": "/path/to/file.py",
            },
        )
        dispatcher.dispatch(event)
        assert renderer.file_previews == [("/path/to/file.py", "file content here")]

    def test_dispatches_edit_preview(self, dispatcher, renderer):
        """edit_preview metadata should route to on_edit_preview."""
        event = AgentExecutionEvent(
            type=EventType.CONTENT,
            metadata={
                "edit_preview": "+ added\n- removed",
                "path": "/path/to/file.py",
            },
        )
        dispatcher.dispatch(event)
        assert renderer.edit_previews == [("/path/to/file.py", "+ added\n- removed")]


class TestEventDispatcherErrorEvents:
    """Tests for error event routing."""

    def test_dispatches_error_with_message(self, dispatcher, renderer):
        """ERROR events with error text should surface via on_status."""
        event = AgentExecutionEvent(
            type=EventType.ERROR,
            error="Model not found: gpt-oss:latest",
            recoverable=False,
        )
        dispatcher.dispatch(event)
        assert any("Model not found" in s for s in renderer.statuses)
        assert dispatcher.error_surfaced is True

    def test_dispatches_error_with_content_fallback(self, dispatcher, renderer):
        """ERROR events without error attribute should use content."""
        event = AgentExecutionEvent(
            type=EventType.ERROR,
            content="Connection refused",
        )
        dispatcher.dispatch(event)
        assert any("Connection refused" in s for s in renderer.statuses)

    def test_dispatches_error_with_default_message(self, dispatcher, renderer):
        """ERROR events with no details should use default message."""
        event = AgentExecutionEvent(type=EventType.ERROR)
        dispatcher.dispatch(event)
        assert any("provider returned an error" in s for s in renderer.statuses)


class TestEventDispatcherThinkingEvents:
    """Tests for thinking/reasoning event routing."""

    def test_dispatches_reasoning_content(self, dispatcher, renderer):
        """reasoning_content metadata should route to on_thinking_content."""
        event = AgentExecutionEvent(
            type=EventType.CONTENT,
            metadata={"reasoning_content": "Let me think about this..."},
        )
        dispatcher.dispatch(event)
        assert len(renderer.thinking_contents) > 0
        assert renderer.thinking_starts == 1

    def test_dispatches_thinking_status(self, dispatcher, renderer):
        """Status messages with thinking prefixes should trigger thinking mode."""
        event = AgentExecutionEvent(
            type=EventType.CONTENT,
            metadata={"status": "\U0001f4ad Thinking..."},
        )
        dispatcher.dispatch(event)
        assert renderer.thinking_starts == 1

    def test_dispatches_regular_status(self, dispatcher, renderer):
        """Non-thinking status messages should route to on_status."""
        event = AgentExecutionEvent(
            type=EventType.CONTENT,
            metadata={"status": "Processing file..."},
        )
        dispatcher.dispatch(event)
        assert renderer.statuses == ["Processing file..."]

    def test_flush_thinking_ends_state(self, dispatcher, renderer):
        """flush_thinking should end thinking state if active."""
        # Enter thinking state
        event = AgentExecutionEvent(
            type=EventType.CONTENT,
            metadata={"reasoning_content": "Thinking..."},
        )
        dispatcher.dispatch(event)
        assert dispatcher.was_thinking is True

        # Flush should end it
        dispatcher.flush_thinking()
        assert dispatcher.was_thinking is False
        assert renderer.thinking_ends == 1


class TestEventDispatcherEdgeCases:
    """Tests for edge cases in event dispatch."""

    def test_suppress_thinking_blocks_reasoning(self, renderer, content_filter, normalizer):
        """When suppress_thinking is True, reasoning should be hidden."""
        dispatcher = EventDispatcher(
            renderer=renderer,
            content_filter=content_filter,
            content_normalizer=normalizer,
            reasoning_normalizer=normalizer,
            suppress_thinking=True,
        )
        event = AgentExecutionEvent(
            type=EventType.CONTENT,
            metadata={"reasoning_content": "Hidden thinking"},
        )
        dispatcher.dispatch(event)
        assert renderer.thinking_starts == 0
        assert len(renderer.thinking_contents) == 0

    def test_empty_event_does_nothing(self, dispatcher, renderer):
        """An event with no content or metadata should be a no-op."""
        event = AgentExecutionEvent(type=EventType.CONTENT)
        dispatcher.dispatch(event)
        assert renderer.content == []
        assert renderer.tool_calls == []
        assert renderer.tool_results == []

    def test_unknown_event_type_ignored(self, dispatcher, renderer):
        """Unknown event types should be silently ignored."""
        event = AgentExecutionEvent(type="unknown_type", content="something")
        dispatcher.dispatch(event)
        # Content should still be handled
        assert renderer.content == ["something"]

    def test_tool_result_with_follow_up_suggestions(self, dispatcher, renderer):
        """Tool results with follow-up suggestions should pass them through."""
        follow_ups = [
            {
                "command": 'graph(mode="neighbors", node="X")',
                "description": "Inspect neighbors.",
            }
        ]
        event = AgentExecutionEvent(
            type=EventType.TOOL_RESULT,
            tool_name="graph",
            metadata={
                "tool_result": {
                    "name": "graph",
                    "success": True,
                    "elapsed": 0.5,
                    "arguments": {},
                    "follow_up_suggestions": follow_ups,
                    "result": "done",
                }
            },
        )
        dispatcher.dispatch(event)
        assert renderer.tool_results[0]["follow_up_suggestions"] == follow_ups

    def test_metadata_tool_start_with_none_value(self, dispatcher, renderer):
        """Tool_start metadata with None value should not crash."""
        event = AgentExecutionEvent(
            type=EventType.TOOL_CALL,
            tool_name="read",
            metadata={"tool_start": None},
        )
        dispatcher.dispatch(event)
        # Should fall back to tool_name
        assert renderer.tool_calls == [("read", {})]

    def test_tool_progress_noop_when_no_on_progress(self, content_filter, normalizer):
        """Tool progress should be a no-op if renderer has no on_tool_progress."""
        renderer = _RecorderRenderer()

        # Create a renderer without on_tool_progress
        class _NoProgressRenderer:
            def __getattr__(self, name):
                raise AttributeError(name)

        no_progress = _NoProgressRenderer()
        # Copy all recorder methods except on_tool_progress
        for attr_name in dir(renderer):
            if attr_name.startswith("_"):
                continue
            if attr_name == "on_tool_progress":
                continue
            setattr(no_progress, attr_name, getattr(renderer, attr_name))

        dispatcher = EventDispatcher(
            renderer=no_progress,
            content_filter=content_filter,
            content_normalizer=normalizer,
            reasoning_normalizer=normalizer,
        )
        event = AgentExecutionEvent(
            type=EventType.TOOL_PROGRESS,
            tool_name="shell",
            metadata={"tool_progress": {"name": "shell", "stdout": "data"}},
        )
        # Should not raise
        dispatcher.dispatch(event)
