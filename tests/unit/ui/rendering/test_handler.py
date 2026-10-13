from __future__ import annotations

import asyncio
from typing import Any

import pytest

from victor.core.errors import AgentError

from victor.framework.events import AgentExecutionEvent, EventType
from victor.ui.rendering.handler import stream_response


class _RecorderRenderer:
    def __init__(self) -> None:
        self.tool_calls: list[tuple[str, dict[str, Any]]] = []
        self.tool_results: list[dict[str, Any]] = []
        self.content: list[str] = []
        self.statuses: list[str] = []

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

    def on_status(self, message: str) -> None:
        self.statuses.append(message)

    def on_file_preview(self, path: str, content: str) -> None:
        pass

    def on_edit_preview(self, path: str, diff: str) -> None:
        pass

    def on_content(self, text: str) -> None:
        self.content.append(text)

    def on_thinking_content(self, text: str) -> None:
        pass

    def on_thinking_start(self) -> None:
        pass

    def on_thinking_end(self) -> None:
        pass

    def had_tool_calls(self) -> bool:
        return bool(self.tool_calls)

    def finalize(self) -> str:
        return "".join(self.content)

    def cleanup(self) -> None:
        pass


class _AgentStreamSource:
    def __init__(self, events: list[AgentExecutionEvent]) -> None:
        self._events = events

    async def stream(self, _message: str):
        for event in self._events:
            yield event


def test_stream_response_normalizes_agent_tool_events() -> None:
    follow_ups = [
        {
            "command": 'graph(mode="neighbors", node="GraphAnalyzer", depth=2)',
            "description": "Inspect nearby graph nodes.",
        }
    ]
    agent = _AgentStreamSource(
        [
            AgentExecutionEvent(
                type=EventType.TOOL_CALL,
                tool_name="graph",
                arguments={"mode": "find", "query": "graph"},
            ),
            AgentExecutionEvent(
                type=EventType.TOOL_RESULT,
                tool_name="graph",
                result="Tool completed successfully.",
                success=True,
                metadata={
                    "tool_result": {
                        "name": "graph",
                        "success": True,
                        "elapsed": 0.25,
                        "arguments": {"mode": "find", "query": "graph"},
                        "follow_up_suggestions": follow_ups,
                        "result": "preview graph output",
                        "original_result": "full graph output",
                        "was_pruned": True,
                    }
                },
            ),
            AgentExecutionEvent(type=EventType.CONTENT, content="analysis complete"),
        ]
    )
    renderer = _RecorderRenderer()

    result = asyncio.run(stream_response(agent, "review graph code", renderer))

    assert renderer.tool_calls == [("graph", {"mode": "find", "query": "graph"})]
    assert renderer.tool_results == [
        {
            "name": "graph",
            "success": True,
            "elapsed": 0.25,
            "arguments": {"mode": "find", "query": "graph"},
            "error": None,
            "follow_up_suggestions": follow_ups,
            "was_pruned": True,
            "original_result": "full graph output",
            "result": "preview graph output",
        }
    ]
    assert result == "analysis complete"


@pytest.mark.parametrize("event_type", [EventType.ERROR, EventType.STREAM_END])
def test_stream_response_surfaces_error_events(event_type) -> None:
    """A failed run preserves partial output, raises, and closes before returning."""
    closed = []

    class Source:
        async def stream(self, _message):
            try:
                yield AgentExecutionEvent(type=EventType.CONTENT, content="Partial result")
                yield AgentExecutionEvent(
                    type=event_type,
                    success=False,
                    error="Run rejected",
                    recoverable=False,
                    metadata={"status": "must not mask failure"},
                )
                pytest.fail("Failed run must not consume later events")
            finally:
                await asyncio.sleep(0)
                closed.append(True)

    renderer = _RecorderRenderer()
    with pytest.raises(AgentError, match="Run rejected") as caught:
        asyncio.run(stream_response(Source(), "hi", renderer))
    assert caught.value.recoverable is False
    assert renderer.finalize() == "Partial result"
    assert renderer.statuses == ["❌ Run rejected"]
    assert closed == [True]


def test_stream_response_successful_terminal_preserves_content() -> None:
    renderer = _RecorderRenderer()
    agent = _AgentStreamSource(
        [
            AgentExecutionEvent(type=EventType.CONTENT, content="Complete"),
            AgentExecutionEvent(type=EventType.STREAM_END, success=True),
        ]
    )
    assert asyncio.run(stream_response(agent, "hi", renderer)) == "Complete"
    assert not renderer.statuses


def test_stream_response_thinking_abort_is_failure(monkeypatch) -> None:
    from victor.agent.response_sanitizer import create_streaming_filter

    content_filter = create_streaming_filter()
    monkeypatch.setattr(content_filter, "should_abort", lambda: True)
    monkeypatch.setattr(
        "victor.ui.rendering.handler.create_streaming_filter",
        lambda **kwargs: content_filter,
    )
    renderer = _RecorderRenderer()
    agent = _AgentStreamSource([AgentExecutionEvent(type=EventType.CONTENT, content="Partial")])
    with pytest.raises(AgentError, match="thinking"):
        asyncio.run(stream_response(agent, "hi", renderer))
    assert renderer.finalize() == "Partial"
    assert len(renderer.statuses) == 1
