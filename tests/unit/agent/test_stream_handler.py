# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
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

"""Tests for stream_handler module."""

import pytest
import asyncio
from typing import AsyncIterator
from victor.agent.stream_handler import (
    StreamHandler,
    StreamResult,
    StreamMetrics,
    StreamBuffer,
)
from victor.providers.base import StreamChunk


async def mock_stream(
    chunks: list[StreamChunk],
) -> AsyncIterator[StreamChunk]:
    """Create an async iterator from a list of chunks."""
    for chunk in chunks:
        yield chunk


class TestStreamHandler:
    """Tests for StreamHandler class."""

    def test_stream_handler_init(self):
        """Test StreamHandler initialization."""
        handler = StreamHandler()
        assert handler.on_content is None
        assert handler.on_tool_call is None
        assert handler.timeout == 300.0

    def test_stream_handler_with_callbacks(self):
        """Test StreamHandler with callbacks."""
        content_received = []

        def on_content(chunk):
            content_received.append(chunk)

        handler = StreamHandler(on_content=on_content)
        assert handler.on_content is not None

    def test_stream_handler_timeout(self):
        """Test StreamHandler timeout setting."""
        handler = StreamHandler(timeout=60.0)
        assert handler.timeout == 60.0


class TestStreamResult:
    """Tests for StreamResult dataclass."""

    def test_stream_result_default(self):
        """Test StreamResult with defaults."""
        result = StreamResult()
        assert result.content == ""
        assert result.tool_calls == []
        assert result.stop_reason is None
        assert result.error is None

    def test_stream_result_with_content(self):
        """Test StreamResult with content."""
        result = StreamResult(content="Hello, World!")
        assert result.content == "Hello, World!"

    def test_stream_result_with_tool_calls(self):
        """Test StreamResult with tool calls."""
        tool_calls = [{"name": "test_tool", "args": {}}]
        result = StreamResult(tool_calls=tool_calls)
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "test_tool"


class TestStreamMetrics:
    """Tests for StreamMetrics dataclass."""

    def test_stream_metrics_default(self):
        """Test StreamMetrics with defaults."""
        metrics = StreamMetrics()
        assert metrics.start_time == 0.0
        assert metrics.first_token_time is None
        assert metrics.total_chunks == 0

    def test_stream_metrics_time_to_first_token(self):
        """Test time to first token calculation."""
        metrics = StreamMetrics(start_time=1.0, first_token_time=1.5)
        assert metrics.time_to_first_token == 0.5

    def test_stream_metrics_time_to_first_token_none(self):
        """Test time to first token when not set."""
        metrics = StreamMetrics(start_time=1.0)
        assert metrics.time_to_first_token is None

    def test_stream_metrics_total_duration(self):
        """Test total duration calculation."""
        metrics = StreamMetrics(start_time=1.0, end_time=5.0)
        assert metrics.total_duration == 4.0

    def test_stream_metrics_tokens_per_second(self):
        """Test tokens per second calculation."""
        metrics = StreamMetrics(start_time=1.0, end_time=2.0, total_content_length=400)
        # 400 chars / 4 = 100 tokens, 100 tokens / 1 sec = 100 tps
        assert metrics.tokens_per_second == 100.0

    def test_stream_metrics_tokens_per_second_zero_duration(self):
        """Test tokens per second with zero duration."""
        metrics = StreamMetrics(start_time=1.0, end_time=1.0)
        assert metrics.tokens_per_second == 0.0


class TestStreamHandlerProcessStream:
    """Tests for StreamHandler.process_stream method."""

    @pytest.mark.asyncio
    async def test_process_stream_content(self):
        """Test processing stream with content chunks."""
        handler = StreamHandler()

        chunks = [
            StreamChunk(content="Hello, "),
            StreamChunk(content="World!"),
            StreamChunk(is_final=True),
        ]

        result = await handler.process_stream(mock_stream(chunks))

        assert result.content == "Hello, World!"
        assert result.metrics.total_chunks == 3
        assert result.metrics.total_content_length == 13

    @pytest.mark.asyncio
    async def test_process_stream_with_callback(self):
        """Test process_stream with content callback."""
        received = []

        def on_content(chunk):
            received.append(chunk)

        handler = StreamHandler(on_content=on_content)

        chunks = [
            StreamChunk(content="Hello"),
            StreamChunk(content="World"),
            StreamChunk(is_final=True),
        ]

        await handler.process_stream(mock_stream(chunks))

        assert received == ["Hello", "World"]

    @pytest.mark.asyncio
    async def test_process_stream_with_tool_calls(self):
        """Test processing stream with tool calls."""
        tool_calls_received = []

        def on_tool_call(tc):
            tool_calls_received.append(tc)

        handler = StreamHandler(on_tool_call=on_tool_call)

        chunks = [
            StreamChunk(content="Using tool..."),
            StreamChunk(tool_calls=[{"name": "test_tool", "args": {"x": 1}}]),
            StreamChunk(is_final=True),
        ]

        result = await handler.process_stream(mock_stream(chunks))

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "test_tool"
        assert len(tool_calls_received) == 1

    @pytest.mark.asyncio
    async def test_process_stream_stop_reason(self):
        """Test process_stream captures stop reason."""
        handler = StreamHandler()

        chunks = [
            StreamChunk(content="Done"),
            StreamChunk(stop_reason="end_turn", is_final=True),
        ]

        result = await handler.process_stream(mock_stream(chunks))

        assert result.stop_reason == "end_turn"

    @pytest.mark.asyncio
    async def test_process_stream_on_complete_callback(self):
        """Test on_complete callback is called."""
        completed = []

        def on_complete(result):
            completed.append(result)

        handler = StreamHandler(on_complete=on_complete)

        chunks = [
            StreamChunk(content="Done", is_final=True),
        ]

        await handler.process_stream(mock_stream(chunks))

        assert len(completed) == 1
        assert completed[0].content == "Done"

    @pytest.mark.asyncio
    async def test_process_stream_callback_error_handled(self):
        """Test that callback errors are handled gracefully."""

        def bad_callback(chunk):
            raise ValueError("Callback error")

        handler = StreamHandler(on_content=bad_callback)

        chunks = [
            StreamChunk(content="Test"),
            StreamChunk(is_final=True),
        ]

        # Should not raise despite callback error
        result = await handler.process_stream(mock_stream(chunks))
        assert result.content == "Test"

    @pytest.mark.asyncio
    async def test_process_stream_metrics(self):
        """Test that metrics are collected during streaming."""
        handler = StreamHandler()

        chunks = [
            StreamChunk(content="Hello"),
            StreamChunk(content=" World"),
            StreamChunk(is_final=True),
        ]

        result = await handler.process_stream(mock_stream(chunks))

        assert result.metrics.total_chunks == 3
        assert result.metrics.total_content_length == 11
        assert result.metrics.first_token_time is not None
        assert result.metrics.end_time > result.metrics.start_time

    @pytest.mark.asyncio
    async def test_cancel_stream(self):
        """Test cancelling stream processing."""
        handler = StreamHandler()

        async def slow_stream():
            for i in range(10):
                if i == 2:
                    handler.cancel()
                yield StreamChunk(content=f"chunk{i}")
                await asyncio.sleep(0.01)

        result = await handler.process_stream(slow_stream())
        # Stream should be cancelled partway through
        assert result.metrics.total_chunks < 10

    def test_reset(self):
        """Test resetting handler state."""
        handler = StreamHandler()
        handler._cancelled = True
        handler._current_content = "test"
        handler._tool_calls = [{"test": "data"}]

        handler.reset()

        assert handler._cancelled is False
        assert handler._current_content == ""
        assert handler._tool_calls == []


class TestStreamBuffer:
    """Tests for StreamBuffer class."""

    def test_buffer_init(self):
        """Test StreamBuffer initialization."""
        buffer = StreamBuffer()
        assert buffer._buffers == {}

    def test_buffer_add_chunk_new(self):
        """Test adding chunk to new buffer."""
        buffer = StreamBuffer()

        chunk = {
            "function": {"name": "test_tool"},
            "index": 0,
        }

        result = buffer.add_chunk("call_1", chunk)
        # Should return None (not complete yet - no arguments)
        assert result is None
        assert "call_1" in buffer._buffers

    def test_buffer_add_chunk_complete(self):
        """Test adding chunk that completes the tool call."""
        buffer = StreamBuffer()

        # First chunk with name
        buffer.add_chunk("call_1", {"function": {"name": "test_tool"}})

        # Second chunk with complete arguments
        result = buffer.add_chunk("call_1", {"function": {"arguments": '{"x": 1}'}})

        # Should be complete (has name and args ending with })
        assert result is not None
        assert result["function"]["name"] == "test_tool"
        assert result["function"]["arguments"] == '{"x": 1}'

    def test_buffer_add_chunk_incremental_args(self):
        """Test adding arguments incrementally."""
        buffer = StreamBuffer()

        buffer.add_chunk("call_1", {"function": {"name": "test"}})
        buffer.add_chunk("call_1", {"function": {"arguments": '{"x":'}})
        result = buffer.add_chunk("call_1", {"function": {"arguments": " 1}"}})

        # Should be complete after full JSON
        assert result is not None
        assert result["function"]["arguments"] == '{"x": 1}'

    def test_buffer_clear(self):
        """Test clearing buffer."""
        buffer = StreamBuffer()
        buffer.add_chunk("call_1", {"function": {"name": "test"}})

        buffer.clear()

        assert buffer._buffers == {}

    def test_buffer_flush(self):
        """Test flushing incomplete buffers."""
        buffer = StreamBuffer()
        buffer.add_chunk("call_1", {"function": {"name": "test1"}})
        buffer.add_chunk("call_2", {"function": {"name": "test2"}})

        flushed = buffer.flush()

        assert len(flushed) == 2
        assert buffer._buffers == {}

    def test_buffer_flush_empty(self):
        """Test flushing empty buffer."""
        buffer = StreamBuffer()
        flushed = buffer.flush()
        assert flushed == []


class TestStreamHandlerEdgeCases:
    """Edge case tests for StreamHandler."""

    @pytest.mark.asyncio
    async def test_process_stream_timeout(self):
        """Test stream timeout handling (covers lines 179-184)."""
        handler = StreamHandler(timeout=0.01)  # Very short timeout

        async def slow_stream():
            yield StreamChunk(content="Start")
            await asyncio.sleep(1)  # Will trigger timeout
            yield StreamChunk(content="End", is_final=True)

        result = await handler.process_stream(slow_stream())
        assert result.error is not None
        assert "timed out" in result.error.lower()

    @pytest.mark.asyncio
    async def test_process_stream_timeout_with_on_error(self):
        """Test timeout calls on_error callback."""
        errors = []

        def on_error(e):
            errors.append(e)

        handler = StreamHandler(timeout=0.01, on_error=on_error)

        async def slow_stream():
            yield StreamChunk(content="Start")
            await asyncio.sleep(1)
            yield StreamChunk(is_final=True)

        await handler.process_stream(slow_stream())
        assert len(errors) == 1
        assert isinstance(errors[0], TimeoutError)

    @pytest.mark.asyncio
    async def test_process_stream_exception(self):
        """Test exception handling during stream (covers lines 185-189)."""
        errors = []

        def on_error(e):
            errors.append(e)

        handler = StreamHandler(on_error=on_error)

        async def error_stream():
            yield StreamChunk(content="Start")
            raise RuntimeError("Stream error")

        result = await handler.process_stream(error_stream())
        assert result.error == "Stream error"
        assert len(errors) == 1
        assert isinstance(errors[0], RuntimeError)

    @pytest.mark.asyncio
    async def test_process_stream_tool_call_callback_error(self):
        """Test tool call callback error is handled (covers lines 167-168)."""

        def bad_tool_callback(tc):
            raise ValueError("Bad callback")

        handler = StreamHandler(on_tool_call=bad_tool_callback)

        chunks = [
            StreamChunk(tool_calls=[{"name": "test", "args": {}}]),
            StreamChunk(is_final=True),
        ]

        # Should complete despite callback error
        result = await handler.process_stream(mock_stream(chunks))
        assert len(result.tool_calls) == 1

    @pytest.mark.asyncio
    async def test_process_stream_on_complete_callback_error(self):
        """Test on_complete callback error is handled (covers lines 205-206)."""

        def bad_complete_callback(result):
            raise ValueError("Complete callback error")

        handler = StreamHandler(on_complete=bad_complete_callback)

        chunks = [
            StreamChunk(content="Done", is_final=True),
        ]

        # Should complete despite callback error
        result = await handler.process_stream(mock_stream(chunks))
        assert result.content == "Done"


class TestStreamMetricsEdgeCases:
    """Edge case tests for StreamMetrics."""

    def test_total_duration_no_times(self):
        """Test total_duration returns 0 when times not set (covers line 51)."""
        metrics = StreamMetrics()
        assert metrics.total_duration == 0.0

    def test_total_duration_only_start(self):
        """Test total_duration with only start time."""
        metrics = StreamMetrics(start_time=1.0)
        assert metrics.total_duration == 0.0

    def test_total_duration_only_end(self):
        """Test total_duration with only end time."""
        metrics = StreamMetrics(end_time=5.0)
        assert metrics.total_duration == 0.0

    def test_time_to_first_token_no_start(self):
        """Test TTFT without start time."""
        metrics = StreamMetrics(first_token_time=1.5)
        assert metrics.time_to_first_token is None


class TestStreamBufferEdgeCases:
    """Edge case tests for StreamBuffer."""

    def test_buffer_array_arguments(self):
        """Test buffer completes with array JSON arguments."""
        buffer = StreamBuffer()
        buffer.add_chunk("call_1", {"function": {"name": "test"}})
        result = buffer.add_chunk("call_1", {"function": {"arguments": "[1, 2, 3]"}})
        assert result is not None
        assert result["function"]["arguments"] == "[1, 2, 3]"

    def test_buffer_empty_function_data(self):
        """Test buffer handles empty function data."""
        buffer = StreamBuffer()
        result = buffer.add_chunk("call_1", {"function": {}})
        assert result is None
        assert "call_1" in buffer._buffers

    def test_buffer_no_function_key(self):
        """Test buffer handles chunk without function key."""
        buffer = StreamBuffer()
        result = buffer.add_chunk("call_1", {"other": "data"})
        assert result is None
        assert "call_1" in buffer._buffers

    def test_buffer_multiple_calls(self):
        """Test buffer handles multiple concurrent tool calls."""
        buffer = StreamBuffer()

        buffer.add_chunk("call_1", {"function": {"name": "tool1"}})
        buffer.add_chunk("call_2", {"function": {"name": "tool2"}})

        result1 = buffer.add_chunk("call_1", {"function": {"arguments": "{}"}})
        result2 = buffer.add_chunk("call_2", {"function": {"arguments": "{}"}})

        assert result1 is not None
        assert result2 is not None
        assert result1["function"]["name"] == "tool1"
        assert result2["function"]["name"] == "tool2"


class TestWireTruthLatency:
    """W3b: StreamMetrics prefers sandhi's wire-measured latency."""

    def test_wire_latency_preferred_over_wall_clock(self):
        from victor.agent.stream_handler import StreamMetrics

        metrics = StreamMetrics(start_time=100.0, first_token_time=105.0, end_time=110.0)
        metrics.record_wire_latency({"duration_ms": 2000, "time_to_first_token_ms": 250})
        assert metrics.time_to_first_token == 0.25
        assert metrics.total_duration == 2.0

    def test_wall_clock_fallback_when_wire_absent(self):
        from victor.agent.stream_handler import StreamMetrics

        metrics = StreamMetrics(start_time=100.0, first_token_time=100.5, end_time=103.0)
        metrics.record_wire_latency({"attempts": 2})  # diagnostics without latency
        assert metrics.time_to_first_token == 0.5
        assert metrics.total_duration == 3.0

    def test_record_wire_latency_tolerates_junk(self):
        from victor.agent.stream_handler import StreamMetrics

        metrics = StreamMetrics()
        metrics.record_wire_latency(None)
        metrics.record_wire_latency({"duration_ms": "x", "time_to_first_token_ms": -5})
        assert metrics.wire_duration_ms is None
        assert metrics.wire_ttft_ms is None
