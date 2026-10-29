"""Tests for StreamingLoopHandler."""

import pytest

from victor.agent.coordinators.streaming_loop_handler import (
    StreamingLoopHandler,
)


async def _async_gen(items):
    """Helper to create async generator from items."""
    for item in items:
        yield item


class TestStreamingLoopHandler:
    """Tests for streaming loop handler."""

    @pytest.mark.asyncio
    async def test_accumulate_string_chunks(self):
        handler = StreamingLoopHandler(max_iterations=10)
        chunks = ["Hello", " ", "World"]
        result = await handler.run_streaming_loop(_async_gen(chunks))
        assert result.content == "Hello World"

    @pytest.mark.asyncio
    async def test_accumulate_dict_chunks(self):
        handler = StreamingLoopHandler(max_iterations=10)
        chunks = [
            {"content": "Hello"},
            {"content": " World"},
            {"finish_reason": "stop"},
        ]
        result = await handler.run_streaming_loop(_async_gen(chunks))
        assert result.content == "Hello World"
        assert result.finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_tool_calls_extracted(self):
        handler = StreamingLoopHandler(max_iterations=10)
        chunks = [
            {"content": "Let me "},
            {"tool_calls": [{"name": "read", "args": {}}]},
            {"content": "read that."},
        ]
        result = await handler.run_streaming_loop(_async_gen(chunks))
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "read"

    @pytest.mark.asyncio
    async def test_callbacks_invoked(self):
        received = []

        async def on_chunk(content):
            received.append(("chunk", content))

        handler = StreamingLoopHandler(max_iterations=10)
        chunks = ["a", "b"]
        await handler.run_streaming_loop(_async_gen(chunks), on_chunk=on_chunk)
        assert len(received) == 2

    @pytest.mark.asyncio
    async def test_should_continue_respects_max(self):
        handler = StreamingLoopHandler(max_iterations=2)
        assert handler.should_continue()

        await handler.run_streaming_loop(_async_gen(["a"]))
        assert handler.should_continue()  # 1 < 2

        await handler.run_streaming_loop(_async_gen(["b"]))
        assert not handler.should_continue()  # 2 >= 2

    @pytest.mark.asyncio
    async def test_accumulate_chunks_simple(self):
        handler = StreamingLoopHandler()
        result = await handler.accumulate_chunks(_async_gen(["Hello", " ", "World"]))
        assert result == "Hello World"

    def test_reset(self):
        handler = StreamingLoopHandler(max_iterations=1)
        handler._iteration = 5
        handler.reset()
        assert handler._iteration == 0
        assert handler.should_continue()
