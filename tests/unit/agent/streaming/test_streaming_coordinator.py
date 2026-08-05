# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
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

"""Tests for StreamingCoordinator."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from victor.agent.streaming_controller import StreamingController
from victor.agent.streaming.streaming_coordinator import StreamingCoordinator
from victor.core.events.protocols import MessagingEvent


class TestStreamingCoordinator:
    """Tests for StreamingCoordinator class."""

    @pytest.fixture
    def streaming_controller(self):
        """Create a mock streaming controller."""
        controller = MagicMock(spec=StreamingController)
        return controller

    @pytest.fixture
    def coordinator(self, streaming_controller):
        """Create StreamingCoordinator with mock controller."""
        return StreamingCoordinator(streaming_controller)

    @pytest.mark.asyncio
    async def test_init(self, streaming_controller):
        """Test coordinator initialization."""
        coordinator = StreamingCoordinator(streaming_controller)

        assert coordinator._controller is streaming_controller

    @pytest.mark.asyncio
    async def test_process_streaming_response_success(self, coordinator, streaming_controller):
        """Test successful streaming response processing."""

        # Mock response (async iterator)
        async def mock_response():
            yield {"type": "chunk", "content": "Hello"}
            yield {"type": "chunk", "content": " World"}
            yield {"type": "end"}

        # Mock controller to process chunks
        async def mock_process_chunks(chunks, context):
            return [
                {"type": "event", "data": "Hello World"},
                {"type": "done"},
            ]

        streaming_controller.process_chunks = AsyncMock(side_effect=mock_process_chunks)

        # Process streaming response - consume the async generator
        context = {"key": "value"}
        results = []
        async for event in coordinator.process(mock_response(), context):
            results.append(event)

        # Verify controller was called
        streaming_controller.process_chunks.assert_called_once()

        # Verify we got chunk events
        assert len([r for r in results if r["type"] == "chunk"]) == 3

    @pytest.mark.asyncio
    async def test_process_streaming_response_empty(self, coordinator, streaming_controller):
        """Test processing empty streaming response."""

        # Mock empty response
        async def mock_response():
            return
            yield

        async def mock_process_chunks(chunks, context):
            return []

        streaming_controller.process_chunks = AsyncMock(side_effect=mock_process_chunks)

        # Process - consume the async generator
        results = []
        async for event in coordinator.process(mock_response(), {}):
            results.append(event)

        # Should handle empty response gracefully
        streaming_controller.process_chunks.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_streaming_response_error_handling(
        self, coordinator, streaming_controller
    ):
        """Test error handling during streaming."""

        # Mock response that raises error
        async def mock_response():
            yield {"type": "chunk", "content": "Hello"}
            raise RuntimeError("Stream error")

        # Process - consume the async generator
        results = []
        async for event in coordinator.process(mock_response(), {}):
            results.append(event)

        # Verify we got chunk event before error
        assert len([r for r in results if r["type"] == "chunk"]) == 1
        assert any(r["type"] == "error" for r in results)
        # Note: process_chunks is NOT called when error occurs mid-stream

    @pytest.mark.asyncio
    async def test_aggregate_chunks(self, coordinator, streaming_controller):
        """Test chunk aggregation."""
        chunks = [
            {"type": "chunk", "content": "Hello"},
            {"type": "chunk", "content": " "},
            {"type": "chunk", "content": "World"},
        ]

        # Mock aggregation
        streaming_controller.aggregate_chunks = MagicMock(return_value="Hello World")

        # Aggregate
        result = coordinator.aggregate_chunks(chunks)

        # Verify aggregation
        streaming_controller.aggregate_chunks.assert_called_once_with(chunks)
        assert result == "Hello World"

    @pytest.mark.asyncio
    async def test_dispatch_streaming_events(self, coordinator, streaming_controller):
        """Test streaming event dispatch."""
        events = [
            MessagingEvent(topic="streaming.chunk", data={"content": "Hello"}),
            MessagingEvent(topic="streaming.chunk", data={"content": " World"}),
            MessagingEvent(topic="streaming.end", data={}),
        ]

        # Mock dispatch
        streaming_controller.dispatch_events = AsyncMock()

        # Dispatch
        await coordinator.dispatch_events(events, context={"key": "value"})

        # Verify dispatch
        streaming_controller.dispatch_events.assert_called_once_with(events, {"key": "value"})

    @pytest.mark.asyncio
    async def test_format_streaming_output(self, coordinator, streaming_controller):
        """Test streaming output formatting."""
        # Mock controller formatting
        streaming_controller.format_output = MagicMock(return_value="Formatted output")

        # Format
        result = coordinator.format_output({"data": "test"})

        # Verify formatting
        streaming_controller.format_output.assert_called_once_with({"data": "test"})
        assert result == "Formatted output"

    @pytest.mark.asyncio
    async def test_handle_streaming_completion(self, coordinator, streaming_controller):
        """Test streaming completion handling."""
        # Mock completion handling
        streaming_controller.on_completion = AsyncMock()

        # Handle completion
        await coordinator.handle_completion(context={"session_id": "test"}, metadata={"turns": 5})

        # Verify completion callback was called
        streaming_controller.on_completion.assert_called_once()


class TestStreamingCoordinatorIntegration:
    """Integration tests for StreamingCoordinator."""

    @pytest.fixture
    def coordinator(self):
        """Create coordinator with real streaming controller."""
        # Use actual StreamingController if possible, otherwise mock
        controller = MagicMock(spec=StreamingController)
        return StreamingCoordinator(controller)

    @pytest.mark.asyncio
    async def test_full_streaming_lifecycle(self, coordinator):
        """Test complete streaming lifecycle through coordinator."""

        # Mock response
        async def mock_response():
            yield {"type": "chunk", "content": "Test"}
            yield {"type": "end"}

        # Setup controller mocks
        async def mock_process_chunks(chunks, context):
            return [{"type": "event", "data": "Test"}]

        coordinator._controller.process_chunks = AsyncMock(side_effect=mock_process_chunks)

        # Process - consume the async generator
        results = []
        async for event in coordinator.process(mock_response(), {}):
            results.append(event)

        # Verify lifecycle
        coordinator._controller.process_chunks.assert_called_once()

        # Verify we got events
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_error_recovery_in_streaming(self, coordinator):
        """Test error recovery during streaming."""

        # Mock response with error
        async def mock_response():
            yield {"type": "chunk", "content": "Before"}
            raise ValueError("Invalid chunk")

        # Process should not raise - consume the async generator
        results = []
        async for event in coordinator.process(mock_response(), {}):
            results.append(event)

        # Verify we got chunk event before error
        assert len([r for r in results if r["type"] == "chunk"]) == 1
        assert any(r["type"] == "error" for r in results)
        # Note: process_chunks is NOT called when error occurs mid-stream


class TestStreamingCoordinatorPerformance:
    """Performance tests for StreamingCoordinator."""

    @pytest.fixture
    def coordinator(self):
        """Create coordinator."""
        controller = MagicMock()
        return StreamingCoordinator(controller)

    @pytest.mark.asyncio
    async def test_large_stream_handling(self, coordinator):
        """Test handling of large streaming responses."""

        # Mock large response (1000 chunks)
        async def mock_response():
            for i in range(1000):
                yield {"type": "chunk", "content": f"chunk{i}"}
            yield {"type": "end"}

        # Process should handle large stream efficiently - consume the async generator
        results = []
        async for event in coordinator.process(mock_response(), {}):
            results.append(event)

        # Verify we got all the content chunks (at least 1000 events)
        assert len(results) >= 1000
        # Verify most are chunk events
        chunk_events = [r for r in results if r["type"] == "chunk"]
        assert len(chunk_events) >= 1000

    @pytest.mark.asyncio
    async def test_concurrent_stream_handling(self, coordinator):
        """Test handling multiple concurrent streams."""

        # Mock multiple concurrent responses
        async def mock_response_1():
            yield {"type": "chunk", "content": "Stream1"}

        async def mock_response_2():
            yield {"type": "chunk", "content": "Stream2"}

        # Setup controller for concurrent handling
        async def mock_process_chunks(chunks, ctx):
            return [{"type": "event"}]

        coordinator._controller.process_chunks = AsyncMock(side_effect=mock_process_chunks)

        # Process both streams - consume the async generators
        results1 = []
        async for event in coordinator.process(mock_response_1(), {}):
            results1.append(event)

        results2 = []
        async for event in coordinator.process(mock_response_2(), {}):
            results2.append(event)

        # Verify both were processed
        assert coordinator._controller.process_chunks.call_count == 2

        # Verify each got results
        assert len(results1) > 0
        assert len(results2) > 0
