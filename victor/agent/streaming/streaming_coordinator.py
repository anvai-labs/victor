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

"""Streaming Coordinator - Coordinates streaming response processing.

This module extracts streaming response processing from AgentOrchestrator:
- Chunk aggregation and formatting
- Streaming event dispatch
- Error handling during streaming
- Response processing lifecycle

Design Principles:
- Single Responsibility: Coordinates streaming response processing only
- Composable: Works with existing StreamingController
- Observable: Delegates to controller for metrics and events
- Testable: Pure functions for aggregation and formatting

Usage:
    coordinator = StreamingCoordinator(streaming_controller)

    # Process streaming response
    async for event in coordinator.process(response, context):
        # Handle streaming events
        pass

    # Aggregate chunks
    result = coordinator.aggregate_chunks(chunks)

    # Dispatch events
    await coordinator.dispatch_events(events, context)
"""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator, Dict, List, TYPE_CHECKING

if TYPE_CHECKING:
    from victor.agent.streaming_controller import StreamingController
    from victor.core.events.protocols import MessagingEvent

logger = logging.getLogger(__name__)


class StreamingCoordinator:
    """Coordinate streaming response processing.

    This coordinator handles the processing of streaming responses from LLMs,
    delegating to StreamingController for metrics and lifecycle management.

    Responsibilities:
    - Chunk aggregation: Combine streaming chunks into complete responses
    - Event dispatch: Emit events for observability
    - Error handling: Gracefully handle streaming errors
    - Output formatting: Format responses for display

    Example:
        coordinator = StreamingCoordinator(streaming_controller)

        # Process a streaming response
        async for event in coordinator.process(response_stream, context):
            if event.type == "chunk":
                print(event.data)
            elif event.type == "done":
                break

        # Aggregate chunks manually
        chunks = [{"content": "Hello"}, {"content": " World"}]
        result = coordinator.aggregate_chunks(chunks)  # "Hello World"
    """

    def __init__(self, streaming_controller: "StreamingController"):
        """Initialize the streaming coordinator.

        Args:
            streaming_controller: Controller for streaming session management
        """
        self._controller = streaming_controller

    async def process(
        self,
        response: AsyncIterator[Dict[str, Any]],
        context: Dict[str, Any],
    ) -> AsyncIterator[Dict[str, Any]]:
        """Process a streaming response into events.

        This method consumes the streaming response, aggregates chunks,
        and yields events for each chunk or completion.

        Args:
            response: Async iterator of streaming chunks
            context: Processing context (e.g., session_id, metadata)

        Yields:
            Dict events with 'type' and 'data' keys:
            - {'type': 'chunk', 'data': 'content'}
            - {'type': 'done', 'data': {...}}
            - {'type': 'error', 'data': 'error message'}

        Example:
            async for event in coordinator.process(response, context):
                if event['type'] == 'chunk':
                    print(event['data'], end='', flush=True)
                elif event['type'] == 'done':
                    print("\nDone!")
        """
        chunks: List[Dict[str, Any]] = []

        try:
            async for chunk in response:
                # Track chunk in controller
                if self._controller:
                    content_length = len(chunk.get("content", ""))
                    self._controller.record_chunk(content_length)

                # Collect chunk
                chunks.append(chunk)

                # Yield chunk event
                yield {"type": "chunk", "data": chunk}

            # Process complete chunks through controller
            if self._controller and hasattr(self._controller, "process_chunks"):
                processed = await self._controller.process_chunks(chunks, context)
                for item in processed:
                    yield item
            else:
                # Default behavior: yield done event
                yield {"type": "done", "data": {"chunks_count": len(chunks)}}

        except Exception as e:
            logger.error(f"Error processing streaming response: {e}")
            # Yield error event
            yield {"type": "error", "data": str(e)}

    def aggregate_chunks(self, chunks: List[Dict[str, Any]]) -> str:
        """Aggregate streaming chunks into a complete response.

        Delegates to the streaming controller if available, otherwise
        performs default aggregation.

        Args:
            chunks: List of chunk dictionaries

        Returns:
            Aggregated string content

        Example:
            chunks = [
                {"type": "chunk", "content": "Hello"},
                {"type": "chunk", "content": " World"},
            ]
            result = coordinator.aggregate_chunks(chunks)
            # Returns: "Hello World"
        """
        # Delegate to controller if available
        if self._controller and hasattr(self._controller, "aggregate_chunks"):
            return self._controller.aggregate_chunks(chunks)

        # Default aggregation
        if not chunks:
            return ""

        # Extract content from chunks
        contents = []
        for chunk in chunks:
            if isinstance(chunk, dict):
                content = chunk.get("content", "")
                if content:
                    contents.append(str(content))
            elif isinstance(chunk, str):
                contents.append(chunk)

        return "".join(contents)

    async def dispatch_events(
        self,
        events: List["MessagingEvent"],
        context: Dict[str, Any],
    ) -> None:
        """Dispatch streaming events for observability.

        This method emits events to registered listeners for monitoring,
        logging, or other observability purposes.

        Args:
            events: List of events to dispatch
            context: Processing context

        Example:
            events = [
                MessagingEvent(topic="streaming.chunk", data={"content": "Hello"}),
                MessagingEvent(topic="streaming.done", data={}),
            ]
            await coordinator.dispatch_events(events, context)
        """
        # Delegate to controller if it has dispatch capability
        if self._controller and hasattr(self._controller, "dispatch_events"):
            await self._controller.dispatch_events(events, context)
        else:
            # Default: just log the events
            for event in events:
                if hasattr(event, "topic"):
                    logger.debug(f"Event: {event.topic} - {event.data}")

    def format_output(self, data: Dict[str, Any]) -> str:
        """Format streaming output for display.

        Args:
            data: Output data to format

        Returns:
            Formatted string

        Example:
            output = coordinator.format_output({"data": "Hello"})
            # Returns: "Hello"
        """
        # Delegate to controller if available
        if self._controller and hasattr(self._controller, "format_output"):
            return self._controller.format_output(data)

        # Default formatting
        if "data" in data:
            return str(data["data"])
        elif "content" in data:
            return str(data["content"])
        else:
            return str(data)

    async def handle_completion(
        self,
        context: Dict[str, Any],
        metadata: Dict[str, Any],
    ) -> None:
        """Handle streaming completion.

        Called when streaming completes successfully or with an error.
        Delegates to controller for metrics cleanup and callbacks.

        Args:
            context: Processing context
            metadata: Completion metadata (e.g., turns, tokens)

        Example:
            await coordinator.handle_completion(
                context={"session_id": "abc123"},
                metadata={"turns": 5, "tokens": 1000}
            )
        """
        # Delegate to controller's on_completion if available
        if self._controller and hasattr(self._controller, "on_completion"):
            await self._controller.on_completion(context, metadata)
        else:
            # Default: just complete the session
            if self._controller:
                session = self._controller.complete_session()
                if session:
                    logger.debug(
                        f"Completed session {session.session_id}: " f"{session.duration:.2f}s"
                    )


def create_streaming_coordinator(
    streaming_controller: "StreamingController",
) -> StreamingCoordinator:
    """Factory function to create a StreamingCoordinator.

    Args:
        streaming_controller: Controller for streaming session management

    Returns:
        Configured StreamingCoordinator instance
    """
    return StreamingCoordinator(streaming_controller)


__all__ = [
    "StreamingCoordinator",
    "create_streaming_coordinator",
]
