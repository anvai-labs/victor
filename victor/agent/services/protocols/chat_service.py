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

"""Chat service protocol.

Defines the interface for chat operations, handling the core flow
of processing user messages and generating responses.
"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Dict,
    Optional,
    Protocol,
    runtime_checkable,
)

if TYPE_CHECKING:
    from victor.providers.base import CompletionResponse, StreamChunk


@runtime_checkable
class ChatServiceProtocol(Protocol):
    """[CANONICAL] Protocol for chat operations service.

    This protocol represents the target architecture for chat operations,
    replacing the facade-driven Coordinator pattern with a state-passed
    Service pattern.

    This protocol follows the Interface Segregation Principle (ISP)
    by focusing only on chat-related operations.

    Methods:
        chat: Process a chat message with optional streaming
        stream_chat: Stream chat response in chunks
        reset_conversation: Reset conversation history

    Example:
        class MyChatService(ChatServiceProtocol):
            def __init__(self, provider, tool_service, context_service):
                self._provider = provider
                self._tools = tool_service
                self._context = context_service

            async def chat(self, user_message: str, **kwargs) -> CompletionResponse:
                # Process message with agentic loop
                pass

            async def stream_chat(self, user_message: str, **kwargs) -> AsyncIterator[StreamChunk]:
                # Stream response chunks
                pass

            def reset_conversation(self) -> None:
                # Clear conversation history
                pass
    """

    async def chat(
        self, user_message: str, *, stream: bool = False, **kwargs
    ) -> "CompletionResponse":
        """Process a chat message with the agentic loop.

        This method handles the core chat flow:
        1. Analyze the user's request
        2. Select appropriate tools
        3. Execute tools as needed
        4. Generate the final response
        5. Handle recovery from errors

        Args:
            user_message: The user's input message
            stream: If True, return a streaming response
            **kwargs: Additional options for the chat (temperature, max_tokens, etc.)

        Returns:
            CompletionResponse with the generated response content

        Raises:
            ProviderError: If the provider fails to generate a response
            ToolExecutionError: If tool execution fails critically
            ContextOverflowError: If context exceeds limits and can't be compacted
        """
        ...

    async def stream_chat(self, user_message: str, **kwargs) -> AsyncIterator["StreamChunk"]:
        """Stream chat response in chunks for real-time feedback.

        Provides incremental response chunks as they're generated,
        enabling real-time feedback and better UX for long responses.

        Args:
            user_message: The user's input message
            **kwargs: Additional options for the chat

        Yields:
            StreamChunk objects with incremental response content

        Raises:
            ProviderError: If the provider fails during streaming
            ToolExecutionError: If tool execution fails critically
        """
        ...

    async def chat_with_planning(
        self,
        user_message: str,
        use_planning: Optional[bool] = None,
    ) -> "CompletionResponse":
        """Process a chat message with optional planning support.

        Args:
            user_message: The user's input message
            use_planning: Force planning on/off. None means auto-detect.

        Returns:
            CompletionResponse with the generated response content
        """
        ...

    async def handle_context_and_iteration_limits(
        self,
        user_message: str,
        max_total_iterations: int,
        max_context: int,
        total_iterations: int,
        last_quality_score: float,
    ) -> tuple[bool, Optional["StreamChunk"]]:
        """Handle context overflow and iteration hard limits during streaming."""
        ...

    def reset_conversation(self) -> None:
        """Reset the conversation history and state.

        Clears all conversation context, effectively starting
        a new conversation session.

        This is useful for:
        - Starting fresh conversations
        - Testing and development
        - Clearing state after errors
        """
        ...

    @staticmethod
    def persist_message(
        role: str,
        content: str,
        memory_manager: Optional[Any] = None,
        memory_session_id: Optional[str] = None,
        usage_logger: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Persist a message to memory and log usage events.

        Handles async-aware thread pool offloading for SQLite I/O
        and logs user_prompt/assistant_response events.

        Args:
            role: Message role (user, assistant, system)
            content: Message content
            memory_manager: Optional memory manager for persistence
            memory_session_id: Optional session ID for memory
            usage_logger: Optional logger for usage events
            metadata: Optional message metadata for persistence
        """
        ...

    def is_healthy(self) -> bool:
        """Check if the chat service is healthy.

        A healthy chat service should:
        - Have a valid provider connection
        - Have tool service available
        - Have context service available

        Returns:
            True if the service is healthy, False otherwise
        """
        ...


@runtime_checkable
class StreamingChatServiceProtocol(Protocol):
    """Extended protocol for streaming-specific chat operations.

    Provides additional methods for advanced streaming scenarios
    like chunk aggregation and cancellation.

    This protocol can be implemented by services that need
    more control over streaming behavior.
    """

    async def aggregate_chunks(
        self,
        chunks: AsyncIterator["StreamChunk"],
        timeout: Optional[float] = None,
    ) -> "CompletionResponse":
        """Aggregate streaming chunks into a complete response.

        Args:
            chunks: Iterator of stream chunks
            timeout: Optional timeout for aggregation

        Returns:
            Aggregated CompletionResponse

        Raises:
            TimeoutError: If aggregation times out
        """
        ...

    def cancel_streaming(self) -> None:
        """Cancel any active streaming operations.

        Useful for:
        - User cancellation requests
        - Timeout scenarios
        - Error recovery
        """
        ...
