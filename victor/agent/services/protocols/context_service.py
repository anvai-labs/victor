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

"""Context service protocol.

Defines the interface for context and state management operations.
"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Protocol,
    Tuple,
    runtime_checkable,
)

if TYPE_CHECKING:
    Message = Dict[str, Any]
else:
    Message = Dict[str, Any]


@runtime_checkable
class ContextMetrics(Protocol):
    """Metrics about the current context state.

    Provides visibility into context size, composition, and health.
    """

    @property
    def total_tokens(self) -> int:
        """Total tokens in context."""
        ...

    @property
    def message_count(self) -> int:
        """Number of messages in context."""
        ...

    @property
    def user_message_count(self) -> int:
        """Number of user messages."""
        ...

    @property
    def assistant_message_count(self) -> int:
        """Number of assistant messages."""
        ...

    @property
    def tool_result_count(self) -> int:
        """Number of tool results in context."""
        ...

    @property
    def system_prompt_tokens(self) -> int:
        """Tokens used by system prompt."""
        ...

    @property
    def utilization_percent(self) -> float:
        """Context utilization as percentage (0-100)."""
        ...


@runtime_checkable
class ContextServiceProtocol(Protocol):
    """[CANONICAL] Protocol for context and state management service.

    This protocol represents the target architecture for context operations,
    replacing the facade-driven Coordinator pattern with a state-passed
    Service pattern.

    This protocol follows the Interface Segregation Principle (ISP)
    by focusing only on context-related operations.

    Methods:
        get_context_metrics: Get current context metrics
        check_context_overflow: Check if context exceeds limits
        compact_context: Compact context to fit within limits
        add_message: Add a message to context
        get_messages: Get messages from context
        clear_messages: Clear messages from context

    Example:
        class MyContextService(ContextServiceProtocol):
            def __init__(self, max_tokens=100000):
                self._max_tokens = max_tokens
                self._messages = []

            async def get_context_metrics(self):
                total_tokens = sum(m.token_count for m in self._messages)
                return ContextMetrics(
                    total_tokens=total_tokens,
                    message_count=len(self._messages),
                    utilization_percent=(total_tokens / self._max_tokens) * 100,
                )

            async def check_context_overflow(self):
                metrics = await self.get_context_metrics()
                return metrics.total_tokens > self._max_tokens

            async def compact_context(self):
                # Implement compaction strategy
                pass
    """

    async def get_context_metrics(self) -> "ContextMetrics":
        """Get current context metrics.

        Returns comprehensive metrics about the current context state
        including token count, message counts, and utilization.

        Returns:
            ContextMetrics with current context information

        Example:
            metrics = await context_service.get_context_metrics()
            print(f"Context: {metrics.total_tokens} tokens, "
                  f"{metrics.utilization_percent:.1f}% utilized")
        """
        ...

    async def check_context_overflow(self) -> bool:
        """Check if context exceeds limits.

        Determines whether the current context size exceeds
        the configured maximum and would benefit from compaction.

        Returns:
            True if context exceeds limits, False otherwise

        Example:
            if await context_service.check_context_overflow():
                await context_service.compact_context()
        """
        ...

    async def compact_context(
        self,
        strategy: str = "tiered",
        min_messages: int = 6,
    ) -> int:
        """Compact context to fit within limits.

        Removes less important messages while preserving
        critical context for conversation continuity.

        Args:
            strategy: Compaction strategy ("simple", "tiered", "semantic", "hybrid")
            min_messages: Minimum messages to retain

        Returns:
            Number of messages removed

        Raises:
            ContextCompactionError: If compaction fails

        Example:
            removed = await context_service.compact_context(strategy="tiered")
            logger.info(f"Compacted context by removing {removed} messages")
        """
        ...

    async def prepare_for_tool_output_injection(
        self,
        estimated_output_tokens: int,
        *,
        provider_name: str = "",
        model_name: str = "",
        task_type: str = "",
        min_messages: int = 6,
        default_strategy: str = "tiered",
    ) -> Dict[str, Any]:
        """Optionally compact before large tool outputs are injected into context.

        The canonical runtime uses this hook to keep pre-injection compaction
        policy owned by ContextService rather than scattered across execution
        helpers.

        Args:
            estimated_output_tokens: Estimated tokens for the incoming tool output
            provider_name: Active provider identifier
            model_name: Active model identifier
            task_type: Current task classification
            min_messages: Minimum messages to retain if compaction runs
            default_strategy: Preferred fallback strategy

        Returns:
            Structured decision/result metadata including whether compaction was
            needed, whether it occurred, the chosen strategy, and the policy reason.
        """
        ...

    def add_message(self, message: "Message") -> None:
        """Add a message to the context.

        Args:
            message: Message to add

        Raises:
            ContextOverflowError: If adding would exceed maximum size
        """
        ...

    def add_messages(self, messages: List["Message"]) -> None:
        """Add multiple messages to the context.

        Args:
            messages: Messages to add

        Raises:
            ContextOverflowError: If adding would exceed maximum size
        """
        ...

    def get_messages(
        self,
        limit: Optional[int] = None,
        role: Optional[str] = None,
    ) -> List["Message"]:
        """Get messages from context.

        Args:
            limit: Maximum number of messages to return
            role: Filter by message role (user, assistant, system, tool)

        Returns:
            List of messages

        Example:
            # Get last 10 messages
            recent = context_service.get_messages(limit=10)

            # Get only user messages
            user_msgs = context_service.get_messages(role="user")
        """
        ...

    def clear_messages(self, retain_system: bool = True) -> None:
        """Clear messages from context.

        Args:
            retain_system: If True, retain system prompt
        """
        ...

    def get_max_tokens(self) -> int:
        """Get the maximum context token limit.

        Returns:
            Maximum tokens allowed in context
        """
        ...

    def set_max_tokens(self, max_tokens: int) -> None:
        """Set the maximum context token limit.

        Args:
            max_tokens: Maximum tokens allowed

        Raises:
            ValueError: If max_tokens is negative
        """
        ...

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text.

        Provides fast token estimation without full tokenization.
        Useful for checking size before adding to context.

        Args:
            text: Text to estimate tokens for

        Returns:
            Estimated token count

        Example:
            text = "This is a sample message"
            if context_service.estimate_tokens(text) > 1000:
                logger.warning("Message is very large")
        """
        ...

    def is_healthy(self) -> bool:
        """Check if the context service is healthy.

        A healthy context service should:
        - Have valid max_tokens configured
        - Not be in overflow state
        - Have message history accessible

        Returns:
            True if the service is healthy, False otherwise
        """
        ...


@runtime_checkable
class ContextCompactionStrategy(Protocol):
    """Protocol for context compaction strategies.

    Different strategies for deciding what to remove when
    compacting context.
    """

    async def select_messages_to_remove(
        self,
        messages: List["Message"],
        target_count: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[int]:
        """Select indices of messages to remove.

        Args:
            messages: Current message list
            target_count: Number of messages to remove
            metadata: Additional metadata for decision making

        Returns:
            List of message indices to remove
        """
        ...

    @property
    def name(self) -> str:
        """Strategy name."""
        ...


@runtime_checkable
class SemanticContextServiceProtocol(Protocol):
    """Extended protocol for semantic context operations.

    Provides advanced context operations using semantic
    analysis and embeddings.
    """

    async def find_relevant_messages(
        self,
        query: str,
        limit: int = 5,
    ) -> List[Tuple["Message", float]]:
        """Find semantically relevant messages to query.

        Uses embeddings to find messages similar to the query,
        useful for context-aware retrieval.

        Args:
            query: Query to find relevant messages for
            limit: Maximum number of messages to return

        Returns:
            List of (message, similarity_score) tuples

        Example:
            relevant = await context_service.find_relevant_messages(
                "What files were modified?",
                limit=3
            )
            for msg, score in relevant:
                print(f"[{score:.2f}] {msg.content[:50]}...")
        """
        ...

    async def summarize_context(self) -> str:
        """Generate a summary of the current context.

        Creates a condensed summary of the conversation
        for use in compaction or context injection.

        Returns:
            Context summary string

        Example:
            summary = await context_service.summarize_context()
            logger.info(f"Context summary: {summary}")
        """
        ...
