"""Protocol definitions for conversation protocols."""

from __future__ import annotations

from typing import (
    Any,
    Dict,
    List,
    Optional,
    Protocol,
    TYPE_CHECKING,
    runtime_checkable,
)

if TYPE_CHECKING:
    from victor.agent.conversation.state_machine import ConversationStage


__all__ = [
    "ConversationControllerProtocol",
    "ConversationStateMachineProtocol",
    "MessageHistoryProtocol",
    "ConversationEmbeddingStoreProtocol",
    "IMessageStore",
    "IContextOverflowHandler",
    "ISessionManager",
    "IEmbeddingManager",
]


@runtime_checkable
class ConversationControllerProtocol(Protocol):
    """Protocol for conversation management.

    Manages message history, context tracking, and compaction.
    """

    def add_message(self, role: str, content: str) -> None:
        """Add a message to conversation history."""
        ...

    def get_messages(self) -> List[Dict[str, Any]]:
        """Get all messages in conversation."""
        ...

    def get_context_metrics(self) -> Any:
        """Get current context utilization metrics."""
        ...

    def compact_if_needed(self) -> bool:
        """Compact conversation if context is nearly full.

        Returns:
            True if compaction occurred
        """
        ...

    def set_system_prompt(self, prompt: str) -> None:
        """Set the system prompt."""
        ...


@runtime_checkable
class ConversationStateMachineProtocol(Protocol):
    """Protocol for conversation state machine.

    Tracks conversation stage (INITIAL, PLANNING, EXECUTING, etc.).
    """

    def get_stage(self) -> "ConversationStage":
        """Get current conversation stage."""
        ...

    def get_current_stage(self) -> "ConversationStage":
        """Get current conversation stage (alias)."""
        ...

    def record_tool_execution(self, tool_name: str, args: Dict[str, Any]) -> None:
        """Record tool execution for stage inference."""
        ...

    def record_message(self, content: str, is_user: bool = True) -> None:
        """Record a message for stage inference."""
        ...


@runtime_checkable
class MessageHistoryProtocol(Protocol):
    """Protocol for message history.

    Manages raw message storage.
    """

    def add_message(self, role: str, content: str, **kwargs: Any) -> Any:
        """Add a message."""
        ...

    def get_messages_for_provider(self) -> List[Any]:
        """Get all messages for provider."""
        ...

    def clear(self) -> None:
        """Clear message history."""
        ...


@runtime_checkable
class ConversationEmbeddingStoreProtocol(Protocol):
    """Protocol for conversation embedding storage.

    Provides semantic search over conversation history using embeddings.
    """

    @property
    def is_initialized(self) -> bool:
        """Check if the store is initialized."""
        ...

    async def initialize(self) -> None:
        """Initialize the embedding store."""
        ...

    async def search_similar(
        self,
        query: str,
        session_id: Optional[str] = None,
        limit: int = 10,
        min_similarity: float = 0.3,
        exclude_message_ids: Optional[List[str]] = None,
    ) -> List[Any]:
        """Search for semantically similar messages.

        Args:
            query: Query text to search for
            session_id: Optional session to scope search
            limit: Maximum number of results
            min_similarity: Minimum similarity threshold (0-1)
            exclude_message_ids: Message IDs to exclude

        Returns:
            List of search results (message_id + similarity)
        """
        ...

    async def delete_session(self, session_id: str) -> int:
        """Delete all embeddings for a session.

        Args:
            session_id: Session ID to delete

        Returns:
            Number of embeddings deleted
        """
        ...

    async def get_stats(self) -> Dict[str, Any]:
        """Get store statistics."""
        ...

    async def close(self) -> None:
        """Clean up resources."""
        ...


class IMessageStore(Protocol):
    """Protocol for message storage and retrieval.

    Defines interface for persisting and retrieving messages.
    Separated from other conversation concerns to follow ISP.
    """

    def add_message(self, role: str, content: str, **metadata) -> None:
        """Add a message to storage.

        Args:
            role: Message role (user, assistant, system, tool)
            content: Message content
            **metadata: Additional message metadata
        """
        ...

    def get_messages(self, limit: Optional[int] = None) -> List[Any]:
        """Retrieve messages.

        Args:
            limit: Optional limit on number of messages

        Returns:
            List of messages
        """
        ...

    def persist(self) -> bool:
        """Persist messages to storage.

        Returns:
            True if persistence succeeded, False otherwise
        """
        ...


class IContextOverflowHandler(Protocol):
    """Protocol for context overflow handling.

    Defines interface for detecting and handling context overflow.
    Separated from IMessageStore to follow ISP.
    """

    def check_overflow(self) -> bool:
        """Check if context has overflowed.

        Returns:
            True if overflow detected, False otherwise
        """
        ...

    def handle_compaction(self) -> Optional[Any]:
        """Handle context compaction.

        Returns:
            Compaction result or None
        """
        ...


class ISessionManager(Protocol):
    """Protocol for session lifecycle management.

    Defines interface for creating and managing sessions.
    Separated to support different session backends.
    """

    def create_session(self) -> str:
        """Create a new session.

        Returns:
            Session ID
        """
        ...

    def recover_session(self, session_id: str) -> bool:
        """Recover an existing session.

        Args:
            session_id: Session ID to recover

        Returns:
            True if recovery succeeded, False otherwise
        """
        ...

    def persist_session(self) -> bool:
        """Persist session state.

        Returns:
            True if persistence succeeded, False otherwise
        """
        ...


class IEmbeddingManager(Protocol):
    """Protocol for embedding and semantic search.

    Defines interface for semantic search over conversations.
    Separated because not all conversations need embeddings.
    """

    def initialize_embeddings(self) -> None:
        """Initialize embedding store."""
        ...

    def semantic_search(self, query: str, k: int = 5) -> List[Any]:
        """Perform semantic search.

        Args:
            query: Search query
            k: Number of results to return

        Returns:
            List of search results
        """
        ...
