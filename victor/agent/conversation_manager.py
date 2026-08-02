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

"""Conversation management facade composing controller, store, and embedding services.

This module provides a unified interface for conversation management by composing:
- MessageStore: Message storage and persistence
- ContextOverflowHandler: Context overflow detection and compaction
- SessionManager: Session lifecycle management
- EmbeddingManager: Embedding and semantic search

The ConversationManager acts as a Facade, simplifying the interface for
conversation operations while delegating to specialized components.

SRP Compliance:
ConversationManager now delegates to focused components instead of
implementing all functionality directly. This improves maintainability
and testability by separating concerns.

Part of TD-002: AgentOrchestrator god class refactoring.
Part of SOLID-based refactoring to eliminate god class anti-pattern.

Usage:
    from victor.agent.conversation_manager import ConversationManager
    from victor.config.settings import Settings

    manager = ConversationManager(settings, provider, model, system_prompt)
    manager.add_user_message("Hello!")
    manager.add_assistant_message("Hi there!")

    # Get context metrics
    metrics = manager.get_context_metrics()

    # Session persistence
    session_id = manager.session_id
    sessions = manager.get_recent_sessions(limit=5)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING

from victor.agent.conversation.controller import (
    ConversationController,
    ConversationConfig,
    ContextMetrics,
)
from victor.agent.conversation.state_machine import ConversationStage
from victor.agent.conversation import (
    MessageStore,
    ContextOverflowHandler,
    SessionManager,
    EmbeddingManager,
)

if TYPE_CHECKING:
    from victor.agent.conversation.store import ConversationStore, ConversationSession
    from victor.agent.conversation_embedding_store import ConversationEmbeddingStore
    from victor.config.settings import Settings
    from victor.storage.embeddings.service import EmbeddingService
    from victor.providers.base import Message


logger = logging.getLogger(__name__)


@dataclass
class ConversationManagerConfig:
    """Configuration for ConversationManager.

    Attributes:
        enable_persistence: Whether to persist conversations to SQLite
        enable_embeddings: Whether to enable LanceDB embedding store
        max_context_chars: Maximum context size in characters
        chars_per_token_estimate: Estimate for token calculation
        enable_stage_tracking: Whether to track conversation stages
        auto_compaction: Whether to automatically compact on overflow
    """

    enable_persistence: bool = True
    enable_embeddings: bool = True
    max_context_chars: int = 200000
    chars_per_token_estimate: int = 3
    enable_stage_tracking: bool = True
    auto_compaction: bool = True


class ConversationManager:
    """Facade for unified conversation management.

    Composes ConversationController, ConversationStore, and ConversationEmbeddingStore
    to provide a simplified interface for conversation operations.

    This class is part of the TD-002 refactoring effort to extract conversation
    management concerns from AgentOrchestrator.

    Responsibilities:
    - Message addition with persistence
    - Context metrics and overflow detection
    - Session management (create, recover, list)
    - Stage tracking delegation
    - Lazy embedding store initialization

    Example:
        manager = ConversationManager(
            settings=settings,
            provider="anthropic",
            model="claude-3-sonnet",
            system_prompt="You are a helpful assistant.",
        )

        manager.add_user_message("Hello!")
        response = await some_llm_call(manager.messages)
        manager.add_assistant_message(response.content)

        if manager.check_context_overflow():
            manager.handle_compaction(current_message)
    """

    def __init__(
        self,
        settings: Optional["Settings"] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        config: Optional[ConversationManagerConfig] = None,
        controller: Optional[ConversationController] = None,
        store: Optional["ConversationStore"] = None,
        embedding_service: Optional["EmbeddingService"] = None,
        retrieval_gateway: Optional[Any] = None,
        session_id: Optional[str] = None,
    ):
        """Initialize ConversationManager using composition.

        Args:
            settings: Application settings
            provider: LLM provider name (e.g., "anthropic", "openai")
            model: Model identifier (e.g., "claude-3-sonnet")
            system_prompt: System prompt to initialize conversation with
            config: Manager configuration options
            controller: Optional pre-configured ConversationController
            store: Optional pre-configured ConversationStore
            embedding_service: Optional embedding service for semantic features
            retrieval_gateway: Optional RetrievalGateway for semantic search
            session_id: Optional session ID to recover/continue
        """
        self._settings = settings
        self._provider = provider
        self._model = model
        self._config = config or ConversationManagerConfig()

        # Build conversation config from manager config
        conv_config = ConversationConfig(
            max_context_chars=self._config.max_context_chars,
            chars_per_token_estimate=self._config.chars_per_token_estimate,
            enable_stage_tracking=self._config.enable_stage_tracking,
        )

        # Initialize controller (required component)
        self._controller = controller or ConversationController(config=conv_config)

        # Set system prompt if provided
        if system_prompt:
            self._controller.set_system_prompt(system_prompt)

        # Initialize specialized components (DIP: Depend on abstractions)
        self._message_store = MessageStore(
            controller=self._controller,
            store=store,
            enable_persistence=self._config.enable_persistence,
        )

        self._context_handler = ContextOverflowHandler(
            controller=self._controller,
            max_context_chars=self._config.max_context_chars,
            chars_per_token=self._config.chars_per_token_estimate,
        )

        self._session_manager = SessionManager(
            store=store,
            enable_persistence=self._config.enable_persistence,
        )

        self._embedding_manager = EmbeddingManager(
            embedding_store=None,  # Will initialize later
            enable_embeddings=self._config.enable_embeddings,
        )

        # Keep references for backward compatibility
        self._store: Optional["ConversationStore"] = store
        self._embedding_store: Optional["ConversationEmbeddingStore"] = None
        self._embedding_service = embedding_service
        self._retrieval_gateway = retrieval_gateway
        self._session_id = session_id
        self._session: Optional["ConversationSession"] = None

        # Initialize persistence if enabled and store provided
        if self._store and self._config.enable_persistence:
            self._initialize_session()

        logger.debug(
            f"ConversationManager initialized: provider={provider}, model={model}, "
            f"persistence={self._config.enable_persistence}"
        )

    def _initialize_session(self) -> None:
        """Initialize or recover conversation session."""
        if not self._store:
            return

        if self._session_id:
            # Try to recover existing session
            self._session = self._store.get_session(self._session_id)
            if self._session:
                logger.info(f"Recovered session: {self._session_id}")
                return

        # Create new session
        project_path = None
        if self._settings:
            project_path = str(getattr(self._settings, "project_path", None) or "")

        self._session = self._store.create_session(
            session_id=self._session_id,
            project_path=project_path,
            provider=self._provider,
            model=self._model,
        )
        self._session_id = self._session.session_id
        logger.info(f"Created new session: {self._session_id}")

        # Ensure .victor/ subdirs exist including .victor/analysis/ for checkpoints
        try:
            from victor.config.settings import ProjectPaths

            ProjectPaths().ensure_project_dirs()
        except Exception:
            pass

    # =========================================================================
    # MESSAGE MANAGEMENT
    # =========================================================================

    def add_message(self, role: str, content: str) -> "Message":
        """Add a message with specified role.

        Delegates to MessageStore (SRP).

        Args:
            role: Message role (user, assistant, system)
            content: Message content

        Returns:
            The created Message object
        """
        return self._message_store.add_message(role, content)

    def add_user_message(self, content: str) -> "Message":
        """Add a user message.

        Delegates to MessageStore (SRP).

        Args:
            content: User message content

        Returns:
            The created Message object
        """
        return self._message_store.add_user_message(content)

    def add_assistant_message(
        self,
        content: str,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
    ) -> "Message":
        """Add an assistant message with optional tool calls.

        Delegates to MessageStore (SRP).

        Args:
            content: Assistant message content
            tool_calls: Optional list of tool calls made by the assistant

        Returns:
            The created Message object
        """
        return self._message_store.add_assistant_message(content, tool_calls)

    def add_tool_result(
        self,
        tool_call_id: str,
        content: str,
    ) -> "Message":
        """Add a tool result message.

        Delegates to MessageStore (SRP).

        Args:
            tool_call_id: ID of the tool call this result is for
            content: Tool result content

        Returns:
            The created Message object
        """
        return self._message_store.add_tool_result(tool_call_id, content)

    def _persist_message(
        self,
        role: str,
        content: str,
        tool_name: Optional[str] = None,
    ) -> None:
        """Persist message to SQLite store if enabled.

        Note: This is kept for backward compatibility but is now handled
        internally by MessageStore.
        """
        # Delegated to MessageStore
        pass

    # =========================================================================
    # MESSAGE ACCESS
    # =========================================================================

    @property
    def messages(self) -> List["Message"]:
        """Get all messages in the conversation.

        Delegates to MessageStore (SRP).

        Returns:
            List of Message objects
        """
        return self._message_store.messages

    def message_count(self) -> int:
        """Get the number of messages in the conversation.

        Delegates to MessageStore (SRP).

        Returns:
            Number of messages
        """
        return self._message_store.message_count

    def get_last_user_message(self) -> Optional[str]:
        """Get the content of the last user message.

        Delegates to MessageStore (SRP).

        Returns:
            Last user message content or None
        """
        return self._message_store.get_last_user_message()

    def get_last_assistant_message(self) -> Optional[str]:
        """Get the content of the last assistant message.

        Delegates to MessageStore (SRP).

        Returns:
            Last assistant message content or None
        """
        return self._message_store.get_last_assistant_message()

    # =========================================================================
    # CONTEXT MANAGEMENT
    # =========================================================================

    def get_context_metrics(self) -> ContextMetrics:
        """Get current context metrics.

        Delegates to ContextOverflowHandler (SRP).

        Returns:
            ContextMetrics with size and overflow information
        """
        return self._context_handler.get_context_metrics()

    def check_context_overflow(self) -> bool:
        """Check if context is at risk of overflow.

        Delegates to ContextOverflowHandler (SRP).

        Returns:
            True if context is dangerously large
        """
        return self._context_handler.check_overflow()

    def handle_compaction(
        self,
        user_message: Optional[str] = None,
        target_messages: Optional[int] = None,
    ) -> int:
        """Trigger context compaction.

        Delegates to ContextOverflowHandler (SRP).

        Args:
            user_message: Current user message for semantic relevance scoring
            target_messages: Target number of messages to keep

        Returns:
            Number of messages removed
        """
        # Delegate to context handler
        metrics = self._context_handler.handle_compaction(
            strategy="semantic" if user_message else "recent",
            target_ratio=0.7,
            preserve_system_prompt=True,
        )

        if metrics:
            # Calculate messages removed
            return self.message_count() - metrics.message_count
        return 0

    def get_memory_context(self, max_tokens: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get token-aware context for LLM calls.

        Delegates to ContextOverflowHandler (SRP).

        Args:
            max_tokens: Maximum tokens for context

        Returns:
            List of messages in provider format
        """
        if self._store and self._session_id:
            return self._store.get_context_messages(
                session_id=self._session_id,
                max_tokens=max_tokens,
            )

        # Fall back to context handler
        return self._context_handler.get_memory_context(max_tokens)

    # =========================================================================
    # STAGE TRACKING
    # =========================================================================

    @property
    def stage(self) -> ConversationStage:
        """Get current conversation stage.

        Returns:
            Current ConversationStage
        """
        return self._controller.stage

    def get_stage_recommended_tools(self) -> Set[str]:
        """Get tools recommended for current conversation stage.

        Returns:
            Set of recommended tool names
        """
        return self._controller.get_stage_recommended_tools()

    # =========================================================================
    # SESSION MANAGEMENT
    # =========================================================================

    @property
    def session_id(self) -> Optional[str]:
        """Get current session ID.

        Delegates to SessionManager (SRP).

        Returns:
            Session ID or None if no session
        """
        return self._session_manager.session_id or self._session_id

    def get_recent_sessions(
        self,
        limit: int = 10,
        project_path: Optional[str] = None,
    ) -> List["ConversationSession"]:
        """List recent conversation sessions.

        Delegates to SessionManager (SRP).

        Args:
            limit: Maximum sessions to return
            project_path: Optional filter by project path

        Returns:
            List of ConversationSession objects
        """
        if not self._store:
            return []

        return self._store.list_sessions(project_path=project_path, limit=limit)

    def recover_session(self, session_id: str) -> bool:
        """Recover a previous session.

        Delegates to SessionManager (SRP).

        Loads session from persistent store and restores messages
        to the controller.

        Args:
            session_id: Session ID to recover

        Returns:
            True if recovery successful, False otherwise
        """
        if not self._store:
            logger.warning("Cannot recover session: no store configured")
            return False

        session = self._store.get_session(session_id)
        if not session:
            logger.warning(f"Session not found: {session_id}")
            return False

        # Reset controller
        self._controller.reset()

        # Restore messages to controller
        # Note: ConversationSession.messages contains ConversationMessage objects
        for msg in session.messages:
            role = msg.role.value if hasattr(msg.role, "value") else str(msg.role)

            # Preserve tool_calls metadata for assistant messages
            kwargs = {}
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                kwargs["tool_calls"] = msg.tool_calls

            self._controller.add_message(role, msg.content, **kwargs)

        # Update store's session_id to reflect recovered session
        if hasattr(self._store, "session_id"):
            self._store.session_id = session_id

        self._session = session
        self._session_id = session_id
        logger.info(f"Recovered session {session_id} with {len(session.messages)} messages")
        return True

    def get_session_stats(self) -> Dict[str, Any]:
        """Get statistics for current session.

        Delegates to SessionManager (SRP).

        Returns:
            Dictionary of session statistics
        """
        if not self._store or not self._session_id:
            return {
                "session_id": self._session_id,
                "message_count": self.message_count(),
                "stage": self.stage.value,
            }

        return self._store.get_session_stats(self._session_id)

    # =========================================================================
    # EMBEDDING STORE
    # =========================================================================

    async def initialize_embedding_store(self) -> bool:
        """Lazy initialize LanceDB embedding store.

        Delegates to EmbeddingManager (SRP).

        Call this method when semantic search capabilities are needed.
        The embedding store is not initialized by default to reduce startup overhead.

        Returns:
            True if initialization successful, False otherwise
        """
        if not self._config.enable_embeddings:
            return False

        if self._embedding_store is not None:
            return True

        try:
            from victor.agent.conversation_embedding_store import (
                ConversationEmbeddingStore,
            )

            # Get or create embedding service
            if self._embedding_service is None:
                from victor.storage.embeddings.service import get_embedding_service

                self._embedding_service = get_embedding_service()

            # Get SQLite DB path from store if available
            sqlite_path = None
            if self._store:
                sqlite_path = self._store.db_path

            self._embedding_store = ConversationEmbeddingStore(
                embedding_service=self._embedding_service,
                sqlite_db_path=sqlite_path,
            )
            await self._embedding_store.initialize()

            # Update embedding manager with the store
            self._embedding_manager._embedding_store = self._embedding_store

            # Wire up embedding store to controller and store
            if self._embedding_service:
                self._controller.set_embedding_service(self._embedding_service)
            if self._store and self._embedding_store:
                self._store.set_embedding_store(self._embedding_store)

            logger.info("Embedding store initialized")
            return True

        except ImportError as e:
            logger.warning(f"Failed to initialize embedding store: {e}")
            return False
        except Exception as e:
            logger.error(f"Embedding store initialization error: {e}")
            return False

    async def search_similar_messages(
        self,
        query: str,
        limit: int = 10,
        min_similarity: float = 0.3,
    ) -> List[Dict[str, Any]]:
        """Search for semantically similar messages.

        Delegates to EmbeddingManager (SRP).

        Requires embedding store to be initialized.

        Args:
            query: Query text to search for
            limit: Maximum results to return
            min_similarity: Minimum similarity threshold

        Returns:
            List of matching messages with similarity scores
        """
        if not self._embedding_store:
            logger.warning(
                "Embedding store not initialized. Call initialize_embedding_store() first."
            )
            return []

        try:
            gateway = self._get_retrieval_gateway()
            if gateway is not None and self._session_id:
                from victor.storage.retrieval.gateway import RetrievalRequest

                items = await gateway.search(
                    RetrievalRequest(
                        query=query,
                        session_id=self._session_id,
                        limit=limit,
                        min_similarity=min_similarity,
                    )
                )
                return [
                    {
                        "message_id": item.message_id,
                        "similarity": item.score,
                        "source": item.source,
                    }
                    for item in items
                ]

            # Fallback: direct embedding store access
            if not self._embedding_store:
                return []
            results = await self._embedding_store.search_similar(
                query=query,
                session_id=self._session_id,
                limit=limit,
                min_similarity=min_similarity,
            )
            if self._store:
                return [
                    {
                        "message_id": r.message_id,
                        "session_id": r.session_id,
                        "similarity": r.similarity,
                        "timestamp": (r.timestamp.isoformat() if r.timestamp else None),
                    }
                    for r in results
                ]
            return [
                {
                    "message_id": r.message_id,
                    "session_id": r.session_id,
                    "similarity": r.similarity,
                }
                for r in results
            ]

        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return []

    def _get_retrieval_gateway(self):
        """Get RetrievalGateway (injected via constructor or None)."""
        return self._retrieval_gateway

    # =========================================================================
    # LIFECYCLE
    # =========================================================================

    def reset(self) -> None:
        """Reset conversation to initial state.

        Clears messages and resets stage tracking.
        Does not delete persisted session data.
        """
        self._controller.reset()
        logger.info("Conversation reset")

    def set_system_prompt(self, prompt: str) -> None:
        """Set or update the system prompt.

        Args:
            prompt: New system prompt
        """
        self._controller.set_system_prompt(prompt)

    def to_dict(self) -> Dict[str, Any]:
        """Export conversation state as dictionary.

        Returns:
            Dictionary representation of conversation
        """
        return {
            **self._controller.to_dict(),
            "session_id": self._session_id,
            "provider": self._provider,
            "model": self._model,
            "persistence_enabled": self._config.enable_persistence,
            "embeddings_enabled": self._config.enable_embeddings,
        }

    async def close(self) -> None:
        """Clean up resources.

        Should be called when the conversation manager is no longer needed.
        """
        if self._embedding_store:
            await self._embedding_store.close()
            self._embedding_store = None

        logger.info("ConversationManager closed")


def create_conversation_manager(
    settings: Optional["Settings"] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    system_prompt: Optional[str] = None,
    enable_persistence: bool = True,
    enable_embeddings: bool = True,
) -> ConversationManager:
    """Factory function to create a ConversationManager.

    This is a convenience function that handles common initialization patterns.

    Args:
        settings: Application settings
        provider: LLM provider name
        model: Model identifier
        system_prompt: System prompt for the conversation
        enable_persistence: Whether to enable SQLite persistence
        enable_embeddings: Whether to enable embedding store

    Returns:
        Configured ConversationManager instance
    """
    config = ConversationManagerConfig(
        enable_persistence=enable_persistence,
        enable_embeddings=enable_embeddings,
    )

    # Create store if persistence enabled
    store = None
    if enable_persistence:
        try:
            from victor.agent.conversation.store import ConversationStore

            store = ConversationStore()
        except Exception as e:
            logger.warning(f"Failed to create ConversationStore: {e}")

    return ConversationManager(
        settings=settings,
        provider=provider,
        model=model,
        system_prompt=system_prompt,
        config=config,
        store=store,
    )
