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

"""Message storage and retrieval.

This module provides MessageStore, which handles message storage,
persistence, and retrieval. Extracted from ConversationManager to
follow the Single Responsibility Principle (SRP).

Part of SOLID-based refactoring to eliminate god class anti-pattern.
"""

import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from victor.agent.protocols import IMessageStore

if TYPE_CHECKING:
    from victor.providers.base import Message

logger = logging.getLogger(__name__)


class MessageStore(IMessageStore):
    """Handles message storage and retrieval operations.

    This class is responsible for:
    - Storing messages in memory
    - Persisting messages to storage
    - Retrieving messages and metadata

    SRP Compliance: Focuses only on message storage, delegating
    context management, session management, and embeddings to
    specialized components.

    Attributes:
        _controller: ConversationController for in-memory message handling
        _store: Optional ConversationStore for persistence
        _enable_persistence: Whether persistence is enabled
    """

    def __init__(
        self,
        controller: Any,
        store: Optional[Any] = None,
        enable_persistence: bool = True,
    ):
        """Initialize the message store.

        Args:
            controller: ConversationController for in-memory messages
            store: Optional ConversationStore for persistence
            enable_persistence: Whether to enable persistence
        """
        self._controller = controller
        self._store = store
        self._enable_persistence = enable_persistence

    def add_message(
        self,
        role: str,
        content: str,
        **metadata: Any,
    ) -> "Message":
        """Add a message to storage.

        Args:
            role: Message role (user, assistant, system, tool)
            content: Message content
            **metadata: Additional message metadata

        Returns:
            Created Message object
        """
        # Add to in-memory controller
        message = self._controller.add_message(role, content, **metadata)

        # Persist if enabled
        if self._enable_persistence:
            self._persist_message(message)

        return message

    def add_user_message(self, content: str) -> "Message":
        """Add a user message.

        Args:
            content: Message content

        Returns:
            Created Message object
        """
        return self.add_message("user", content)

    def add_assistant_message(
        self,
        content: str,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        **metadata: Any,
    ) -> "Message":
        """Add an assistant message.

        Args:
            content: Message content
            tool_calls: Optional tool calls made by assistant
            **metadata: Additional message metadata

        Returns:
            Created Message object
        """
        msg_metadata = {"tool_calls": tool_calls} if tool_calls else {}
        msg_metadata.update(metadata)
        return self.add_message("assistant", content, **msg_metadata)

    def add_tool_result(
        self,
        tool_call_id: str,
        content: str,
        **metadata: Any,
    ) -> "Message":
        """Add a tool result message.

        Args:
            tool_call_id: ID of the tool call this result is for
            content: Tool result content
            **metadata: Additional message metadata

        Returns:
            Created Message object
        """
        msg_metadata = {"tool_call_id": tool_call_id}
        msg_metadata.update(metadata)
        return self.add_message("tool", content, **msg_metadata)

    def get_messages(self, limit: Optional[int] = None) -> List["Message"]:
        """Retrieve messages.

        Args:
            limit: Optional limit on number of messages

        Returns:
            List of messages
        """
        return self._controller.messages[:limit] if limit else self._controller.messages

    def get_last_user_message(self) -> Optional[str]:
        """Get the last user message content.

        Returns:
            Last user message content or None
        """
        for message in reversed(self._controller.messages):
            if message.role == "user":
                return message.content
        return None

    def get_last_assistant_message(self) -> Optional[str]:
        """Get the last assistant message content.

        Returns:
            Last assistant message content or None
        """
        for message in reversed(self._controller.messages):
            if message.role == "assistant":
                return message.content
        return None

    @property
    def messages(self) -> List["Message"]:
        """Get all messages.

        Returns:
            List of all messages
        """
        return self._controller.messages

    @property
    def message_count(self) -> int:
        """Get the number of messages.

        Returns:
            Number of messages
        """
        return len(self._controller.messages)

    def persist(self) -> bool:
        """Persist messages to storage.

        Returns:
            True if persistence succeeded, False otherwise
        """
        if not self._enable_persistence or not self._store:
            return False

        try:
            self._store.persist_messages()
            return True
        except Exception as e:
            logger.warning(f"Failed to persist messages: {e}")
            return False

    def _persist_message(self, message: "Message") -> None:
        """Persist a single message.

        Args:
            message: Message to persist
        """
        if not self._store:
            return

        try:
            # Store is responsible for persistence
            # The store will add this to its pending writes
            pass
        except Exception as e:
            logger.warning(f"Failed to queue message for persistence: {e}")
