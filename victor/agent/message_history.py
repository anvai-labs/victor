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

"""Simple in-memory message history for conversations.

This module provides basic in-memory conversation history management.

The primary class is `MessageHistory`.

For persistent storage:
- File-based: see `victor.agent.session.SessionPersistence`
- SQLite-based with token management: see `victor.agent.conversation.store.ConversationStore`

"""

import logging
from copy import deepcopy
from typing import Any, Dict, List, Optional

from victor.providers.base import Message

logger = logging.getLogger(__name__)


class _TrackedList(list):
    """List subclass that logs when non-Message items are inserted.

    Temporary instrumentation to find where raw strings leak into _messages.
    """

    def append(self, item: Any) -> None:
        if not isinstance(item, Message):
            import traceback

            logger.warning(
                "TRACKED: non-Message appended: type=%s value=%r caller=\n%s",
                type(item).__name__,
                str(item)[:80],
                "".join(traceback.format_stack()[-4:-1]),
            )
        super().append(item)

    def extend(self, items: Any) -> None:
        # Check if items is a string (would iterate chars)
        if isinstance(items, str):
            import traceback

            logger.error(
                "TRACKED: string passed to extend (would iterate chars!): "
                "len=%d value=%r caller=\n%s",
                len(items),
                items[:80],
                "".join(traceback.format_stack()[-4:-1]),
            )
            return  # Block the corruption
        for item in items:
            if not isinstance(item, Message):
                import traceback

                logger.warning(
                    "TRACKED: non-Message in extend: type=%s value=%r caller=\n%s",
                    type(item).__name__,
                    str(item)[:80],
                    "".join(traceback.format_stack()[-4:-1]),
                )
        super().extend(items)

    def insert(self, index: int, item: Any) -> None:
        if not isinstance(item, Message):
            import traceback

            logger.warning(
                "TRACKED: non-Message inserted at %d: type=%s caller=\n%s",
                index,
                type(item).__name__,
                "".join(traceback.format_stack()[-4:-1]),
            )
        super().insert(index, item)

    def __setitem__(self, index: Any, item: Any) -> None:
        if isinstance(item, list):
            for i in item:
                if not isinstance(i, Message):
                    import traceback

                    logger.warning(
                        "TRACKED: non-Message in slice assign: type=%s caller=\n%s",
                        type(i).__name__,
                        "".join(traceback.format_stack()[-4:-1]),
                    )
        elif not isinstance(item, Message):
            import traceback

            logger.warning(
                "TRACKED: non-Message set at index %s: type=%s caller=\n%s",
                index,
                type(item).__name__,
                "".join(traceback.format_stack()[-4:-1]),
            )
        super().__setitem__(index, item)


class MessageHistory:
    """Simple in-memory message history for conversations.

    Responsibilities:
    - Maintain conversation history in memory
    - Add system prompts
    - Format messages for providers
    - Manage context window limits (by message count)

    For persistent storage, see SessionPersistence or ConversationStore.

    """

    def __init__(
        self,
        system_prompt: str = "",
        max_history_messages: int = 100000,
    ):
        """Initialize conversation manager.

        Args:
            system_prompt: Initial system prompt for the conversation
            max_history_messages: Maximum messages to retain (default 100k for historical analysis)
        """
        self._system_prompt = system_prompt
        self._system_added = False
        self.__actual_messages: List[Message] = _TrackedList()
        self._preview_messages: List[Dict[str, Any]] = []
        self._max_history = max_history_messages

    @property
    def _messages(self) -> List[Message]:
        return self.__actual_messages

    @_messages.setter
    def _messages(self, value: Any) -> None:
        import traceback

        if not isinstance(value, _TrackedList):
            has_str = (
                any(isinstance(v, str) for v in value) if hasattr(value, "__iter__") else False
            )
            logger.info(
                "MESSAGES REPLACED with %s (len=%d, has_str=%s) from:\n%s",
                type(value).__name__,
                len(value) if hasattr(value, "__len__") else -1,
                has_str,
                "".join(traceback.format_stack()[-4:-1]),
            )
        self.__actual_messages = (
            _TrackedList(value) if not isinstance(value, _TrackedList) else value
        )

    @property
    def messages(self) -> List[Message]:
        """Get all messages in conversation history."""
        return self._messages.copy()

    @property
    def system_prompt(self) -> str:
        """Get the current system prompt."""
        return self._system_prompt

    @system_prompt.setter
    def system_prompt(self, value: str) -> None:
        """Set the system prompt."""
        self._system_prompt = value

    def add_message(self, role: str, content: str, **kwargs: Any) -> Message:
        """Add a message to conversation history.

        Args:
            role: Message role (system, user, assistant, tool)
            content: Message content
            **kwargs: Additional message fields (tool_calls, tool_call_id, name)

        Returns:
            The created Message object
        """
        message = Message(role=role, content=content, **kwargs)
        self._messages.append(message)
        self._trim_history()
        # Instrumentation: check for non-Message items after each add
        non_msg = sum(1 for m in self._messages if not isinstance(m, Message))
        if non_msg > 0:
            types = {type(m).__name__ for m in self._messages if not isinstance(m, Message)}
            logger.debug(
                "NON-MESSAGE DETECTED after add_message(%s): %d non-Message items, types=%s, "
                "total=%d, caller=%s",
                role,
                non_msg,
                types,
                len(self._messages),
                __import__("traceback").format_stack()[-3].strip(),
            )
        return message

    def append_message(self, message: Any) -> None:
        """Append a pre-constructed message with type validation.

        Ensures only Message objects enter _messages. Converts dicts and
        strings defensively to prevent HTTP 400 from malformed history.

        Args:
            message: Message object, dict with 'role' key, or string
        """
        if isinstance(message, Message):
            self._messages.append(message)
        elif isinstance(message, dict) and "role" in message:
            safe_keys = {k: v for k, v in message.items() if k in Message.model_fields}
            self._messages.append(Message(**safe_keys))
            logger.debug(
                "append_message: converted dict to Message (role=%s, keys=%s)",
                message.get("role"),
                list(message.keys()),
            )
        elif isinstance(message, str):
            logger.warning("Converting raw string to Message in history")
            self._messages.append(Message(role="assistant", content=message))
        else:
            logger.error("Dropping non-Message object from history: %s", type(message).__name__)
            return
        self._trim_history()

    def add_user_message(self, content: str, metadata: Any = None) -> Message:
        """Add a user message to conversation history."""
        if metadata is not None:
            return self.add_message("user", content, metadata=metadata)
        return self.add_message("user", content)

    def add_assistant_message(
        self,
        content: str,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        metadata: Any = None,
    ) -> Message:
        """Add an assistant message to conversation history."""
        kwargs: Dict[str, Any] = {}
        if tool_calls is not None:
            kwargs["tool_calls"] = tool_calls
        if metadata is not None:
            kwargs["metadata"] = metadata
        return self.add_message("assistant", content, **kwargs)

    def add_tool_result(
        self, tool_call_id: str, content: str, tool_name: Optional[str] = None
    ) -> Message:
        """Add a tool result message to conversation history.

        Args:
            tool_call_id: ID of the tool call being responded to
            content: Tool execution result
            tool_name: Optional name of the tool

        Returns:
            The created Message object
        """
        return self.add_message("tool", content, tool_call_id=tool_call_id, name=tool_name)

    def ensure_system_prompt(self) -> None:
        """Ensure system prompt is added to the beginning of conversation."""
        if not self._system_added and self._system_prompt:
            # Insert at the beginning
            system_message = Message(role="system", content=self._system_prompt)
            self._messages.insert(0, system_message)
            self._system_added = True

    def get_messages_for_provider(self) -> List[Message]:
        """Get messages formatted for sending to provider.

        Ensures system prompt is included and history is within limits.

        Returns:
            List of messages ready for provider
        """
        self.ensure_system_prompt()
        return self._messages.copy()

    def add_preview_message(
        self,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Store a replay-only preview message outside provider-facing history."""
        if not content:
            return

        preview_metadata = deepcopy(metadata or {})
        self._preview_messages.append(
            {
                "role": role,
                "content": content,
                "metadata": preview_metadata,
                # Anchor after the current provider-facing history length.
                "after_message_index": len(self._messages),
            }
        )

    @property
    def preview_messages(self) -> List[Dict[str, Any]]:
        """Return replay-only preview messages in insertion order."""
        return deepcopy(self._preview_messages)

    def clear(self) -> None:
        """Clear conversation history."""
        self._messages.clear()
        self._preview_messages.clear()
        self._system_added = False

    def _trim_history(self) -> None:
        """Trim history to max_history_messages, preserving system prompt."""
        if len(self._messages) <= self._max_history:
            return

        # Keep system message if present
        system_msg = None
        if self._messages and self._messages[0].role == "system":
            system_msg = self._messages[0]

        # Trim from the front (oldest messages), keeping recent ones
        excess = len(self._messages) - self._max_history
        kept = self._messages[excess:]
        self._messages = _TrackedList(kept)

        # Re-add system message at front if it was removed
        if system_msg and (not self._messages or self._messages[0].role != "system"):
            self._messages.insert(0, system_msg)

    def get_last_user_message(self) -> Optional[str]:
        """Get the content of the last user message."""
        for msg in reversed(self._messages):
            if msg.role == "user":
                return msg.content
        return None

    def get_latest_user_message(self) -> Optional[str]:
        """Alias for get_last_user_message — satisfies planning_runtime API contract."""
        return self.get_last_user_message()

    def get_last_assistant_message(self) -> Optional[str]:
        """Get the content of the last assistant message."""
        for msg in reversed(self._messages):
            if msg.role == "assistant":
                return msg.content
        return None

    def message_count(self) -> int:
        """Get the number of messages in history."""
        return len(self._messages)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize conversation state to dictionary."""
        return {
            "system_prompt": self._system_prompt,
            "system_added": self._system_added,
            "messages": [msg.to_dict() for msg in self._messages],
            "preview_messages": self.preview_messages,
            "max_history": self._max_history,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MessageHistory":
        """Deserialize conversation state from dictionary."""
        manager = cls(
            system_prompt=data.get("system_prompt", ""),
            max_history_messages=data.get("max_history", 100),
        )
        manager._system_added = data.get("system_added", False)
        for msg_data in data.get("messages", []):
            manager._messages.append(Message(**msg_data))
        preview_messages = data.get("preview_messages", [])
        if isinstance(preview_messages, list):
            manager._preview_messages = [msg for msg in preview_messages if isinstance(msg, dict)]
        return manager
