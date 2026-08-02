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

"""Conversation management protocol for vertical extensions.

Defines the contract that vertical-specific conversation managers
must implement. External verticals should depend on this protocol
rather than importing internal coordinator classes.

Usage::

    from victor_contracts.verticals.protocols.conversation import (
        ConversationManagerProtocol,
    )

    class MyConversationManager:
        '''Implements ConversationManagerProtocol.'''

        def add_message(self, role: str, content: str, **metadata):
            ...
"""

from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


@runtime_checkable
class ConversationManagerProtocol(Protocol):
    """Protocol for vertical-specific conversation management.

    Verticals that need enhanced conversation tracking should implement
    this protocol. The framework's ConversationCoordinator satisfies
    this protocol and can be wrapped by vertical-specific managers.
    """

    def add_message(self, role: str, content: str, **metadata: Any) -> str:
        """Record a message in the conversation history.

        Args:
            role: The message role (e.g., "user", "assistant", "system").
            content: The message content.
            **metadata: Additional metadata to attach to the message.

        Returns:
            A unique message identifier.
        """
        ...

    def get_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Retrieve conversation history.

        Args:
            limit: Maximum number of recent messages to return.
                   None returns all messages.

        Returns:
            List of message dicts with at least 'role' and 'content' keys.
        """
        ...

    def clear_history(self) -> None:
        """Clear all conversation history."""
        ...

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the conversation state.

        Returns:
            Dict with conversation summary data (e.g., message counts,
            active topics, turn statistics).
        """
        ...

    def get_observability_data(self) -> Dict[str, Any]:
        """Get observability/telemetry data for the conversation.

        Returns:
            Dict with metrics and telemetry data suitable for logging
            or monitoring dashboards.
        """
        ...
