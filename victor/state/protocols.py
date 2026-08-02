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

"""Unified state management protocols.

This module defines the canonical protocols for state management across
the Victor framework. All state systems MUST implement these protocols
for unified access and debugging.

SOLID Principles:
- SRP: Protocols define interface only, no implementation
- OCP: Extensible via protocol composition
- LSP: All implementations are substitutable
- ISP: Focused, minimal interfaces
- DIP: High-level modules depend on these protocols

Usage:
    from victor.state.protocols import IStateManager, StateScope

    class MyStateManager:
        scope: StateScope = StateScope.WORKFLOW

        async def get(self, key: str, default: Any = None) -> Any:
            # Implementation
            ...
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


class StateScope(str, Enum):
    """State management scope.

    Defines the four scopes for state management:
    - WORKFLOW: Single workflow execution (replaces ExecutionContext)
    - CONVERSATION: Multi-turn conversation (replaces ConversationStateMachine)
    - TEAM: Multi-agent team coordination (replaces TeamContext)
    - GLOBAL: Cross-cutting application state
    """

    WORKFLOW = "workflow"
    CONVERSATION = "conversation"
    TEAM = "team"
    GLOBAL = "global"


@runtime_checkable
class IStateManager(Protocol):
    """Canonical state management protocol.

    ALL state systems MUST implement this protocol for unified access.
    This enables GlobalStateManager facade to provide consistent interface
    across all scopes.

    SOLID: ISP (Focused CRUD + lifecycle interface)

    Attributes:
        scope: The StateScope this manager handles

    Methods:
        get: Retrieve a value by key
        set: Store a value by key
        delete: Remove a value by key
        exists: Check if key exists
        keys: List all keys matching pattern
        get_all: Get all state as dictionary
        update: Update multiple keys at once
        clear: Remove all state
        snapshot: Create immutable snapshot
        restore: Restore from snapshot
        add_observer: Register state change observer
        remove_observer: Unregister state change observer
    """

    scope: StateScope

    async def get(self, key: str, default: Any = None) -> Any:
        """Get a value by key.

        Args:
            key: The key to retrieve
            default: Default value if key doesn't exist

        Returns:
            The value associated with key, or default if not found
        """
        ...

    async def set(self, key: str, value: Any) -> None:
        """Set a value by key.

        Args:
            key: The key to set
            value: The value to store
        """
        ...

    async def delete(self, key: str) -> None:
        """Delete a value by key.

        Args:
            key: The key to delete
        """
        ...

    async def exists(self, key: str) -> bool:
        """Check if key exists.

        Args:
            key: The key to check

        Returns:
            True if key exists, False otherwise
        """
        ...

    async def keys(self, pattern: str = "*") -> List[str]:
        """Get all keys matching pattern.

        Args:
            pattern: Glob pattern to match keys (default: "*" for all)

        Returns:
            List of keys matching pattern
        """
        ...

    async def get_all(self) -> Dict[str, Any]:
        """Get all state as dictionary.

        Returns:
            Dictionary of all key-value pairs
        """
        ...

    async def update(self, updates: Dict[str, Any]) -> None:
        """Update multiple keys at once.

        Args:
            updates: Dictionary of key-value pairs to update
        """
        ...

    async def clear(self) -> None:
        """Clear all state."""
        ...

    async def snapshot(self) -> Dict[str, Any]:
        """Create immutable snapshot for checkpointing.

        Returns:
            Dictionary snapshot of current state
        """
        ...

    async def restore(self, snapshot: Dict[str, Any]) -> None:
        """Restore from snapshot.

        Args:
            snapshot: Snapshot dictionary to restore from
        """
        ...

    def add_observer(self, observer: IStateObserver) -> None:
        """Add state change observer.

        Args:
            observer: Observer to notify of state changes
        """
        ...

    def remove_observer(self, observer: IStateObserver) -> None:
        """Remove state change observer.

        Args:
            observer: Observer to remove
        """
        ...


@runtime_checkable
class IStateObserver(Protocol):
    """State change observer protocol.

    Observers are notified when state changes occur.
    Used by StateTracer and debugging tools.

    SOLID: ISP (Single notification method)

    Methods:
        on_state_changed: Called when state changes
    """

    async def on_state_changed(
        self,
        scope: StateScope,
        key: str,
        old_value: Any,
        new_value: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Called when state changes.

        Args:
            scope: The StateScope where change occurred
            key: The key that changed
            old_value: Previous value
            new_value: New value
            metadata: Optional metadata about the change
        """
        ...


__all__ = [
    "StateScope",
    "IStateManager",
    "IStateObserver",
]
