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

"""State manager implementations for each scope.

This module provides concrete implementations of IStateManager protocol
for each StateScope. These managers directly replace legacy state systems.

SOLID Principles:
- SRP: Each manager handles one scope only
- OCP: Extensible via protocol implementation
- LSP: All managers are substitutable via IStateManager
- ISP: Implements focused IStateManager interface
- DIP: High-level modules depend on IStateManager abstraction

Thread Safety:
- All methods are protected by asyncio.Lock to prevent data corruption
  under concurrent access (e.g., parallel team formations).
- Observer notifications happen outside the lock to prevent deadlocks,
  but use a snapshot of the observer list captured inside the lock.

Replacements:
- WorkflowStateManager replaces ExecutionContext
- ConversationStateManager replaces ConversationStateMachine
- TeamStateManager replaces TeamContext
- GlobalStateManager provides application-wide state

Usage:
    from victor.state.managers import WorkflowStateManager

    manager = WorkflowStateManager()
    await manager.set("key", "value")
    value = await manager.get("key")
"""

from __future__ import annotations

import asyncio
import fnmatch
import logging
from typing import Any, Dict, List

from victor.state.protocols import IStateManager, IStateObserver, StateScope

logger = logging.getLogger(__name__)


class BaseStateManager:
    """Base state manager with full lock protection on all operations.

    All concrete state managers inherit from this class. The asyncio.Lock
    ensures safe concurrent access from parallel team formations and
    multi-agent workflows.

    Note on re-entrancy: asyncio.Lock is NOT re-entrant. Methods that
    compose other locked methods (e.g. update() calling set()) use
    private _unlocked variants to avoid deadlock.
    """

    def __init__(self, scope: StateScope, scope_label: str) -> None:
        self.scope: StateScope = scope
        self._state: Dict[str, Any] = {}
        self._observers: List[IStateObserver] = []
        self._lock = asyncio.Lock()
        self._scope_label = scope_label

        logger.debug("%s initialized", scope_label)

    async def get(self, key: str, default: Any = None) -> Any:
        """Get a value by key.

        Args:
            key: The key to retrieve
            default: Default value if key doesn't exist

        Returns:
            The value associated with key, or default if not found
        """
        async with self._lock:
            return self._state.get(key, default)

    async def set(self, key: str, value: Any, notify: bool = True) -> None:
        """Set a value by key.

        Args:
            key: The key to set
            value: The value to store
            notify: Whether to notify observers (default: True)
        """
        async with self._lock:
            old_value, observers = self._set_unlocked(key, value)
            if old_value is _UNCHANGED:
                return

        if notify and observers:
            await self._notify_observers(observers, key, old_value, value)

        logger.debug("%s state set: %s = %s", self._scope_label, key, str(value)[:50])

    def _set_unlocked(self, key: str, value: Any) -> tuple:
        """Set a value without acquiring the lock. Must be called under lock.

        Returns:
            (old_value, observers_snapshot) or (_UNCHANGED, []) if no change.
        """
        old_value = self._state.get(key)

        if old_value == value and key in self._state:
            return (_UNCHANGED, [])

        self._state[key] = value
        observers = list(self._observers)
        return (old_value, observers)

    async def delete(self, key: str) -> None:
        """Delete a value by key.

        Args:
            key: The key to delete
        """
        async with self._lock:
            if key not in self._state:
                return
            old_value = self._state.pop(key)
            observers = list(self._observers)

        await self._notify_observers(observers, key, old_value, None)

    async def exists(self, key: str) -> bool:
        """Check if key exists.

        Args:
            key: The key to check

        Returns:
            True if key exists, False otherwise
        """
        async with self._lock:
            return key in self._state

    async def keys(self, pattern: str = "*") -> List[str]:
        """Get all keys matching pattern.

        Args:
            pattern: Glob pattern to match keys (default: "*" for all)

        Returns:
            List of keys matching pattern
        """
        async with self._lock:
            if pattern == "*":
                return list(self._state.keys())
            return [k for k in self._state.keys() if fnmatch.fnmatch(k, pattern)]

    async def get_all(self) -> Dict[str, Any]:
        """Get all state as dictionary.

        Returns:
            Dictionary of all key-value pairs (shallow copy)
        """
        async with self._lock:
            return dict(self._state)

    async def update(self, updates: Dict[str, Any]) -> None:
        """Update multiple keys atomically.

        Uses _set_unlocked to avoid re-entrant lock acquisition.
        Observer notifications are batched after all keys are set.

        Args:
            updates: Dictionary of key-value pairs to update
        """
        pending_notifications: list = []

        async with self._lock:
            for key, value in updates.items():
                old_value, observers = self._set_unlocked(key, value)
                if old_value is not _UNCHANGED and observers:
                    pending_notifications.append((observers, key, old_value, value))

        for observers, key, old_value, value in pending_notifications:
            await self._notify_observers(observers, key, old_value, value)

    async def clear(self) -> None:
        """Clear all state."""
        async with self._lock:
            self._state.clear()
        logger.debug("%s state cleared", self._scope_label)

    async def snapshot(self) -> Dict[str, Any]:
        """Create immutable snapshot for checkpointing.

        Returns:
            Dictionary snapshot of current state
        """
        async with self._lock:
            return dict(self._state)

    async def restore(self, snapshot: Dict[str, Any]) -> None:
        """Restore from snapshot.

        Args:
            snapshot: Snapshot dictionary to restore from
        """
        async with self._lock:
            self._state = dict(snapshot)
        logger.debug(
            "%s state restored from snapshot (%d keys)",
            self._scope_label,
            len(snapshot),
        )

    def add_observer(self, observer: IStateObserver) -> None:
        """Add state change observer.

        Args:
            observer: Observer to notify of state changes
        """
        if observer not in self._observers:
            self._observers.append(observer)

    def remove_observer(self, observer: IStateObserver) -> None:
        """Remove state change observer.

        Args:
            observer: Observer to remove
        """
        if observer in self._observers:
            self._observers.remove(observer)

    async def _notify_observers(
        self,
        observers: List[IStateObserver],
        key: str,
        old_value: Any,
        new_value: Any,
    ) -> None:
        """Notify a snapshot of observers outside the lock.

        Args:
            observers: Snapshot of observer list captured inside the lock
            key: The key that changed
            old_value: Previous value
            new_value: New value
        """
        await asyncio.gather(
            *[
                observer.on_state_changed(
                    scope=self.scope,
                    key=key,
                    old_value=old_value,
                    new_value=new_value,
                )
                for observer in observers
            ],
            return_exceptions=True,
        )


class _Sentinel:
    """Sentinel object to distinguish 'no change' from None."""

    pass


_UNCHANGED = _Sentinel()


class WorkflowStateManager(BaseStateManager):
    """State manager for workflow scope.

    Replaces ExecutionContext directly (no adapter).
    Manages state for single workflow executions.

    Example:
        >>> manager = WorkflowStateManager()
        >>> await manager.set("task_id", "task-123")
        >>> task_id = await manager.get("task_id")
    """

    def __init__(self) -> None:
        super().__init__(StateScope.WORKFLOW, "Workflow")


class ConversationStateManager(BaseStateManager):
    """State manager for conversation scope.

    Replaces ConversationStateMachine directly (no adapter).
    Manages state for multi-turn conversations.

    Example:
        >>> manager = ConversationStateManager()
        >>> await manager.set("stage", "gathering")
        >>> stage = await manager.get("stage")
    """

    def __init__(self) -> None:
        super().__init__(StateScope.CONVERSATION, "Conversation")


class TeamStateManager(BaseStateManager):
    """State manager for team scope.

    Replaces TeamContext directly (no adapter).
    Manages state for multi-agent team coordination.

    Example:
        >>> manager = TeamStateManager()
        >>> await manager.set("coordinator", "agent-1")
        >>> coordinator = await manager.get("coordinator")
    """

    def __init__(self) -> None:
        super().__init__(StateScope.TEAM, "Team")


class GlobalStateManagerImpl(BaseStateManager):
    """State manager for global scope.

    Manages cross-cutting application state.
    Used for configuration, settings, and global state.

    Example:
        >>> manager = GlobalStateManagerImpl()
        >>> await manager.set("config", {"debug": True})
        >>> config = await manager.get("config")
    """

    def __init__(self) -> None:
        super().__init__(StateScope.GLOBAL, "Global")


__all__ = [
    "WorkflowStateManager",
    "ConversationStateManager",
    "TeamStateManager",
    "GlobalStateManagerImpl",
]
