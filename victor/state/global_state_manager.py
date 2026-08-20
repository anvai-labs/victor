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

"""Global state manager facade for unified state access.

This module provides GlobalStateManager, a unified facade that coordinates
all state managers across different scopes. It provides a single entry point
for state operations across the entire application.

SOLID Principles:
- SRP: Unified state access coordination only
- OCP: Extensible via new scope registration
- LSP: Substitutable with individual managers
- ISP: Focused on state CRUD and lifecycle
- DIP: Depends on IStateManager abstraction

Usage:
    from victor.state import get_global_manager, StateScope

    state = get_global_manager()

    # Set a value in workflow scope
    await state.set("task_id", "task-123", scope=StateScope.WORKFLOW)

    # Get a value from conversation scope
    stage = await state.get("stage", scope=StateScope.CONVERSATION)

    # Create checkpoint across all scopes
    checkpoint = await state.create_checkpoint()

    # Restore checkpoint
    await state.restore_checkpoint(checkpoint)
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

from victor.state.protocols import IStateManager, StateScope

logger = logging.getLogger(__name__)


class GlobalStateManager:
    """Unified facade for all state management.

    Provides a single entry point for state operations across all scopes.
    Coordinates multiple IStateManager instances and provides unified
    checkpoint/rollback functionality.

    SOLID: SRP (unified access only), DIP (depends on IStateManager abstraction)

    Attributes:
        _managers: Dictionary mapping StateScope to IStateManager instances
        _tracer: Optional StateTracer for transition tracking

    Example:
        >>> manager = GlobalStateManager()
        >>> manager.register_manager(StateScope.WORKFLOW, WorkflowStateManager())
        >>> await manager.set("key", "value", scope=StateScope.WORKFLOW)
        >>> value = await manager.get("key", scope=StateScope.WORKFLOW)
    """

    def __init__(self) -> None:
        """Initialize the global state manager."""
        self._managers: Dict[StateScope, IStateManager] = {}
        self._tracer: Optional[Any] = None
        self._lock = asyncio.Lock()

        logger.info("GlobalStateManager initialized")

    def register_manager(self, scope: StateScope, manager: IStateManager) -> None:
        """Register a state manager for a specific scope.

        Args:
            scope: The StateScope this manager handles
            manager: The IStateManager instance to register

        Raises:
            ValueError: If a manager is already registered for this scope
        """
        if scope in self._managers:
            logger.warning(
                f"Manager already registered for scope {scope.value}, "
                f"replacing with new manager"
            )

        self._managers[scope] = manager
        logger.info(f"Registered manager for scope: {scope.value}")

    def set_tracer(self, tracer: Any) -> None:
        """Set the state tracer for transition tracking.

        Args:
            tracer: StateTracer instance for recording state transitions
        """
        self._tracer = tracer
        logger.info("State tracer registered with GlobalStateManager")

    async def get(
        self,
        key: str,
        scope: StateScope = StateScope.GLOBAL,
        default: Any = None,
    ) -> Any:
        """Get a value by key from specified scope.

        Args:
            key: The key to retrieve
            scope: The StateScope to query (default: GLOBAL)
            default: Default value if key doesn't exist

        Returns:
            The value associated with key, or default if not found

        Raises:
            ValueError: If no manager is registered for the specified scope
        """
        manager = self._managers.get(scope)
        if not manager:
            raise ValueError(f"No manager registered for scope: {scope.value}")

        return await manager.get(key, default)

    async def set(
        self,
        key: str,
        value: Any,
        scope: StateScope = StateScope.GLOBAL,
    ) -> None:
        """Set a value by key in specified scope.

        Args:
            key: The key to set
            value: The value to store
            scope: The StateScope to update (default: GLOBAL)

        Raises:
            ValueError: If no manager is registered for the specified scope
        """
        manager = self._managers.get(scope)
        if not manager:
            raise ValueError(f"No manager registered for scope: {scope.value}")

        async with self._lock:
            old_value = await manager.get(key)
            await manager.set(key, value)

            if self._tracer:
                self._tracer.record_transition(
                    scope=scope.value,
                    key=key,
                    old_value=old_value,
                    new_value=value,
                )

    async def delete(
        self,
        key: str,
        scope: StateScope = StateScope.GLOBAL,
    ) -> None:
        """Delete a value by key from specified scope.

        Args:
            key: The key to delete
            scope: The StateScope to delete from (default: GLOBAL)

        Raises:
            ValueError: If no manager is registered for the specified scope
        """
        manager = self._managers.get(scope)
        if not manager:
            raise ValueError(f"No manager registered for scope: {scope.value}")

        async with self._lock:
            old_value = await manager.get(key)
            await manager.delete(key)

            if self._tracer:
                self._tracer.record_transition(
                    scope=scope.value,
                    key=key,
                    old_value=old_value,
                    new_value=None,
                )

    async def exists(
        self,
        key: str,
        scope: StateScope = StateScope.GLOBAL,
    ) -> bool:
        """Check if key exists in specified scope.

        Args:
            key: The key to check
            scope: The StateScope to check (default: GLOBAL)

        Returns:
            True if key exists, False otherwise

        Raises:
            ValueError: If no manager is registered for the specified scope
        """
        manager = self._managers.get(scope)
        if not manager:
            raise ValueError(f"No manager registered for scope: {scope.value}")

        return await manager.exists(key)

    async def keys(
        self,
        scope: StateScope = StateScope.GLOBAL,
        pattern: str = "*",
    ) -> List[str]:
        """Get all keys matching pattern from specified scope.

        Args:
            scope: The StateScope to query (default: GLOBAL)
            pattern: Glob pattern to match keys (default: "*" for all)

        Returns:
            List of keys matching pattern

        Raises:
            ValueError: If no manager is registered for the specified scope
        """
        manager = self._managers.get(scope)
        if not manager:
            raise ValueError(f"No manager registered for scope: {scope.value}")

        return await manager.keys(pattern)

    async def get_all(self, scope: StateScope = StateScope.GLOBAL) -> Dict[str, Any]:
        """Get all state from specified scope as dictionary.

        Args:
            scope: The StateScope to query (default: GLOBAL)

        Returns:
            Dictionary of all key-value pairs in the scope

        Raises:
            ValueError: If no manager is registered for the specified scope
        """
        manager = self._managers.get(scope)
        if not manager:
            raise ValueError(f"No manager registered for scope: {scope.value}")

        return await manager.get_all()

    async def update(
        self,
        updates: Dict[str, Any],
        scope: StateScope = StateScope.GLOBAL,
    ) -> None:
        """Update multiple keys at once in specified scope.

        Args:
            updates: Dictionary of key-value pairs to update
            scope: The StateScope to update (default: GLOBAL)

        Raises:
            ValueError: If no manager is registered for the specified scope
        """
        manager = self._managers.get(scope)
        if not manager:
            raise ValueError(f"No manager registered for scope: {scope.value}")

        async with self._lock:
            if self._tracer:
                for key, value in updates.items():
                    old_value = await manager.get(key)
                    self._tracer.record_transition(
                        scope=scope.value,
                        key=key,
                        old_value=old_value,
                        new_value=value,
                    )

            await manager.update(updates)

    async def clear(self, scope: StateScope = StateScope.GLOBAL) -> None:
        """Clear all state from specified scope.

        Args:
            scope: The StateScope to clear (default: GLOBAL)

        Raises:
            ValueError: If no manager is registered for the specified scope
        """
        manager = self._managers.get(scope)
        if not manager:
            raise ValueError(f"No manager registered for scope: {scope.value}")

        async with self._lock:
            await manager.clear()
        logger.info(f"Cleared all state from scope: {scope.value}")

    async def create_checkpoint(self) -> Dict[str, Any]:
        """Create checkpoint across all scopes.

        Creates snapshots of all registered managers and returns
        a unified checkpoint dictionary.

        Returns:
            Dictionary mapping scope values to their snapshots

        Example:
            >>> checkpoint = await state.create_checkpoint()
            >>> # checkpoint = {
            >>> #     "workflow": {...},
            >>> #     "conversation": {...},
            >>> #     "team": {...},
            >>> #     "global": {...}
            >>> # }
        """
        checkpoint: Dict[str, Any] = {}

        async with self._lock:
            for scope, manager in self._managers.items():
                try:
                    snapshot = await manager.snapshot()
                    checkpoint[scope.value] = snapshot
                except Exception as e:
                    logger.error(f"Failed to create snapshot for scope {scope.value}: {e}")
                    raise

        logger.info(f"Created checkpoint across {len(checkpoint)} scopes")
        return checkpoint

    async def restore_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Restore checkpoint across all scopes.

        Restores all registered managers from the provided checkpoint
        dictionary.

        Args:
            checkpoint: Checkpoint dictionary from create_checkpoint()

        Raises:
            ValueError: If checkpoint contains data for unregistered scopes

        Example:
            >>> await state.restore_checkpoint(checkpoint)
        """
        async with self._lock:
            for scope_value, snapshot in checkpoint.items():
                try:
                    scope = StateScope(scope_value)
                except ValueError:
                    logger.warning(f"Invalid scope in checkpoint: {scope_value}")
                    continue

                manager = self._managers.get(scope)
                if not manager:
                    raise ValueError(f"Cannot restore scope {scope.value}: no manager registered")

                try:
                    await manager.restore(snapshot)
                except Exception as e:
                    logger.error(f"Failed to restore snapshot for scope {scope.value}: {e}")
                    raise

        logger.info(f"Restored checkpoint across {len(checkpoint)} scopes")

    async def get_cross_scope_state(self) -> Dict[str, Dict[str, Any]]:
        """Get all state from all scopes.

        Returns a unified dictionary containing all state from all
        registered managers.

        Returns:
            Dictionary mapping scope values to their state dictionaries

        Example:
            >>> all_state = await state.get_cross_scope_state()
            >>> # all_state = {
            >>> #     "workflow": {"task_id": "..."},
            >>> #     "conversation": {"stage": "..."},
            >>> #     "team": {"coordinator": "..."},
            >>> #     "global": {"config": {...}}
            >>> # }
        """
        cross_scope_state: Dict[str, Dict[str, Any]] = {}

        for scope, manager in self._managers.items():
            try:
                state = await manager.get_all()
                cross_scope_state[scope.value] = state
            except Exception as e:
                logger.error(f"Failed to get state from scope {scope.value}: {e}")
                raise

        return cross_scope_state

    def get_registered_scopes(self) -> List[StateScope]:
        """Get list of registered scopes.

        Returns:
            List of StateScope enums that have registered managers
        """
        return list(self._managers.keys())

    def has_scope(self, scope: StateScope) -> bool:
        """Check if a scope has a registered manager.

        Args:
            scope: The StateScope to check

        Returns:
            True if a manager is registered for this scope, False otherwise
        """
        return scope in self._managers

    def get_manager(self, scope: StateScope) -> Optional[IStateManager]:
        """Get the manager for a specific scope.

        Args:
            scope: The StateScope to get the manager for

        Returns:
            The IStateManager instance for the scope, or None if not registered
        """
        return self._managers.get(scope)


__all__ = ["GlobalStateManager"]
