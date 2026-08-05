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

"""Factory functions for global state manager initialization.

This module provides factory functions for creating and initializing
the GlobalStateManager with all scope managers registered.

SOLID Principles:
- SRP: Factory functions only (no business logic)
- OCP: Extensible via new manager types
- DIP: Depends on GlobalStateManager abstraction

Usage:
    from victor.state.factory import get_global_manager, reset_global_manager

    # Get or create global state manager
    state = get_global_manager()

    # Use unified state access
    await state.set("key", "value", scope=StateScope.WORKFLOW)

    # Reset for testing
    reset_global_manager()
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, Optional

from victor.state.global_state_manager import GlobalStateManager
from victor.state.managers import (
    ConversationStateManager,
    GlobalStateManagerImpl,
    TeamStateManager,
    WorkflowStateManager,
)
from victor.state.protocols import StateScope
from victor.state.tracer import StateTracer

logger = logging.getLogger(__name__)

_global_manager: Optional[GlobalStateManager] = None
_tracer: Optional[StateTracer] = None
_factory_lock = threading.Lock()


def get_global_manager() -> GlobalStateManager:
    """Get or create global state manager instance.

    Implements singleton pattern - returns the same instance on subsequent calls.

    First call initializes all scope managers:
    - WorkflowStateManager (replaces ExecutionContext)
    - ConversationStateManager (replaces ConversationStateMachine)
    - TeamStateManager (replaces TeamContext)
    - GlobalStateManagerImpl (global scope)

    Returns:
        GlobalStateManager instance with all scopes registered

    Example:
        >>> from victor.state.factory import get_global_manager
        >>> state = get_global_manager()
        >>> await state.set("key", "value", scope=StateScope.WORKFLOW)
    """
    global _global_manager, _tracer

    if _global_manager is None:
        with _factory_lock:
            if _global_manager is None:
                manager = GlobalStateManager()

                # Register all scope managers
                manager.register_manager(StateScope.WORKFLOW, WorkflowStateManager())
                manager.register_manager(StateScope.CONVERSATION, ConversationStateManager())
                manager.register_manager(StateScope.TEAM, TeamStateManager())
                manager.register_manager(StateScope.GLOBAL, GlobalStateManagerImpl())

                _global_manager = manager
                logger.info("GlobalStateManager initialized with all scope managers")

    return _global_manager


def set_tracer(tracer: StateTracer) -> None:
    """Set the state tracer for the global state manager.

    Args:
        tracer: StateTracer instance for recording state transitions

    Example:
        >>> from victor.state.factory import get_global_manager, set_tracer
        >>> from victor.core.events import ObservabilityBus as EventBus
        >>> event_bus = EventBus()
        >>> tracer = StateTracer(event_bus)
        >>> set_tracer(tracer)
    """
    global _global_manager, _tracer

    if _global_manager is None:
        _global_manager = get_global_manager()

    _global_manager.set_tracer(tracer)
    _tracer = tracer

    logger.info("State tracer set on GlobalStateManager")


def get_tracer() -> Optional[StateTracer]:
    """Get the current state tracer instance.

    Returns:
        StateTracer instance if set, None otherwise
    """
    return _tracer


async def initialize_with_existing(
    workflow_state: Optional[Dict[str, Any]] = None,
    conversation_state: Optional[Dict[str, Any]] = None,
    team_state: Optional[Dict[str, Any]] = None,
    global_state: Optional[Dict[str, Any]] = None,
) -> GlobalStateManager:
    """Initialize global state manager with existing state data.

    Migration helper for transitioning from legacy state systems.
    Initializes the global manager and imports existing state data.

    Args:
        workflow_state: Existing workflow state (from ExecutionContext)
        conversation_state: Existing conversation state (from ConversationStateMachine)
        team_state: Existing team state (from TeamContext)
        global_state: Existing global state

    Returns:
        GlobalStateManager instance with imported state

    Example:
        >>> # Migrate from ExecutionContext
        >>> old_context = ExecutionContext()
        >>> old_context.task_id = "task-123"
        >>>
        >>> # Import to new system
        >>> state = await initialize_with_existing(
        ...     workflow_state={"task_id": old_context.task_id}
        ... )
    """
    manager = get_global_manager()

    # Import workflow state
    if workflow_state:
        try:
            wf_mgr = manager.get_manager(StateScope.WORKFLOW)
            if wf_mgr:
                await wf_mgr.update(workflow_state)
                logger.info(f"Imported {len(workflow_state)} keys to workflow scope")
        except Exception as e:
            logger.error(f"Failed to import workflow state: {e}")
            raise

    # Import conversation state
    if conversation_state:
        try:
            conv_mgr = manager.get_manager(StateScope.CONVERSATION)
            if conv_mgr:
                await conv_mgr.update(conversation_state)
                logger.info(f"Imported {len(conversation_state)} keys to conversation scope")
        except Exception as e:
            logger.error(f"Failed to import conversation state: {e}")
            raise

    # Import team state
    if team_state:
        try:
            team_mgr = manager.get_manager(StateScope.TEAM)
            if team_mgr:
                await team_mgr.update(team_state)
                logger.info(f"Imported {len(team_state)} keys to team scope")
        except Exception as e:
            logger.error(f"Failed to import team state: {e}")
            raise

    # Import global state
    if global_state:
        try:
            global_mgr = manager.get_manager(StateScope.GLOBAL)
            if global_mgr:
                await global_mgr.update(global_state)
                logger.info(f"Imported {len(global_state)} keys to global scope")
        except Exception as e:
            logger.error(f"Failed to import global state: {e}")
            raise

    return manager


def reset_global_manager() -> None:
    """Reset the global state manager instance.

    Deletes the singleton instance and creates a fresh one on next call
    to get_global_manager().

    Useful for testing and isolated environments.

    Example:
        >>> from victor.state.factory import reset_global_manager, get_global_manager
        >>>
        >>> # Reset for test isolation
        >>> reset_global_manager()
        >>> state = get_global_manager()
        >>> # Fresh instance for testing
    """
    global _global_manager, _tracer

    with _factory_lock:
        _global_manager = None
        _tracer = None

    logger.debug("GlobalStateManager reset")


def is_initialized() -> bool:
    """Check if global state manager has been initialized.

    Returns:
        True if get_global_manager() has been called at least once,
        False otherwise
    """
    return _global_manager is not None


__all__ = [
    "get_global_manager",
    "set_tracer",
    "get_tracer",
    "initialize_with_existing",
    "reset_global_manager",
    "is_initialized",
]
