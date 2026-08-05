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

"""Unified state management system.

This package provides unified state management across the Victor framework,
replacing fragmented state systems (ExecutionContext, ConversationStateMachine,
TeamContext, CopyOnWriteState) with a single, consistent API.

Architecture:
- IStateManager protocol: Canonical interface for all state managers
- StateScope enum: Four scopes (WORKFLOW, CONVERSATION, TEAM, GLOBAL)
- StateTracer: Records state transitions for debugging
- GlobalStateManager: Unified facade for all scopes

SOLID Principles:
- SRP: Each component has single responsibility
- OCP: Extensible via protocols and factory pattern
- LSP: All managers are substitutable via IStateManager
- ISP: Focused, minimal interfaces
- DIP: High-level modules depend on IStateManager abstraction

Usage:
    from victor.state import get_global_manager, StateScope

    # Get the global state manager
    state = get_global_manager()

    # Set a value
    await state.set("my_key", "value", scope=StateScope.WORKFLOW)

    # Get a value
    value = await state.get("my_key", scope=StateScope.WORKFLOW)

    # Create checkpoint
    checkpoint = await state.create_checkpoint()

    # Restore checkpoint
    await state.restore_checkpoint(checkpoint)
"""

# Protocols
from victor.state.protocols import IStateManager, IStateObserver, StateScope

# Tracer
from victor.state.tracer import StateTransition, StateTracer

# Managers
from victor.state.managers import (
    ConversationStateManager,
    GlobalStateManagerImpl,
    TeamStateManager,
    WorkflowStateManager,
)

# Global State Manager
from victor.state.global_state_manager import GlobalStateManager

# Factory
from victor.state.factory import (
    get_global_manager,
    initialize_with_existing,
    is_initialized,
    reset_global_manager,
    set_tracer,
)

# Re-export main items for convenience
__all__ = [
    # Protocols
    "StateScope",
    "IStateManager",
    "IStateObserver",
    # Tracer
    "StateTransition",
    "StateTracer",
    # Managers
    "WorkflowStateManager",
    "ConversationStateManager",
    "TeamStateManager",
    "GlobalStateManagerImpl",
    # Global State Manager
    "GlobalStateManager",
    # Factory
    "get_global_manager",
    "set_tracer",
    "initialize_with_existing",
    "reset_global_manager",
    "is_initialized",
]
