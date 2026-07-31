# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
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

"""Base formation strategy for multi-agent coordination.

MIGRATION NOTICE: TeamContext state storage is migrating to canonical system.

For team state storage, use the canonical state management system:
    - victor.state.TeamStateManager - Team scope state
    - victor.state.get_global_manager() - Unified access to all scopes

The TeamContext class now uses TeamStateManager internally for
shared_state storage.

---

Legacy Documentation:

This module defines the abstract base for all formation strategies,
following the Open/Closed Principle (OCP) and Strategy pattern.
"""

import threading
from abc import ABC, abstractmethod
from typing import Any, Awaitable, Callable, Dict, List, Optional

from victor.teams.types import AgentMessage, MemberResult


class TeamContext:
    """Simple context for team execution.

    MIGRATION: This class now integrates with TeamStateManager for
    canonical state storage. The shared_state dict is kept for
    backward compatibility but is synced to TeamStateManager.

    Attributes:
        team_id: Team identifier
        formation: Formation pattern being used
        shared_state: Shared state between agents (DEPRECATED - use state_manager)
        state_manager: Optional TeamStateManager for canonical state storage
        metadata: Additional metadata

    Example:
        # OLD (using shared_state dict):
        context = TeamContext("team-1", "orchestration")
        context.shared_state["coordinator"] = "agent-1"

        # NEW (using canonical state manager):
        from victor.state import TeamStateManager

        mgr = TeamStateManager()
        await mgr.set("coordinator", "agent-1")

        # OR with TeamContext integration:
        context = TeamContext("team-1", "orchestration", state_manager=mgr)
        await context.set("coordinator", "agent-1")  # Uses manager
    """

    def __init__(
        self,
        team_id: str,
        formation: str,
        shared_state: Optional[Dict[str, Any]] = None,
        state_manager: Optional[Any] = None,
        lsp_capability: Optional[Any] = None,
        **metadata: Any,
    ):
        self.team_id = team_id
        self.formation = formation
        self.shared_state = shared_state or {}
        self._lsp = lsp_capability  # Language Server Protocol capability
        self.metadata = metadata
        self._state_manager = state_manager
        self._lock = threading.Lock()

        # ADR-023 member durability (opt-in; set by UnifiedTeamCoordinator when a
        # checkpointer is configured). None → no checkpoint/resume, unchanged behavior.
        # checkpoint_hook: awaited after each member completes with
        #   (index, member_result, results_so_far, shared_state).
        # resume_completed: seeds a resumed run — {member_ids, member_results,
        #   shared_state, last_output, last_agent_id}.
        self.checkpoint_hook: Optional[Callable[..., Awaitable[None]]] = None
        self.resume_completed: Optional[Dict[str, Any]] = None

        # ADR-023 per-member streaming (opt-in; set by UnifiedTeamCoordinator when a
        # member event sink is present on the stream context var). Awaited by a formation
        # around each member with (kind, member_id, index, *, success=, content=). None →
        # no member events emitted, unchanged behavior.
        self.member_event_hook: Optional[Callable[..., Awaitable[None]]] = None

        # Initialize manager with existing shared_state
        if self._state_manager and self.shared_state:
            self._sync_to_manager()

    def _sync_to_manager(self) -> None:
        """Sync shared_state to the canonical state manager.

        Called once during __init__. Uses direct _state access since
        the async manager API is unavailable in a sync constructor.
        This is safe because no concurrent access exists at construction time.
        """
        if not self._state_manager:
            return

        try:
            for key, value in self.shared_state.items():
                self._state_manager._state[key] = value
        except Exception as e:
            import logging

            logger = logging.getLogger(__name__)
            logger.warning(f"Failed to sync state to manager: {e}")

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from shared state.

        DEPRECATED: Use state_manager.get() instead.

        Args:
            key: Key to retrieve
            default: Default value if key doesn't exist

        Returns:
            Value associated with key, or default
        """
        with self._lock:
            if self._state_manager:
                try:
                    return self._state_manager._state.get(key, default)
                except Exception:
                    pass
            return self.shared_state.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set a value in shared state.

        DEPRECATED: Use state_manager.set() instead.

        Args:
            key: Key to set
            value: Value to store
        """
        with self._lock:
            if self._state_manager:
                try:
                    self._state_manager._state[key] = value
                except Exception:
                    pass
            self.shared_state[key] = value

    def update(self, updates: Dict[str, Any]) -> None:
        """Update multiple values in shared state.

        DEPRECATED: Use state_manager.update() instead.

        Args:
            updates: Dictionary of key-value pairs to update
        """
        with self._lock:
            if self._state_manager:
                try:
                    self._state_manager._state.update(updates)
                except Exception:
                    pass
            self.shared_state.update(updates)

    @property
    def lsp(self) -> Optional[Any]:
        """Get the LSP capability for code intelligence in team context.

        Returns:
            LSPCapability instance or None
        """
        return self._lsp

    def set_lsp(self, lsp_capability: Any) -> None:
        """Set the LSP capability for team context.

        Enables language intelligence features for team coordination
        and code-related operations.

        Args:
            lsp_capability: LSPCapability instance
        """
        self._lsp = lsp_capability


class BaseFormationStrategy(ABC):
    """Abstract base for formation strategies.

    This class defines the contract for all formation strategies,
    allowing the coordinator to delegate execution logic while
    remaining independent of specific implementations (OCP).

    Implementations must define how to:
    - Execute agents in the formation pattern
    - Handle results from agents
    - Manage context flow between agents
    """

    @abstractmethod
    async def execute(
        self,
        agents: List[Any],
        context: TeamContext,
        task: AgentMessage,
    ) -> List[MemberResult]:
        """Execute agents using this formation strategy.

        Args:
            agents: List of agents to execute
            context: Team context with shared state
            task: Task message to process

        Returns:
            List of results from each agent
        """
        pass

    @abstractmethod
    def validate_context(self, context: "TeamContext") -> bool:
        """Validate that context has required fields for this formation.

        Args:
            context: Team context to validate

        Returns:
            True if context is valid for this formation
        """
        pass

    def get_required_roles(self) -> Optional[List[str]]:
        """Get required roles for this formation (if any).

        Returns:
            List of required role names, or None if any role is acceptable
        """
        return None

    def supports_early_termination(self) -> bool:
        """Check if this formation supports early termination.

        Returns:
            True if formation can terminate before all agents complete
        """
        return False

    def consumes_context_agents(self) -> bool:
        """Whether this formation reads role-named agents from the context.

        Most formations operate on the ``agents`` list passed to ``execute()``.
        A few (e.g. reflection) instead pull named role agents — "generator",
        "critic" — out of ``context`` (see :meth:`get_required_roles`). When this
        returns True, the coordinator binds members to those roles and injects
        them into the context before execution.

        Returns:
            True if ``execute()`` reads agents from context by role name.
        """
        return False

    async def prepare_context(
        self,
        context: "TeamContext",
        task: AgentMessage,
    ) -> "TeamContext":
        """Prepare context before execution.

        Args:
            context: Initial team context
            task: Task message

        Returns:
            Prepared context with formation-specific initialization
        """
        return context

    async def process_results(
        self,
        results: List[MemberResult],
        context: "TeamContext",
    ) -> List[MemberResult]:
        """Process results after execution.

        Args:
            results: Raw results from agents
            context: Team context after execution

        Returns:
            Processed results
        """
        return results
