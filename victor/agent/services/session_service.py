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

"""Session service implementation.

Extracts session management from the AgentOrchestrator into
a focused, single-responsibility service following SOLID principles.

This service handles:
- Session creation and initialization
- Session state management
- Session persistence and restoration
- Session cleanup and disposal
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)
_UNSET = object()


class SessionInfoImpl:
    """Implementation of session information."""

    def __init__(
        self,
        session_id: str,
        created_at: datetime,
        last_activity: datetime,
        message_count: int = 0,
        tool_calls: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.session_id = session_id
        self.created_at = created_at
        self.last_activity = last_activity
        self.message_count = message_count
        self.tool_calls = tool_calls
        self.metadata = metadata or {}


class SessionService:
    """[CANONICAL] Service for session lifecycle and management.

    The target implementation for session operations following the
    state-passed architectural pattern.
    """

    def __init__(
        self,
        session_state_manager: Any,
        lifecycle_manager: Optional[Any] = None,
        memory_manager: Optional[Any] = None,
        checkpoint_manager: Optional[Any] = None,
        cost_tracker: Optional[Any] = None,
        session_timeout_seconds: int = 3600,
    ):
        """Initialize the session service.

        Args:
            session_state_manager: Manager for execution state tracking
            lifecycle_manager: Optional manager for lifecycle operations (may be None during init)
            memory_manager: Optional manager for persistent session storage
            checkpoint_manager: Optional manager for checkpoint save/restore
            cost_tracker: Optional tracker for session costs
            session_timeout_seconds: Default session timeout

        Note:
            lifecycle_manager is optional because it's created by component_assembler
            after SessionService initialization. The service will function without it,
            and can be set later via set_lifecycle_manager() if needed.
        """
        self._session_state = session_state_manager
        self._lifecycle_manager = lifecycle_manager  # May be None initially
        self._memory_manager = memory_manager
        self._checkpoint_manager = checkpoint_manager
        self._cost_tracker = cost_tracker
        self._session_timeout = session_timeout_seconds

        self._current_session: Optional[SessionInfoImpl] = None
        self._memory_session_id: Optional[str] = None
        self._logger = logging.getLogger(f"{__name__}.{id(self)}")

    def set_lifecycle_manager(self, lifecycle_manager: Any) -> None:
        """Set the lifecycle manager after it's been created.

        Called by component_assembler after lifecycle_manager creation.
        """
        self._lifecycle_manager = lifecycle_manager

    def bind_runtime_components(
        self,
        *,
        lifecycle_manager: Optional[Any] = _UNSET,
        memory_manager: Optional[Any] = _UNSET,
        checkpoint_manager: Optional[Any] = _UNSET,
        cost_tracker: Optional[Any] = _UNSET,
        memory_session_id: Optional[str] = _UNSET,
    ) -> None:
        """Bind live runtime collaborators after bootstrap."""
        if lifecycle_manager is not _UNSET:
            self._lifecycle_manager = lifecycle_manager
        if memory_manager is not _UNSET:
            self._memory_manager = memory_manager
        if checkpoint_manager is not _UNSET:
            self._checkpoint_manager = checkpoint_manager
        if cost_tracker is not _UNSET:
            self._cost_tracker = cost_tracker
        if memory_session_id is not _UNSET:
            self._memory_session_id = memory_session_id

    async def create_session(
        self,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Create a new session."""
        if not session_id:
            session_id = f"session-{uuid.uuid4().hex[:16]}"

        now = datetime.now()
        self._current_session = SessionInfoImpl(
            session_id=session_id,
            created_at=now,
            last_activity=now,
            metadata=metadata or {},
        )

        # Reset state for new session
        self._session_state.reset()

        # Initialize memory session if available
        if self._memory_manager:
            try:
                self._memory_session_id = self._memory_manager.create_session(
                    project_path=getattr(self._memory_manager, "_project_path", None),
                )
            except Exception as e:
                self._logger.warning(f"Failed to create memory session: {e}")
                self._memory_session_id = None

        self._logger.info(f"Created session: {session_id}")
        return session_id

    async def get_session(self, session_id: str) -> Optional[SessionInfoImpl]:
        """Get the active session info if it matches the requested ID."""
        current = self._current_session
        if current is None or current.session_id != session_id:
            return None
        return current

    async def update_session(self, session_id: str, metadata: Dict[str, Any]) -> bool:
        """Update metadata for the active session."""
        current = self._current_session
        if current is None or current.session_id != session_id:
            return False
        current.metadata.update(metadata)
        self.update_activity()
        return True

    async def close_session(self, session_id: str) -> bool:
        """Close the active session if it matches the requested ID."""
        current = self._current_session
        if current is None or current.session_id != session_id:
            return False
        await self.end_session()
        return True

    async def list_sessions(
        self,
        state: Optional[str] = None,
        limit: int = 100,
    ) -> List[SessionInfoImpl]:
        """List active sessions currently loaded in-memory."""
        if limit <= 0:
            return []
        current = self._current_session
        if current is None:
            return []
        if state and state != "active":
            return []
        return [current]

    async def delete_session(self, session_id: str) -> bool:
        """Delete the active session state."""
        return await self.close_session(session_id)

    async def get_session_metrics(self, session_id: str) -> Dict[str, Any]:
        """Get detailed metrics for the active session."""
        current = self._current_session
        if current is None or current.session_id != session_id:
            return {}
        return {
            "message_count": current.message_count,
            "tool_calls": current.tool_calls,
            "tool_calls_used": self._session_state.tool_calls_used,
            "token_usage": self._session_state.get_token_usage(),
            "metadata": dict(current.metadata),
        }

    def recover_session(self, session_id: str) -> bool:
        """Recover a previous session."""
        if not self._memory_manager:
            return False

        if self._lifecycle_manager:
            success = self._lifecycle_manager.recover_session(
                session_id=session_id,
                memory_manager=self._memory_manager,
            )

            if success:
                self._memory_session_id = session_id
                self._current_session = SessionInfoImpl(
                    session_id=session_id,
                    created_at=datetime.now(),
                    last_activity=datetime.now(),
                )
                return True
        return False

    async def maybe_auto_checkpoint(self) -> Optional[str]:
        """Trigger auto-checkpoint if threshold met."""
        if not self._checkpoint_manager:
            return None

        state = self._get_checkpoint_state()
        try:
            return await self._checkpoint_manager.maybe_auto_checkpoint(
                session_id=self._memory_session_id or "default",
                state=state,
            )
        except Exception as e:
            self._logger.debug(f"Auto-checkpoint failed: {e}")
            return None

    async def save_checkpoint(
        self,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> Optional[str]:
        """Save a checkpoint of the current session state."""
        if not self._checkpoint_manager:
            self._logger.debug("Checkpoint save skipped - manager not initialized")
            return None

        state = self._get_checkpoint_state()
        try:
            checkpoint_id = await self._checkpoint_manager.save_checkpoint(
                session_id=self._memory_session_id or "default",
                state=state,
                description=description,
                tags=tags,
            )
            self._logger.info(f"Checkpoint saved: {checkpoint_id[:20]}...")
            return checkpoint_id
        except Exception as e:
            self._logger.warning(f"Failed to save checkpoint: {e}")
            return None

    async def restore_checkpoint(self, checkpoint_id: str) -> bool:
        """Restore session state from a checkpoint."""
        if not self._checkpoint_manager:
            self._logger.warning("Cannot restore - checkpoint manager not initialized")
            return False

        try:
            state = await self._checkpoint_manager.restore_checkpoint(checkpoint_id)
            return self._apply_checkpoint_state(state)
        except Exception as e:
            self._logger.warning(f"Failed to restore checkpoint: {e}")
            return False

    def get_memory_context(
        self,
        max_tokens: Optional[int] = None,
        messages: Optional[List[Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Get token-aware context from memory."""
        if not self._memory_manager or not self._memory_session_id:
            if messages:
                return [msg.model_dump() if hasattr(msg, "model_dump") else msg for msg in messages]
            return []

        try:
            return self._memory_manager.get_context_messages(
                session_id=self._memory_session_id,
                max_tokens=max_tokens,
            )
        except Exception as e:
            self._logger.warning(f"Failed to get memory context: {e}")
            if messages:
                return [msg.model_dump() if hasattr(msg, "model_dump") else msg for msg in messages]
            return []

    def get_session_stats(self) -> Dict[str, Any]:
        """Get comprehensive session statistics."""
        base_stats: Dict[str, Any] = {
            "enabled": bool(self._memory_manager),
            "session_id": self.get_current_session_id(),
            "memory_session_id": self._memory_session_id,
            "tool_calls_used": self._session_state.tool_calls_used,
            "token_usage": self._session_state.get_token_usage(),
        }

        if self._memory_manager and self._memory_session_id:
            try:
                memory_stats = self._memory_manager.get_session_stats(self._memory_session_id)
                if memory_stats:
                    base_stats.update(memory_stats)
            except Exception as e:
                self._logger.warning(f"Failed to get memory stats: {e}")

        return base_stats

    def _get_checkpoint_state(self) -> Dict[str, Any]:
        """Build state dict for checkpointing."""
        return {
            "session_id": self.get_current_session_id(),
            "tool_calls_used": self._session_state.tool_calls_used,
            "token_usage": self._session_state.get_token_usage(),
            "observed_files": list(self._session_state.observed_files),
        }

    def get_current_session_id(self) -> Optional[str]:
        """Get the current active session ID."""
        return self._current_session.session_id if self._current_session else None

    # ==========================================================================
    # Session Lifecycle
    # ==========================================================================

    async def end_session(self) -> None:
        """End the current session.

        Cleans up session resources, saves checkpoints if available,
        and clears the current session state.

        Example:
            await service.end_session()
        """
        if not self._current_session:
            self._logger.warning("No active session to end")
            return

        session_id = self._current_session.session_id

        # Try to save checkpoint before ending
        if self._checkpoint_manager:
            try:
                await self.maybe_auto_checkpoint()
            except Exception as e:
                self._logger.warning(f"Failed to save checkpoint on end: {e}")

        # Clear session
        self._current_session = None
        self._memory_session_id = None

        self._logger.info(f"Ended session: {session_id}")

    async def reset_session(self) -> None:
        """Reset the current session state.

        Resets the execution state for the current session without
        ending the session. Useful for starting fresh while keeping
        the same session ID.

        Example:
            await service.reset_session()
        """
        if not self._current_session:
            self._logger.warning("No active session to reset")
            return

        # Reset execution state
        self._session_state.reset()

        # Update activity
        self.update_activity()

        self._logger.info(f"Reset session: {self._current_session.session_id}")

    # ==========================================================================
    # Session Properties
    # ==========================================================================

    @property
    def session_id(self) -> Optional[str]:
        """Get the current session ID.

        Returns the session ID of the active session, or None if no
        active session.

        Returns:
            Session ID string or None

        Example:
            sid = service.session_id
            print(f"Current session: {sid}")
        """
        return self.get_current_session_id()

    @property
    def memory_session_id(self) -> Optional[str]:
        """Get the memory session ID.

        Returns the memory manager's session ID for the current session,
        or None if no active memory session.

        Returns:
            Memory session ID string or None

        Example:
            mid = service.memory_session_id
            print(f"Memory session: {mid}")
        """
        return self._memory_session_id

    @property
    def is_active(self) -> bool:
        """Check if a session is currently active.

        Returns True if there is an active session (not ended or timed out).

        Returns:
            True if session is active, False otherwise

        Example:
            if service.is_active:
                print("Session is active")
        """
        return self._current_session is not None and not self.is_session_timeout()

    # ==========================================================================
    # Budget Properties
    # ==========================================================================

    @property
    def tool_calls_used(self) -> int:
        """Get the number of tool calls used in this session.

        Returns:
            Number of tool calls executed

        Example:
            calls = service.tool_calls_used
            print(f"Tool calls used: {calls}")
        """
        return self._session_state.tool_calls_used

    @property
    def remaining_budget(self) -> int:
        """Get the remaining tool budget for this session.

        Returns:
            Number of tool calls remaining

        Example:
            remaining = service.remaining_budget
            print(f"Remaining budget: {remaining}")
        """
        # Assuming a default budget of 100 if not set
        max_budget = getattr(self._session_state, "max_budget", 100)
        return max(0, max_budget - self._session_state.tool_calls_used)

    @property
    def is_budget_exhausted(self) -> bool:
        """Check if the tool budget is exhausted.

        Returns:
            True if no more tool calls allowed

        Example:
            if service.is_budget_exhausted:
                print("Budget exhausted!")
        """
        return self.remaining_budget == 0

    # ==========================================================================
    # Session Summary and Cost Tracking
    # ==========================================================================

    def get_session_summary(self) -> Dict[str, Any]:
        """Get a summary of the current session.

        Returns:
            Dictionary with session summary including:
            - session_id
            - duration (seconds)
            - message_count
            - tool_calls_used
            - token_usage
            - is_active

        Example:
            summary = service.get_session_summary()
            print(f"Session duration: {summary['duration_seconds']}s")
        """
        if not self._current_session:
            return {
                "session_id": None,
                "duration_seconds": 0,
                "message_count": 0,
                "tool_calls_used": 0,
                "token_usage": {},
                "is_active": False,
            }

        age = self.get_session_age()

        return {
            "session_id": self._current_session.session_id,
            "duration_seconds": age,
            "message_count": self._current_session.message_count,
            "tool_calls_used": self._current_session.tool_calls,
            "token_usage": self._session_state.get_token_usage(),
            "is_active": self.is_active,
        }

    def get_session_cost_summary(self) -> Dict[str, Any]:
        """Get cost summary for the current session.

        Returns:
            Dictionary with cost information:
            - total_cost (estimated)
            - input_tokens
            - output_tokens
            - total_tokens
            - tool_calls

        Example:
            costs = service.get_session_cost_summary()
            print(f"Total cost: ${costs['total_cost']:.4f}")
        """
        token_usage = self._session_state.get_token_usage()

        # Simple cost estimation (can be enhanced with actual pricing)
        input_tokens = token_usage.get("input", 0)
        output_tokens = token_usage.get("output", 0)
        total_tokens = input_tokens + output_tokens

        # Approximate costs (adjust based on actual provider pricing)
        # Using generic rates: $0.001/1K input tokens, $0.002/1K output tokens
        input_cost = (input_tokens / 1000) * 0.001
        output_cost = (output_tokens / 1000) * 0.002
        total_cost = input_cost + output_cost

        return {
            "total_cost": total_cost,
            "input_cost": input_cost,
            "output_cost": output_cost,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "tool_calls": (self._current_session.tool_calls if self._current_session else 0),
        }

    def get_session_cost_formatted(self) -> str:
        """Get a formatted cost summary string.

        Returns:
            Human-readable cost summary

        Example:
            summary = service.get_session_cost_formatted()
            print(summary)
            # "Session Cost: $0.0150 (1500 input + 500 output tokens, 5 tool calls)"
        """
        costs = self.get_session_cost_summary()

        return (
            f"Session Cost: ${costs['total_cost']:.4f} "
            f"({costs['input_tokens']} input + {costs['output_tokens']} output tokens, "
            f"{costs['tool_calls']} tool calls)"
        )

    # ==========================================================================
    # Session Persistence and State Management
    # ==========================================================================

    def get_session_state(self) -> Dict[str, Any]:
        """Get current session state as dictionary.

        Returns a snapshot of the current session state including
        session info, state data, and metadata. Useful for
        serialization or persistence.

        Returns:
            Dictionary with complete session state

        Example:
            state = service.get_session_state()
            # Save to file, database, etc.
        """
        if not self._current_session:
            return {
                "session_id": None,
                "created_at": None,
                "last_activity": None,
                "message_count": 0,
                "tool_calls": 0,
                "metadata": {},
                "is_active": False,
            }

        return {
            "session_id": self._current_session.session_id,
            "created_at": self._current_session.created_at.isoformat(),
            "last_activity": self._current_session.last_activity.isoformat(),
            "message_count": self._current_session.message_count,
            "tool_calls": self._current_session.tool_calls,
            "metadata": self._current_session.metadata,
            "memory_session_id": self._memory_session_id,
            "tool_calls_used": self._session_state.tool_calls_used,
            "token_usage": self._session_state.get_token_usage(),
            "observed_files": list(self._session_state.observed_files),
            "is_active": True,
        }

    def load_session_state(self, state: Dict[str, Any]) -> bool:
        """Load session state from dictionary.

        Restores session from a previously saved state dictionary.
        Useful for session persistence and restoration.

        Args:
            state: Session state dictionary from get_session_state()

        Returns:
            True if loaded successfully, False otherwise

        Example:
            state = load_from_file("session.json")
            success = service.load_session_state(state)
        """
        if not state or not state.get("is_active", False):
            self._logger.warning("Invalid or inactive session state")
            return False

        try:
            # Restore session info
            self._current_session = SessionInfoImpl(
                session_id=state["session_id"],
                created_at=datetime.fromisoformat(state["created_at"]),
                last_activity=datetime.fromisoformat(state["last_activity"]),
                message_count=state.get("message_count", 0),
                tool_calls=state.get("tool_calls", 0),
                metadata=state.get("metadata", {}),
            )

            # Restore memory session ID
            self._memory_session_id = state.get("memory_session_id")

            # Restore observed files (may be read-only property)
            if "observed_files" in state:
                try:
                    self._session_state.observed_files = set(state["observed_files"])
                except (AttributeError, TypeError):
                    # Property is read-only or not settable - skip restoration
                    self._logger.debug("observed_files is read-only, skipping restoration")

            self._logger.info(f"Loaded session state: {state['session_id']}")
            return True

        except (KeyError, ValueError) as e:
            self._logger.error(f"Failed to load session state: {e}")
            return False

    def _apply_checkpoint_state(self, state: Dict[str, Any]) -> bool:
        """Apply checkpoint state to session.

        Applies a previously saved checkpoint state to the current session,
        restoring execution state, token usage, and other tracked data.

        Args:
            state: Checkpoint state dictionary

        Returns:
            True if applied successfully, False otherwise

        Example:
            if service._apply_checkpoint_state(checkpoint):
                print("Checkpoint restored successfully")
        """
        if not state:
            self._logger.warning("No checkpoint state to apply")
            return False

        try:
            # Restore tool calls used
            if "tool_calls_used" in state:
                self._session_state.tool_calls_used = state["tool_calls_used"]

            # Restore token usage
            if "token_usage" in state:
                # Create a token usage dict if needed
                token_usage = state["token_usage"]
                if hasattr(self._session_state, "update_token_usage"):
                    self._session_state.update_token_usage(
                        token_usage.get("input", 0), token_usage.get("output", 0)
                    )
                elif hasattr(self._session_state, "set_token_usage"):
                    self._session_state.set_token_usage(
                        token_usage.get("input", 0), token_usage.get("output", 0)
                    )

            # Restore observed files if possible
            if "observed_files" in state:
                try:
                    if hasattr(self._session_state, "observed_files"):
                        self._session_state.observed_files = set(state["observed_files"])
                except (AttributeError, TypeError):
                    # Property is read-only or not settable
                    self._logger.debug("observed_files is read-only, skipping restoration")

            self._logger.info(
                f"Applied checkpoint state for session: {state.get('session_id', 'unknown')}"
            )
            return True

        except Exception as e:
            self._logger.error(f"Failed to apply checkpoint state: {e}")
            return False

    def update_token_usage(self, input_tokens: int, output_tokens: int) -> None:
        """Update token usage statistics.

        Updates the session's token usage tracking with the given
        input and output token counts.

        Args:
            input_tokens: Number of input tokens consumed
            output_tokens: Number of output tokens consumed

        Example:
            service.update_token_usage(1500, 500)
        """
        if hasattr(self._session_state, "update_token_usage"):
            self._session_state.update_token_usage(input_tokens, output_tokens)
        elif hasattr(self._session_state, "set_token_usage"):
            self._session_state.set_token_usage(input_tokens, output_tokens)
        else:
            # Fallback: track in a simple dict
            if not hasattr(self._session_state, "_token_usage"):
                self._session_state._token_usage = {"input": 0, "output": 0}

            self._session_state._token_usage["input"] += input_tokens
            self._session_state._token_usage["output"] += output_tokens

        self._logger.debug(f"Updated token usage: +{input_tokens} input, +{output_tokens} output")

    def reset_token_usage(self) -> None:
        """Reset token usage statistics.

        Clears all token usage tracking for the current session,
        resetting both input and output token counts to zero.

        Example:
            service.reset_token_usage()
        """
        if hasattr(self._session_state, "reset_token_usage"):
            self._session_state.reset_token_usage()
        elif hasattr(self._session_state, "_token_usage"):
            self._session_state._token_usage = {"input": 0, "output": 0}
        else:
            # Initialize if not present
            self._session_state._token_usage = {"input": 0, "output": 0}

        self._logger.info("Reset token usage statistics")

    # ==========================================================================
    # Session Metrics and Monitoring
    # ==========================================================================

    def get_session_age(self) -> float:
        """Get session age in seconds.

        Returns the number of seconds since the session was created.
        Returns 0 if no active session.

        Returns:
            Session age in seconds

        Example:
            age = service.get_session_age()
            # Returns: 3600.0 (1 hour old)
        """
        if not self._current_session:
            return 0.0

        now = datetime.now()
        delta = now - self._current_session.created_at
        return delta.total_seconds()

    def get_session_idle_time(self) -> float:
        """Get session idle time in seconds.

        Returns the number of seconds since the last activity.
        Returns 0 if no active session.

        Returns:
            Idle time in seconds

        Example:
            idle = service.get_session_idle_time()
            # Returns: 1800.0 (30 minutes idle)
        """
        if not self._current_session:
            return 0.0

        now = datetime.now()
        delta = now - self._current_session.last_activity
        return delta.total_seconds()

    def is_session_timeout(self) -> bool:
        """Check if session has timed out.

        Returns True if the session has been idle for longer than
        the configured timeout period.

        Returns:
            True if session timed out, False otherwise

        Example:
            if service.is_session_timeout():
                # Handle timeout
        """
        idle_time = self.get_session_idle_time()
        return idle_time > self._session_timeout

    def get_session_metadata(self) -> Dict[str, Any]:
        """Get session metadata.

        Returns the metadata dictionary for the current session.
        Returns empty dict if no active session.

        Returns:
            Session metadata dictionary

        Example:
            metadata = service.get_session_metadata()
            # {"user_id": "123", "task": "code-review"}
        """
        if not self._current_session:
            return {}
        return dict(self._current_session.metadata)

    def update_session_metadata(self, metadata: Dict[str, Any]) -> None:
        """Update session metadata.

        Merges the provided metadata with existing session metadata.
        New keys overwrite existing keys.

        Args:
            metadata: Metadata to merge into session

        Example:
            service.update_session_metadata({"task": "new-task"})
        """
        if not self._current_session:
            self._logger.warning("No active session to update metadata")
            return

        self._current_session.metadata.update(metadata)
        self._logger.debug(f"Updated session metadata: {list(metadata.keys())}")

    def update_activity(self) -> None:
        """Update last activity timestamp to current time.

        Should be called when the session is used to prevent timeout.
        This is called automatically during most session operations.

        Example:
            service.update_activity()
        """
        if self._current_session:
            self._current_session.last_activity = datetime.now()

    def get_session_info(self) -> Optional[Dict[str, Any]]:
        """Get complete session information.

        Returns comprehensive information about the current session
        including ID, timestamps, metrics, and metadata.

        Returns:
            Dictionary with complete session info, or None if no active session

        Example:
            info = service.get_session_info()
            # {
            #   "session_id": "session-abc123",
            #   "created_at": "2025-04-17T10:30:00",
            #   "last_activity": "2025-04-17T11:30:00",
            #   "age_seconds": 3600.0,
            #   "idle_seconds": 600.0,
            #   "is_timeout": False,
            #   "message_count": 10,
            #   "tool_calls": 5,
            #   "metadata": {...}
            # }
        """
        if not self._current_session:
            return None

        now = datetime.now()
        age_delta = now - self._current_session.created_at
        idle_delta = now - self._current_session.last_activity

        return {
            "session_id": self._current_session.session_id,
            "created_at": self._current_session.created_at.isoformat(),
            "last_activity": self._current_session.last_activity.isoformat(),
            "age_seconds": age_delta.total_seconds(),
            "idle_seconds": idle_delta.total_seconds(),
            "is_timeout": self.is_session_timeout(),
            "message_count": self._current_session.message_count,
            "tool_calls": self._current_session.tool_calls,
            "metadata": self._current_session.metadata,
            "memory_session_id": self._memory_session_id,
        }

    def is_healthy(self) -> bool:
        """Check if the session service is healthy.

        Returns:
            True if the service is healthy
        """
        return True

    # ==========================================================================
    # Serialization and Deserialization
    # ==========================================================================

    def to_dict(self) -> Dict[str, Any]:
        """Convert current session to dictionary for serialization.

        Returns:
            Dictionary representation of current session

        Example:
            data = service.to_dict()
            # Save to file, database, etc.
        """
        if not self._current_session:
            return {
                "session_id": None,
                "created_at": None,
                "last_activity": None,
                "message_count": 0,
                "tool_calls": 0,
                "metadata": {},
                "memory_session_id": None,
            }

        return {
            "session_id": self._current_session.session_id,
            "created_at": self._current_session.created_at.isoformat(),
            "last_activity": self._current_session.last_activity.isoformat(),
            "message_count": self._current_session.message_count,
            "tool_calls": self._current_session.tool_calls,
            "metadata": self._current_session.metadata,
            "memory_session_id": self._memory_session_id,
            "tool_calls_used": self._session_state.tool_calls_used,
            "token_usage": self._session_state.get_token_usage(),
            "is_active": self.is_active,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionService":
        """Create SessionService instance from dictionary.

        Args:
            data: Dictionary representation from to_dict()

        Returns:
            New SessionService instance with restored state

        Example:
            service = SessionService.from_dict(data)
        """
        import time

        # Create minimal dependencies
        class MockSessionState:
            def __init__(self, data):
                self.tool_calls_used = data.get("tool_calls_used", 0)
                self.observed_files = set()

            def reset(self):
                self.tool_calls_used = 0
                self.observed_files.clear()

            def get_token_usage(self):
                return data.get("token_usage", {"input": 0, "output": 0})

        session_state = MockSessionState(data)

        # Create service
        service = cls(
            session_state_manager=session_state,
            lifecycle_manager=None,
            memory_manager=None,
            checkpoint_manager=None,
        )

        # Restore session if active
        if data.get("session_id") and data.get("is_active", False):
            from datetime import datetime

            service._current_session = SessionInfoImpl(
                session_id=data["session_id"],
                created_at=(
                    datetime.fromisoformat(data["created_at"])
                    if data.get("created_at")
                    else datetime.now()
                ),
                last_activity=(
                    datetime.fromisoformat(data["last_activity"])
                    if data.get("last_activity")
                    else datetime.now()
                ),
                message_count=data.get("message_count", 0),
                tool_calls=data.get("tool_calls", 0),
                metadata=data.get("metadata", {}),
            )
            service._memory_session_id = data.get("memory_session_id")

        return service

    # ==========================================================================
    # Recent Sessions
    # ==========================================================================

    def get_recent_sessions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent conversation sessions for recovery.

        Args:
            limit: Maximum number of sessions to return

        Returns:
            List of session metadata dictionaries

        Example:
            recent = service.get_recent_sessions(limit=5)
            for session in recent:
                print(f"Session: {session['session_id']}")
        """
        if not self._memory_manager:
            return []

        try:
            sessions = self._memory_manager.list_sessions(limit=limit)
            return [
                {
                    "session_id": s.session_id,
                    "created_at": (
                        s.created_at.isoformat()
                        if hasattr(s, "created_at") and s.created_at
                        else None
                    ),
                    "last_activity": (
                        s.last_activity.isoformat()
                        if hasattr(s, "last_activity") and s.last_activity
                        else None
                    ),
                    "project_path": getattr(s, "project_path", None),
                    "provider": getattr(s, "provider", None),
                    "model": getattr(s, "model", None),
                    "message_count": len(s.messages) if hasattr(s, "messages") else 0,
                }
                for s in sessions
            ]
        except Exception as e:
            self._logger.warning(f"Failed to get recent sessions: {e}")
            return []

    # ==========================================================================
    # Embedding Store Initialization
    # ==========================================================================

    @staticmethod
    def init_conversation_embedding_store(
        memory_manager: Any,
    ) -> tuple[Optional[Any], Optional[Any]]:
        """Initialize LanceDB embedding store for semantic conversation retrieval.

        Uses the module-level singleton to prevent duplicate initialization.
        The singleton pattern ensures that intelligent_prompt_builder and other
        components share the same instance.

        Args:
            memory_manager: Memory manager to wire the embedding store to

        Returns:
            Tuple of (conversation_embedding_store, pending_semantic_cache).
            Either or both may be None if initialization fails.

        Example:
            store, cache = SessionService.init_conversation_embedding_store(memory_mgr)
            if store:
                print("Embedding store initialized")
        """
        if memory_manager is None:
            return None, None

        conversation_embedding_store = None
        pending_semantic_cache = None

        try:
            # Try to get or create embedding store from memory manager
            if hasattr(memory_manager, "get_embedding_store"):
                conversation_embedding_store = memory_manager.get_embedding_store()
            elif hasattr(memory_manager, "embedding_store"):
                conversation_embedding_store = memory_manager.embedding_store

            # Try to get semantic cache
            if hasattr(memory_manager, "get_semantic_cache"):
                pending_semantic_cache = memory_manager.get_semantic_cache()
            elif hasattr(memory_manager, "semantic_cache"):
                pending_semantic_cache = memory_manager.semantic_cache

        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.warning(f"Failed to initialize embedding store: {e}")

        return conversation_embedding_store, pending_semantic_cache

    # ==========================================================================
    # Background Tasks
    # ==========================================================================

    @staticmethod
    def create_background_task(
        coro: Any,
        name: str,
        background_tasks: set,
        bg_task_lock: Any = None,
    ) -> Optional[Any]:
        """Create and track a background task for graceful shutdown.

        Args:
            coro: The coroutine to run as a background task
            name: Name for the task (for logging)
            background_tasks: Set tracking active background tasks
            bg_task_lock: Optional lock protecting concurrent add/discard

        Returns:
            The created task, or None if no event loop is available

        Example:
            async def my_background_task():
                while True:
                    await asyncio.sleep(1)
                    # Do work

            task = SessionService.create_background_task(
                my_background_task(),
                "my-task",
                background_tasks,
            )
        """
        import asyncio
        import threading

        try:
            loop = asyncio.get_running_loop()
            task = loop.create_task(coro, name=name)

            if bg_task_lock:
                with bg_task_lock:
                    background_tasks.add(task)
            else:
                background_tasks.add(task)

            def _discard_task(t: Any) -> None:
                if bg_task_lock:
                    with bg_task_lock:
                        background_tasks.discard(t)
                else:
                    background_tasks.discard(t)

            task.add_done_callback(_discard_task)

            logger = logging.getLogger(__name__)
            logger.debug(f"Created background task: {name}")

            return task

        except RuntimeError:
            # No event loop running
            if asyncio.iscoroutine(coro):
                coro.close()
            logger = logging.getLogger(__name__)
            logger.warning(f"Failed to create background task '{name}': no event loop")
            return None
