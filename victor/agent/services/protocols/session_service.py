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

"""Session service protocol.

Defines the interface for session lifecycle and management operations.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Protocol, runtime_checkable

if TYPE_CHECKING:
    from uuid import UUID


@runtime_checkable
class SessionInfo(Protocol):
    """Information about a session.

    Provides metadata about the current session state.
    """

    @property
    def session_id(self) -> str:
        """Unique session identifier."""
        ...

    @property
    def created_at(self) -> datetime:
        """Session creation timestamp."""
        ...

    @property
    def last_activity(self) -> datetime:
        """Last activity timestamp."""
        ...

    @property
    def message_count(self) -> int:
        """Number of messages in session."""
        ...

    @property
    def tool_calls(self) -> int:
        """Number of tool calls in session."""
        ...

    @property
    def metadata(self) -> Dict[str, Any]:
        """Additional session metadata."""
        ...


@runtime_checkable
class SessionState(Protocol):
    """Session state enumeration.

    Represents the current state of a session.
    """

    ACTIVE = "active"
    IDLE = "idle"
    SUSPENDED = "suspended"
    CLOSED = "closed"


@runtime_checkable
class SessionServiceProtocol(Protocol):
    """[CANONICAL] Protocol for session lifecycle and management service.

    This protocol represents the target architecture for session operations,
    replacing the facade-driven Coordinator pattern with a state-passed
    Service pattern.

    This protocol follows the Interface Segregation Principle (ISP)
    by focusing only on session-related operations.

    Methods:
        create_session: Create a new session
        get_session: Get existing session info
        update_session: Update session metadata
        close_session: Close and cleanup a session
        list_sessions: List all active sessions

    Example:
        class MySessionService(SessionServiceProtocol):
            def __init__(self, store):
                self._store = store

            async def create_session(self, metadata=None):
                session_id = str(uuid.uuid4())
                info = SessionInfo(
                    session_id=session_id,
                    created_at=datetime.now(),
                    metadata=metadata or {}
                )
                await self._store.save(session_id, info)
                return session_id
    """

    async def create_session(
        self,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Create a new session.

        Initializes a new session with the provided metadata.

        Args:
            metadata: Optional session metadata

        Returns:
            Session ID for the new session

        Example:
            session_id = await session_service.create_session(
                metadata={"user_id": "123", "project": "myproject"}
            )
        """
        ...

    async def get_session(self, session_id: str) -> Optional["SessionInfo"]:
        """Get information about a session.

        Retrieves session information including state, metrics,
        and metadata.

        Args:
            session_id: Session ID to retrieve

        Returns:
            SessionInfo if session exists, None otherwise

        Example:
            info = await session_service.get_session(session_id)
            if info:
                print(f"Session: {info.message_count} messages")
        """
        ...

    async def update_session(
        self,
        session_id: str,
        metadata: Dict[str, Any],
    ) -> bool:
        """Update session metadata.

        Updates the session metadata with the provided values.
        Merges with existing metadata rather than replacing.

        Args:
            session_id: Session ID to update
            metadata: Metadata to update

        Returns:
            True if session was updated, False if not found

        Example:
            await session_service.update_session(
                session_id,
                {"last_tool": "read", "stage": "planning"}
            )
        """
        ...

    async def close_session(self, session_id: str) -> bool:
        """Close and cleanup a session.

        Marks the session as closed and performs cleanup
        of resources associated with the session.

        Args:
            session_id: Session ID to close

        Returns:
            True if session was closed, False if not found

        Example:
            await session_service.close_session(session_id)
        """
        ...

    def get_current_session_id(self) -> Optional[str]:
        """Get the current active session ID.

        Returns the session ID for the currently active session
        in this context.

        Returns:
            Current session ID, or None if no active session

        Example:
            session_id = session_service.get_current_session_id()
            if session_id:
                # Use current session
                pass
        """
        ...

    async def list_sessions(
        self,
        state: Optional[str] = None,
        limit: int = 100,
    ) -> List["SessionInfo"]:
        """List sessions.

        Lists sessions, optionally filtered by state.

        Args:
            state: Optional state filter (ACTIVE, IDLE, CLOSED, etc.)
            limit: Maximum number of sessions to return

        Returns:
            List of session information

        Example:
            active = await session_service.list_sessions(state="active")
            print(f"Active sessions: {len(active)}")
        """
        ...

    async def delete_session(self, session_id: str) -> bool:
        """Delete a session completely.

        Permanently deletes the session and all associated data.
        Use with caution - this cannot be undone.

        Args:
            session_id: Session ID to delete

        Returns:
            True if session was deleted, False if not found

        Example:
            await session_service.delete_session(old_session_id)
        """
        ...

    async def get_session_metrics(
        self,
        session_id: str,
    ) -> Dict[str, Any]:
        """Get detailed metrics for a session.

        Returns comprehensive metrics about the session including
        message counts, tool usage, timing, and errors.

        Args:
            session_id: Session ID to get metrics for

        Returns:
            Dictionary with session metrics

        Example:
            metrics = await session_service.get_session_metrics(session_id)
            print(f"Messages: {metrics['message_count']}")
            print(f"Tool calls: {metrics['tool_calls']}")
        """
        ...

    def recover_session(self, session_id: str) -> bool:
        """Recover a previous session from persistent memory."""
        ...

    async def maybe_auto_checkpoint(self) -> Optional[str]:
        """Create an automatic checkpoint if configured thresholds are met."""
        ...

    async def save_checkpoint(
        self,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> Optional[str]:
        """Save a checkpoint of the current session state."""
        ...

    async def restore_checkpoint(self, checkpoint_id: str) -> bool:
        """Restore the current session state from a checkpoint."""
        ...

    def get_memory_context(
        self,
        max_tokens: Optional[int] = None,
        messages: Optional[List[Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Get token-aware memory context for the active session."""
        ...

    def get_recent_sessions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """List recent sessions for recovery UX."""
        ...

    def get_session_stats(self) -> Dict[str, Any]:
        """Get summary statistics for the current session."""
        ...

    def is_healthy(self) -> bool:
        """Check if the session service is healthy.

        A healthy session service should:
        - Have a current session or be able to create one
        - Have storage backend accessible
        - Not be in an error state

        Returns:
            True if the service is healthy, False otherwise
        """
        ...


@runtime_checkable
class SessionPersistenceProtocol(Protocol):
    """Extended protocol for session persistence operations.

    Provides methods for saving and restoring complete session state.
    """

    async def save_session(
        self,
        session_id: str,
        state: Dict[str, Any],
    ) -> bool:
        """Save complete session state.

        Serializes and persists the entire session state
        for later restoration.

        Args:
            session_id: Session ID to save
            state: Complete session state

        Returns:
            True if save succeeded, False otherwise

        Example:
            state = {
                "messages": messages,
                "context": context,
                "metadata": metadata
            }
            await session_service.save_session(session_id, state)
        """
        ...

    async def restore_session(
        self,
        session_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Restore complete session state.

        Loads and deserializes a previously saved session state.

        Args:
            session_id: Session ID to restore

        Returns:
            Session state if found, None otherwise

        Example:
            state = await session_service.restore_session(session_id)
            if state:
                messages = state["messages"]
                context = state["context"]
        """
        ...

    async def list_saved_sessions(self) -> List[str]:
        """List all saved session IDs.

        Returns a list of session IDs that have persisted state.

        Returns:
            List of session IDs

        Example:
            sessions = await session_service.list_saved_sessions()
            print(f"Saved sessions: {sessions}")
        """
        ...


@runtime_checkable
class SessionTimeoutProtocol(Protocol):
    """Protocol for session timeout management.

    Provides methods for managing session timeouts and idle detection.
    """

    async def update_last_activity(self, session_id: str) -> None:
        """Update last activity timestamp for a session.

        Should be called on each user interaction or operation.

        Args:
            session_id: Session ID to update

        Example:
            await session_service.update_last_activity(session_id)
        """
        ...

    async def check_session_timeout(
        self,
        session_id: str,
        timeout_seconds: int,
    ) -> bool:
        """Check if a session has timed out.

        Determines if the session has been idle longer than
        the specified timeout.

        Args:
            session_id: Session ID to check
            timeout_seconds: Timeout threshold in seconds

        Returns:
            True if session has timed out, False otherwise

        Example:
            if await session_service.check_session_timeout(session_id, 1800):
                # Session idle for 30 minutes
                await session_service.close_session(session_id)
        """
        ...

    async def get_idle_sessions(
        self,
        timeout_seconds: int,
    ) -> List[str]:
        """Get sessions that have been idle beyond timeout.

        Args:
            timeout_seconds: Timeout threshold in seconds

        Returns:
            List of session IDs that have timed out

        Example:
            idle = await session_service.get_idle_sessions(1800)
            for session_id in idle:
                await session_service.close_session(session_id)
        """
        ...


@runtime_checkable
class SessionExportProtocol(Protocol):
    """Protocol for session export functionality.

    Provides methods for exporting session data in various formats.
    """

    async def export_session(
        self,
        session_id: str,
        format: str = "json",
    ) -> str:
        """Export session data.

        Exports session data in the specified format.

        Args:
            session_id: Session ID to export
            format: Export format ("json", "yaml", "markdown")

        Returns:
            Exported session data as string

        Example:
            json_data = await session_service.export_session(
                session_id,
                format="json"
            )
        """
        ...

    async def export_session_conversation(
        self,
        session_id: str,
    ) -> str:
        """Export session conversation as formatted text.

        Exports the conversation in a human-readable format
        suitable for sharing or documentation.

        Args:
            session_id: Session ID to export

        Returns:
            Formatted conversation text

        Example:
            conversation = await session_service.export_session_conversation(
                session_id
            )
            print(conversation)
        """
        ...
