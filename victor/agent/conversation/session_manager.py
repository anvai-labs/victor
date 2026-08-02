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

"""Session lifecycle management.

This module provides SessionManager, which handles session creation,
recovery, and management. Extracted from ConversationManager to
follow the Single Responsibility Principle (SRP).

Part of SOLID-based refactoring to eliminate god class anti-pattern.
"""

import logging
from typing import Any, Dict, List, Optional

from victor.agent.protocols import ISessionManager

logger = logging.getLogger(__name__)


class SessionManager(ISessionManager):
    """Manages session lifecycle.

    This class is responsible for:
    - Creating new sessions
    - Recovering existing sessions
    - Getting session statistics
    - Listing recent sessions

    SRP Compliance: Focuses only on session management, delegating
    message storage, context management, and embeddings to specialized components.

    Attributes:
        _store: Optional ConversationStore for persistence
        _enable_persistence: Whether persistence is enabled
    """

    def __init__(
        self,
        store: Optional[Any] = None,
        enable_persistence: bool = True,
    ):
        """Initialize the session manager.

        Args:
            store: Optional ConversationStore for persistence
            enable_persistence: Whether to enable persistence
        """
        self._store = store
        self._enable_persistence = enable_persistence

    def create_session(self) -> str:
        """Create a new session.

        Returns:
            Session ID
        """
        if not self._store or not self._enable_persistence:
            logger.warning("Cannot create session: persistence disabled or no store")
            return ""

        try:
            session_id = self._store.create_session()
            logger.info(f"Created session: {session_id}")
            return session_id
        except Exception as e:
            logger.error(f"Failed to create session: {e}")
            return ""

    def recover_session(self, session_id: str) -> bool:
        """Recover an existing session.

        Args:
            session_id: Session ID to recover

        Returns:
            True if recovery succeeded, False otherwise
        """
        if not self._store or not self._enable_persistence:
            logger.warning("Cannot recover session: persistence disabled or no store")
            return False

        try:
            session = self._store.get_session(session_id)
            if session:
                logger.info(f"Recovered session: {session_id}")
                return True
            else:
                logger.warning(f"Session not found: {session_id}")
                return False
        except Exception as e:
            logger.error(f"Failed to recover session {session_id}: {e}")
            return False

    def persist_session(self) -> bool:
        """Persist session state.

        Returns:
            True if persistence succeeded, False otherwise
        """
        if not self._store or not self._enable_persistence:
            return False

        try:
            self._store.persist_messages()
            return True
        except Exception as e:
            logger.warning(f"Failed to persist session: {e}")
            return False

    def get_recent_sessions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent sessions.

        Args:
            limit: Maximum number of sessions to return

        Returns:
            List of session dictionaries
        """
        if not self._store or not self._enable_persistence:
            return []

        try:
            sessions = self._store.list_sessions(limit=limit)
            return [session.to_dict() for session in sessions]
        except Exception as e:
            logger.error(f"Failed to list sessions: {e}")
            return []

    def get_session_stats(self) -> Dict[str, Any]:
        """Get session statistics.

        Returns:
            Dictionary with session statistics
        """
        if not self._store or not self._enable_persistence:
            return {
                "total_sessions": 0,
                "active_sessions": 0,
                "persistence_enabled": False,
            }

        try:
            sessions = self._store.list_sessions(limit=1000)
            return {
                "total_sessions": len(sessions),
                "active_sessions": len([s for s in sessions if s.is_active]),
                "persistence_enabled": True,
            }
        except Exception as e:
            logger.error(f"Failed to get session stats: {e}")
            return {
                "total_sessions": 0,
                "active_sessions": 0,
                "persistence_enabled": True,
                "error": str(e),
            }

    @property
    def session_id(self) -> Optional[str]:
        """Get current session ID.

        Returns:
            Current session ID or None
        """
        if not self._store or not self._enable_persistence:
            return None

        try:
            return self._store.session_id
        except Exception:
            return None

    def set_session_id(self, session_id: str) -> None:
        """Set current session ID.

        Args:
            session_id: Session ID to set
        """
        if self._store:
            self._store.session_id = session_id
