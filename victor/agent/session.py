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

"""File-based session persistence for saving and loading conversation state.

This module provides JSON file-based session persistence:
- Save conversation state to disk
- Load previous sessions
- List available sessions
- Auto-save functionality
- Session metadata (timestamp, model, provider, etc.)

The primary class is `SessionPersistence`.

For in-memory message history, see `victor.agent.message_history.MessageHistory`.
For SQLite-based persistence with token management, see `victor.agent.conversation.store.ConversationStore`.

Usage:
    from victor.agent.session import SessionPersistence

    # Save a session
    persistence = SessionPersistence()
    session_id = persistence.save_session(
        conversation=message_history,
        model="claude-sonnet-4-20250514",
        provider="anthropic",
        profile="default"
    )

    # List sessions
    sessions = persistence.list_sessions()

    # Load a session
    data = persistence.load_session(session_id)
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class SessionMetadata:
    """Metadata about a saved session."""

    session_id: str
    created_at: str  # ISO format timestamp
    updated_at: str  # ISO format timestamp
    model: str
    provider: str
    profile: str
    message_count: int
    title: str = ""  # Auto-generated or user-provided title
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionMetadata":
        """Create from dictionary."""
        return cls(
            session_id=data.get("session_id", ""),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
            model=data.get("model", ""),
            provider=data.get("provider", ""),
            profile=data.get("profile", ""),
            message_count=data.get("message_count", 0),
            title=data.get("title", ""),
            tags=data.get("tags", []),
        )


@dataclass
class Session:
    """A saved conversation session."""

    metadata: SessionMetadata
    conversation: Dict[str, Any]  # Serialized MessageHistory state
    conversation_state: Optional[Dict[str, Any]] = None  # ConversationStateMachine state
    tool_selection_stats: Optional[Dict[str, Any]] = None  # Tool usage statistics

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "metadata": self.metadata.to_dict(),
            "conversation": self.conversation,
            "conversation_state": self.conversation_state,
            "tool_selection_stats": self.tool_selection_stats,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Session":
        """Create from dictionary."""
        return cls(
            metadata=SessionMetadata.from_dict(data.get("metadata", {})),
            conversation=data.get("conversation", {}),
            conversation_state=data.get("conversation_state"),
            tool_selection_stats=data.get("tool_selection_stats"),
        )


class SessionPersistence:
    """File-based session persistence for Victor conversations.

    Sessions are stored as JSON files in {project}/.victor/sessions/ by default.
    Each session includes the conversation history, metadata, and optionally
    tool usage statistics.
    """

    def __init__(self, session_dir: Optional[Path] = None):
        """Initialize session manager.

        Args:
            session_dir: Directory to store sessions (default: {project}/.victor/sessions/)
        """
        from victor.config.settings import get_project_paths

        self.session_dir = session_dir or get_project_paths().sessions_dir
        self._ensure_session_dir()

    def _ensure_session_dir(self) -> None:
        """Ensure session directory exists."""
        self.session_dir.mkdir(parents=True, exist_ok=True)

    def _generate_session_id(self) -> str:
        """Generate a unique session ID based on timestamp."""
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def _generate_title(self, conversation_data: Dict[str, Any]) -> str:
        """Generate a title from the first user message.

        Args:
            conversation_data: Serialized conversation manager state

        Returns:
            A title string (truncated to 50 chars)
        """
        messages = conversation_data.get("messages", [])
        for msg in messages:
            if msg.get("role") == "user":
                content = str(msg.get("content", ""))
                # Truncate to 50 chars
                if len(content) > 50:
                    return content[:47] + "..."
                return content
        return "Untitled Session"

    def save_session(
        self,
        conversation: Any,  # MessageHistory or dict
        model: str,
        provider: str,
        profile: str = "default",
        session_id: Optional[str] = None,
        title: Optional[str] = None,
        tags: Optional[List[str]] = None,
        conversation_state: Optional[Any] = None,  # ConversationStateMachine or dict
        tool_selection_stats: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Save a session to disk.

        Args:
            conversation: MessageHistory instance or dict
            model: Model name used in the session
            provider: Provider name
            profile: Profile name
            session_id: Optional custom session ID (auto-generated if not provided)
            title: Optional custom title (auto-generated from first message if not provided)
            tags: Optional list of tags for categorization
            conversation_state: Optional ConversationStateMachine state
            tool_selection_stats: Optional tool usage statistics

        Returns:
            The session ID
        """
        # Convert conversation to dict if needed
        if hasattr(conversation, "to_dict"):
            conversation_data = conversation.to_dict()
        else:
            conversation_data = conversation

        # Convert conversation state to dict if needed
        state_data = None
        if conversation_state is not None:
            if hasattr(conversation_state, "to_dict"):
                state_data = conversation_state.to_dict()
            else:
                state_data = conversation_state

        # Generate session ID if not provided
        if session_id is None:
            session_id = self._generate_session_id()

        # Generate title if not provided
        if title is None:
            title = self._generate_title(conversation_data)

        now = datetime.now().isoformat()
        metadata = SessionMetadata(
            session_id=session_id,
            created_at=now,
            updated_at=now,
            model=model,
            provider=provider,
            profile=profile,
            message_count=len(conversation_data.get("messages", [])),
            title=title,
            tags=tags or [],
        )

        session = Session(
            metadata=metadata,
            conversation=conversation_data,
            conversation_state=state_data,
            tool_selection_stats=tool_selection_stats,
        )

        # Save to file
        session_file = self.session_dir / f"{session_id}.json"
        with open(session_file, "w", encoding="utf-8") as f:
            json.dump(session.to_dict(), f, indent=2, ensure_ascii=False)

        logger.info(f"Session saved: {session_id} ({title})")
        return session_id

    def load_session(self, session_id: str) -> Optional[Session]:
        """Load a session from disk.

        Args:
            session_id: The session ID to load

        Returns:
            Session object or None if not found
        """
        session_file = self.session_dir / f"{session_id}.json"
        if not session_file.exists():
            logger.warning(f"Session not found: {session_id}")
            return None

        try:
            with open(session_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return Session.from_dict(data)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to load session {session_id}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error loading session {session_id}: {e}")
            return None

    def list_sessions(
        self,
        limit: int = 20,
        tags: Optional[List[str]] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ) -> List[SessionMetadata]:
        """List available sessions.

        Args:
            limit: Maximum number of sessions to return
            tags: Filter by tags (any match)
            provider: Filter by provider name
            model: Filter by model name

        Returns:
            List of SessionMetadata objects, sorted by updated_at (newest first)
        """
        sessions: List[SessionMetadata] = []

        for session_file in self.session_dir.glob("*.json"):
            try:
                with open(session_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                metadata = SessionMetadata.from_dict(data.get("metadata", {}))

                # Apply filters
                if tags and not any(tag in metadata.tags for tag in tags):
                    continue
                if provider and metadata.provider != provider:
                    continue
                if model and metadata.model != model:
                    continue

                sessions.append(metadata)
            except Exception as e:
                logger.warning(f"Failed to read session file {session_file}: {e}")
                continue

        # Sort by updated_at descending
        sessions.sort(key=lambda s: s.updated_at, reverse=True)

        return sessions[:limit]

    def delete_session(self, session_id: str) -> bool:
        """Delete a session.

        Args:
            session_id: The session ID to delete

        Returns:
            True if deleted, False if not found
        """
        session_file = self.session_dir / f"{session_id}.json"
        if not session_file.exists():
            return False

        try:
            session_file.unlink()
            logger.info(f"Session deleted: {session_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete session {session_id}: {e}")
            return False

    def update_session(
        self,
        session_id: str,
        conversation: Any,
        conversation_state: Optional[Any] = None,
        tool_selection_stats: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Update an existing session with new conversation state.

        Args:
            session_id: The session ID to update
            conversation: Updated MessageHistory
            conversation_state: Updated ConversationStateMachine state
            tool_selection_stats: Updated tool usage statistics

        Returns:
            True if updated, False if session not found
        """
        session = self.load_session(session_id)
        if session is None:
            return False

        # Convert conversation to dict if needed
        if hasattr(conversation, "to_dict"):
            conversation_data = conversation.to_dict()
        else:
            conversation_data = conversation

        # Convert conversation state to dict if needed
        state_data = None
        if conversation_state is not None:
            if hasattr(conversation_state, "to_dict"):
                state_data = conversation_state.to_dict()
            else:
                state_data = conversation_state

        # Update session
        session.metadata.updated_at = datetime.now().isoformat()
        session.metadata.message_count = len(conversation_data.get("messages", []))
        session.conversation = conversation_data
        if state_data is not None:
            session.conversation_state = state_data
        if tool_selection_stats is not None:
            session.tool_selection_stats = tool_selection_stats

        # Save to file
        session_file = self.session_dir / f"{session_id}.json"
        with open(session_file, "w", encoding="utf-8") as f:
            json.dump(session.to_dict(), f, indent=2, ensure_ascii=False)

        logger.info(f"Session updated: {session_id}")
        return True

    def get_latest_session(self) -> Optional[Session]:
        """Get the most recent session.

        Returns:
            The most recent Session or None if no sessions exist
        """
        sessions = self.list_sessions(limit=1)
        if not sessions:
            return None
        return self.load_session(sessions[0].session_id)


# Default singleton instance
_default_persistence: Optional[SessionPersistence] = None


def get_session_manager() -> SessionPersistence:
    """Get the default session persistence instance.

    Note: Function name kept for backward compatibility.
    """
    global _default_persistence
    from victor.config.settings import get_project_paths

    current_session_dir = get_project_paths().sessions_dir
    if _default_persistence is None or _default_persistence.session_dir != current_session_dir:
        _default_persistence = SessionPersistence()
    return _default_persistence
