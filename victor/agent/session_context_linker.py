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

"""Cross-session context linking.

Provides session resume context and cross-session semantic retrieval,
enabling intelligent session resumption with ledger, execution state,
and compaction history restoration.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING, Union

if TYPE_CHECKING:
    from victor.agent.sqlite_session_persistence import SQLiteSessionPersistence
    from victor.agent.conversation.store import ConversationStore
    from victor.storage.embeddings.service import EmbeddingService

logger = logging.getLogger(__name__)


@dataclass
class SessionResumeContext:
    """Context assembled when resuming a session."""

    ledger: Optional[Any] = None
    compaction_summaries: List[str] = field(default_factory=list)
    execution_state: Optional[Any] = None
    resume_summary: str = ""


class SessionContextLinker:
    """Links session state across save/load boundaries.

    Builds rich resume context from persisted session data including
    ledger, execution state, and compaction history. Optionally supports
    cross-session semantic search via embedding service.

    Supports both SQLiteSessionPersistence (deprecated) and ConversationStore (canonical).
    """

    def __init__(
        self,
        session_persistence: Optional[
            Union["SQLiteSessionPersistence", "ConversationStore"]
        ] = None,
        conversation_store: Optional["ConversationStore"] = None,
        embedding_service: Optional["EmbeddingService"] = None,
    ):
        self._conversation_store: Optional["ConversationStore"] = None

        # Normalize to single persistence parameter
        # If session_persistence is ConversationStore, use it directly
        # If session_persistence is SQLiteSessionPersistence, use it with deprecation warning
        # If conversation_store is provided, use it (new canonical path)

        if conversation_store is not None:
            # New canonical path
            self._persistence = conversation_store
            self._persistence_type = "ConversationStore"
            self._conversation_store = conversation_store
        elif session_persistence is not None:
            # Check type
            if isinstance(session_persistence, str):
                # Type check failed - might be a string representation
                self._persistence = session_persistence
                self._persistence_type = "unknown"
            elif hasattr(session_persistence, "__class__"):
                class_name = session_persistence.__class__.__name__
                if class_name == "ConversationStore":
                    self._persistence = session_persistence
                    self._persistence_type = "ConversationStore"
                    self._conversation_store = session_persistence
                elif class_name == "SQLiteSessionPersistence":
                    self._persistence = session_persistence
                    self._persistence_type = "SQLiteSessionPersistence"
                    warnings.warn(
                        "SQLiteSessionPersistence is deprecated and will be removed in v0.10.0. "
                        "Use ConversationStore instead. "
                        "Pass ConversationStore as conversation_store parameter.",
                        DeprecationWarning,
                        stacklevel=2,
                    )
                else:
                    self._persistence = session_persistence
                    self._persistence_type = class_name
            else:
                self._persistence = session_persistence
                self._persistence_type = "unknown"
        else:
            raise ValueError("Either session_persistence or conversation_store must be provided")

        self._embedding_service = embedding_service

    def build_resume_context(self, session_id: str) -> SessionResumeContext:
        """Load session and build rich resume context.

        Args:
            session_id: Session to resume

        Returns:
            SessionResumeContext with restored state and summary
        """
        try:
            # Handle both persistence types
            if self._persistence_type == "ConversationStore":
                # Use ConversationStore.load_session()
                session_data = self._persistence.load_session(session_id)
            else:
                # Use SQLiteSessionPersistence.load_session()
                session_data = self._persistence.load_session(session_id)
        except Exception as e:
            logger.warning(f"Failed to load session {session_id}: {e}")
            return SessionResumeContext(resume_summary="[Session data unavailable]")

        if session_data is None:
            return SessionResumeContext(resume_summary="[Session not found]")

        # Restore ledger
        ledger = None
        ledger_data = session_data.get("session_ledger")
        if ledger_data:
            try:
                from victor.agent.session_ledger import SessionLedger

                ledger = SessionLedger.from_dict(ledger_data)
            except Exception as e:
                logger.warning(f"Failed to restore ledger: {e}")

        # Restore execution state
        execution_state = None
        exec_data = session_data.get("execution_state")
        if exec_data:
            try:
                from victor.agent.session_state_manager import ExecutionState

                execution_state = ExecutionState.from_dict(exec_data)
            except Exception as e:
                logger.warning(f"Failed to restore execution state: {e}")

        # Restore compaction context from persisted hierarchy
        compaction_summaries: List[str] = []
        hierarchy_data = session_data.get("compaction_hierarchy")
        if hierarchy_data:
            try:
                from victor.agent.compaction_hierarchy import (
                    HierarchicalCompactionManager,
                )

                hm = HierarchicalCompactionManager.from_dict(hierarchy_data)
                active_ctx = hm.get_active_context()
                if active_ctx:
                    compaction_summaries = [active_ctx]
            except Exception as e:
                logger.warning(f"Failed to restore compaction summaries: {e}")

        # Build resume summary
        summary_parts = ["[Resumed session"]

        metadata = session_data.get("metadata", {})
        if metadata.get("title"):
            summary_parts.append(f"\"{metadata['title']}\"")
        if metadata.get("updated_at"):
            summary_parts.append(f"last active: {metadata['updated_at']}")

        if ledger:
            files = ledger.get_files_read()
            if files:
                file_list = ", ".join(sorted(files.keys())[:5])
                summary_parts.append(f"read: {file_list}")

            decisions = [e.summary for e in ledger.entries if e.category == "decision"]
            if decisions:
                summary_parts.append(f"decided: {'; '.join(decisions[:3])}")

            pending = [
                e.summary
                for e in ledger.entries
                if e.category == "pending_action" and not e.resolved
            ]
            if pending:
                summary_parts.append(f"pending: {'; '.join(pending[:3])}")

        if execution_state:
            tool_calls = getattr(execution_state, "tool_calls_used", 0)
            if tool_calls:
                summary_parts.append(f"{tool_calls} tool calls used")

        preview_messages = session_data.get("conversation", {}).get("preview_messages", [])
        if isinstance(preview_messages, list) and preview_messages:
            preview_paths: List[str] = []
            seen_paths: set[str] = set()
            for preview in preview_messages:
                if not isinstance(preview, dict):
                    continue
                metadata = preview.get("metadata", {})
                if not isinstance(metadata, dict):
                    continue
                preview_path = metadata.get("preview_path")
                if not isinstance(preview_path, str) or not preview_path:
                    continue
                if preview_path in seen_paths:
                    continue
                seen_paths.add(preview_path)
                preview_paths.append(preview_path)

            if preview_paths:
                listed_paths = ", ".join(preview_paths[:3])
                extra_count = max(0, len(preview_paths) - 3)
                if extra_count:
                    listed_paths = f"{listed_paths} (+{extra_count} more)"
                summary_parts.append(f"previews: {listed_paths}")
            else:
                summary_parts.append(f"{len(preview_messages)} previews available")

        resume_summary = ". ".join(summary_parts) + ".]"

        return SessionResumeContext(
            ledger=ledger,
            compaction_summaries=compaction_summaries,
            execution_state=execution_state,
            resume_summary=resume_summary,
        )

    def find_related_sessions(self, query: str, limit: int = 3) -> List[Dict[str, Any]]:
        """Semantic search across sessions for cross-session linking.

        Args:
            query: Search query
            limit: Maximum results

        Returns:
            List of session metadata dicts
        """
        if not self._conversation_store:
            return []

        try:
            search_fn = getattr(self._conversation_store, "search", None)
            if search_fn:
                results = search_fn(query, limit=limit)
                return results if isinstance(results, list) else []
        except Exception as e:
            logger.debug(f"Cross-session search failed: {e}")

        return []

    def build_cross_session_context(
        self,
        query: str,
        session_ids: List[str],
        max_chars: int = 4000,
    ) -> str:
        """Retrieve relevant context from related sessions.

        Args:
            query: Current query for relevance
            session_ids: Sessions to search
            max_chars: Maximum context size

        Returns:
            Combined relevant context string
        """
        if not self._conversation_store:
            return ""

        parts = []
        chars_used = 0

        for sid in session_ids:
            try:
                session_data = self._persistence.load_session(sid)
                if not session_data:
                    continue

                metadata = session_data.get("metadata", {})
                title = metadata.get("title", "Untitled")
                entry = f"[From session '{title}' ({sid[:8]})]"

                # Get ledger summary if available
                ledger_data = session_data.get("session_ledger")
                if ledger_data:
                    from victor.agent.session_ledger import SessionLedger

                    ledger = SessionLedger.from_dict(ledger_data)
                    rendered = ledger.render(max_chars=500)
                    if rendered:
                        entry += f" {rendered}"

                if chars_used + len(entry) > max_chars:
                    break

                parts.append(entry)
                chars_used += len(entry)

            except Exception as e:
                logger.debug(f"Failed to load cross-session context from {sid}: {e}")

        return " | ".join(parts) if parts else ""
