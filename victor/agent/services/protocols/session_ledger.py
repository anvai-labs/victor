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

"""Session ledger protocol for structured session state tracking."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


@runtime_checkable
class SessionLedgerProtocol(Protocol):
    """Protocol for structured session state tracking.

    The session ledger maintains an append-only log of high-signal events
    (file reads, modifications, decisions, recommendations, pending actions)
    that survives compaction and is rendered into context each turn.
    """

    def record_file_read(self, path: str, summary: str, turn_index: int) -> None:
        """Record a file read event."""
        ...

    def record_file_modified(self, path: str, change_summary: str, turn_index: int) -> None:
        """Record a file modification event."""
        ...

    def record_decision(self, decision: str, turn_index: int) -> None:
        """Record a decision made during the session."""
        ...

    def record_recommendation(self, recommendation: str, turn_index: int) -> None:
        """Record a recommendation made during the session."""
        ...

    def record_pending_action(self, action: str, turn_index: int) -> None:
        """Record a pending action to be completed."""
        ...

    def resolve_pending_action(self, action_key: str) -> None:
        """Mark a pending action as resolved."""
        ...

    def update_from_tool_result(
        self, tool_name: str, args: Dict[str, Any], result: str, turn_index: int
    ) -> None:
        """Update ledger from a tool execution result."""
        ...

    def update_from_assistant_response(self, content: str, turn_index: int) -> None:
        """Update ledger from an assistant response, extracting decisions/recommendations."""
        ...

    def render(self, max_chars: Optional[int] = None) -> str:
        """Render the ledger as a context string for injection into the conversation."""
        ...

    def get_recent_actionable_items(self, limit: int = 5) -> List[Any]:
        """Get recent actionable items (decisions, recommendations, pending actions)."""
        ...

    def get_files_read(self) -> Dict[str, str]:
        """Get mapping of file paths to their summaries."""
        ...

    def to_dict(self) -> Dict[str, Any]:
        """Serialize ledger state."""
        ...

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionLedgerProtocol":
        """Deserialize ledger state."""
        ...
