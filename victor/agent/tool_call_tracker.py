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

"""Tool call tracker to prevent redundant operations.

This module prevents redundant tool calls by tracking recent operations
and detecting when tools are called with overlapping intent.

This is DISTINCT from victor.tools.deduplication/, which handles tool
DEFINITION deduplication (resolving naming conflicts between native,
LangChain, and MCP tools). This module tracks RUNTIME tool CALLS.

Examples of redundancy:
- grep(mode='semantic', query='tool registration') followed by grep(mode='regex', pattern='register_tool')
- read_file(path='foo.py') followed by read_file(path='foo.py')
- list_directory(path='.') followed by list_directory(path='.')

Key distinction:
- victor/agent/tool_call_tracker.py → Tracks runtime tool CALLS (prevents redundant operations)
- victor/tools/deduplication/ → Deduplicates tool DEFINITIONS (resolves naming conflicts)
"""

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Set

from victor.agent.file_state import FileStateSnapshot, capture_file_state
from victor.tools.tool_names import get_canonical_name

logger = logging.getLogger(__name__)


@dataclass
class TrackedToolCall:
    """Record of a tool call for redundancy tracking.

    Different from victor.agent.tool_calling.base.ToolCall - this is specifically
    for tracking recent calls to detect redundant operations.
    """

    tool_name: str
    args: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    file_state: Optional[FileStateSnapshot] = None


class ToolCallTracker:
    """Tracks recent tool calls to detect and prevent redundant operations.

    Uses semantic similarity and argument overlap to identify redundant calls.
    """

    def __init__(
        self,
        window_size: int = 10,
        similarity_threshold: float = 0.7,
        read_redundancy_ttl_seconds: float = 300.0,
    ):
        """Initialize tool call tracker.

        Args:
            window_size: Number of recent tool calls to track
            similarity_threshold: Threshold for considering calls similar (0.0-1.0)
            read_redundancy_ttl_seconds: Only dedup unchanged rereads within this TTL window
        """
        from victor.core.utils.content_hasher import ContentHasher

        self.window_size = window_size
        self.similarity_threshold = similarity_threshold
        self.read_redundancy_ttl_seconds = read_redundancy_ttl_seconds
        self.recent_calls: Deque[TrackedToolCall] = deque(maxlen=window_size)
        # Use ContentHasher for consistent hashing across components
        # Tool calls need exact matching (no normalization) for precise deduplication
        self._hasher = ContentHasher(
            normalize_whitespace=False,
            case_insensitive=False,
            hash_length=16,
        )

        # Query synonyms for semantic matching
        self.query_synonyms: Dict[str, Set[str]] = {
            "tool registration": {
                "register tool",
                "tool registry",
                "@tool",
                "register_tool",
                "ToolRegistry",
            },
            "provider": {"llm provider", "model provider", "baseprovider"},
            "error handling": {
                "exception",
                "try catch",
                "try except",
                "error recovery",
            },
        }

    def add_call(self, tool_name: str, args: Dict[str, Any]) -> None:
        """Add a tool call to the tracker.

        Args:
            tool_name: Name of the tool
            args: Tool arguments
        """
        canonical_tool_name = get_canonical_name(tool_name)
        file_state = None
        if canonical_tool_name == "read":
            file_state = capture_file_state(
                args.get("path") or args.get("file_path") or args.get("file")
            )
        call = TrackedToolCall(tool_name=canonical_tool_name, args=args, file_state=file_state)
        self.recent_calls.append(call)
        logger.debug(f"Tracking tool call: {canonical_tool_name}({self._format_args(args)})")

    def is_redundant(self, tool_name: str, args: Dict[str, Any], explain: bool = False) -> bool:
        """Check if a tool call is redundant given recent history.

        Args:
            tool_name: Name of the tool to call
            args: Tool arguments
            explain: If True, log explanation for why call is redundant

        Returns:
            True if the call is likely redundant, False otherwise
        """
        canonical_tool_name = get_canonical_name(tool_name)

        if not self.recent_calls:
            return False

        # Check for exact duplicates
        for recent in self.recent_calls:
            if recent.tool_name == canonical_tool_name and recent.args == args:
                if canonical_tool_name == "read":
                    return self._is_recent_unchanged_read(recent, args)
                if explain:
                    logger.info(
                        f"Redundant tool call detected: {canonical_tool_name}({self._format_args(args)}) "
                        f"was called {(time.time() - recent.timestamp):.1f}s ago"
                    )
                return True

        # Check for semantic overlap
        if self._has_semantic_overlap(canonical_tool_name, args):
            if explain:
                logger.info(
                    f"Semantically redundant tool call: {canonical_tool_name}({self._format_args(args)}) "
                    "overlaps with recent call"
                )
            return True

        return False

    def _has_semantic_overlap(self, tool_name: str, args: Dict[str, Any]) -> bool:
        """Check if tool call semantically overlaps with recent calls.

        Args:
            tool_name: Tool name
            args: Tool arguments

        Returns:
            True if there's semantic overlap with recent calls
        """
        tool_name = get_canonical_name(tool_name)
        # Check for grep/search redundancy
        if tool_name in ("grep", "code_search", "semantic_code_search"):
            return self._check_search_redundancy(tool_name, args)

        # Check for file operation redundancy
        if tool_name in ("read", "edit", "write"):
            return self._check_file_redundancy(tool_name, args)

        # Check for list/ls redundancy
        if tool_name in ("ls", "tree"):
            return self._check_list_redundancy(tool_name, args)

        return False

    def _check_search_redundancy(self, tool_name: str, args: Dict[str, Any]) -> bool:
        """Check if search/grep call is redundant.

        Blocks exact duplicate queries only.
        Different queries (even with overlapping words) are allowed —
        the agent often needs multiple targeted searches to explore a codebase.
        """
        query = args.get("query") or args.get("pattern")
        if not query:
            return False

        query_lower = str(query).lower().strip()
        # Also consider mode — same query with different mode is not redundant
        mode = args.get("mode", "")

        # Check recent grep/search calls
        for recent in self.recent_calls:
            if recent.tool_name not in ("grep", "code_search", "semantic_code_search"):
                continue

            recent_query = recent.args.get("query") or recent.args.get("pattern")
            if not recent_query:
                continue

            recent_query_lower = str(recent_query).lower().strip()
            recent_mode = recent.args.get("mode", "")

            # Block exact same query with same mode (but NOT synonyms - allow related searches)
            if query_lower == recent_query_lower and mode == recent_mode:
                return True

        return False

    def _check_file_redundancy(self, tool_name: str, args: Dict[str, Any]) -> bool:
        """Check if file operation is redundant.

        Detects patterns like:
        - Reading same file twice
        - Writing to file just read
        """
        path = args.get("path") or args.get("file_path") or args.get("file")
        if not path:
            return False

        # Check recent file operations
        for recent in self.recent_calls:
            if recent.tool_name not in ("read", "edit", "write"):
                continue

            recent_path = (
                recent.args.get("path") or recent.args.get("file_path") or recent.args.get("file")
            )
            if not recent_path:
                continue

            # Same file accessed
            if path == recent_path:
                # Read after read is redundant (unless large file with different offsets)
                if tool_name == "read" and recent.tool_name == "read":
                    # Allow if different offsets/limits specified
                    if args.get("offset") != recent.args.get("offset") or args.get(
                        "limit"
                    ) != recent.args.get("limit"):
                        return False
                    return self._is_recent_unchanged_read(recent, args)

        return False

    def _is_recent_unchanged_read(self, recent: TrackedToolCall, args: Dict[str, Any]) -> bool:
        """Only deduplicate rereads when the file snapshot is unchanged."""
        if (time.time() - recent.timestamp) >= self.read_redundancy_ttl_seconds:
            return False

        current_path = args.get("path") or args.get("file_path") or args.get("file")
        current_state = capture_file_state(current_path)
        if recent.file_state is None and current_state is None:
            return True
        if recent.file_state is None or current_state is None:
            return False
        return recent.file_state == current_state

    def _check_list_redundancy(self, tool_name: str, args: Dict[str, Any]) -> bool:
        """Check if list/directory operation is redundant.

        Detects patterns like:
        - Listing same directory multiple times
        """
        path = args.get("path", ".")

        # Check recent list operations
        for recent in self.recent_calls:
            if recent.tool_name not in ("ls", "tree"):
                continue

            recent_path = recent.args.get("path", ".")

            # Same directory listed
            if path == recent_path:
                return True

        return False

    def _queries_are_synonyms(self, query1: str, query2: str) -> bool:
        """Check if two queries are synonyms.

        Args:
            query1: First query (lowercase)
            query2: Second query (lowercase)

        Returns:
            True if queries are synonyms
        """
        for concept, synonyms in self.query_synonyms.items():
            # Check if both queries relate to same concept
            query1_matches = concept in query1 or any(syn in query1 for syn in synonyms)
            query2_matches = concept in query2 or any(syn in query2 for syn in synonyms)

            if query1_matches and query2_matches:
                return True

        return False

    def _format_args(self, args: Dict[str, Any]) -> str:
        """Format args for logging.

        Args:
            args: Tool arguments

        Returns:
            Formatted string representation
        """
        formatted = []
        for key, value in args.items():
            if isinstance(value, str) and len(value) > 50:
                value = value[:47] + "..."
            formatted.append(f"{key}={repr(value)}")
        return ", ".join(formatted)

    def get_recent_calls(self, limit: Optional[int] = None) -> List[TrackedToolCall]:
        """Get recent tool calls.

        Args:
            limit: Maximum number of calls to return

        Returns:
            List of recent tool calls (most recent first)
        """
        calls = list(reversed(self.recent_calls))
        if limit:
            calls = calls[:limit]
        return calls

    def clear(self) -> None:
        """Clear all tracked tool calls."""
        self.recent_calls.clear()
        logger.debug("Cleared tool call tracker")

    def invalidate_for_modified_path(self, path: str) -> int:
        """Drop potentially stale call history after a file modification.

        Repeated searches should be allowed after edits because earlier search
        results may no longer reflect the current codebase. File-specific read
        history for the modified path is also cleared.

        Args:
            path: Modified file path

        Returns:
            Number of tracked calls removed
        """

        try:
            normalized_target = str(Path(path).expanduser().resolve())
        except Exception:
            normalized_target = str(path)

        retained: Deque[TrackedToolCall] = deque(maxlen=self.window_size)
        removed = 0

        for recent in self.recent_calls:
            if recent.tool_name in {"grep", "code_search", "semantic_code_search"}:
                removed += 1
                continue

            recent_path = (
                recent.args.get("path") or recent.args.get("file_path") or recent.args.get("file")
            )
            if recent_path:
                try:
                    normalized_recent = str(Path(str(recent_path)).expanduser().resolve())
                except Exception:
                    normalized_recent = str(recent_path)
                if normalized_recent == normalized_target:
                    removed += 1
                    continue

            retained.append(recent)

        if removed:
            self.recent_calls = retained
            logger.debug(
                "Invalidated %d tracked tool call(s) after modifying %s",
                removed,
                normalized_target,
            )

        return removed

    def get_duplicate_count(self) -> int:
        """Get number of exact duplicate calls in recent history.

        Returns:
            Count of duplicate calls
        """
        seen = set()
        duplicates = 0

        for call in self.recent_calls:
            key = (call.tool_name, frozenset(call.args.items()))
            if key in seen:
                duplicates += 1
            else:
                seen.add(key)

        return duplicates


# Global singleton instance (legacy - prefer DI container)
_tool_call_tracker: Optional[ToolCallTracker] = None


def get_tool_call_tracker() -> ToolCallTracker:
    """Get or create the tool call tracker.

    Resolution order:
    1. Check DI container (preferred)
    2. Fall back to module-level singleton (legacy)

    Returns:
        ToolCallTracker instance
    """
    global _tool_call_tracker

    # Try DI container first
    try:
        from victor.core.container import get_container
        from victor.agent.protocols import ToolCallTrackerProtocol

        container = get_container()
        if container.is_registered(ToolCallTrackerProtocol):
            return container.get(ToolCallTrackerProtocol)
    except Exception:
        pass  # Fall back to legacy singleton

    # Legacy fallback
    if _tool_call_tracker is None:
        _tool_call_tracker = ToolCallTracker()
    return _tool_call_tracker


def reset_tool_call_tracker() -> None:
    """Reset the global tool call tracker (for testing).

    Note: This only resets the legacy module-level singleton. If using DI
    container, use reset_container() as well.
    """
    global _tool_call_tracker
    _tool_call_tracker = None


def is_redundant_call(tool_name: str, args: Dict[str, Any]) -> bool:
    """Convenience function to check if a tool call is redundant.

    Args:
        tool_name: Name of the tool
        args: Tool arguments

    Returns:
        True if redundant, False otherwise
    """
    tracker = get_tool_call_tracker()
    return tracker.is_redundant(tool_name, args, explain=True)


def track_call(tool_name: str, args: Dict[str, Any]) -> None:
    """Convenience function to track a tool call.

    Args:
        tool_name: Name of the tool
        args: Tool arguments
    """
    tracker = get_tool_call_tracker()
    tracker.add_call(tool_name, args)
