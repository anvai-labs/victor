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

"""State inspection and diffing for workflow debugging.

This module provides state snapshot, diffing, and query capabilities
for debugging StateGraph workflows.

Key Classes:
    StateSnapshot: Snapshot of workflow state at a point in time
    StateDiff: Difference between two state snapshots
    StateInspector: Inspects workflow state for debugging (SRP)

Example:
    from victor.framework.debugging.inspector import StateInspector

    inspector = StateInspector()

    # Capture snapshot
    snapshot = inspector.capture_snapshot(state, "analyze")

    # Compare states
    diff = inspector.compare_states(before_state, after_state)

    # Query state
    value = inspector.get_value(state, "user.name")
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set


@dataclass
class StateSnapshot:
    """Snapshot of workflow state at a point in time.

    Captures state for inspection and diffing.

    Attributes:
        timestamp: When snapshot was taken
        node_id: Current node ID
        state: Full state dictionary
        state_summary: Summary with keys and types
        size_bytes: Approximate size in memory
        metadata: Additional metadata

    Example:
        snapshot = StateSnapshot.capture(
            state={"task": "test", "errors": 0},
            node_id="analyze"
        )
    """

    timestamp: float
    node_id: str
    state: Dict[str, Any]
    state_summary: Dict[str, str]  # key -> type
    size_bytes: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def capture(cls, state: Dict[str, Any], node_id: str) -> "StateSnapshot":
        """Capture a state snapshot.

        Args:
            state: Current workflow state
            node_id: Current node ID

        Returns:
            StateSnapshot instance
        """
        # Calculate approximate size
        size_bytes = sum(
            len(k) + len(str(v)) if not isinstance(v, (dict, list)) else 0 for k, v in state.items()
        )

        # Create summary
        summary = {k: type(v).__name__ for k, v in state.items()}

        return cls(
            timestamp=time.time(),
            node_id=node_id,
            state=state.copy(),
            state_summary=summary,
            size_bytes=size_bytes,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation of snapshot
        """
        return {
            "timestamp": self.timestamp,
            "node_id": self.node_id,
            "state_summary": self.state_summary,
            "size_bytes": self.size_bytes,
            "metadata": self.metadata,
        }


@dataclass
class StateDiff:
    """Difference between two state snapshots.

    Attributes:
        before_keys: Keys present in before state
        after_keys: Keys present in after state
        added_keys: Keys added in after
        removed_keys: Keys removed in after
        changed_keys: Keys with changed values
        unchanged_keys: Unchanged keys

    Example:
        diff = StateDiff.compare(
            before={"task": "test", "errors": 0},
            after={"task": "test", "errors": 5, "new_key": "value"}
        )

        print(diff.added_keys)  # {"new_key"}
        print(diff.changed_keys)  # {"errors": (0, 5)}
    """

    before_keys: Set[str]
    after_keys: Set[str]
    added_keys: Set[str]
    removed_keys: Set[str]
    changed_keys: Dict[str, tuple[Any, Any]]  # key -> (old, new)
    unchanged_keys: Set[str]

    @classmethod
    def compare(cls, before: Dict[str, Any], after: Dict[str, Any]) -> "StateDiff":
        """Compare two state dictionaries.

        Args:
            before: State before
            after: State after

        Returns:
            StateDiff with differences
        """
        before_keys = set(before.keys())
        after_keys = set(after.keys())

        added_keys = after_keys - before_keys
        removed_keys = before_keys - after_keys
        common_keys = before_keys & after_keys

        changed_keys = {}
        unchanged_keys = set()

        for key in common_keys:
            if before[key] != after[key]:
                changed_keys[key] = (before[key], after[key])
            else:
                unchanged_keys.add(key)

        return cls(
            before_keys=before_keys,
            after_keys=after_keys,
            added_keys=added_keys,
            removed_keys=removed_keys,
            changed_keys=changed_keys,
            unchanged_keys=unchanged_keys,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation of diff
        """
        return {
            "added_keys": list(self.added_keys),
            "removed_keys": list(self.removed_keys),
            "changed_keys": [
                {"key": k, "old": v[0], "new": v[1]} for k, v in self.changed_keys.items()
            ],
            "unchanged_count": len(self.unchanged_keys),
        }

    def has_changes(self) -> bool:
        """Check if there are any changes.

        Returns:
            True if there are changes
        """
        return bool(self.added_keys or self.removed_keys or self.changed_keys)


class StateInspector:
    """Inspects workflow state for debugging (SRP).

    Provides state snapshot, diffing, and query capabilities.

    Attributes:
        _snapshots: History of state snapshots
        _max_snapshots: Maximum snapshots to keep

    Example:
        inspector = StateInspector()

        # Capture snapshot
        snapshot = inspector.capture_snapshot(state, "analyze")

        # Compare states
        diff = inspector.compare_states(before_state, after_state)

        # Query state
        value = inspector.get_value(state, "user.name")
    """

    def __init__(self, max_snapshots: int = 100) -> None:
        """Initialize state inspector.

        Args:
            max_snapshots: Maximum snapshots to keep in memory
        """
        self._snapshots: List[StateSnapshot] = []
        self._max_snapshots = max_snapshots

    def capture_snapshot(self, state: Dict[str, Any], node_id: str) -> StateSnapshot:
        """Capture a state snapshot.

        Args:
            state: Current workflow state
            node_id: Current node ID

        Returns:
            StateSnapshot instance
        """
        snapshot = StateSnapshot.capture(state, node_id)

        self._snapshots.append(snapshot)

        # Limit history
        if len(self._snapshots) > self._max_snapshots:
            self._snapshots.pop(0)

        return snapshot

    def compare_states(self, before: Dict[str, Any], after: Dict[str, Any]) -> StateDiff:
        """Compare two state dictionaries.

        Args:
            before: State before
            after: State after

        Returns:
            StateDiff with differences
        """
        return StateDiff.compare(before, after)

    def get_value(self, state: Dict[str, Any], key_path: str, default: Any = None) -> Any:
        """Get a value from state by key path.

        Supports nested key paths with dot notation:
            "user.profile.name" -> state["user"]["profile"]["name"]

        Args:
            state: State dictionary
            key_path: Dot-separated key path
            default: Default value if key not found

        Returns:
            Value at key path or default
        """
        keys = key_path.split(".")
        value = state

        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
                if value is None:
                    return default
            else:
                return default

        return value

    def get_state_summary(self, state: Dict[str, Any]) -> Dict[str, str]:
        """Get summary of state keys and types.

        Args:
            state: State dictionary

        Returns:
            Dictionary mapping keys to type names
        """
        return {k: type(v).__name__ for k, v in state.items()}

    def get_snapshots(self, limit: Optional[int] = None) -> List[StateSnapshot]:
        """Get state snapshots.

        Args:
            limit: Optional limit on number of snapshots

        Returns:
            List of snapshots
        """
        if limit:
            return self._snapshots[-limit:]
        return self._snapshots.copy()

    def get_snapshot_history(self, session_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get snapshot history as dictionaries.

        Args:
            session_id: Optional session ID filter (not used currently)

        Returns:
            List of snapshot dictionaries
        """
        return [s.to_dict() for s in self._snapshots]

    def clear_snapshots(self) -> None:
        """Clear all snapshots."""
        self._snapshots.clear()

    def get_large_state_keys(self, state: Dict[str, Any], threshold_bytes: int = 1024) -> List[str]:
        """Find state keys with large values.

        Args:
            state: State dictionary
            threshold_bytes: Size threshold in bytes

        Returns:
            List of keys exceeding threshold
        """
        large_keys = []

        for key, value in state.items():
            size = len(str(value)) if not isinstance(value, (dict, list)) else 0
            if size > threshold_bytes:
                large_keys.append(key)

        return large_keys
