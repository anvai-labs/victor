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

"""State merging strategies for hybrid orchestration.

This module provides strategies for merging state from multi-agent teams
back into workflow graphs. It handles conflicts when both the team and
graph modify the same keys.

Design Principles (SOLID):
    - Single Responsibility: Each strategy handles one merging approach
    - Open/Closed: Extensible via custom merge functions
    - Liskov Substitution: All strategies implement MergeStrategy protocol
    - Interface Segregation: Small, focused protocols
    - Dependency Inversion: Depend on abstractions (protocols) not concretions

Key Concepts:
    - MergeStrategy: Protocol defining merge behavior
    - StateMergeError: Raised when merge fails validation
    - Merge modes: TEAM_WINS, GRAPH_WINS, MERGE, ERROR

Example:
    from victor.framework.state_merging import (
        dict_merge_strategy,
        list_merge_strategy,
        custom_merge_strategy,
        StateMergeError,
    )

    # Default dict merge (team wins on conflicts)
    merged = dict_merge_strategy(graph_state, team_state)

    # Custom merge with conflict handler
    def resolve_conflicts(key, graph_val, team_val):
        if key == "tool_calls":
            return graph_val + team_val  # Concatenate
        return team_val  # Team wins by default

    merged = custom_merge_strategy(
        graph_state,
        team_state,
        conflict_resolver=resolve_conflicts
    )
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    runtime_checkable,
)

logger = logging.getLogger(__name__)


class MergeMode(str, Enum):
    """Mode for handling state merge conflicts.

    Attributes:
        TEAM_WINS: Team state overrides graph state on conflicts
        GRAPH_WINS: Graph state is preserved on conflicts
        MERGE: Attempt to merge values (if types are compatible)
        ERROR: Raise StateMergeError on conflicts
    """

    TEAM_WINS = "team_wins"
    GRAPH_WINS = "graph_wins"
    MERGE = "merge"
    ERROR = "error"


@dataclass
class StateMergeError(Exception):
    """Raised when state merge fails.

    Attributes:
        message: Error message
        key: Key that caused the conflict
        graph_value: Value from graph state
        team_value: Value from team state
    """

    message: str
    key: Optional[str] = None
    graph_value: Optional[Any] = None
    team_value: Optional[Any] = None

    def __str__(self) -> str:
        """Format error message."""
        if self.key:
            return (
                f"{self.message} (key={self.key}, "
                f"graph_value={self.graph_value}, team_value={self.team_value})"
            )
        return self.message


@runtime_checkable
class MergeStrategy(Protocol):
    """Protocol for state merge strategies.

    All merge strategies must implement this protocol.
    """

    def merge(
        self,
        graph_state: Dict[str, Any],
        team_state: Dict[str, Any],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Merge team state into graph state.

        Args:
            graph_state: Current workflow graph state
            team_state: State produced by team execution
            **kwargs: Additional strategy-specific parameters

        Returns:
            Merged state dictionary

        Raises:
            StateMergeError: If merge fails and mode is ERROR
        """
        ...


class BaseMergeStrategy(ABC):
    """Base class for merge strategies (SRP: Single Responsibility)."""

    def __init__(self, mode: MergeMode = MergeMode.TEAM_WINS):
        """Initialize merge strategy.

        Args:
            mode: How to handle conflicts (default: TEAM_WINS)
        """
        self.mode = mode

    @abstractmethod
    def merge(
        self,
        graph_state: Dict[str, Any],
        team_state: Dict[str, Any],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Merge team state into graph state.

        Args:
            graph_state: Current workflow graph state
            team_state: State produced by team execution
            **kwargs: Additional strategy-specific parameters

        Returns:
            Merged state dictionary

        Raises:
            StateMergeError: If merge fails and mode is ERROR
        """
        ...

    def _handle_conflict(
        self,
        key: str,
        graph_value: Any,
        team_value: Any,
    ) -> Any:
        """Handle merge conflict based on mode.

        Args:
            key: Key that conflicts
            graph_value: Value from graph state
            team_value: Value from team state

        Returns:
            Resolved value

        Raises:
            StateMergeError: If mode is ERROR
        """
        if self.mode == MergeMode.TEAM_WINS:
            return team_value
        elif self.mode == MergeMode.GRAPH_WINS:
            return graph_value
        elif self.mode == MergeMode.ERROR:
            raise StateMergeError(
                message=f"Merge conflict for key '{key}'",
                key=key,
                graph_value=graph_value,
                team_value=team_value,
            )
        elif self.mode == MergeMode.MERGE:
            # Attempt to merge compatible types
            return self._attempt_merge(key, graph_value, team_value)
        else:
            # Default to team wins
            return team_value

    def _attempt_merge(self, key: str, graph_value: Any, team_value: Any) -> Any:
        """Attempt to merge conflicting values.

        Args:
            key: Key that conflicts
            graph_value: Value from graph state
            team_value: Value from team state

        Returns:
            Merged value or team_value as fallback
        """
        # Merge dicts recursively
        if isinstance(graph_value, dict) and isinstance(team_value, dict):
            return dict_merge_strategy.merge(graph_value, team_value, mode=self.mode)

        # Merge lists by concatenation
        if isinstance(graph_value, list) and isinstance(team_value, list):
            return graph_value + team_value

        # Merge sets by union
        if isinstance(graph_value, set) and isinstance(team_value, set):
            return graph_value | team_value

        # Can't merge, return team value
        logger.warning(
            f"Cannot merge incompatible types for key '{key}': "
            f"{type(graph_value)} vs {type(team_value)}, using team value"
        )
        return team_value


class DictMergeStrategy(BaseMergeStrategy):
    """Strategy for merging dictionary states.

    Recursively merges nested dictionaries and handles conflicts
    based on the configured merge mode.
    """

    def merge(
        self,
        graph_state: Dict[str, Any],
        team_state: Dict[str, Any],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Merge team state into graph state.

        Args:
            graph_state: Current workflow graph state
            team_state: State produced by team execution
            **kwargs: Additional parameters (mode: MergeMode)

        Returns:
            Merged state dictionary
        """
        # Create a copy to avoid mutating original
        merged = dict(graph_state)

        # Override mode if provided
        mode = kwargs.get("mode")
        if mode:
            self.mode = mode if isinstance(mode, MergeMode) else MergeMode(mode)

        for key, team_value in team_state.items():
            if key not in merged:
                # New key, add it
                merged[key] = team_value
            else:
                # Key exists in both, handle conflict
                graph_value = merged[key]

                # Both are dicts, recurse
                if isinstance(graph_value, dict) and isinstance(team_value, dict):
                    merged[key] = self.merge(graph_value, team_value, **kwargs)
                else:
                    # Handle conflict
                    merged[key] = self._handle_conflict(key, graph_value, team_value)

        return merged


class ListMergeStrategy(BaseMergeStrategy):
    """Strategy for merging list-based states.

    Merges lists by concatenation, with options for deduplication.
    """

    def merge(
        self,
        graph_state: Dict[str, Any],
        team_state: Dict[str, Any],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Merge team state into graph state.

        Args:
            graph_state: Current workflow graph state
            team_state: State produced by team execution
            **kwargs: Additional parameters:
                - mode: MergeMode (default: TEAM_WINS)
                - deduplicate: Whether to deduplicate lists (default: False)
                - list_keys: Keys to treat as lists (default: auto-detect)

        Returns:
            Merged state dictionary
        """
        merged = dict(graph_state)

        # Override mode if provided
        mode = kwargs.get("mode")
        if mode:
            self.mode = mode if isinstance(mode, MergeMode) else MergeMode(mode)

        deduplicate = kwargs.get("deduplicate", False)
        list_keys = kwargs.get("list_keys", None)

        for key, team_value in team_state.items():
            if key not in merged:
                # New key, add it
                merged[key] = team_value
            else:
                graph_value = merged[key]

                # Check if this should be treated as a list merge
                is_list_merge = (
                    (list_keys and key in list_keys)
                    or isinstance(graph_value, list)
                    and isinstance(team_value, list)
                )

                if is_list_merge:
                    # Merge lists
                    merged_list = graph_value + team_value
                    if deduplicate:
                        # Remove duplicates while preserving order
                        seen = set()
                        unique_list = []
                        for item in merged_list:
                            if item not in seen:
                                seen.add(item)
                                unique_list.append(item)
                        merged[key] = unique_list
                    else:
                        merged[key] = merged_list
                else:
                    # Handle conflict normally
                    merged[key] = self._handle_conflict(key, graph_value, team_value)

        return merged


class CustomMergeStrategy(BaseMergeStrategy):
    """Strategy for custom merge logic.

    Uses a user-provided conflict resolver function for maximum flexibility.
    """

    def __init__(
        self,
        conflict_resolver: Optional[Callable[[str, Any, Any], Any]] = None,
        mode: MergeMode = MergeMode.TEAM_WINS,
    ):
        """Initialize custom merge strategy.

        Args:
            conflict_resolver: Function to resolve conflicts
                Signature: (key: str, graph_value: Any, team_value: Any) -> Any
            mode: Fallback mode if conflict_resolver returns None
        """
        super().__init__(mode=mode)
        self.conflict_resolver = conflict_resolver

    def merge(
        self,
        graph_state: Dict[str, Any],
        team_state: Dict[str, Any],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Merge team state into graph state.

        Args:
            graph_state: Current workflow graph state
            team_state: State produced by team execution
            **kwargs: Additional parameters (not used)

        Returns:
            Merged state dictionary
        """
        merged = dict(graph_state)

        for key, team_value in team_state.items():
            if key not in merged:
                # New key, add it
                merged[key] = team_value
            else:
                graph_value = merged[key]

                # Both are dicts, check if we should recurse
                if isinstance(graph_value, dict) and isinstance(team_value, dict):
                    # If custom resolver exists, try it first
                    if self.conflict_resolver:
                        resolved = self.conflict_resolver(key, graph_value, team_value)
                        if resolved is not None:
                            merged[key] = resolved
                            continue
                        # Resolver returned None, fall through to recursion

                    # Recursively merge nested dicts
                    merged[key] = self.merge(graph_value, team_value, **kwargs)
                else:
                    # Use custom resolver if provided
                    if self.conflict_resolver:
                        resolved = self.conflict_resolver(key, graph_value, team_value)
                        if resolved is not None:
                            merged[key] = resolved
                            continue

                    # Fallback to base conflict handling
                    merged[key] = self._handle_conflict(key, graph_value, team_value)

        return merged


class SelectiveMergeStrategy(BaseMergeStrategy):
    """Strategy for selective merging of specific keys.

    Only merges specified keys from team state, ignoring others.
    """

    def __init__(
        self,
        keys_to_merge: List[str],
        mode: MergeMode = MergeMode.TEAM_WINS,
        recursive: bool = True,
    ):
        """Initialize selective merge strategy.

        Args:
            keys_to_merge: List of keys to merge from team state
            mode: How to handle conflicts
            recursive: Whether to recursively merge nested dicts
        """
        super().__init__(mode=mode)
        self.keys_to_merge = set(keys_to_merge)
        self.recursive = recursive

    def merge(
        self,
        graph_state: Dict[str, Any],
        team_state: Dict[str, Any],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Merge team state into graph state.

        Only merges keys that are in keys_to_merge. Other keys from
        team_state are ignored.

        Args:
            graph_state: Current workflow graph state
            team_state: State produced by team execution
            **kwargs: Additional parameters (mode: MergeMode)

        Returns:
            Merged state dictionary
        """
        merged = dict(graph_state)

        # Override mode if provided
        mode = kwargs.get("mode")
        if mode:
            self.mode = mode if isinstance(mode, MergeMode) else MergeMode(mode)

        for key in self.keys_to_merge:
            if key not in team_state:
                continue

            team_value = team_state[key]

            if key not in merged:
                # New key, add it
                merged[key] = team_value
            else:
                graph_value = merged[key]

                # Both are dicts and recursive is enabled
                if (
                    self.recursive
                    and isinstance(graph_value, dict)
                    and isinstance(team_value, dict)
                ):
                    # Recursively merge only selected keys
                    nested_strategy = SelectiveMergeStrategy(
                        keys_to_merge=list(team_value.keys()),
                        mode=self.mode,
                        recursive=True,
                    )
                    merged[key] = nested_strategy.merge(graph_value, team_value, **kwargs)
                else:
                    # Handle conflict
                    merged[key] = self._handle_conflict(key, graph_value, team_value)

        return merged


def validate_merged_state(
    merged_state: Dict[str, Any],
    required_keys: Optional[List[str]] = None,
    forbidden_keys: Optional[List[str]] = None,
    validators: Optional[Dict[str, Callable[[Any], bool]]] = None,
) -> bool:
    """Validate merged state before continuing execution.

    Args:
        merged_state: State to validate
        required_keys: Keys that must be present
        forbidden_keys: Keys that must NOT be present
        validators: Custom validators per key (key -> validator function)

    Returns:
        True if valid

    Raises:
        StateMergeError: If validation fails
    """
    errors = []

    # Check required keys
    if required_keys:
        missing = [key for key in required_keys if key not in merged_state]
        if missing:
            errors.append(f"Missing required keys: {missing}")

    # Check forbidden keys
    if forbidden_keys:
        found = [key for key in forbidden_keys if key in merged_state]
        if found:
            errors.append(f"Forbidden keys present: {found}")

    # Run custom validators
    if validators:
        for key, validator in validators.items():
            if key in merged_state:
                try:
                    if not validator(merged_state[key]):
                        errors.append(f"Validation failed for key '{key}'")
                except Exception as e:
                    errors.append(f"Validator error for key '{key}': {e}")

    if errors:
        raise StateMergeError(message=f"State validation failed: {'; '.join(errors)}")

    return True


# =============================================================================
# Default Strategy Instances
# =============================================================================

# Default dict merge strategy (team wins on conflicts)
dict_merge_strategy = DictMergeStrategy(mode=MergeMode.TEAM_WINS)

# Default list merge strategy (concatenates lists)
list_merge_strategy = ListMergeStrategy(mode=MergeMode.TEAM_WINS)

# Default custom merge (no resolver, falls back to team wins)
custom_merge_strategy = CustomMergeStrategy(mode=MergeMode.TEAM_WINS)


def create_merge_strategy(
    strategy_type: str = "dict",
    mode: MergeMode = MergeMode.TEAM_WINS,
    **kwargs: Any,
) -> MergeStrategy:
    """Factory function for creating merge strategies.

    Args:
        strategy_type: Type of strategy ("dict", "list", "custom", "selective")
        mode: How to handle conflicts
        **kwargs: Strategy-specific parameters:
            - conflict_resolver: For "custom" type
            - keys_to_merge: For "selective" type
            - deduplicate: For "list" type
            - list_keys: For "list" type

    Returns:
        Configured merge strategy instance

    Raises:
        ValueError: If strategy_type is unknown
    """
    if strategy_type == "dict":
        return DictMergeStrategy(mode=mode)
    elif strategy_type == "list":
        return ListMergeStrategy(mode=mode, **kwargs)
    elif strategy_type == "custom":
        return CustomMergeStrategy(
            conflict_resolver=kwargs.get("conflict_resolver"),
            mode=mode,
        )
    elif strategy_type == "selective":
        return SelectiveMergeStrategy(
            keys_to_merge=kwargs.get("keys_to_merge", []),
            mode=mode,
            recursive=kwargs.get("recursive", True),
        )
    else:
        raise ValueError(
            f"Unknown merge strategy type: {strategy_type}. "
            f"Valid options: dict, list, custom, selective"
        )


__all__ = [
    # Protocols
    "MergeStrategy",
    # Exceptions
    "StateMergeError",
    # Enums
    "MergeMode",
    # Strategies
    "BaseMergeStrategy",
    "DictMergeStrategy",
    "ListMergeStrategy",
    "CustomMergeStrategy",
    "SelectiveMergeStrategy",
    # Functions
    "validate_merged_state",
    "create_merge_strategy",
    # Default instances
    "dict_merge_strategy",
    "list_merge_strategy",
    "custom_merge_strategy",
]
