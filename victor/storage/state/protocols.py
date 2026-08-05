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

"""State machine protocols for dependency inversion.

These protocols define the contracts that state machine implementations
must follow, enabling loose coupling and testability.

Design Pattern: Protocol-based dependency inversion
- Components depend on protocols, not concrete implementations
- Enables testing with mock implementations
- Supports multiple state machine backends
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Protocol, Set, runtime_checkable


@runtime_checkable
class StageProtocol(Protocol):
    """Protocol for stage definitions.

    Stages define the phases a state machine can be in,
    along with associated tools and transition rules.
    """

    @property
    def name(self) -> str:
        """Stage identifier."""
        ...

    @property
    def description(self) -> str:
        """Human-readable description."""
        ...

    @property
    def tools(self) -> Set[str]:
        """Tools associated with this stage."""
        ...

    @property
    def next_stages(self) -> Set[str]:
        """Valid stages to transition to."""
        ...


@runtime_checkable
class StateProtocol(Protocol):
    """Protocol for state machine implementations.

    Any class implementing this protocol can be used as a state machine
    in Victor's orchestration layer.
    """

    def get_stage(self) -> str:
        """Get the current stage name.

        Returns:
            Current stage identifier.
        """
        ...

    def transition_to(self, stage: str, confidence: float = 1.0) -> bool:
        """Attempt to transition to a new stage.

        Args:
            stage: Target stage name.
            confidence: Confidence in this transition (0.0 to 1.0).

        Returns:
            True if transition succeeded, False otherwise.
        """
        ...

    def get_stage_tools(self) -> Set[str]:
        """Get tools for the current stage.

        Returns:
            Set of tool names relevant to current stage.
        """
        ...

    def reset(self) -> None:
        """Reset to initial state."""
        ...


@runtime_checkable
class TransitionValidatorProtocol(Protocol):
    """Protocol for transition validation strategies.

    Validators determine whether a state transition should be allowed.
    Multiple validators can be composed for complex validation logic.
    """

    def validate(
        self,
        current_stage: str,
        target_stage: str,
        context: Dict[str, Any],
    ) -> tuple[bool, Optional[str]]:
        """Validate a transition.

        Args:
            current_stage: Current stage name.
            target_stage: Proposed target stage.
            context: Additional context for validation.

        Returns:
            Tuple of (is_valid, error_message).
            error_message is None if valid.
        """
        ...


@runtime_checkable
class StateObserverProtocol(Protocol):
    """Protocol for state change observers.

    Observers are notified of state transitions and can react
    to state changes without modifying the state machine.
    """

    def on_transition(
        self,
        old_stage: str,
        new_stage: str,
        context: Dict[str, Any],
    ) -> None:
        """Called when a transition occurs.

        Args:
            old_stage: Stage being exited.
            new_stage: Stage being entered.
            context: Transition context.
        """
        ...


@runtime_checkable
class StageDetectorProtocol(Protocol):
    """Protocol for automatic stage detection.

    Detectors analyze context to suggest the appropriate stage
    based on tool usage, message content, etc.
    """

    def detect_stage(
        self,
        context: Dict[str, Any],
    ) -> Optional[tuple[str, float]]:
        """Detect the appropriate stage from context.

        Args:
            context: Context including tool history, message content, etc.

        Returns:
            Tuple of (stage_name, confidence) or None if no detection.
        """
        ...
