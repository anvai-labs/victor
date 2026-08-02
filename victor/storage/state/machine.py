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

"""Generic state machine implementation.

Provides a configurable state machine that can be used for any
state-based workflow, not just conversations.

Design Patterns:
- State Pattern: Encapsulates state-specific behavior
- Strategy Pattern: Pluggable validators and detectors
- Observer Pattern: Transition notifications

Example:
    from victor.storage.state import StateMachine, StateConfig

    config = StateConfig(
        stages=["DRAFT", "REVIEW", "APPROVED", "REJECTED"],
        initial_stage="DRAFT",
        transitions={
            "DRAFT": ["REVIEW"],
            "REVIEW": ["APPROVED", "REJECTED", "DRAFT"],
            "APPROVED": [],
            "REJECTED": ["DRAFT"],
        },
    )

    machine = StateMachine(config)
    machine.transition_to("REVIEW")
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set

from victor.storage.state.protocols import (
    StateObserverProtocol,
    StateProtocol,
    StageDetectorProtocol,
    TransitionValidatorProtocol,
)

logger = logging.getLogger(__name__)


@dataclass
class StateConfig:
    """Configuration for a state machine.

    Attributes:
        stages: List of valid stage names.
        initial_stage: Starting stage.
        transitions: Dict mapping stage to valid next stages.
        stage_tools: Optional dict mapping stage to tool names.
        cooldown_seconds: Minimum time between transitions.
        backward_threshold: Confidence required for backward transitions.
    """

    stages: List[str]
    initial_stage: str
    transitions: Dict[str, List[str]] = field(default_factory=dict)
    stage_tools: Dict[str, Set[str]] = field(default_factory=dict)
    cooldown_seconds: float = 0.0
    backward_threshold: float = 0.8

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.initial_stage not in self.stages:
            raise ValueError(f"Initial stage '{self.initial_stage}' not in stages: {self.stages}")

        for stage, targets in self.transitions.items():
            if stage not in self.stages:
                raise ValueError(f"Transition source '{stage}' not in stages")
            for target in targets:
                if target not in self.stages:
                    raise ValueError(f"Transition target '{target}' (from {stage}) not in stages")


@dataclass
class StateTransition:
    """Record of a state transition.

    Attributes:
        from_stage: Stage transitioned from.
        to_stage: Stage transitioned to.
        timestamp: When transition occurred.
        confidence: Confidence in the transition.
        context: Additional context.
    """

    from_stage: str
    to_stage: str
    timestamp: float = field(default_factory=time.time)
    confidence: float = 1.0
    context: Dict[str, Any] = field(default_factory=dict)


class StateMachine:
    """Generic configurable state machine.

    Implements StateProtocol and provides:
    - Configurable stages and transitions
    - Transition validation with cooldown
    - Observer notifications
    - History tracking

    Example:
        config = StateConfig(
            stages=["A", "B", "C"],
            initial_stage="A",
            transitions={"A": ["B"], "B": ["C"], "C": ["A"]},
        )
        machine = StateMachine(config)

        machine.add_observer(my_observer)
        machine.transition_to("B")
    """

    def __init__(
        self,
        config: StateConfig,
        validators: Optional[List[TransitionValidatorProtocol]] = None,
        detectors: Optional[List[StageDetectorProtocol]] = None,
    ) -> None:
        """Initialize the state machine.

        Args:
            config: State machine configuration.
            validators: Optional transition validators.
            detectors: Optional stage detectors.
        """
        self._config = config
        self._current_stage = config.initial_stage
        self._stage_index = {stage: i for i, stage in enumerate(config.stages)}
        self._validators = validators or []
        self._detectors = detectors or []
        self._observers: List[StateObserverProtocol] = []
        self._history: List[StateTransition] = []
        self._last_transition_time: float = 0.0
        self._transition_count: int = 0

    def get_stage(self) -> str:
        """Get the current stage name.

        Returns:
            Current stage identifier.
        """
        return self._current_stage

    def get_stage_index(self) -> int:
        """Get the index of the current stage.

        Returns:
            Index in the stages list.
        """
        return self._stage_index[self._current_stage]

    def get_stage_tools(self) -> Set[str]:
        """Get tools for the current stage.

        Returns:
            Set of tool names, or empty set if not configured.
        """
        return self._config.stage_tools.get(self._current_stage, set())

    def get_valid_transitions(self) -> List[str]:
        """Get valid next stages from current stage.

        Returns:
            List of stage names that can be transitioned to.
        """
        return self._config.transitions.get(self._current_stage, [])

    def can_transition_to(self, stage: str) -> bool:
        """Check if transition to stage is valid.

        Args:
            stage: Target stage name.

        Returns:
            True if transition is allowed.
        """
        if stage not in self._stage_index:
            return False
        return stage in self.get_valid_transitions()

    def transition_to(
        self,
        stage: str,
        confidence: float = 1.0,
        context: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Attempt to transition to a new stage.

        Args:
            stage: Target stage name.
            confidence: Confidence in this transition (0.0 to 1.0).
            context: Optional context for validators and observers.

        Returns:
            True if transition succeeded, False otherwise.
        """
        context = context or {}
        old_stage = self._current_stage

        # Check if stage exists
        if stage not in self._stage_index:
            logger.warning(f"Invalid stage: {stage}")
            return False

        # No-op if already in target stage
        if stage == old_stage:
            return True

        # Check transition validity
        if not self.can_transition_to(stage):
            logger.debug(
                f"Transition not allowed: {old_stage} -> {stage}. "
                f"Valid: {self.get_valid_transitions()}"
            )
            return False

        # Check cooldown
        if self._config.cooldown_seconds > 0:
            elapsed = time.time() - self._last_transition_time
            if elapsed < self._config.cooldown_seconds:
                logger.debug(
                    f"Transition blocked by cooldown: {elapsed:.1f}s < "
                    f"{self._config.cooldown_seconds}s"
                )
                return False

        # Check backward transition threshold
        old_index = self._stage_index[old_stage]
        new_index = self._stage_index[stage]
        if new_index < old_index and confidence < self._config.backward_threshold:
            logger.debug(
                f"Backward transition requires higher confidence: "
                f"{confidence} < {self._config.backward_threshold}"
            )
            return False

        # Run validators
        for validator in self._validators:
            is_valid, error = validator.validate(old_stage, stage, context)
            if not is_valid:
                logger.debug(f"Validator rejected transition: {error}")
                return False

        # Perform transition
        logger.info(f"State transition: {old_stage} -> {stage} (confidence: {confidence:.2f})")
        self._current_stage = stage
        self._last_transition_time = time.time()
        self._transition_count += 1

        # Record history
        transition = StateTransition(
            from_stage=old_stage,
            to_stage=stage,
            confidence=confidence,
            context=context,
        )
        self._history.append(transition)

        # Notify observers
        for observer in self._observers:
            try:
                observer.on_transition(old_stage, stage, context)
            except Exception as e:
                logger.warning(f"Observer error: {e}")

        return True

    def detect_and_transition(self, context: Dict[str, Any]) -> bool:
        """Use detectors to automatically transition.

        Args:
            context: Context for detection.

        Returns:
            True if a transition occurred.
        """
        best_stage: Optional[str] = None
        best_confidence: float = 0.0

        for detector in self._detectors:
            result = detector.detect_stage(context)
            if result:
                stage, confidence = result
                if confidence > best_confidence and self.can_transition_to(stage):
                    best_stage = stage
                    best_confidence = confidence

        if best_stage and best_confidence > 0.5:  # Minimum threshold
            return self.transition_to(best_stage, best_confidence, context)

        return False

    def reset(self) -> None:
        """Reset to initial state."""
        self._current_stage = self._config.initial_stage
        self._last_transition_time = 0.0
        self._transition_count = 0
        self._history.clear()

    def add_observer(self, observer: StateObserverProtocol) -> Callable[[], None]:
        """Add a transition observer.

        Args:
            observer: Observer to add.

        Returns:
            Function to remove the observer.
        """
        self._observers.append(observer)

        def remove() -> None:
            if observer in self._observers:
                self._observers.remove(observer)

        return remove

    def add_validator(self, validator: TransitionValidatorProtocol) -> None:
        """Add a transition validator.

        Args:
            validator: Validator to add.
        """
        self._validators.append(validator)

    def add_detector(self, detector: StageDetectorProtocol) -> None:
        """Add a stage detector.

        Args:
            detector: Detector to add.
        """
        self._detectors.append(detector)

    @property
    def history(self) -> List[StateTransition]:
        """Get transition history."""
        return list(self._history)

    @property
    def transition_count(self) -> int:
        """Get total number of transitions."""
        return self._transition_count

    def to_dict(self) -> Dict[str, Any]:
        """Serialize state machine to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "current_stage": self._current_stage,
            "transition_count": self._transition_count,
            "history": [
                {
                    "from": t.from_stage,
                    "to": t.to_stage,
                    "timestamp": t.timestamp,
                    "confidence": t.confidence,
                }
                for t in self._history[-10:]  # Last 10 transitions
            ],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], config: StateConfig) -> "StateMachine":
        """Restore state machine from dictionary.

        Args:
            data: Dictionary with state data.
            config: State machine configuration.

        Returns:
            Restored StateMachine instance.
        """
        machine = cls(config)
        if "current_stage" in data:
            machine._current_stage = data["current_stage"]
        if "transition_count" in data:
            machine._transition_count = data["transition_count"]
        return machine
