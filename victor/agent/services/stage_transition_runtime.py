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

"""Service-owned stage-transition batching runtime.

This module provides ``StageTransitionCoordinator`` as the canonical runtime
implementation for batched conversation-stage transitions. The historical
``victor.agent.coordinators.stage_transition_coordinator`` module remains only
as a compatibility import path that re-exports these definitions.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from victor.core.shared_types import ConversationStage

if TYPE_CHECKING:
    from victor.agent.services.stage_transition_strategies import (
        TransitionStrategyProtocol,
    )

logger = logging.getLogger(__name__)

__all__ = [
    "StageTransitionCoordinator",
    "TransitionDecision",
    "TransitionResult",
    "TurnContext",
]


class TransitionDecision(str, Enum):
    """Decision result from transition evaluation."""

    NO_TRANSITION = "no_transition"
    HEURISTIC_TRANSITION = "heuristic"
    EDGE_MODEL_TRANSITION = "edge_model"
    COOLDOWN_SKIP = "cooldown_skip"
    HIGH_CONFIDENCE_SKIP = "high_confidence_skip"


@dataclass
class TurnContext:
    """Context for a single turn (batching window)."""

    turn_id: str
    start_time: float
    tools_executed: List[Tuple[str, Dict[str, Any]]] = field(default_factory=list)
    current_stage: ConversationStage = ConversationStage.INITIAL

    def add_tool(self, tool_name: str, args: Dict[str, Any]) -> None:
        """Add a tool execution to this turn."""
        self.tools_executed.append((tool_name, args))

    @property
    def unique_tools(self) -> set[str]:
        """Get unique tool names executed this turn."""
        return {tool for tool, _ in self.tools_executed}

    @property
    def tool_count(self) -> int:
        """Get total tool executions this turn."""
        return len(self.tools_executed)


@dataclass
class TransitionResult:
    """Result of a transition evaluation."""

    decision: TransitionDecision
    new_stage: Optional[ConversationStage]
    confidence: float
    reason: str
    edge_model_called: bool = False
    calibration_applied: bool = False


class StageTransitionCoordinator:
    """Coordinates stage transitions with batching and runtime optimizations."""

    def __init__(
        self,
        state_machine: Any,
        strategy: "TransitionStrategyProtocol",
        cooldown_seconds: float = 2.0,
        min_tools_for_transition: int = 5,
    ):
        """Initialize the coordinator."""
        self._state_machine = state_machine
        self._strategy = strategy
        self._cooldown_seconds = cooldown_seconds
        self._min_tools_for_transition = min_tools_for_transition

        self._current_turn: Optional[TurnContext] = None
        self._last_transition_time: float = 0.0
        self._transition_count: int = 0

        logger.debug(
            "StageTransitionCoordinator initialized: strategy=%s, cooldown=%ss, min_tools=%s",
            type(strategy).__name__,
            cooldown_seconds,
            min_tools_for_transition,
        )

    def begin_turn(self) -> None:
        """Mark the start of a new turn."""
        self._current_turn = TurnContext(
            turn_id=str(uuid.uuid4())[:8],
            start_time=time.time(),
            current_stage=self._state_machine.get_stage(),
        )
        logger.debug(
            "Turn %s started, stage=%s",
            self._current_turn.turn_id,
            self._current_turn.current_stage,
        )

    def record_tool(self, tool_name: str, args: Dict[str, Any]) -> None:
        """Record a tool execution without immediate transition evaluation."""
        if self._current_turn is None:
            logger.warning("record_tool called before begin_turn, creating turn")
            self.begin_turn()

        self._current_turn.add_tool(tool_name, args)
        logger.debug(
            "Recorded tool %s in turn %s, total tools=%s",
            tool_name,
            self._current_turn.turn_id,
            self._current_turn.tool_count,
        )

    def end_turn(self) -> Optional[ConversationStage]:
        """Process the batched turn and return the new stage if one occurred."""
        if self._current_turn is None:
            logger.warning("end_turn called without active turn")
            return None

        turn = self._current_turn
        logger.debug(
            "Turn %s ending, tools=%s, unique=%s, stage=%s",
            turn.turn_id,
            turn.tool_count,
            len(turn.unique_tools),
            turn.current_stage,
        )

        if self._is_in_cooldown():
            time_since_last = time.time() - self._last_transition_time
            logger.debug(
                "In cooldown period (%.1fs < %ss), skipping transition evaluation",
                time_since_last,
                self._cooldown_seconds,
            )
            self._current_turn = None
            return None

        result = self._evaluate_transition(turn)

        if self._should_calibrate(result):
            result = self._apply_calibration(result)

        new_stage = None
        if result.decision in (
            TransitionDecision.HEURISTIC_TRANSITION,
            TransitionDecision.EDGE_MODEL_TRANSITION,
            TransitionDecision.HIGH_CONFIDENCE_SKIP,
        ):
            self._execute_transition(result)
            new_stage = result.new_stage

        self._current_turn = None
        return new_stage

    def should_skip_edge_model(
        self,
        detected_stage: ConversationStage,
        current_stage: ConversationStage,
    ) -> bool:
        """Check if the edge model should be skipped."""
        if self._is_in_cooldown():
            return True

        if detected_stage == current_stage:
            return True

        if self._current_turn:
            overlap = self._calculate_stage_overlap(detected_stage)
            if overlap >= self._min_tools_for_transition:
                logger.debug(
                    "High heuristic confidence (overlap=%s >= %s), skipping edge model",
                    overlap,
                    self._min_tools_for_transition,
                )
                return True

        return False

    def _is_in_cooldown(self) -> bool:
        """Check if currently in cooldown period."""
        time_since_last = time.time() - self._last_transition_time
        return time_since_last < self._cooldown_seconds

    def _calculate_stage_overlap(self, stage: ConversationStage) -> int:
        """Calculate tool overlap with a stage."""
        if not self._current_turn:
            return 0

        stage_tools = self._state_machine._get_tools_for_stage(stage)
        turn_tools = self._current_turn.unique_tools
        return len(turn_tools & stage_tools)

    def _evaluate_transition(self, turn: TurnContext) -> TransitionResult:
        """Evaluate if a transition should occur using the configured strategy."""
        try:
            return self._strategy.detect_transition(
                current_stage=turn.current_stage,
                tools_executed=turn.tools_executed,
                state_machine=self._state_machine,
                coordinator=self,
            )
        except Exception as e:
            logger.error("Error in transition strategy: %s", e)
            return TransitionResult(
                decision=TransitionDecision.NO_TRANSITION,
                new_stage=None,
                confidence=0.0,
                reason=f"Strategy error: {e}",
                edge_model_called=False,
            )

    def _should_calibrate(self, result: TransitionResult) -> bool:
        """Check if read-only exploration calibration should be applied."""
        from victor.agent.action_authorizer import ActionIntent

        if result.new_stage != ConversationStage.EXECUTION:
            return False

        if result.confidence < 0.95:
            return False

        if getattr(self._state_machine, "_action_intent", None) == ActionIntent.WRITE_ALLOWED:
            return False

        files_read = len(self._state_machine.state.observed_files)
        files_modified = len(self._state_machine.state.modified_files)
        return files_read > 10 and files_modified == 0

    def _apply_calibration(self, result: TransitionResult) -> TransitionResult:
        """Apply read-only exploration calibration to the result."""
        files_read = len(self._state_machine.state.observed_files)
        logger.warning(
            "Edge model calibration: %s (%.2f) -> ANALYSIS. Reason: Agent has read %s "
            "files without any edits. High confidence EXECUTION is likely biased/overconfident.",
            result.new_stage,
            result.confidence,
            files_read,
        )
        return TransitionResult(
            decision=result.decision,
            new_stage=ConversationStage.ANALYSIS,
            confidence=0.7,
            reason="Calibration applied: read-only exploration",
            edge_model_called=result.edge_model_called,
            calibration_applied=True,
        )

    def _execute_transition(self, result: TransitionResult) -> None:
        """Execute the stage transition."""
        if not result.new_stage or not self._current_turn:
            return

        old_stage = self._current_turn.current_stage
        if result.new_stage == old_stage:
            return

        logger.info(
            "Stage transition: %s -> %s (confidence: %.2f, reason: %s, edge_model=%s, "
            "calibration=%s)",
            old_stage,
            result.new_stage,
            result.confidence,
            result.reason,
            result.edge_model_called,
            result.calibration_applied,
        )

        self._state_machine._transition_to(result.new_stage, result.confidence)
        self._last_transition_time = time.time()
        self._transition_count += 1

    def get_statistics(self) -> Dict[str, Any]:
        """Get coordinator statistics."""
        return {
            "transition_count": self._transition_count,
            "last_transition_time": self._last_transition_time,
            "current_turn_id": (self._current_turn.turn_id if self._current_turn else None),
            "current_turn_tools": (self._current_turn.tool_count if self._current_turn else 0),
            "strategy": type(self._strategy).__name__,
        }
