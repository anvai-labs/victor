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

"""Service-owned strategies for stage-transition batching runtime.

This module provides the canonical transition strategies used by
``victor.agent.services.stage_transition_runtime``. The historical
``victor.agent.coordinators.transition_strategies`` module remains only as a
compatibility import path that re-exports these definitions.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

from victor.core.shared_types import ConversationStage

if TYPE_CHECKING:
    from victor.agent.services.stage_transition_runtime import (
        StageTransitionCoordinator,
        TransitionDecision,
        TransitionResult,
    )

logger = logging.getLogger(__name__)

__all__ = [
    "TransitionStrategyProtocol",
    "HeuristicOnlyTransitionStrategy",
    "EdgeModelTransitionStrategy",
    "HybridTransitionStrategy",
    "create_transition_strategy",
]


class TransitionStrategyProtocol(ABC):
    """Protocol for stage-transition strategies."""

    @abstractmethod
    def detect_transition(
        self,
        current_stage: ConversationStage,
        tools_executed: List[Tuple[str, Dict[str, Any]]],
        state_machine: Any,
        coordinator: Any,
    ) -> Any:
        """Detect if a stage transition should occur."""

    @abstractmethod
    def requires_edge_model(self) -> bool:
        """Whether this strategy uses the edge model."""
        ...


class HeuristicOnlyTransitionStrategy:
    """Use only heuristic detection, no edge model."""

    def detect_transition(
        self,
        current_stage: ConversationStage,
        tools_executed: List[Tuple[str, Dict[str, Any]]],
        state_machine: Any,
        coordinator: Any,
    ) -> Any:
        """Detect transition using heuristic only."""
        detected = state_machine._detect_stage_from_tools()

        if detected and detected != current_stage:
            stage_tools = state_machine._get_tools_for_stage(detected)
            unique_tools = {tool for tool, _ in tools_executed}
            overlap = len(unique_tools & stage_tools)
            confidence = min(0.9, 0.6 + (overlap * 0.1))

            from victor.agent.services.stage_transition_runtime import (
                TransitionDecision,
                TransitionResult,
            )

            return TransitionResult(
                decision=TransitionDecision.HEURISTIC_TRANSITION,
                new_stage=detected,
                confidence=confidence,
                reason=f"Heuristic detection (overlap={overlap})",
                edge_model_called=False,
            )

        from victor.agent.services.stage_transition_runtime import (
            TransitionDecision,
            TransitionResult,
        )

        return TransitionResult(
            decision=TransitionDecision.NO_TRANSITION,
            new_stage=None,
            confidence=0.0,
            reason="No heuristic detection",
            edge_model_called=False,
        )

    def requires_edge_model(self) -> bool:
        return False


class EdgeModelTransitionStrategy:
    """Always use edge model for stage detection."""

    def __init__(self, edge_model_enabled: bool = True):
        self._edge_model_enabled = edge_model_enabled

    def detect_transition(
        self,
        current_stage: ConversationStage,
        tools_executed: List[Tuple[str, Dict[str, Any]]],
        state_machine: Any,
        coordinator: Any,
    ) -> Any:
        """Detect transition using edge model."""
        if not self._edge_model_enabled:
            return HeuristicOnlyTransitionStrategy().detect_transition(
                current_stage=current_stage,
                tools_executed=tools_executed,
                state_machine=state_machine,
                coordinator=coordinator,
            )

        edge_stage, edge_confidence = state_machine._try_edge_model_transition(
            heuristic_stage=current_stage,
            heuristic_confidence=0.5,
        )

        from victor.agent.services.stage_transition_runtime import (
            TransitionDecision,
            TransitionResult,
        )

        if edge_stage and edge_stage != current_stage and edge_confidence > 0.6:
            return TransitionResult(
                decision=TransitionDecision.EDGE_MODEL_TRANSITION,
                new_stage=edge_stage,
                confidence=edge_confidence,
                reason=f"Edge model detection (confidence={edge_confidence:.2f})",
                edge_model_called=True,
            )

        return TransitionResult(
            decision=TransitionDecision.NO_TRANSITION,
            new_stage=None,
            confidence=edge_confidence if edge_stage else 0.0,
            reason="Edge model: no transition detected",
            edge_model_called=True,
        )

    def requires_edge_model(self) -> bool:
        return True


class HybridTransitionStrategy:
    """Combine heuristic + edge model for accuracy and performance."""

    def __init__(self, edge_model_enabled: bool = True):
        self._edge_model_enabled = edge_model_enabled

    def detect_transition(
        self,
        current_stage: ConversationStage,
        tools_executed: List[Tuple[str, Dict[str, Any]]],
        state_machine: Any,
        coordinator: Any,
    ) -> Any:
        """Detect transition using hybrid approach."""
        detected = state_machine._detect_stage_from_tools()

        if detected and detected != current_stage:
            stage_tools = state_machine._get_tools_for_stage(detected)
            unique_tools = {tool for tool, _ in tools_executed}
            overlap = len(unique_tools & stage_tools)
            min_tools = coordinator._min_tools_for_transition

            if overlap >= min_tools:
                confidence = min(0.9, 0.6 + (overlap * 0.1))

                from victor.agent.services.stage_transition_runtime import (
                    TransitionDecision,
                    TransitionResult,
                )

                return TransitionResult(
                    decision=TransitionDecision.HIGH_CONFIDENCE_SKIP,
                    new_stage=detected,
                    confidence=confidence,
                    reason=f"High heuristic confidence (overlap={overlap} >= {min_tools})",
                    edge_model_called=False,
                )

            if self._edge_model_enabled:
                if coordinator.should_skip_edge_model(detected, current_stage):
                    confidence = min(0.7, 0.6 + (overlap * 0.1))

                    from victor.agent.services.stage_transition_runtime import (
                        TransitionDecision,
                        TransitionResult,
                    )

                    return TransitionResult(
                        decision=TransitionDecision.COOLDOWN_SKIP,
                        new_stage=detected,
                        confidence=confidence,
                        reason="Edge model skipped (cooldown or high confidence)",
                        edge_model_called=False,
                    )

                edge_stage, edge_confidence = state_machine._try_edge_model_transition(
                    heuristic_stage=detected,
                    heuristic_confidence=0.6 + (overlap * 0.1),
                )

                if edge_stage and edge_confidence > 0.6:
                    from victor.agent.services.stage_transition_runtime import (
                        TransitionDecision,
                        TransitionResult,
                    )

                    return TransitionResult(
                        decision=TransitionDecision.EDGE_MODEL_TRANSITION,
                        new_stage=edge_stage,
                        confidence=edge_confidence,
                        reason=f"Edge model override (confidence={edge_confidence:.2f})",
                        edge_model_called=True,
                    )

            confidence = min(0.7, 0.6 + (overlap * 0.1))

            from victor.agent.services.stage_transition_runtime import (
                TransitionDecision,
                TransitionResult,
            )

            return TransitionResult(
                decision=TransitionDecision.HEURISTIC_TRANSITION,
                new_stage=detected,
                confidence=confidence,
                reason=f"Heuristic fallback (overlap={overlap})",
                edge_model_called=False,
            )

        from victor.agent.services.stage_transition_runtime import (
            TransitionDecision,
            TransitionResult,
        )

        return TransitionResult(
            decision=TransitionDecision.NO_TRANSITION,
            new_stage=None,
            confidence=0.0,
            reason="No transition detected",
            edge_model_called=False,
        )

    def requires_edge_model(self) -> bool:
        return True


def create_transition_strategy(
    strategy_type: str = "hybrid",
    edge_model_enabled: bool = True,
) -> TransitionStrategyProtocol:
    """Factory function to create transition strategies."""
    if strategy_type == "heuristic":
        return HeuristicOnlyTransitionStrategy()
    if strategy_type == "edge_model":
        return EdgeModelTransitionStrategy(edge_model_enabled=edge_model_enabled)
    if strategy_type == "hybrid":
        return HybridTransitionStrategy(edge_model_enabled=edge_model_enabled)
    raise ValueError(
        f"Unknown strategy type: {strategy_type}. " f"Valid types: heuristic, edge_model, hybrid"
    )
