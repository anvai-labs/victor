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

"""Centralized RL Hook Registry for event-driven learner activation.

This module provides a SOLID-compliant event dispatch system for RL learners.
Instead of manually triggering learners at each call site, components emit
events which are dispatched to subscribed learners.

Design Principles:
    - Single Responsibility: Each hook handles one event type
    - Open/Closed: New events can be added without modifying existing code
    - Dependency Inversion: Components depend on event abstractions, not learners

Events:
    - TOOL_EXECUTED: Tool completed execution (triggers tool_selector)
    - MODE_TRANSITION: Mode changed (triggers mode_transition)
    - GROUNDING_CHECK: Grounding verification (triggers grounding_threshold)
    - CACHE_ACCESS: Cache hit/miss (triggers cache_eviction)
    - WORKFLOW_STEP: Workflow step completed (triggers workflow_execution)
    - QUALITY_ASSESSED: Quality score computed (triggers quality_weights)
    - TEAM_COMPLETED: Team execution finished (triggers team_composition)
    - CONTINUATION: Continuation attempt (triggers continuation_patience, continuation_prompts)
    - SEMANTIC_MATCH: Semantic search result (triggers semantic_threshold)
    - MODEL_SELECTED: Model selection made (triggers model_selector)

Usage:
    from victor.framework.rl.hooks import RLHookRegistry, RLEvent, RLEventType

    # Get global registry
    registry = get_rl_hooks()

    # Emit an event (components do this)
    registry.emit(RLEvent(
        type=RLEventType.TOOL_EXECUTED,
        tool_name="read",
        task_type="analysis",
        success=True,
        quality_score=0.85,
        provider="anthropic",
        model="claude-3-5-sonnet"
    ))

    # Registry dispatches to subscribed learners automatically
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, TYPE_CHECKING
from victor.core.constants import DEFAULT_VERTICAL

if TYPE_CHECKING:
    from victor.framework.rl.coordinator import RLCoordinator
    from victor.framework.rl.base import RLOutcome

logger = logging.getLogger(__name__)


class RLEventType(str, Enum):
    """Types of events that trigger RL learner updates.

    Each event type maps to one or more learners that should be notified.
    """

    # Tool-related events
    TOOL_EXECUTED = "tool_executed"
    """A tool completed execution. Triggers: tool_selector"""

    TOOL_SELECTED = "tool_selected"
    """A tool was selected (before execution). For epsilon tracking."""

    # Mode events
    MODE_TRANSITION = "mode_transition"
    """Mode changed (explore→plan→build). Triggers: mode_transition"""

    # Quality events
    GROUNDING_CHECK = "grounding_check"
    """Grounding verification completed. Triggers: grounding_threshold"""

    QUALITY_ASSESSED = "quality_assessed"
    """Quality score computed. Triggers: quality_weights"""

    SEMANTIC_MATCH = "semantic_match"
    """Semantic search/match result. Triggers: semantic_threshold"""

    # Cache events
    CACHE_ACCESS = "cache_access"
    """Cache hit or miss. Triggers: cache_eviction"""

    # Workflow events
    WORKFLOW_STEP = "workflow_step"
    """Workflow step completed. Triggers: workflow_execution"""

    WORKFLOW_COMPLETED = "workflow_completed"
    """Entire workflow finished. Triggers: workflow_execution"""

    # Team events
    TEAM_COMPLETED = "team_completed"
    """Multi-agent team execution finished. Triggers: team_composition"""

    # Continuation events
    CONTINUATION_ATTEMPT = "continuation_attempt"
    """Continuation attempt made. Triggers: continuation_patience"""

    CONTINUATION_PROMPT = "continuation_prompt"
    """Continuation prompt used. Triggers: continuation_prompts"""

    # Model selection
    MODEL_SELECTED = "model_selected"
    """Model was selected for task. Triggers: model_selector"""

    # Cross-vertical
    VERTICAL_SWITCH = "vertical_switch"
    """Switched between verticals. Triggers: cross_vertical"""

    # Prompt template
    PROMPT_USED = "prompt_used"
    """Prompt template was used. Triggers: prompt_template"""

    # Prompt optimization
    GEPA_EVOLUTION = "gepa_evolution"
    """GEPA prompt evolution cycle completed. Triggers: prompt_optimizer"""

    PROMPT_CANDIDATE_USED = "prompt_candidate_used"
    """An evolved prompt candidate was served for a turn and its outcome is known.
    Carries the served candidate identity (prompt_section, prompt_candidate_hash)
    + the turn's completion score, so prompt_optimizer can update the candidate's
    Thompson posterior. Triggers: prompt_optimizer"""


# Register domain event mapping with canonical EventType taxonomy
def _register_rl_event_taxonomy() -> None:
    try:
        from victor.core.events.taxonomy import EventTaxonomyRegistry
        from victor.framework.events import EventType

        EventTaxonomyRegistry.register_domain(
            "rl",
            RLEventType,
            {
                RLEventType.TOOL_EXECUTED: EventType.TOOL_RESULT,
                RLEventType.TOOL_SELECTED: EventType.TOOL_CALL,
                RLEventType.MODE_TRANSITION: EventType.STAGE_CHANGE,
                RLEventType.GROUNDING_CHECK: EventType.PROGRESS,
                RLEventType.QUALITY_ASSESSED: EventType.PROGRESS,
                RLEventType.WORKFLOW_COMPLETED: EventType.STREAM_END,
                RLEventType.TEAM_COMPLETED: EventType.MILESTONE,
                RLEventType.GEPA_EVOLUTION: EventType.MILESTONE,
                RLEventType.PROMPT_CANDIDATE_USED: EventType.PROGRESS,
            },
        )
    except ImportError:
        pass


_register_rl_event_taxonomy()


@dataclass
class RLEvent:
    """Event data structure for RL hook dispatch.

    Contains all context needed for learner updates.
    """

    type: RLEventType
    """Event type determines which learners are notified."""

    # Common context
    provider: Optional[str] = None
    model: Optional[str] = None
    task_type: Optional[str] = None
    vertical: str = DEFAULT_VERTICAL

    # Outcome data
    success: Optional[bool] = None
    quality_score: Optional[float] = None

    # Event-specific data
    tool_name: Optional[str] = None
    mode_from: Optional[str] = None
    mode_to: Optional[str] = None
    cache_hit: Optional[bool] = None
    workflow_name: Optional[str] = None
    workflow_step: Optional[str] = None
    team_id: Optional[str] = None
    team_formation: Optional[str] = None
    threshold_value: Optional[float] = None
    similarity_score: Optional[float] = None

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    # Exploration tracking
    was_exploration: bool = False
    """True if this action was chosen via exploration (epsilon), not exploitation."""

    epsilon_value: Optional[float] = None
    """Current epsilon value when action was taken."""


# Mapping from event types to learner names
EVENT_TO_LEARNER: Dict[RLEventType, List[str]] = {
    RLEventType.TOOL_EXECUTED: ["tool_selector"],
    RLEventType.TOOL_SELECTED: ["tool_selector"],
    RLEventType.MODE_TRANSITION: ["mode_transition"],
    RLEventType.GROUNDING_CHECK: ["grounding_threshold"],
    RLEventType.QUALITY_ASSESSED: ["quality_weights"],
    RLEventType.SEMANTIC_MATCH: ["semantic_threshold"],
    RLEventType.CACHE_ACCESS: ["cache_eviction"],
    RLEventType.WORKFLOW_STEP: ["workflow_execution"],
    RLEventType.WORKFLOW_COMPLETED: ["workflow_execution"],
    RLEventType.TEAM_COMPLETED: ["team_composition"],
    RLEventType.CONTINUATION_ATTEMPT: ["continuation_patience"],
    RLEventType.CONTINUATION_PROMPT: ["continuation_prompts"],
    RLEventType.MODEL_SELECTED: ["model_selector"],
    RLEventType.VERTICAL_SWITCH: ["cross_vertical"],
    RLEventType.PROMPT_USED: ["prompt_template"],
    RLEventType.PROMPT_CANDIDATE_USED: ["prompt_optimizer"],
}


class RLHookRegistry:
    """Centralized registry for RL event dispatch.

    Implements the Observer pattern for decoupled learner activation.
    Components emit events; this registry dispatches to appropriate learners.

    Features:
        - Event-driven learner activation
        - Epsilon/exploration tracking per event
        - Metrics collection for observability
        - Async-safe event buffering
        - Learner subscription management
    """

    def __init__(self, coordinator: Optional["RLCoordinator"] = None):
        """Initialize hook registry.

        Args:
            coordinator: RL coordinator for accessing learners
        """
        self._coordinator = coordinator
        self._subscribers: Dict[RLEventType, Set[str]] = {}
        self._custom_handlers: Dict[RLEventType, List[Callable[[RLEvent], None]]] = {}

        # Metrics tracking
        self._event_counts: Dict[RLEventType, int] = {}
        self._exploration_counts: Dict[str, int] = {}  # learner -> exploration count
        self._exploitation_counts: Dict[str, int] = {}  # learner -> exploitation count
        self._epsilon_history: List[tuple] = []  # (timestamp, learner, epsilon)

        # Track whether we've warned about missing coordinator (only warn once)
        self._warned_no_coordinator = False

        # Initialize default subscriptions
        self._init_default_subscriptions()

        logger.debug("RLHookRegistry initialized")

    def _init_default_subscriptions(self) -> None:
        """Set up default event-to-learner mappings."""
        for event_type, learners in EVENT_TO_LEARNER.items():
            self._subscribers[event_type] = set(learners)

    def set_coordinator(self, coordinator: "RLCoordinator") -> None:
        """Set the RL coordinator for learner access.

        Args:
            coordinator: RL coordinator instance
        """
        self._coordinator = coordinator
        self._warned_no_coordinator = False  # Reset warning flag
        logger.debug("RLHookRegistry connected to coordinator")

    def subscribe(self, event_type: RLEventType, learner_name: str) -> None:
        """Subscribe a learner to an event type.

        Args:
            event_type: Event type to subscribe to
            learner_name: Name of learner to notify
        """
        if event_type not in self._subscribers:
            self._subscribers[event_type] = set()
        self._subscribers[event_type].add(learner_name)
        logger.debug(f"RL: Learner '{learner_name}' subscribed to {event_type.value}")

    def unsubscribe(self, event_type: RLEventType, learner_name: str) -> None:
        """Unsubscribe a learner from an event type.

        Args:
            event_type: Event type to unsubscribe from
            learner_name: Name of learner to remove
        """
        if event_type in self._subscribers:
            self._subscribers[event_type].discard(learner_name)

    def add_handler(self, event_type: RLEventType, handler: Callable[[RLEvent], None]) -> None:
        """Add a custom event handler.

        Allows extension without modifying learner code.

        Args:
            event_type: Event type to handle
            handler: Callback function
        """
        if event_type not in self._custom_handlers:
            self._custom_handlers[event_type] = []
        self._custom_handlers[event_type].append(handler)

    def emit(self, event: RLEvent) -> None:
        """Emit an event to all subscribed learners.

        This is the main entry point for components to trigger RL updates.

        Args:
            event: Event data to dispatch
        """
        # Track metrics
        self._event_counts[event.type] = self._event_counts.get(event.type, 0) + 1

        # Track exploration vs exploitation
        if event.was_exploration:
            for learner in self._subscribers.get(event.type, []):
                self._exploration_counts[learner] = self._exploration_counts.get(learner, 0) + 1
        else:
            for learner in self._subscribers.get(event.type, []):
                self._exploitation_counts[learner] = self._exploitation_counts.get(learner, 0) + 1

        # Track epsilon values
        if event.epsilon_value is not None:
            for learner in self._subscribers.get(event.type, []):
                self._epsilon_history.append((event.timestamp, learner, event.epsilon_value))

        # Dispatch to subscribed learners
        if self._coordinator is None:
            # Only warn once about missing coordinator (expected during early startup)
            if not self._warned_no_coordinator:
                logger.debug(
                    "RL: No coordinator set, events will be dispatched once coordinator is connected"
                )
                self._warned_no_coordinator = True
            return

        # Check if coordinator is closed (prevents "Cannot operate on closed database" errors)
        if hasattr(self._coordinator, "is_closed") and self._coordinator.is_closed:
            logger.debug("RL: Coordinator is closed, skipping event dispatch")
            return

        learners = self._subscribers.get(event.type, set())
        for learner_name in learners:
            try:
                self._dispatch_to_learner(learner_name, event)
            except Exception as e:
                # Don't log warnings for closed database errors (expected during shutdown)
                if "closed database" in str(e).lower():
                    logger.debug(f"RL: Skipping dispatch to '{learner_name}' - database closed")
                else:
                    logger.warning(f"RL: Failed to dispatch to '{learner_name}': {e}")

        # Dispatch to custom handlers
        handlers = self._custom_handlers.get(event.type, [])
        for handler in handlers:
            try:
                handler(event)
            except Exception as e:
                logger.warning(f"RL: Custom handler failed: {e}")

        logger.debug(f"RL: Dispatched {event.type.value} to {len(learners)} learners")

    def _dispatch_to_learner(self, learner_name: str, event: RLEvent) -> None:
        """Dispatch event to a specific learner.

        Args:
            learner_name: Name of learner
            event: Event data
        """
        from victor.framework.rl.base import RLOutcome

        # Convert event to outcome
        outcome = RLOutcome(
            provider=event.provider or "",
            model=event.model or "",
            task_type=event.task_type or "general",
            success=event.success if event.success is not None else True,
            quality_score=(event.quality_score if event.quality_score is not None else 0.5),
            metadata={
                "event_type": event.type.value,
                "tool_name": event.tool_name,
                "mode_from": event.mode_from,
                "mode_to": event.mode_to,
                "workflow_name": event.workflow_name,
                "team_id": event.team_id,
                "was_exploration": event.was_exploration,
                "epsilon_value": event.epsilon_value,
                **event.metadata,
            },
            vertical=event.vertical,
        )

        # Record outcome via coordinator
        self._coordinator.record_outcome(learner_name, outcome, event.vertical)

    def get_exploration_rate(self, learner_name: str) -> float:
        """Get current exploration rate for a learner.

        Args:
            learner_name: Name of learner

        Returns:
            Exploration rate (0-1), or 0 if no data
        """
        explore = self._exploration_counts.get(learner_name, 0)
        exploit = self._exploitation_counts.get(learner_name, 0)
        total = explore + exploit

        if total == 0:
            return 0.0

        return explore / total

    def get_metrics(self) -> Dict[str, Any]:
        """Get registry metrics for observability.

        Returns:
            Dict with event counts, exploration rates, epsilon history
        """
        return {
            "event_counts": dict(self._event_counts),
            "exploration_counts": dict(self._exploration_counts),
            "exploitation_counts": dict(self._exploitation_counts),
            "exploration_rates": {
                name: self.get_exploration_rate(name)
                for name in set(self._exploration_counts) | set(self._exploitation_counts)
            },
            "epsilon_history_size": len(self._epsilon_history),
            "subscribers": {
                event_type.value: list(learners)
                for event_type, learners in self._subscribers.items()
            },
        }

    def get_epsilon_trend(self, learner_name: str, limit: int = 100) -> List[tuple]:
        """Get epsilon value trend for a learner.

        Args:
            learner_name: Name of learner
            limit: Max entries to return

        Returns:
            List of (timestamp, epsilon) tuples
        """
        entries = [(ts, eps) for ts, name, eps in self._epsilon_history if name == learner_name]
        return entries[-limit:]


# Global singleton
_hook_registry: Optional[RLHookRegistry] = None


def get_rl_hooks(coordinator: Optional["RLCoordinator"] = None) -> RLHookRegistry:
    """Get the global RL hook registry.

    Args:
        coordinator: Optional coordinator to set (only on first call)

    Returns:
        Global RLHookRegistry instance
    """
    global _hook_registry

    if _hook_registry is None:
        _hook_registry = RLHookRegistry(coordinator)
    elif coordinator is not None:
        _hook_registry.set_coordinator(coordinator)

    return _hook_registry


def reset_rl_hooks() -> None:
    """Reset the global hook registry (for testing)."""
    global _hook_registry
    _hook_registry = None
