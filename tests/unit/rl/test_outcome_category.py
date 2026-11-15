# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
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

"""FEP-0046: outcome_category classification and dispatch invariants.

The event taxonomy partitions RLEventType into task-level and
learner-internal sets; dispatch must tag the derived outcome without touching
``task_type`` (the spec's Option A invariant). The SQL backfill constants in
``victor.core.schema`` must mirror the in-code sets exactly.
"""

from typing import Any, List

import pytest

from victor.core.rl.hooks import (
    LEARNER_INTERNAL_EVENTS,
    OUTCOME_CATEGORY_LEARNER_INTERNAL,
    OUTCOME_CATEGORY_TASK,
    RLEvent,
    RLEventType,
    RLHookRegistry,
    TASK_LEVEL_EVENTS,
    classify_event_category,
)
from victor.core.schema import (
    RL_OUTCOME_CATEGORY_INTERNAL_EVENT_TYPES,
    RL_OUTCOME_CATEGORY_INTERNAL_LEARNER_IDS,
)
from victor.framework.rl.base import RLOutcome
from victor.framework.rl.coordinator import LEARNER_INTERNAL_LEARNERS


class _RecordingCoordinator:
    """Minimal coordinator stub capturing dispatched outcomes."""

    def __init__(self) -> None:
        self.recorded: List[tuple] = []
        self.is_closed = False

    def record_outcome(self, learner_name: str, outcome: Any, vertical: str = "coding") -> None:
        self.recorded.append((learner_name, outcome))


def test_event_sets_partition_enum_exactly():
    """Every RLEventType is classified once — no gaps, no overlap.

    Negative guard: the next added event type without a classification fails
    here, forcing a conscious decision.
    """
    all_events = set(RLEventType)
    assert TASK_LEVEL_EVENTS | LEARNER_INTERNAL_EVENTS == all_events
    assert not TASK_LEVEL_EVENTS & LEARNER_INTERNAL_EVENTS


@pytest.mark.parametrize("event_type", sorted(TASK_LEVEL_EVENTS, key=lambda e: e.value))
def test_task_level_events_classified(event_type: RLEventType):
    assert classify_event_category(event_type) == OUTCOME_CATEGORY_TASK


@pytest.mark.parametrize("event_type", sorted(LEARNER_INTERNAL_EVENTS, key=lambda e: e.value))
def test_learner_internal_events_classified(event_type: RLEventType):
    assert classify_event_category(event_type) == OUTCOME_CATEGORY_LEARNER_INTERNAL


def test_dispatch_tags_category_and_preserves_task_type():
    """The conflation fix itself: a failed grounding check during a
    code_generation task must land as learner_internal while task_type keeps
    naming the underlying task."""
    coordinator = _RecordingCoordinator()
    registry = RLHookRegistry(coordinator=coordinator)
    registry.subscribe(RLEventType.GROUNDING_CHECK, "grounding_threshold")

    registry.emit(
        RLEvent(
            type=RLEventType.GROUNDING_CHECK,
            provider="zai",
            model="glm-5.2",
            task_type="code_generation",
            success=False,
            quality_score=0.4,
        )
    )

    assert len(coordinator.recorded) == 1
    learner_name, outcome = coordinator.recorded[0]
    assert learner_name == "grounding_threshold"
    assert outcome.outcome_category == OUTCOME_CATEGORY_LEARNER_INTERNAL
    # Option A invariant: task_type is NOT rewritten to a pseudo type.
    assert outcome.task_type == "code_generation"
    assert outcome.success is False
    assert outcome.metadata["event_type"] == "grounding_check"


def test_dispatch_task_event_stays_task():
    coordinator = _RecordingCoordinator()
    registry = RLHookRegistry(coordinator=coordinator)
    registry.subscribe(RLEventType.TOOL_EXECUTED, "tool_selector")

    registry.emit(
        RLEvent(
            type=RLEventType.TOOL_EXECUTED,
            tool_name="read",
            task_type="analysis",
            success=True,
            quality_score=0.9,
        )
    )

    _, outcome = coordinator.recorded[0]
    assert outcome.outcome_category == OUTCOME_CATEGORY_TASK
    assert outcome.task_type == "analysis"


def test_outcome_dict_roundtrip_carries_category():
    outcome = RLOutcome(
        provider="p",
        model="m",
        task_type="analysis",
        success=True,
        quality_score=0.7,
        outcome_category=OUTCOME_CATEGORY_LEARNER_INTERNAL,
    )
    restored = RLOutcome.from_dict(outcome.to_dict())
    assert restored.outcome_category == OUTCOME_CATEGORY_LEARNER_INTERNAL


def test_from_dict_legacy_defaults_to_none():
    """Pre-FEP dicts have no category key; None defers to insert-time resolution."""
    legacy = {
        "provider": "p",
        "model": "m",
        "task_type": "analysis",
        "success": True,
        "quality_score": 0.7,
        "metadata": "{}",
    }
    assert RLOutcome.from_dict(legacy).outcome_category is None


def test_schema_backfill_constants_mirror_code_sets():
    """The v10 SQL backfill lists must stay in sync with the taxonomy."""
    assert set(RL_OUTCOME_CATEGORY_INTERNAL_EVENT_TYPES) == {
        event.value for event in LEARNER_INTERNAL_EVENTS
    }
    assert set(RL_OUTCOME_CATEGORY_INTERNAL_LEARNER_IDS) == set(LEARNER_INTERNAL_LEARNERS)
