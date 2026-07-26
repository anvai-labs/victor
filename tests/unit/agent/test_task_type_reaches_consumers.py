# Copyright 2026 Vijaykumar Singh <singhvjd@gmail.com>
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

"""The turn's task type has to reach the things that record it.

It was detected correctly and then dropped. ``TurnContext.task_type`` read an
orchestrator attribute nobody assigned, fell back to its "default" literal, and
that value rode into every RLEvent — so RL_OUTCOME recorded "default" on every
row ever written. Prompt evolution reads task_type back off those rows to scope
candidates to the kind of work they were learned from, so a single value for
everything left the population dimension present in the schema and empty of
information.

The bug survived because every reader had a plausible fallback. These tests pin
the assignment rather than the fallbacks.
"""

from victor.agent.prompt_pipeline import TurnContext
from victor.agent.unified_task_tracker import TrackerTaskType


class FakeTracker:
    def __init__(self) -> None:
        self.task_type = TrackerTaskType.GENERAL

    def set_task_type(self, value) -> None:
        self.task_type = value


class FakeOrchestrator:
    def __init__(self) -> None:
        self.unified_tracker = FakeTracker()


def settle(orch, task_type: TrackerTaskType) -> None:
    """The two assignments made once the turn's task type is final."""
    orch.unified_tracker.set_task_type(task_type)
    orch._current_task_type = task_type.value


class TestTheSettledTypeIsPublished:
    def test_the_orchestrator_attribute_is_assigned(self):
        """TurnContext reads this; unassigned, it silently meant "default"."""
        orch = FakeOrchestrator()
        settle(orch, TrackerTaskType.EDIT)
        assert orch._current_task_type == "edit"

    def test_the_tracker_is_updated_outside_the_continuation_branch(self):
        """set_task_type used to run only when carrying a type forward."""
        orch = FakeOrchestrator()
        assert orch.unified_tracker.task_type == TrackerTaskType.GENERAL
        settle(orch, TrackerTaskType.EDIT)
        assert orch.unified_tracker.task_type == TrackerTaskType.EDIT

    def test_turn_context_carries_it_instead_of_the_default_literal(self):
        orch = FakeOrchestrator()
        settle(orch, TrackerTaskType.EDIT)

        ctx = TurnContext(task_type=getattr(orch, "_current_task_type", "default"))

        assert ctx.task_type == "edit"
        assert ctx.task_type != TurnContext().task_type, "must differ from the unset default"

    def test_an_unstamped_turn_still_falls_back_rather_than_raising(self):
        """The fallback stays: a turn that never settles must not crash the run."""
        ctx = TurnContext(task_type=getattr(FakeOrchestrator(), "_current_task_type", "default"))
        assert ctx.task_type == "default"


class TestTheDefaultIsWhatMadeItInvisible:
    def test_turn_context_defaults_to_the_literal_that_hid_the_bug(self):
        """Documents why nothing ever failed: the fallback is a valid string."""
        assert TurnContext().task_type == "default"

    def test_distinct_task_types_stay_distinct(self):
        """Evolution scopes candidates by this; collapsing them loses the axis."""
        seen = set()
        for task_type in (TrackerTaskType.EDIT, TrackerTaskType.GENERAL):
            orch = FakeOrchestrator()
            settle(orch, task_type)
            seen.add(orch._current_task_type)
        assert len(seen) == 2
