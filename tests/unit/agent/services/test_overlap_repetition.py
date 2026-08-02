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

"""Word-overlap repetition decision used by the streaming executor.

Regression for the spin where the same narration repeated many turns without
breaking: a moderate-overlap turn used to decay the counter, so an oscillating
loop never reached the force-completion threshold.
"""

# The overlap-repetition helper moved to the shared turn_policy module (consolidated so the
# headless and streaming loops share one content-repetition detector).
from victor.agent.turn_policy import (
    evaluate_overlap_repetition as _evaluate_overlap_repetition,
)


def test_near_duplicate_forces_on_first_occurrence() -> None:
    count, action = _evaluate_overlap_repetition(0.9, repetition_count=0)
    assert action == "near_duplicate"
    assert count == 1


def test_two_high_overlap_turns_force_completion() -> None:
    count, action = _evaluate_overlap_repetition(0.6, repetition_count=0)
    assert action == "accumulating"
    assert count == 1

    count, action = _evaluate_overlap_repetition(0.6, repetition_count=count)
    assert action == "high_overlap"
    assert count == 2


def test_moderate_overlap_holds_count_without_decay() -> None:
    # The key fix: a 0.3 < overlap <= 0.5 turn between two high-overlap turns
    # must NOT reset/decay the accumulated spin signal.
    count, action = _evaluate_overlap_repetition(0.6, repetition_count=0)
    assert (count, action) == (1, "accumulating")

    count, action = _evaluate_overlap_repetition(0.45, repetition_count=count)
    assert action == "hold"
    assert count == 1  # held, not decayed

    count, action = _evaluate_overlap_repetition(0.6, repetition_count=count)
    assert action == "high_overlap"
    assert count == 2


def test_distinct_content_resets_count() -> None:
    count, action = _evaluate_overlap_repetition(0.2, repetition_count=3)
    assert action == "reset"
    assert count == 0


def test_oscillating_loop_eventually_breaks() -> None:
    """A loop alternating 0.55 / 0.4 overlap converges instead of spinning forever."""
    count = 0
    forced_at = None
    for i, overlap in enumerate([0.55, 0.4, 0.55, 0.4, 0.55]):
        count, action = _evaluate_overlap_repetition(overlap, count)
        if action in ("near_duplicate", "high_overlap"):
            forced_at = i
            break
    assert forced_at is not None, "oscillating loop should break, not spin forever"
