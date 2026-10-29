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

"""ContextPruningLearner: Thompson-sampled pruning actions per context bucket."""

from pathlib import Path

import pytest

from victor.core.database import get_database, reset_database
from victor.framework.rl.coordinator import RLCoordinator
from victor.framework.rl.learners.context_pruning import (
    PRUNING_CONFIGS,
    ContextPruningLearner,
    PruningAction,
)


@pytest.fixture
def coordinator(tmp_path: Path):
    reset_database()
    db_path = tmp_path / "rl_test.db"
    get_database(db_path)
    yield RLCoordinator(storage_path=tmp_path, db_path=db_path)
    reset_database()


@pytest.fixture
def learner(coordinator) -> ContextPruningLearner:
    lrn = coordinator.get_learner("context_pruning")
    lrn.seed_rng(7)
    return lrn


def test_is_wired_through_the_coordinator(learner):
    assert isinstance(learner, ContextPruningLearner)


def test_configs_tighten_monotonically():
    order = [
        PruningAction.NO_PRUNE,
        PruningAction.LIGHT,
        PruningAction.MODERATE,
        PruningAction.AGGRESSIVE,
    ]
    cfgs = [PRUNING_CONFIGS[a] for a in order]
    for looser, tighter in zip(cfgs, cfgs[1:]):
        assert tighter.compaction_threshold < looser.compaction_threshold
        assert tighter.tool_result_max_chars < looser.tool_result_max_chars
        assert tighter.min_messages_keep <= looser.min_messages_keep


def test_recommendation_is_a_known_action(learner):
    rec = learner.get_recommendation(context_utilization=0.85, tool_call_count=12)
    assert rec is not None
    assert PruningAction(getattr(rec.value, "value", rec.value)) in PRUNING_CONFIGS


def test_outcomes_accumulate_in_stats(learner):
    assert learner.get_stats()["total_decisions"] == 0
    for _ in range(4):
        learner.record_outcome(0.85, 12, PruningAction.LIGHT.value, True, 1500)
    learner.record_outcome(0.85, 12, PruningAction.AGGRESSIVE.value, False, 4000)
    stats = learner.get_stats()
    assert stats["total_decisions"] == 5
    assert stats["total_tokens_saved"] == 4 * 1500 + 4000
    assert stats["action_stats"][PruningAction.LIGHT.value]["visits"] == 4
    assert stats["best_action"] == PruningAction.LIGHT.value


def test_stats_filter_by_provider_type(learner):
    learner.record_outcome(0.85, 12, PruningAction.LIGHT.value, True, 100, provider_type="local")
    assert learner.get_stats("cloud")["action_stats"] == {}
    assert PruningAction.LIGHT.value in learner.get_stats("local")["action_stats"]


def test_reset_clears_learned_state(learner):
    learner.record_outcome(0.85, 12, PruningAction.LIGHT.value, True, 100)
    learner.reset()
    stats = learner.get_stats()
    assert stats["total_decisions"] == 0 and stats["action_stats"] == {}
