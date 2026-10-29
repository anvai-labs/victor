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

"""ContinuationPromptLearner: per provider:model:task continuation limits."""

from pathlib import Path

import pytest

from victor.core.database import get_database, reset_database
from victor.framework.rl.base import RLOutcome
from victor.framework.rl.coordinator import RLCoordinator
from victor.framework.rl.learners.continuation_prompts import ContinuationPromptLearner


@pytest.fixture
def coordinator(tmp_path: Path):
    reset_database()
    db_path = tmp_path / "rl_test.db"
    get_database(db_path)
    yield RLCoordinator(storage_path=tmp_path, db_path=db_path)
    reset_database()


@pytest.fixture
def learner(coordinator) -> ContinuationPromptLearner:
    return coordinator.get_learner("continuation_prompts")


def _outcome(success=True, quality=0.8, **meta) -> RLOutcome:
    meta.setdefault("continuation_prompts_used", 2)
    meta.setdefault("max_prompts_configured", 6)
    return RLOutcome(
        provider="anthropic",
        model="m",
        task_type="action",
        success=success,
        quality_score=quality,
        metadata=meta,
    )


def test_is_wired_through_the_coordinator(learner):
    assert isinstance(learner, ContinuationPromptLearner)
    assert learner.name == "continuation_prompts"


def test_no_recommendation_without_history(learner):
    assert learner.get_recommendation("anthropic", "m", "action") is None


def test_needs_three_sessions_before_recommending(learner):
    learner.record_outcome(_outcome())
    learner.record_outcome(_outcome())
    assert learner.get_recommendation("anthropic", "m", "action") is None
    learner.record_outcome(_outcome())
    rec = learner.get_recommendation("anthropic", "m", "action")
    assert rec is not None and rec.sample_size == 3 and not rec.is_baseline
    assert isinstance(rec.value, int) and rec.value > 0


def test_contexts_are_isolated(learner):
    for _ in range(3):
        learner.record_outcome(_outcome())
    assert learner.get_recommendation("anthropic", "m", "analysis") is None
    assert learner.get_recommendation("openai", "m", "action") is None


def test_stuck_loops_lower_confidence(coordinator):
    healthy = coordinator.get_learner("continuation_prompts")
    for _ in range(5):
        healthy.record_outcome(_outcome())
    good = healthy.get_recommendation("anthropic", "m", "action")
    for _ in range(5):
        healthy.record_outcome(
            RLOutcome(
                provider="anthropic",
                model="stuck",
                task_type="action",
                success=False,
                quality_score=0.1,
                metadata={"stuck_loop_detected": True, "max_prompts_configured": 6},
            )
        )
    bad = healthy.get_recommendation("anthropic", "stuck", "action")
    assert bad.confidence < good.confidence


def test_reward_shaping(learner):
    assert learner._compute_reward(_outcome(success=True, quality=1.0)) == pytest.approx(1.0)
    stuck = learner._compute_reward(_outcome(stuck_loop_detected=True))
    forced = learner._compute_reward(_outcome(forced_completion=True))
    assert stuck < forced < learner._compute_reward(_outcome())
    assert (
        learner._compute_reward(
            _outcome(success=False, quality=0.0, stuck_loop_detected=True, forced_completion=True)
        )
        == -1.0
    )
