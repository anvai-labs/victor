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

"""SemanticThresholdLearner: similarity thresholds per embedding_model:task:tool."""

from pathlib import Path

import pytest

from victor.core.database import get_database, reset_database
from victor.framework.rl.base import RLOutcome
from victor.framework.rl.coordinator import RLCoordinator
from victor.framework.rl.learners.semantic_threshold import SemanticThresholdLearner


@pytest.fixture
def coordinator(tmp_path: Path):
    reset_database()
    db_path = tmp_path / "rl_test.db"
    get_database(db_path)
    yield RLCoordinator(storage_path=tmp_path, db_path=db_path)
    reset_database()


@pytest.fixture
def learner(coordinator) -> SemanticThresholdLearner:
    return coordinator.get_learner("semantic_threshold")


def _search(results=8, quality=0.8, threshold=0.5, tool="code_search", **flags) -> RLOutcome:
    return RLOutcome(
        provider="ignored",
        model="ignored",
        task_type="analysis",
        success=results > 0,
        quality_score=quality,
        metadata={
            "embedding_model": "bge-small",
            "tool_name": tool,
            "results_count": results,
            "threshold_used": threshold,
            **flags,
        },
    )


def test_is_wired_through_the_coordinator(learner):
    assert isinstance(learner, SemanticThresholdLearner)


def test_needs_five_searches_before_recommending(learner):
    for _ in range(4):
        learner.record_outcome(_search())
    assert learner.get_recommendation("bge-small", "code_search", "analysis") is None
    learner.record_outcome(_search())
    rec = learner.get_recommendation("bge-small", "code_search", "analysis")
    assert rec is not None and rec.sample_size == 5
    assert 0.0 < rec.value < 1.0


def test_contexts_are_keyed_by_model_task_and_tool(learner):
    for _ in range(5):
        learner.record_outcome(_search())
    assert learner.get_recommendation("bge-small", "semantic_code_search", "analysis") is None
    assert learner.get_recommendation("other-model", "code_search", "analysis") is None


def test_zero_result_searches_lower_confidence_and_threshold(learner):
    for _ in range(6):
        learner.record_outcome(_search())
    good = learner.get_recommendation("bge-small", "code_search", "analysis")
    for _ in range(6):
        learner.record_outcome(_search(results=0, quality=0.1, tool="t2", false_negatives=True))
    bad = learner.get_recommendation("bge-small", "t2", "analysis")
    assert bad.confidence < good.confidence
    assert bad.value <= good.value  # misses push the threshold down, never up


def test_rl_mode_none_skips_writes(learner, monkeypatch):
    import victor.config.settings as settings

    monkeypatch.setattr(settings, "load_settings", lambda: type("S", (), {"rl_mode": "none"})())
    for _ in range(6):
        learner.record_outcome(_search())
    assert learner.get_recommendation("bge-small", "code_search", "analysis") is None


def test_reward_shaping(learner):
    best = learner._compute_reward(_search(results=8, quality=1.0))
    miss = learner._compute_reward(_search(results=0, quality=0.0, false_negatives=True))
    noisy = learner._compute_reward(_search(results=8, quality=1.0, false_positives=True))
    assert miss < noisy < best
    assert -1.0 <= miss and best <= 1.0
