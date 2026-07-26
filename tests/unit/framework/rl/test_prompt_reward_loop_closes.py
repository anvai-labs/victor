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

"""The FEP-0017 loop must actually turn, end to end.

Each link — the serve gate, the served-hash ledger, reward resolution — has its
own unit test. None of them asserts the property the FEP exists to guarantee:
that a candidate which starts inert (``alpha=beta=1, sample_count=0``, the state
GEPA writes) can be served, rewarded, and end up served again *on merit* rather
than by exploration.

That gap mattered: a prompt-evolution audit concluded from database state alone
that the loop was still open, when in fact it closes — the candidates it looked
at were blocked by the benchmark gate pinned at the bottom of this module. A
test of the whole cycle is what distinguishes "the loop is broken" from "these
candidates never entered it".
"""

import random
import sqlite3
from datetime import datetime

import pytest

from victor.framework.rl.base import RLOutcome
from victor.framework.rl.learners.prompt_optimizer import (
    PromptCandidate,
    PromptOptimizerLearner,
    should_serve_candidate,
)

SECTION = "COMPLETION_GUIDANCE"
PROVIDER = "moonshot"
CANDIDATE_HASH = "cafebabe1234"


@pytest.fixture
def learner():
    conn = sqlite3.connect(":memory:")
    try:
        yield PromptOptimizerLearner(name="loop_test", db_connection=conn)
    finally:
        conn.close()


def _fresh_candidate(learner, **overrides) -> PromptCandidate:
    """A candidate in exactly the state ``evolve()`` leaves behind."""
    candidate = PromptCandidate(
        section_name=SECTION,
        provider=PROVIDER,
        text="TASK COMPLETION (MANDATORY): signal completion once.",
        text_hash=CANDIDATE_HASH,
        generation=1,
        parent_hash="deadbeef0000",
    )
    for key, value in overrides.items():
        setattr(candidate, key, value)
    learner._candidates.setdefault(learner._candidate_key(SECTION, PROVIDER), []).append(candidate)
    return candidate


def _recommend(learner):
    return learner.get_recommendation(
        provider=PROVIDER, model="kimi-k3", task_type="default", section_name=SECTION
    )


def _reward(learner, *, success: bool, score: float) -> None:
    learner.record_outcome(
        RLOutcome(
            provider=PROVIDER,
            model="kimi-k3",
            task_type="default",
            success=success,
            quality_score=score,
            timestamp=datetime.now(),
            metadata={"prompt_section": SECTION, "prompt_candidate_hash": CANDIDATE_HASH},
        )
    )


class _DrawsLow:
    """An RNG whose draw always succeeds — isolates the gate from chance.

    A pending candidate's epsilon is scaled by
    PENDING_BENCHMARK_EXPLORATION_FACTOR, so even ``exploration_epsilon=1.0``
    leaves a coin flip. These tests assert *reachability*, not probability;
    the rate itself is covered by test_pending_explores_at_a_reduced_rate.
    """

    @staticmethod
    def random() -> float:
        return 0.0


class TestLoopCloses:
    def test_a_fresh_candidate_starts_inert(self, learner):
        candidate = _fresh_candidate(learner)
        assert (candidate.alpha, candidate.beta_val, candidate.sample_count) == (1.0, 1.0, 0)

    def test_exploration_serves_what_confidence_alone_never_would(self, learner):
        _fresh_candidate(learner)
        rec = _recommend(learner)
        # The chicken-and-egg the FEP names: no samples means no confidence,
        # so the exploit path can never bootstrap a new candidate.
        assert (
            should_serve_candidate(rec, exploration_enabled=False, exploration_epsilon=0.0) is False
        )
        assert (
            should_serve_candidate(rec, exploration_enabled=True, exploration_epsilon=1.0) is True
        )

    def test_serve_then_reward_moves_the_posterior(self, learner):
        candidate = _fresh_candidate(learner)
        rec = _recommend(learner)
        learner.record_served(SECTION, PROVIDER, rec.metadata["prompt_candidate_hash"])

        _reward(learner, success=True, score=0.9)

        assert candidate.sample_count == 1
        assert candidate.alpha > 1.0
        assert candidate.scores["completion_score"] > 0.0

    def test_full_cycle_ends_with_the_candidate_served_on_merit(self, learner):
        """Inert → explored → rewarded → served by confidence, exploration off."""
        _fresh_candidate(learner)
        rec = _recommend(learner)
        assert should_serve_candidate(rec, exploration_enabled=True, exploration_epsilon=1.0)
        learner.record_served(SECTION, PROVIDER, rec.metadata["prompt_candidate_hash"])

        for _ in range(6):
            _reward(learner, success=True, score=0.9)

        proven = _recommend(learner)
        assert proven.is_baseline is False
        # Exploration OFF: this is the exploit path standing on earned evidence.
        assert (
            should_serve_candidate(proven, exploration_enabled=False, exploration_epsilon=0.0)
            is True
        )

    def test_failures_push_the_posterior_the_other_way(self, learner):
        candidate = _fresh_candidate(learner)
        rec = _recommend(learner)
        learner.record_served(SECTION, PROVIDER, rec.metadata["prompt_candidate_hash"])

        for _ in range(5):
            _reward(learner, success=False, score=0.1)

        assert candidate.beta_val > candidate.alpha
        assert candidate.sample_count == 5

    def test_benchmark_approved_candidate_serves_without_any_samples(self, learner):
        _fresh_candidate(learner, benchmark_passed=True, benchmark_runs=4)
        rec = _recommend(learner)
        assert rec.metadata["benchmark_passed"] is True
        # The suite-validation path must not be blocked by is_baseline.
        assert (
            should_serve_candidate(rec, exploration_enabled=False, exploration_epsilon=0.0) is True
        )


class TestBenchmarkGateHasAnExit:
    """A benchmark-gated candidate must be able to earn its way in.

    Excluding pending candidates from recommendation entirely made the gate a
    wall: nothing in the runtime benchmarks them, so the only exit was an
    operator running ``victor benchmark --prompt-candidate-hash …`` by hand. On
    a default config that stranded three of nine evolvable sections
    (CONCISE_MODE_GUIDANCE, COMPLETION_GUIDANCE, GROUNDING_RULES) — and is why
    two of the three candidates in the 2026-07-25 audit sat permanently at
    ``sample_count=0``.

    The gate now lives in the *serve* decision: pending candidates are reachable
    through exploration alone, and only when nothing servable exists.
    """

    def test_pending_candidate_is_recommendable(self, learner):
        _fresh_candidate(learner, requires_benchmark=True, benchmark_passed=False)
        assert _recommend(learner) is not None

    def test_pending_candidate_is_never_served_on_merit(self, learner):
        candidate = _fresh_candidate(learner, requires_benchmark=True, benchmark_passed=False)
        # Even with a posterior that would make any normal candidate proven.
        candidate.alpha, candidate.sample_count = 50.0, 40
        rec = _recommend(learner)
        assert rec.confidence > 0.6
        assert (
            should_serve_candidate(rec, exploration_enabled=False, exploration_epsilon=0.0) is False
        )

    def test_exploration_can_reach_it(self, learner):
        _fresh_candidate(learner, requires_benchmark=True, benchmark_passed=False)
        rec = _recommend(learner)
        assert (
            should_serve_candidate(
                rec, exploration_enabled=True, exploration_epsilon=1.0, rng=_DrawsLow()
            )
            is True
        )

    def test_pending_explores_at_a_reduced_rate(self, learner):
        """Unvalidated candidates reach live traffic more rarely than merely-unsampled ones."""
        _fresh_candidate(learner, requires_benchmark=True, benchmark_passed=False)
        rec = _recommend(learner)
        rng = random.Random(0)
        served = sum(
            should_serve_candidate(rec, exploration_enabled=True, exploration_epsilon=0.4, rng=rng)
            for _ in range(4000)
        )
        # 0.4 * PENDING_BENCHMARK_EXPLORATION_FACTOR (0.5) = 0.2
        assert 0.15 < served / 4000 < 0.25

    def test_a_passing_benchmark_restores_the_merit_path(self, learner):
        candidate = _fresh_candidate(learner, requires_benchmark=True, benchmark_passed=False)
        rec = _recommend(learner)
        assert (
            should_serve_candidate(rec, exploration_enabled=False, exploration_epsilon=0.0) is False
        )

        candidate.benchmark_passed = True

        rec = _recommend(learner)
        assert (
            should_serve_candidate(rec, exploration_enabled=False, exploration_epsilon=0.0) is True
        )

    def test_a_validated_candidate_is_never_displaced_by_a_pending_one(self, learner):
        """An approved candidate must keep priority — pending is a fallback, not a peer."""
        _fresh_candidate(learner, requires_benchmark=True, benchmark_passed=False)
        approved = PromptCandidate(
            section_name=SECTION,
            provider=PROVIDER,
            text="benchmark-approved text",
            text_hash="approved00001",
            generation=2,
            parent_hash="deadbeef0000",
            benchmark_passed=True,
            benchmark_runs=3,
        )
        learner._candidates[learner._candidate_key(SECTION, PROVIDER)].append(approved)

        rec = _recommend(learner)
        assert rec.metadata["prompt_candidate_hash"] == "approved00001"

    def test_pending_can_complete_the_whole_loop(self, learner):
        """The property the deadlock denied: inert + gated -> served -> rewarded."""
        candidate = _fresh_candidate(learner, requires_benchmark=True, benchmark_passed=False)
        rec = _recommend(learner)
        assert should_serve_candidate(
            rec, exploration_enabled=True, exploration_epsilon=1.0, rng=_DrawsLow()
        )
        learner.record_served(SECTION, PROVIDER, rec.metadata["prompt_candidate_hash"])

        _reward(learner, success=True, score=0.9)

        assert candidate.sample_count == 1
        assert candidate.alpha > 1.0
