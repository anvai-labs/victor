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

"""Approval must mean "better than the prompt we ship", not "cleared 50%".

The old gate compared a candidate's absolute pass rate to a fixed threshold, so
it approved candidates that were *worse* than the seed whenever the benchmark
was easy, and rejected ones that beat the seed whenever it was hard. Neither
outcome is about the candidate.
"""

import sqlite3

import pytest

from victor.agent.optimization_injector import BASELINE_CANDIDATE_HASH
from victor.evaluation.harness import (
    PromptCandidateEvaluationRun,
    PromptCandidateEvaluationSpec,
    PromptCandidateEvaluationSuiteResult,
)
from victor.evaluation.protocol import (
    BenchmarkType,
    EvaluationConfig,
    EvaluationResult,
    TaskResult,
    TaskStatus,
)
from victor.framework.rl.learners.prompt_optimizer import (
    PromptCandidate,
    PromptOptimizerLearner,
)

SECTION = "COMPLETION_GUIDANCE"
PROVIDER = "zai"


def base_config() -> EvaluationConfig:
    return EvaluationConfig(benchmark=BenchmarkType.HUMAN_EVAL, model="glm-5.2", provider=PROVIDER)


def make_arm(candidate_hash: str, outcomes: list[bool]) -> PromptCandidateEvaluationRun:
    """One arm over tasks task-0..N, passing where ``outcomes`` is True."""
    config = EvaluationConfig(
        benchmark=BenchmarkType.HUMAN_EVAL,
        model="glm-5.2",
        provider=PROVIDER,
        prompt_candidate_hash=candidate_hash,
        prompt_section_name=SECTION,
    )
    return PromptCandidateEvaluationRun(
        spec=PromptCandidateEvaluationSpec(
            section_name=SECTION, prompt_candidate_hash=candidate_hash, provider=PROVIDER
        ),
        config=config,
        result=EvaluationResult(
            config=config,
            task_results=[
                TaskResult(
                    task_id=f"task-{i}",
                    status=TaskStatus.PASSED if ok else TaskStatus.FAILED,
                )
                for i, ok in enumerate(outcomes)
            ],
        ),
        label=f"{SECTION}:{candidate_hash}",
    )


def suite(*arms: PromptCandidateEvaluationRun) -> PromptCandidateEvaluationSuiteResult:
    return PromptCandidateEvaluationSuiteResult(base_config=base_config(), runs=list(arms))


def outcomes(pattern: str) -> list[bool]:
    """'PPFF' -> [True, True, False, False]. Keeps the arms readable."""
    return [ch == "P" for ch in pattern]


@pytest.fixture
def learner():
    """A learner that already knows the candidate hashes the arms will name.

    ``sync_evaluation_suite`` only approves a candidate it can record against, so
    an arm for an unknown hash is dropped before any gate runs.
    """
    conn = sqlite3.connect(":memory:")
    try:
        learner = PromptOptimizerLearner(name="test", db_connection=conn)
        learner._candidates[learner._candidate_key(SECTION, PROVIDER)] = [
            PromptCandidate(
                section_name=SECTION,
                provider=PROVIDER,
                text=f"candidate {name}",
                text_hash=name,
                generation=1,
                parent_hash="seed",
                requires_benchmark=True,
            )
            for name in ("cand-worse", "cand-better", "cand-marginal", "cand-tied", "cand-a")
        ]
        yield learner
    finally:
        conn.close()


class TestTheGateIsComparative:
    def test_a_winner_worse_than_the_seed_is_rejected_however_high_its_rate(self, learner):
        """The failure the absolute threshold could not see.

        The candidate passes 60% — comfortably over a 0.5 bar — while losing to
        the seed on the same tasks. The old gate approved exactly this.
        """
        baseline = make_arm(BASELINE_CANDIDATE_HASH, outcomes("PPPPPPPPFF"))
        weaker = make_arm("cand-worse", outcomes("PPPPPPFFFF"))

        result = learner.sync_evaluation_suite(suite(baseline, weaker), min_discordant=2)

        decision = result.decisions[0]
        assert decision.score == pytest.approx(0.6), "clears the old absolute bar"
        assert decision.paired_contrast.effect == -2
        assert decision.passed is False
        assert result.approved_prompt_candidate_hash is None

    def test_a_candidate_that_beats_the_seed_on_shared_tasks_is_approved(self, learner):
        baseline = make_arm(BASELINE_CANDIDATE_HASH, outcomes("PFFFFFFFFF"))
        better = make_arm("cand-better", outcomes("PPPPPPPFFF"))

        result = learner.sync_evaluation_suite(suite(baseline, better), min_discordant=6)

        decision = result.decisions[0]
        assert decision.paired_contrast.effect == 6
        assert decision.passed is True
        assert result.approved_prompt_candidate_hash == "cand-better"

    def test_too_few_disagreements_is_not_a_result(self, learner):
        """A one-task lead over forty tasks is noise, whichever way it leans."""
        baseline = make_arm(BASELINE_CANDIDATE_HASH, outcomes("PPPPPPPPPF"))
        marginal = make_arm("cand-marginal", outcomes("PPPPPPPPPP"))

        result = learner.sync_evaluation_suite(suite(baseline, marginal), min_discordant=8)

        decision = result.decisions[0]
        assert decision.paired_contrast.effect == 1
        assert decision.paired_contrast.discordant == 1
        assert decision.passed is False, "one discordant task cannot approve a prompt"

    def test_a_tie_does_not_pass(self, learner):
        baseline = make_arm(BASELINE_CANDIDATE_HASH, outcomes("PPFFPPFF"))
        tied = make_arm("cand-tied", outcomes("PPFFFFPP"))

        result = learner.sync_evaluation_suite(suite(baseline, tied), min_discordant=2)

        assert result.decisions[0].paired_contrast.effect == 0
        assert result.decisions[0].passed is False


class TestTheBaselineIsNeverACandidate:
    def test_the_seed_is_not_ranked_scored_or_approved(self, learner):
        """Ranking the referent alongside the arms lets it win its own experiment."""
        baseline = make_arm(BASELINE_CANDIDATE_HASH, outcomes("PPPPPPPPPP"))
        candidate = make_arm("cand-a", outcomes("PPPPPFFFFF"))

        result = learner.sync_evaluation_suite(suite(baseline, candidate), min_discordant=2)

        hashes = [d.prompt_candidate_hash for d in result.decisions]
        assert BASELINE_CANDIDATE_HASH not in hashes
        assert result.best_prompt_candidate_hash == "cand-a"
        assert result.promoted_prompt_candidate_hash is None

    def test_a_suite_of_only_a_baseline_decides_nothing(self, learner):
        result = learner.sync_evaluation_suite(suite(make_arm(BASELINE_CANDIDATE_HASH, [True])))
        assert result.decisions == []
        assert result.approved_prompt_candidate_hash is None


class TestWithoutABaseline:
    def test_it_falls_back_to_the_absolute_gate_and_says_so(self, learner, caplog):
        candidate = make_arm("cand-a", outcomes("PPPF"))

        with caplog.at_level("WARNING"):
            result = learner.sync_evaluation_suite(suite(candidate), min_pass_rate=0.5)

        assert result.decisions[0].paired_contrast is None
        assert result.decisions[0].passed is True
        assert "no baseline arm" in caplog.text
        assert "--include-baseline" in caplog.text


class TestReporting:
    def test_the_contrast_reaches_the_serialized_output(self, learner):
        baseline = make_arm(BASELINE_CANDIDATE_HASH, outcomes("PFFFFFFFFF"))
        better = make_arm("cand-better", outcomes("PPPPPPPFFF"))

        payload = learner.sync_evaluation_suite(suite(baseline, better), min_discordant=6).to_dict()

        contrast = payload["decisions"][0]["paired_contrast"]
        assert contrast["effect"] == 6
        assert contrast["n_paired"] == 10
        assert "mcnemar_p" in contrast

    def test_a_run_without_a_contrast_serializes_as_null(self, learner):
        payload = learner.sync_evaluation_suite(suite(make_arm("cand-a", [True]))).to_dict()
        assert payload["decisions"][0]["paired_contrast"] is None
