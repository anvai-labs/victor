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

"""A prompt candidate has to be compared with something to be worth promoting.

Candidates were scored on their own absolute pass rate against a fixed
threshold, which cannot answer the question being asked: a candidate could clear
50% while being worse than the prompt it replaces. Both arms run the identical
task set, so pairing cancels task difficulty and only the disagreements carry
signal.
"""

import pytest

from victor.evaluation.harness import PairedContrast
from victor.evaluation.protocol import EvaluationResult, TaskResult, TaskStatus


def arm(**task_outcomes: bool) -> EvaluationResult:
    """An evaluation arm from ``task_id=passed`` pairs."""
    return EvaluationResult(
        config=None,
        task_results=[
            TaskResult(
                task_id=task_id,
                status=TaskStatus.PASSED if passed else TaskStatus.FAILED,
            )
            for task_id, passed in task_outcomes.items()
        ],
    )


def contrast(variant_only: int, baseline_only: int, both: int = 0, neither: int = 0):
    return PairedContrast(
        n_paired=variant_only + baseline_only + both + neither,
        variant_only_pass=variant_only,
        baseline_only_pass=baseline_only,
        both_pass=both,
        both_fail=neither,
    )


class TestPairing:
    def test_it_counts_the_four_cells(self):
        baseline = arm(t1=True, t2=True, t3=False, t4=False)
        variant = arm(t1=True, t2=False, t3=True, t4=False)

        c = PairedContrast.from_results(baseline, variant)

        assert (c.n_paired, c.both_pass, c.both_fail) == (4, 1, 1)
        assert c.variant_only_pass == 1  # t3
        assert c.baseline_only_pass == 1  # t2
        assert c.effect == 0

    def test_only_shared_tasks_are_paired(self):
        """An arm that ran extra tasks must not contribute them unpaired."""
        baseline = arm(t1=True, t2=False, only_in_baseline=True)
        variant = arm(t1=True, t2=True, only_in_variant=False)

        c = PairedContrast.from_results(baseline, variant)

        assert c.n_paired == 2
        assert c.effect == 1

    def test_disjoint_task_sets_raise_rather_than_read_as_a_tie(self):
        """Silently returning zeros would be indistinguishable from a real tie."""
        with pytest.raises(ValueError, match="shared task set"):
            PairedContrast.from_results(arm(a=True), arm(b=True))

    def test_a_later_attempt_at_the_same_task_wins(self):
        """Resumed runs append rather than replace."""
        result = EvaluationResult(
            config=None,
            task_results=[
                TaskResult(task_id="t1", status=TaskStatus.FAILED),
                TaskResult(task_id="t1", status=TaskStatus.PASSED),
            ],
        )
        c = PairedContrast.from_results(arm(t1=False), result)
        assert c.variant_only_pass == 1


class TestMcNemar:
    """Exact binomial on the discordant split; verified against scipy."""

    @pytest.mark.parametrize(
        "variant_only,baseline_only,expected",
        [
            (0, 0, 1.0),
            (5, 5, 1.0),
            (1, 0, 1.0),
            (10, 0, 0.001953125),
            (8, 1, 0.0390625),
            (11, 3, 0.057373046875),
        ],
    )
    def test_exact_two_sided_p(self, variant_only, baseline_only, expected):
        assert contrast(variant_only, baseline_only).mcnemar_p == pytest.approx(expected)

    def test_concordant_tasks_do_not_move_the_p_value(self):
        """Task difficulty cancels; only disagreements carry information."""
        bare = contrast(8, 1)
        padded = contrast(8, 1, both=200, neither=200)
        assert bare.mcnemar_p == padded.mcnemar_p

    def test_direction_does_not_change_significance(self):
        assert contrast(3, 11).mcnemar_p == contrast(11, 3).mcnemar_p
        assert contrast(3, 11).effect == -8

    def test_p_never_exceeds_one(self):
        for v in range(6):
            assert contrast(v, v).mcnemar_p <= 1.0

    def test_summary_reads_as_a_decision_aid(self):
        assert contrast(11, 3, both=20, neither=6).summary() == "+8/40 (disc 11/3, p=0.06)"
        assert contrast(1, 5).summary().startswith("-4/")


class TestSuiteArtifactsCarryPerTaskOutcomes:
    """A saved suite must be pairable, or the contrast only works in memory.

    ``to_dict`` wrote aggregate metrics alone while ``from_dict`` read a
    ``task_results`` key nobody produced, so a suite round-tripped to zero tasks.
    Pass rates cannot answer a paired question: pairing needs to know *which*
    tasks each arm passed.
    """

    @staticmethod
    def _suite():
        from victor.evaluation.harness import (
            PromptCandidateEvaluationRun,
            PromptCandidateEvaluationSpec,
            PromptCandidateEvaluationSuiteResult,
        )
        from victor.evaluation.protocol import BenchmarkType, EvaluationConfig

        def one(candidate_hash, outcomes):
            config = EvaluationConfig(
                benchmark=BenchmarkType.HUMAN_EVAL,
                model="m",
                provider="p",
                prompt_candidate_hash=candidate_hash,
                prompt_section_name="S",
            )
            return PromptCandidateEvaluationRun(
                spec=PromptCandidateEvaluationSpec(
                    section_name="S", prompt_candidate_hash=candidate_hash
                ),
                config=config,
                result=EvaluationResult(
                    config=config,
                    task_results=[
                        TaskResult(
                            task_id=f"t{i}",
                            status=TaskStatus.PASSED if ok else TaskStatus.FAILED,
                        )
                        for i, ok in enumerate(outcomes)
                    ],
                ),
                label=f"S:{candidate_hash}",
            )

        baseline = one("__baseline__", [True, False, False, True])
        candidate = one("cand", [True, True, True, False])
        return (
            PromptCandidateEvaluationSuiteResult(
                base_config=baseline.config, runs=[baseline, candidate]
            ),
            baseline,
            candidate,
        )

    def test_per_task_outcomes_survive_the_round_trip(self):
        from victor.evaluation.harness import PromptCandidateEvaluationSuiteResult

        suite, _, _ = self._suite()
        restored = PromptCandidateEvaluationSuiteResult.from_dict(suite.to_dict())

        assert [len(run.result.task_results) for run in restored.runs] == [4, 4]

    def test_a_contrast_from_disk_equals_the_one_from_memory(self):
        from victor.evaluation.harness import PromptCandidateEvaluationSuiteResult

        suite, baseline, candidate = self._suite()
        restored = PromptCandidateEvaluationSuiteResult.from_dict(suite.to_dict())

        from_disk = PairedContrast.from_results(restored.runs[0].result, restored.runs[1].result)
        in_memory = PairedContrast.from_results(baseline.result, candidate.result)
        assert from_disk == in_memory

    def test_the_writer_is_the_inverse_of_the_reader(self):
        """One mapping, not two: a second copy drifts field by field."""
        from victor.evaluation.harness import task_result_from_artifact, task_result_to_artifact

        original = TaskResult(task_id="t1", status=TaskStatus.PASSED, tests_passed=3, tests_total=4)
        restored = task_result_from_artifact(task_result_to_artifact(original))

        assert restored.task_id == original.task_id
        assert restored.status == original.status
        assert (restored.tests_passed, restored.tests_total) == (3, 4)

    def test_the_real_run_runner_uses_the_shared_writer(self):
        from victor.evaluation.harness import task_result_to_artifact
        from victor.evaluation.real_run_runner import RealRunBenchmarkRunner

        task = TaskResult(task_id="t1", status=TaskStatus.PASSED)
        assert RealRunBenchmarkRunner._task_result_to_artifact(
            None, task
        ) == task_result_to_artifact(task)


class TestThrottledTasksAreNotEvidence:
    """A task the agent never got a response for says nothing about the prompt.

    Three arms run back to back exhausted one provider's quota and the last arm
    recorded 0/24 in 128 seconds — ten explicit 429s against a prompt that never
    ran. Scored as a loss, that gave an untested candidate a permanent "failed
    benchmark" record. It is the mutator's invisible 429 again: a quota failure
    wearing the costume of a quality signal.
    """

    @staticmethod
    def throttled(task_id: str) -> TaskResult:
        return TaskResult(
            task_id=task_id,
            status=TaskStatus.FAILED,
            error_message="[df4c8825] rate limited (429) (provider=moonshot)",
        )

    @staticmethod
    def ran(task_id: str, passed: bool) -> TaskResult:
        return TaskResult(
            task_id=task_id,
            status=TaskStatus.PASSED if passed else TaskStatus.FAILED,
            error_message="" if passed else "AssertionError",
        )

    def test_a_throttled_task_is_recognised(self):
        from victor.evaluation.harness import was_throttled

        assert was_throttled(self.throttled("t1")) is True
        assert was_throttled(self.ran("t2", False)) is False
        assert was_throttled(self.ran("t3", True)) is False

    def test_a_throttled_pair_is_dropped_not_counted_as_a_loss(self):
        baseline = EvaluationResult(
            config=None, task_results=[self.ran("t1", True), self.ran("t2", True)]
        )
        variant = EvaluationResult(
            config=None, task_results=[self.ran("t1", True), self.throttled("t2")]
        )

        c = PairedContrast.from_results(baseline, variant)

        assert c.n_paired == 1, "the throttled task must not be paired"
        assert c.baseline_only_pass == 0, "a 429 is not the candidate losing"
        assert c.effect == 0

    def test_the_valid_tasks_still_count(self):
        """Dropping the pair, not the arm — partial quota loss keeps its signal."""
        baseline = EvaluationResult(
            config=None,
            task_results=[self.ran("t1", False), self.ran("t2", False), self.ran("t3", True)],
        )
        variant = EvaluationResult(
            config=None,
            task_results=[self.ran("t1", True), self.ran("t2", True), self.throttled("t3")],
        )

        c = PairedContrast.from_results(baseline, variant)

        assert c.n_paired == 2
        assert c.variant_only_pass == 2
        assert c.effect == 2

    def test_an_arm_lost_entirely_to_throttling_raises_rather_than_scoring_zero(self):
        baseline = EvaluationResult(config=None, task_results=[self.ran("t1", True)])
        dead = EvaluationResult(config=None, task_results=[self.throttled("t1")])

        with pytest.raises(ValueError, match="shared task set"):
            PairedContrast.from_results(baseline, dead)


class TestALeadMustBeatChance:
    """Volume of disagreement is not the same question as asymmetry of it.

    The gate asked for a positive effect over enough discordant tasks, and
    approved a candidate at 8 versus 6: fourteen disagreements is ample
    evidence, and a lead of two is still a coin flip. The exact test scored it
    p=0.79. Under the null the effect has mean zero and standard deviation
    sqrt(discordant), so that is the bar a real lead has to clear.
    """

    @staticmethod
    def split(variant_only, baseline_only):
        return PairedContrast(
            n_paired=variant_only + baseline_only,
            variant_only_pass=variant_only,
            baseline_only_pass=baseline_only,
            both_pass=0,
            both_fail=0,
        )

    def test_the_floor_is_the_null_standard_deviation(self):
        assert self.split(7, 7).noise_floor == pytest.approx(14**0.5)
        assert self.split(5, 5).noise_floor == pytest.approx(10**0.5)

    @pytest.mark.parametrize("v,b", [(8, 6), (6, 4), (4, 5), (7, 7)])
    def test_leads_inside_the_noise_are_refused(self, v, b):
        """Every result this session landed here."""
        c = self.split(v, b)
        assert c.beats_noise() is False
        assert c.mcnemar_p > 0.5, "sanity: the exact test agrees these are noise"

    @pytest.mark.parametrize("v,b", [(8, 1), (10, 0), (11, 3), (25, 10)])
    def test_real_leads_still_pass(self, v, b):
        c = self.split(v, b)
        assert c.beats_noise() is True
        assert c.mcnemar_p < 0.2, "sanity: the exact test agrees these are signal"

    def test_a_tie_never_beats_the_floor(self):
        assert self.split(0, 0).beats_noise() is False
        assert self.split(6, 6).beats_noise() is False

    def test_a_loss_never_passes_however_lopsided(self):
        assert self.split(0, 10).beats_noise() is False

    def test_more_tasks_demand_a_bigger_absolute_lead(self):
        """The floor scales with evidence rather than being a fixed number."""
        assert self.split(3, 0).beats_noise() is True  # 3 vs floor 1.7
        assert self.split(3, 22).noise_floor > self.split(3, 0).noise_floor
