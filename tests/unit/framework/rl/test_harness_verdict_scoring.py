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

"""A graded session must be scored by its grade, not by how tidily it ran.

``completion_score`` was computed as ``1 - failure_rate * 1.5`` from tool-call
failures — a proxy for *looking careful*, not for solving the task. Evaluation
sessions are graded by a harness, so for those the verdict is available and must
win. An artifact on disk shows the two disagreeing outright: ``"status":
"failed"`` alongside ``"completion_score": "1.0"``.

Sessions with no verdict (interactive work, which nothing grades) keep the
proxy, now labelled as such on the trace.
"""

import json

import pytest

from victor.framework.rl.learners.prompt_optimizer import (
    HarnessVerdict,
    PromptOptimizerLearner,
)


def _task(**overrides):
    task = {
        "task_id": "astropy__astropy-1",
        "session_id": "sess-1",
        "status": "passed",
        "tests_passed": 4,
        "tests_total": 4,
    }
    task.update(overrides)
    return task


class TestVerdictFromTask:
    def test_passed_scores_one(self):
        verdict = PromptOptimizerLearner._verdict_from_task(_task(), "swe_bench")
        assert verdict == HarnessVerdict(1.0, True, "astropy__astropy-1", "swe_bench")

    def test_failed_with_no_tests_scores_zero(self):
        verdict = PromptOptimizerLearner._verdict_from_task(
            _task(status="failed", tests_passed=0, tests_total=0), "swe_bench"
        )
        assert verdict.completion_score == 0.0
        assert verdict.success is False

    def test_partial_test_pass_earns_partial_credit(self):
        # Fixing 8 of 10 tests is better evidence than fixing none.
        verdict = PromptOptimizerLearner._verdict_from_task(
            _task(status="failed", tests_passed=8, tests_total=10), "swe_bench"
        )
        assert verdict.completion_score == pytest.approx(0.8)
        assert verdict.success is False

    def test_a_non_passing_run_never_scores_one(self):
        verdict = PromptOptimizerLearner._verdict_from_task(
            _task(status="failed", tests_passed=10, tests_total=10), "swe_bench"
        )
        assert verdict.completion_score < 1.0
        assert verdict.success is False

    def test_soft_completion_score_is_ignored(self):
        # The exact contradiction observed on disk. Reading the artifact's own
        # completion_score back would reimport the proxy this change replaces.
        verdict = PromptOptimizerLearner._verdict_from_task(
            _task(status="failed", tests_passed=0, tests_total=0, completion_score="1.0"),
            "dr3_eval",
        )
        assert verdict.completion_score == 0.0

    def test_error_status_scores_zero(self):
        verdict = PromptOptimizerLearner._verdict_from_task(
            _task(status="error", tests_passed=0, tests_total=0), "swe_bench"
        )
        assert verdict.completion_score == 0.0
        assert verdict.success is False

    def test_missing_session_id_cannot_be_joined(self):
        assert PromptOptimizerLearner._verdict_from_task(_task(session_id=""), "b") is None
        assert PromptOptimizerLearner._verdict_from_task(_task(session_id=None), "b") is None

    def test_missing_status_cannot_be_graded(self):
        assert PromptOptimizerLearner._verdict_from_task(_task(status=""), "b") is None

    def test_unparseable_test_counts_fall_back_to_zero(self):
        verdict = PromptOptimizerLearner._verdict_from_task(
            _task(status="failed", tests_passed="lots", tests_total="many"), "b"
        )
        assert verdict.completion_score == 0.0


class TestScoreSession:
    def test_harness_verdict_wins(self):
        verdict = HarnessVerdict(0.8, False, "t", "swe_bench")
        # A flawless tool-call record would have scored this 1.0 via the proxy.
        assert PromptOptimizerLearner._score_session(verdict, 0.0) == (0.8, False, "harness")

    def test_verdict_overrides_a_tidy_run_that_solved_nothing(self):
        verdict = HarnessVerdict(0.0, False, "t", "swe_bench")
        score, success, source = PromptOptimizerLearner._score_session(verdict, 0.0)
        assert (score, success, source) == (0.0, False, "harness")

    def test_verdict_rewards_a_messy_run_that_succeeded(self):
        # Three recoverable errors on the way to a passing patch still passed.
        verdict = HarnessVerdict(1.0, True, "t", "swe_bench")
        assert PromptOptimizerLearner._score_session(verdict, 0.75) == (1.0, True, "harness")

    def test_ungraded_session_keeps_the_proxy(self):
        score, success, source = PromptOptimizerLearner._score_session(None, 0.2)
        assert source == "tool_failure_proxy"
        assert success is True
        assert score == pytest.approx(0.7)

    def test_proxy_never_goes_negative(self):
        score, success, source = PromptOptimizerLearner._score_session(None, 1.0)
        assert score == 0.0
        assert success is False
        assert source == "tool_failure_proxy"


class TestHarnessVerdictLoading:
    def _artifact(self, tmp_path, name, payload):
        path = tmp_path / name
        path.write_text(json.dumps(payload))
        return path

    def _learner(self):
        learner = PromptOptimizerLearner.__new__(PromptOptimizerLearner)
        learner._harness_verdict_cache = None
        return learner

    def test_loads_verdicts_keyed_by_session(self, tmp_path):
        self._artifact(
            tmp_path,
            "eval_swe_bench_1.json",
            {
                "config": {"benchmark": "swe_bench"},
                "tasks": [
                    _task(session_id="s1", status="passed"),
                    _task(session_id="s2", status="failed", tests_passed=1, tests_total=4),
                ],
            },
        )
        verdicts = self._learner()._harness_verdicts(eval_dir=tmp_path)
        assert verdicts["s1"].completion_score == 1.0
        assert verdicts["s2"].completion_score == pytest.approx(0.25)
        assert verdicts["s2"].benchmark == "swe_bench"

    def test_tasks_without_a_session_id_are_skipped(self, tmp_path):
        self._artifact(
            tmp_path,
            "eval_x_1.json",
            {"tasks": [_task(session_id=""), _task(session_id="s1")]},
        )
        verdicts = self._learner()._harness_verdicts(eval_dir=tmp_path)
        assert list(verdicts) == ["s1"]

    def test_unreadable_and_malformed_artifacts_are_tolerated(self, tmp_path):
        (tmp_path / "eval_broken.json").write_text("{not json")
        (tmp_path / "eval_list.json").write_text("[1, 2, 3]")
        self._artifact(tmp_path, "eval_ok.json", {"tasks": [_task(session_id="s1")]})
        assert list(self._learner()._harness_verdicts(eval_dir=tmp_path)) == ["s1"]

    def test_missing_directory_yields_no_verdicts(self, tmp_path):
        assert self._learner()._harness_verdicts(eval_dir=tmp_path / "nope") == {}

    def test_result_is_cached_across_calls(self, tmp_path):
        self._artifact(tmp_path, "eval_a.json", {"tasks": [_task(session_id="s1")]})
        learner = self._learner()
        first = learner._harness_verdicts(eval_dir=tmp_path)
        # A later artifact must not appear: trace collection calls this per
        # collector and should not rescan thousands of files each time.
        self._artifact(tmp_path, "eval_b.json", {"tasks": [_task(session_id="s2")]})
        assert learner._harness_verdicts(eval_dir=tmp_path) is first
