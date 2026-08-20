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

"""A statistical verdict is only meaningful when the measurement was valid.

The FEP-0025 loop needed a human in the loop because four failure modes all
read as an ambiguous red: a result that was never recorded (cross-provider
attribution), an arm that died to quota, arms that ran on different providers,
and a measurement on the wrong tree. Each looks like "the candidate lost" when
the truth is "the measurement was broken." A CI gate cannot route an ambiguous
red, so it cannot be automated.

The fix is precedence: assert validity first, and only apply the statistical
outcome to a valid run. Every invalid run becomes a *classified* red that
routes to a specific fix (re-run, re-run-with-quota, fix-infra), never a
statistical verdict. Then green and refused-statistical are the only remaining
outcomes, and both are unambiguous.
"""

from __future__ import annotations

from victor.evaluation.suite_verdict import (
    ArmValidity,
    SuiteVerdict,
    classify_suite_verdict,
)


def _arms(*providers: str, total: int = 60, throttled: int = 0) -> list[ArmValidity]:
    return [
        ArmValidity(label=f"arm-{p}", provider=p, total_tasks=total, throttled_tasks=throttled)
        for p in providers
    ]


class TestValidityPrecedesStatistics:
    """An invalid run never produces a statistical verdict."""

    def test_green_requires_a_valid_run_and_a_statistical_win(self):
        verdict, _ = classify_suite_verdict(
            _arms("deepseek", "deepseek"),
            has_baseline=True,
            best_recorded=True,
            approved=True,
            min_tasks=60,
        )
        assert verdict is SuiteVerdict.GREEN

    def test_a_valid_run_that_lost_is_refused_not_red_infra(self):
        verdict, reason = classify_suite_verdict(
            _arms("deepseek", "deepseek"),
            has_baseline=True,
            best_recorded=True,
            approved=False,
            min_tasks=60,
        )
        assert verdict is SuiteVerdict.RED_REFUSED
        assert "did not clear" in reason

    def test_no_baseline_is_not_a_statistical_loss(self):
        verdict, _ = classify_suite_verdict(
            _arms("deepseek"), has_baseline=False, best_recorded=True, approved=False, min_tasks=60
        )
        assert verdict is SuiteVerdict.RED_NO_BASELINE


class TestTheSilentFailureModesBecomeClassifiedReds:
    """Each ambiguity that needed a human now names itself."""

    def test_an_unrecorded_winner_is_infra_not_a_refusal(self):
        """The #718 defect: result couldn't be attributed (cross-provider),
        recorded=False forced decision.passed false. It must read as infra."""
        verdict, reason = classify_suite_verdict(
            _arms("deepseek", "deepseek"),
            has_baseline=True,
            best_recorded=False,
            approved=False,
            min_tasks=60,
        )
        assert verdict is SuiteVerdict.RED_UNRECORDED
        assert "not attributable" in reason

    def test_an_incomplete_arm_is_red_incomplete(self):
        arms = [
            ArmValidity(label="baseline", provider="deepseek", total_tasks=60, throttled_tasks=0),
            ArmValidity(label="cand", provider="deepseek", total_tasks=12, throttled_tasks=0),
        ]
        verdict, _ = classify_suite_verdict(
            arms, has_baseline=True, best_recorded=True, approved=False, min_tasks=60
        )
        assert verdict is SuiteVerdict.RED_INCOMPLETE

    def test_a_quota_killed_arm_is_red_throttled(self):
        verdict, _ = classify_suite_verdict(
            _arms("deepseek", "deepseek", total=60, throttled=55),
            has_baseline=True,
            best_recorded=True,
            approved=False,
            min_tasks=60,
        )
        assert verdict is SuiteVerdict.RED_THROTTLED

    def test_mixed_provider_arms_are_an_invalid_contrast(self):
        verdict, _ = classify_suite_verdict(
            [ArmValidity("baseline", "moonshot", 60, 0), ArmValidity("cand", "deepseek", 60, 0)],
            has_baseline=True,
            best_recorded=True,
            approved=True,
            min_tasks=60,
        )
        assert verdict is SuiteVerdict.RED_CROSS_PROVIDER


class TestPrecedenceOrder:
    """Validity is checked in an order that names the most likely culprit."""

    def test_throttle_beats_incomplete_when_an_arm_is_both(self):
        # A quota-killed arm is also incomplete (few good tasks); throttle is the
        # cause, so it must name throttle, not the symptom (incompleteness).
        verdict, _ = classify_suite_verdict(
            _arms("deepseek", "deepseek", total=12, throttled=12),
            has_baseline=True,
            best_recorded=True,
            approved=False,
            min_tasks=60,
        )
        assert verdict is SuiteVerdict.RED_THROTTLED

    def test_cross_provider_beats_statistics_even_if_recorded_and_approved(self):
        verdict, _ = classify_suite_verdict(
            [ArmValidity("baseline", "moonshot", 60, 0), ArmValidity("cand", "deepseek", 60, 0)],
            has_baseline=True,
            best_recorded=True,
            approved=True,
            min_tasks=60,
        )
        assert verdict is SuiteVerdict.RED_CROSS_PROVIDER
