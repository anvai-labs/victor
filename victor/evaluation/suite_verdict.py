# Copyright 2026 Vijaykumar Singh <singhvjd@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""Classified verdict for a prompt-candidate benchmark suite.

A statistical verdict (the candidate passed/failed the comparative gate) is
only meaningful when the measurement underneath it was *valid*. This module
asserts validity first — baseline presence, throttle, completeness, provider
consistency, recording — and only then applies the statistical outcome,
producing one classified verdict.

Classification is what makes a CI gate automatable. The FEP-0025 loop needed a
human because four failure modes all read as an ambiguous "the candidate lost":
a result that was never recorded (cross-provider attribution, #718), an arm
that died to quota, arms that ran on different providers, and a measurement on
the wrong tree. Each is really "the measurement was broken", and each now names
itself, so a red routes to a specific fix (re-run, re-run-with-quota,
fix-infra) instead of a human diagnosis.

Usage: :func:`classify_suite_verdict` is a pure function over the per-arm
validity and the sync decision — easy to test and reason about.
:func:`verdict_from_sync` extracts those inputs from a real suite + sync result.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class SuiteVerdict(Enum):
    """A single, routable verdict for a benchmark suite.

    ``GREEN`` and ``RED_REFUSED`` are the only statistical outcomes, and both
    presuppose a valid measurement. Every other value is an invalid-measurement
    red that routes to a fix, never a statistical claim.
    """

    GREEN = "green"  # valid + cleared the comparative gate -> promote/merge
    RED_REFUSED = "red_refused"  # valid + did not clear -> re-evolve
    RED_INCOMPLETE = "red_incomplete"  # an arm did not reach min_tasks -> re-run
    RED_THROTTLED = "red_throttled"  # an arm was quota-killed -> re-run with quota
    RED_UNRECORDED = "red_unrecorded"  # winner result not attributable -> infra bug
    RED_CROSS_PROVIDER = "red_cross_provider"  # arms used different providers -> invalid contrast
    RED_NO_BASELINE = "red_no_baseline"  # no baseline arm -> no contrast possible


@dataclass(frozen=True)
class ArmValidity:
    """Validity facts about one arm, extracted from its run."""

    label: str
    provider: str
    total_tasks: int
    throttled_tasks: int


def classify_suite_verdict(
    arms: list[ArmValidity],
    *,
    has_baseline: bool,
    best_recorded: bool,
    approved: bool,
    min_tasks: int,
    max_throttled_share: float = 0.25,
) -> tuple[SuiteVerdict, str]:
    """Classify a suite as a trustworthy verdict plus a machine/human reason.

    Validity is checked before statistics, in order of the most likely culprit:
    a quota-killed arm is named ``RED_THROTTLED`` (the cause) rather than
    ``RED_INCOMPLETE`` (the symptom of running few tasks); a mixed-provider
    contrast is rejected even if a candidate happened to clear the gate. An
    invalid run never returns a statistical verdict, because the alternative —
    an infra failure reading as "candidate lost" — is the ambiguity that makes a
    gate impossible to automate.
    """
    if not has_baseline or len(arms) < 2:
        return SuiteVerdict.RED_NO_BASELINE, "no baseline arm; nothing to contrast against"

    for arm in arms:
        if arm.total_tasks and arm.throttled_tasks / arm.total_tasks > max_throttled_share:
            return (
                SuiteVerdict.RED_THROTTLED,
                f"{arm.label} was rate-limited on {arm.throttled_tasks}/{arm.total_tasks} tasks",
            )
    for arm in arms:
        if arm.total_tasks < min_tasks:
            return (
                SuiteVerdict.RED_INCOMPLETE,
                f"{arm.label} ran {arm.total_tasks}/{min_tasks} tasks",
            )

    providers = {arm.provider for arm in arms if arm.provider}
    if len(providers) > 1:
        return (
            SuiteVerdict.RED_CROSS_PROVIDER,
            f"arms used mixed providers: {sorted(providers)}",
        )

    if not best_recorded:
        return (
            SuiteVerdict.RED_UNRECORDED,
            "the best candidate's result was not attributable (recording failed)",
        )
    if approved:
        return SuiteVerdict.GREEN, "valid measurement; candidate cleared the comparative gate"
    return (
        SuiteVerdict.RED_REFUSED,
        "valid measurement; candidate did not clear the comparative gate",
    )


def verdict_from_sync(
    suite: Any,
    sync_result: Any,
    *,
    min_tasks: int,
    max_throttled_share: float = 0.25,
    baseline_hash: str = "__baseline__",
) -> tuple[SuiteVerdict, str]:
    """Bridge a real suite + sync result into :func:`classify_suite_verdict`.

    Extracts per-arm validity from the suite's runs and the recording/approval
    outcome from the sync result. Kept thin so the trust logic stays testable
    in :func:`classify_suite_verdict`.
    """
    from victor.evaluation.harness import was_throttled

    runs = list(getattr(suite, "runs", []) or [])
    arms: list[ArmValidity] = []
    has_baseline = False
    for run in runs:
        config = getattr(run, "config", None)
        result = getattr(run, "result", None)
        task_results = list(getattr(result, "task_results", []) or [])
        total = len(task_results)
        throttled = sum(1 for t in task_results if was_throttled(t))
        hash_ = getattr(config, "prompt_candidate_hash", "") or ""
        if hash_ == baseline_hash:
            has_baseline = True
        arms.append(
            ArmValidity(
                label=getattr(run, "label", hash_ or "arm"),
                provider=getattr(config, "provider", "") or "",
                total_tasks=total,
                throttled_tasks=throttled,
            )
        )

    decisions = list(getattr(sync_result, "decisions", []) or [])
    best = next((d for d in decisions if getattr(d, "rank", 1) == 1), None)
    best_recorded = bool(best and getattr(best, "recorded", False))
    approved = getattr(sync_result, "approved_prompt_candidate_hash", None) is not None

    return classify_suite_verdict(
        arms,
        has_baseline=has_baseline,
        best_recorded=best_recorded,
        approved=approved,
        min_tasks=min_tasks,
        max_throttled_share=max_throttled_share,
    )
