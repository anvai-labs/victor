# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""R2 — canonical default reward derivation + parity with the legacy inline path."""

from __future__ import annotations

import pytest

from victor.framework.rl.reward import (
    DEFAULT_FAILURE_REWARD,
    DEFAULT_SUCCESS_REWARD,
    reward_from_outcome,
    reward_from_signals,
)


def _legacy_inline_reward(success: bool, quality_score):
    """The exact formula previously inlined in tool_selection_runtime (pre-R2)."""
    return float(quality_score) if quality_score is not None else (1.0 if success else 0.3)


@pytest.mark.parametrize(
    "success,quality",
    [
        (True, 0.0),
        (True, 0.5),
        (True, 1.0),
        (False, 0.2),
        (False, 0.9),
        (True, None),
        (False, None),
    ],
)
def test_parity_with_legacy_inline(success, quality):
    # In-range quality + the no-quality defaults must match the old inline path exactly.
    assert reward_from_signals(success=success, quality_score=quality) == _legacy_inline_reward(
        success, quality
    )


def test_no_quality_defaults():
    assert reward_from_signals(success=True) == DEFAULT_SUCCESS_REWARD == 1.0
    assert reward_from_signals(success=False) == DEFAULT_FAILURE_REWARD == 0.3


def test_explicit_quality_wins():
    assert reward_from_signals(success=False, quality_score=0.7) == 0.7
    assert reward_from_signals(success=True, quality_score=0.1) == 0.1


def test_quality_clamped_to_unit_interval():
    assert reward_from_signals(success=True, quality_score=1.5) == 1.0
    assert reward_from_signals(success=True, quality_score=-0.2) == 0.0


def test_bad_quality_falls_back_to_success_default():
    # A non-numeric quality must not raise; fall back to the success/failure default.
    assert reward_from_signals(success=True, quality_score=object()) == 1.0  # type: ignore[arg-type]
    assert reward_from_signals(success=False, quality_score=object()) == 0.3  # type: ignore[arg-type]


def test_reward_from_outcome_reads_fields():
    class _O:
        success = True
        quality_score = 0.42

    assert reward_from_outcome(_O()) == 0.42

    class _Fail:
        success = False
        quality_score = None

    assert reward_from_outcome(_Fail()) == 0.3


def test_base_default_compute_reward_is_concrete_and_canonical():
    # De-abstracted in R2: BaseLearner._compute_reward now has a concrete default
    # delegating to the canonical reward. (_compute_reward ignores self, so we can
    # invoke it unbound without constructing a learner.)
    from victor.framework.rl.base import BaseLearner

    assert "_compute_reward" not in BaseLearner.__abstractmethods__

    class _O:
        success = True
        quality_score = 0.8

    o = _O()
    assert BaseLearner._compute_reward(None, o) == reward_from_outcome(o) == 0.8
