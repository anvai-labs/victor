# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
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

"""Canonical default reward derivation for RL outcomes (R2).

Single home for "reward from an outcome's success + quality". Before this module,
reward-from-outcome was derived in several places — including an ad-hoc inline
``quality_score or (1.0 if success else 0.3)`` in the tool-selection runtime — each
a separate, drifting source of truth.

Learners MAY still override ``_compute_reward`` for domain-specific *shaping*
(e.g. ``tool_selector`` weights success/completion/grounding/efficiency). That is
legitimate and untouched. What this module removes is the *unmanaged default*: any
site that just needs "reward from success + quality" should call these helpers
rather than reinventing the formula. See ``docs/architecture/observability-axes.md``
(the reward-derivation rule). Note: ``ImplicitFeedback.compute_reward`` is a
distinct, richer multi-signal reducer (task_completed/tool_success/grounding/…)
operating on accumulated session state, not a single outcome — it is intentionally
separate from this default.
"""

from __future__ import annotations

from typing import Any, Optional

#: Reward for a successful outcome that carries no explicit quality score.
DEFAULT_SUCCESS_REWARD = 1.0
#: Reward for a failed outcome that carries no explicit quality score.
DEFAULT_FAILURE_REWARD = 0.3


def reward_from_signals(*, success: bool, quality_score: Optional[float] = None) -> float:
    """Canonical default reward in ``[0.0, 1.0]`` from success + optional quality.

    Preserves the historical default: an explicit ``quality_score`` wins (clamped to
    ``[0, 1]``); otherwise success maps to ``1.0`` and failure to ``0.3``.

    Args:
        success: Whether the operation succeeded.
        quality_score: Optional explicit quality in ``[0, 1]`` (e.g. from grounding).

    Returns:
        Reward in ``[0.0, 1.0]``.
    """
    if quality_score is not None:
        try:
            q = float(quality_score)
        except (TypeError, ValueError):
            return DEFAULT_SUCCESS_REWARD if success else DEFAULT_FAILURE_REWARD
        return max(0.0, min(1.0, q))
    return DEFAULT_SUCCESS_REWARD if success else DEFAULT_FAILURE_REWARD


def reward_from_outcome(outcome: Any) -> float:
    """Canonical default reward from an ``RLOutcome`` (its success + quality_score)."""
    return reward_from_signals(
        success=bool(getattr(outcome, "success", False)),
        quality_score=getattr(outcome, "quality_score", None),
    )
