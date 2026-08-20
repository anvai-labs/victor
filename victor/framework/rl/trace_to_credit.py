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

"""Derive segment-level process reward from a trace and feed it to the learners (Vision P6).

The credit-assignment framework (``victor/framework/rl/credit_assignment.py``, 9 methods) was fully
built but **unused** — the learners consumed only outcome-level reward. This module closes that loop:
it turns an ``AgenticExecutionTrace`` into per-action process rewards, densifies them via the existing
``CreditAssignmentIntegration``, and blends the mean process reward into the outcome's quality score
so every learner benefits without touching any individual learner (the blend happens once, at
outcome construction). Fully backward-compatible: no trace / no tool activity ⇒ quality unchanged.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Weight of the segment process reward when blended with outcome quality (0 = ignore, 1 = replace).
DEFAULT_SEGMENT_WEIGHT = 0.3


def _field(tc: Any, key: str, default: Any) -> Any:
    """Read a tool-call field from either a dataclass-like object or a dict."""
    if isinstance(tc, dict):
        return tc.get(key, default)
    return getattr(tc, key, default)


def compute_segment_rewards_from_trace(
    trace: Any, *, integration: Optional[Any] = None
) -> Dict[int, float]:
    """Per-action densified process reward ``{index: credit}`` from a trace (empty if no tool activity).

    Base reward is +1 for a successful tool call, -1 for a failed one; GAE (per-step advantage)
    densifies that over the trajectory. This is the first production *use* of the credit-assignment
    framework (Vision P6).
    """
    tool_calls = list(getattr(trace, "tool_calls", None) or [])
    if not tool_calls:
        return {}

    try:
        from victor.framework.rl.credit_assignment import (
            ActionMetadata,
            CreditAssignmentIntegration,
            CreditMethodology,
        )

        trajectory = [
            ActionMetadata(
                agent_id="agent",
                step_index=i,
                tool_name=_field(tc, "name", None) or _field(tc, "tool_name", None),
                timestamp=float(_field(tc, "timestamp", 0.0) or 0.0),
            )
            for i, tc in enumerate(tool_calls)
        ]
        rewards = [1.0 if _field(tc, "success", True) else -1.0 for tc in tool_calls]

        ca = integration or CreditAssignmentIntegration()
        # GAE: per-step advantage densification over the action trajectory (MONTE_CARLO expects
        # pre-segmented TrajectorySegment inputs, not per-action metadata).
        signals = ca.assign_credit(trajectory, rewards, methodology=CreditMethodology.GAE)
        return {i: float(getattr(s, "credit", 0.0)) for i, s in enumerate(signals)}
    except Exception as exc:  # never break outcome recording on a credit-assignment hiccup
        logger.debug("segment-reward derivation skipped (%s)", exc)
        return {}


def blend_quality_with_segments(
    quality_score: float,
    segment_rewards: Optional[Dict[int, float]],
    *,
    weight: float = DEFAULT_SEGMENT_WEIGHT,
) -> float:
    """Blend outcome quality with the mean segment process reward, clamped to [0, 1].

    Segment credit lives in roughly [-1, 1]; it is mapped to [0, 1] before blending. With no segment
    rewards the quality score is returned unchanged (baseline behavior).
    """
    if not segment_rewards:
        return quality_score
    mean_credit = sum(segment_rewards.values()) / len(segment_rewards)
    seg01 = max(0.0, min(1.0, (mean_credit + 1.0) / 2.0))
    blended = (1.0 - weight) * quality_score + weight * seg01
    return max(0.0, min(1.0, blended))
