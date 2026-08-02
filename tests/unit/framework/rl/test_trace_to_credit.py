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

"""Tests for segment-level RL-from-traces (Vision P6)."""

from types import SimpleNamespace

from victor.framework.rl.trace_to_credit import (
    blend_quality_with_segments,
    compute_segment_rewards_from_trace,
)


def _trace(tool_calls):
    return SimpleNamespace(tool_calls=tool_calls)


# --- segment reward derivation -------------------------------------------------------------------


def test_no_tool_calls_returns_empty():
    assert compute_segment_rewards_from_trace(_trace([])) == {}
    assert compute_segment_rewards_from_trace(SimpleNamespace(tool_calls=None)) == {}


def test_dict_style_tool_calls_produce_per_action_credit():
    trace = _trace(
        [
            {"name": "read", "success": True, "timestamp": 1.0},
            {"name": "edit", "success": True, "timestamp": 2.0},
            {"name": "test", "success": False, "timestamp": 3.0},
        ]
    )
    rewards = compute_segment_rewards_from_trace(trace)
    assert len(rewards) == 3 and all(isinstance(v, float) for v in rewards.values())


def test_object_style_tool_calls_supported():
    trace = _trace([SimpleNamespace(name="read", success=True, timestamp=1.0)])
    rewards = compute_segment_rewards_from_trace(trace)
    assert len(rewards) == 1


# --- blend (backward-compatible) -----------------------------------------------------------------


def test_blend_none_or_empty_reproduces_baseline():
    assert blend_quality_with_segments(0.7, None) == 0.7
    assert blend_quality_with_segments(0.7, {}) == 0.7


def test_blend_moves_toward_segment_mean_and_clamps():
    # all-positive segments (credit≈+1 → seg01≈1.0) pull quality up
    up = blend_quality_with_segments(0.5, {0: 1.0, 1: 1.0}, weight=0.3)
    assert 0.5 < up <= 1.0
    # all-negative segments (credit≈-1 → seg01≈0.0) pull quality down
    down = blend_quality_with_segments(0.5, {0: -1.0, 1: -1.0}, weight=0.3)
    assert 0.0 <= down < 0.5
    # result always clamped to [0, 1]
    assert 0.0 <= blend_quality_with_segments(1.0, {0: 1.0}, weight=1.0) <= 1.0
