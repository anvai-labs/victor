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

"""RLM-2: the tool-outcome RL signal channel (victor/tools/rl_signal.py).

The executor seeds a fresh contextvar holder per execution; tools attach
structured outcome fields; the emit drains them. These tests pin the channel
contract: merge semantics, drain-clears-state, and the unseeded no-op.
"""

from __future__ import annotations

from victor.tools.rl_signal import (
    attach_tool_outcome_metadata,
    drain_tool_outcome_metadata,
    seed_tool_outcome_metadata,
)


def test_channel_seed_attach_drain_roundtrip():
    seed_tool_outcome_metadata()
    attach_tool_outcome_metadata(error_category="timeout", return_code=-1)
    attach_tool_outcome_metadata(duration_ms=12.5)  # second attach merges
    assert drain_tool_outcome_metadata() == {
        "error_category": "timeout",
        "return_code": -1,
        "duration_ms": 12.5,
    }


def test_channel_drain_clears_state():
    seed_tool_outcome_metadata()
    attach_tool_outcome_metadata(error_category="exit_code")
    drain_tool_outcome_metadata()
    # Negative: a stale payload must never leak into a later drain.
    assert drain_tool_outcome_metadata() == {}


def test_channel_attach_without_seed_is_noop():
    drain_tool_outcome_metadata()  # ensure clean state
    attach_tool_outcome_metadata(error_category="blocked")  # must not raise
    assert drain_tool_outcome_metadata() == {}
