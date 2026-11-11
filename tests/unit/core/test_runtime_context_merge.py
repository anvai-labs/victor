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

"""Shared runtime-context override merge (buffered state + streaming topology).

One precedence rule for both paths: derived plans may add keys but never
replace caller-bound budget keys (seam-duplication audit Cluster 4)."""

from victor.framework.agentic_loop import AgenticLoop
from victor.core.runtime_context import (
    CALLER_BOUND_OVERRIDE_KEYS,
    collect_runtime_context_overrides,
    merge_runtime_context_overrides,
)


def test_caller_bound_keys_are_budget_keys():
    assert set(CALLER_BOUND_OVERRIDE_KEYS) == {"tool_budget", "iteration_budget"}


def test_merge_keeps_existing_bound_keys_over_derived():
    existing = {"tool_budget": 134, "iteration_budget": 64, "caller_marker": "keep"}
    merged = merge_runtime_context_overrides(existing, {"tool_budget": 4, "iteration_budget": 2})

    assert merged["tool_budget"] == 134
    assert merged["iteration_budget"] == 64
    assert merged["caller_marker"] == "keep"


def test_merge_adds_derived_keys_absent_from_caller():
    merged = merge_runtime_context_overrides(
        {"tool_budget": 134}, {"formation_hint": "parallel", "team_name": "feature_team"}
    )

    assert merged == {"tool_budget": 134, "formation_hint": "parallel", "team_name": "feature_team"}


def test_merge_does_not_mutate_inputs():
    existing = {"tool_budget": 134}
    derived = {"formation_hint": "parallel"}

    merged = merge_runtime_context_overrides(existing, derived)

    assert existing == {"tool_budget": 134}
    assert derived == {"formation_hint": "parallel"}
    assert merged["tool_budget"] == 134
    assert merged["formation_hint"] == "parallel"


def test_collect_merges_legacy_and_generic_state_keys():
    state = {
        "topology_overrides": {"tool_budget": 4},
        "runtime_context_overrides": {"tool_budget": 134},
    }

    assert collect_runtime_context_overrides(state) == {"tool_budget": 134}


def test_collect_returns_none_without_override_state():
    assert collect_runtime_context_overrides({}) is None


def test_agentic_loop_class_helpers_delegate_to_shared_merge():
    state = {
        "topology_overrides": {"formation_hint": "parallel"},
        "runtime_context_overrides": {"tool_budget": 134},
    }

    assert AgenticLoop._get_runtime_context_overrides(state) == {
        "formation_hint": "parallel",
        "tool_budget": 134,
    }
    merged = AgenticLoop._merge_runtime_context_overrides(
        state, {"tool_budget": 4, "iteration_budget": 2}
    )
    assert merged == {
        "formation_hint": "parallel",
        "tool_budget": 134,
        "iteration_budget": 2,
    }
