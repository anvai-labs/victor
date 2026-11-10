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

"""Shared runtime-context override merge helpers.

One precedence rule for every merge of derived (topology / planning) runtime
overrides onto caller-supplied overrides — the buffered AgenticLoop state
merge and the streaming topology merge share it (seam-duplication audit
Cluster 4, streaming run-context parity).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Mapping, Optional

logger = logging.getLogger(__name__)

RUNTIME_CONTEXT_OVERRIDES_STATE_KEY = "runtime_context_overrides"
LEGACY_TOPOLOGY_OVERRIDES_STATE_KEY = "topology_overrides"

# Budget keys a caller (or the first derived plan) declared in
# runtime_context_overrides bind for the whole run: later derived plans may
# add keys but never replace these (see merge_runtime_context_overrides).
CALLER_BOUND_OVERRIDE_KEYS = ("tool_budget", "iteration_budget")


def collect_runtime_context_overrides(state: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """Return merged generic runtime overrides with legacy topology fallback."""
    merged: Dict[str, Any] = {}
    legacy_overrides = state.get(LEGACY_TOPOLOGY_OVERRIDES_STATE_KEY)
    if isinstance(legacy_overrides, dict):
        merged.update(legacy_overrides)
    runtime_overrides = state.get(RUNTIME_CONTEXT_OVERRIDES_STATE_KEY)
    if isinstance(runtime_overrides, dict):
        merged.update(runtime_overrides)
    return merged or None


def merge_runtime_context_overrides(
    existing: Mapping[str, Any],
    overrides: Mapping[str, Any],
) -> Dict[str, Any]:
    """Merge derived overrides ONTO existing overrides; caller-bound keys win.

    Budget keys already present in ``existing`` bind for the whole run: derived
    plans re-emit the budgets they were handed (topology planning grounds its
    decision from the loop-build service snapshot), and letting those echoes
    win silently lowered a caller's funded ``runtime_context_overrides`` budget
    — a 134-call override collapsed to the 50-call service default and the run
    starved mid-flight while still funded. Derived keys that are absent
    (formation hints, team names, worktree paths) still merge normally.
    """
    merged = dict(existing)
    for key, value in dict(overrides).items():
        if key in CALLER_BOUND_OVERRIDE_KEYS and key in merged:
            logger.debug(
                "Runtime override merge keeps existing %s=%s over derived %s",
                key,
                merged[key],
                value,
            )
            continue
        merged[key] = value
    return merged
