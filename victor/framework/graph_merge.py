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

"""Parallel branch state-merging helpers for StateGraph."""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Literal, Optional, Set

logger = logging.getLogger(__name__)

StateMerger = Callable[[Dict[str, Any], List[Dict[str, Any]]], Dict[str, Any]]
StateMergeStrategy = Literal["custom", "last_write_wins", "strict"]


def default_state_merger(
    base_state: Dict[str, Any],
    branch_states: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Merge parallel branch results by sequential dict.update."""
    merged = dict(base_state)
    conflicting_keys: Set[str] = set()
    for branch_state in branch_states:
        for key, value in branch_state.items():
            if key not in merged:
                continue
            try:
                if merged[key] != value:
                    conflicting_keys.add(str(key))
            except Exception:
                conflicting_keys.add(str(key))
        merged.update(branch_state)
    if conflicting_keys:
        logger.warning(
            "Parallel state merge conflict on keys %s; applying last-write-wins semantics",
            sorted(conflicting_keys),
        )
    return merged


def strict_state_merger(
    base_state: Dict[str, Any],
    branch_states: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Merge branch states but fail fast on conflicting parallel writes."""
    merged = dict(base_state)
    conflicting_keys: Set[str] = set()

    for branch_state in branch_states:
        for key, value in branch_state.items():
            if key not in merged:
                continue
            try:
                if merged[key] != value:
                    conflicting_keys.add(str(key))
            except Exception:
                conflicting_keys.add(str(key))

        if conflicting_keys:
            break

        merged.update(branch_state)

    if conflicting_keys:
        raise ValueError(
            "Parallel state merge conflict on keys "
            f"{sorted(conflicting_keys)}; strict_state_merger requires an explicit resolver"
        )

    return merged


def resolve_state_merger(
    strategy: StateMergeStrategy,
    custom_merger: Optional[StateMerger] = None,
) -> StateMerger:
    """Resolve the configured branch merge policy."""
    if strategy == "strict":
        return strict_state_merger
    if strategy == "last_write_wins":
        return default_state_merger
    if strategy == "custom":
        if custom_merger is None:
            raise ValueError(
                "PerformanceConfig.parallel_state_merge_strategy='custom' requires "
                "PerformanceConfig.custom_state_merger"
            )
        return custom_merger
    raise ValueError(f"Unsupported parallel state merge strategy: {strategy}")
