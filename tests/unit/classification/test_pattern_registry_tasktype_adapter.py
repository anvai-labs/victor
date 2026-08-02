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

"""tool-supply P7 follow-on — pattern_registry.TaskType collapses onto the registry.

The largest task-type adapter (``victor.classification.pattern_registry.TaskType``)
was deferred from the first P7 pass because it carries vertical-scoped/specialized
members (kubernetes, terraform, data_profiling, …). This battery pins its
``to_canonical()`` / ``from_canonical()`` adapter against the single
``TaskTypeRegistry`` authority: every member must resolve to a type registered as
either a core type or a vertical override — no orphans.
"""

from __future__ import annotations

import pytest

from victor.classification.pattern_registry import TaskType as PatternTaskType
from victor.framework.task_types import (
    canonicalize_task_type,
    get_task_type_registry,
    setup_vertical_task_types,
)


@pytest.fixture(autouse=True)
def _ensure_vertical_types_registered():
    """Vertical task types (kubernetes, data_profiling, …) must be registered so the
    no-orphan guard can recognize members that resolve to a vertical-scoped type.

    ``setup_vertical_task_types`` is idempotent (re-registration overwrites), so this
    is safe to call per test and does not reset the singleton for other tests.
    """
    setup_vertical_task_types()


def _is_registered(registry, canonical: str) -> bool:
    """A canonical name is known if it is a core type or any vertical override."""
    if registry.get(canonical) is not None:
        return True
    return any(
        registry.get(canonical, vertical=vertical) is not None
        for vertical in registry.list_verticals()
    )


def test_every_member_maps_to_a_registered_canonical():
    registry = get_task_type_registry()
    for member in PatternTaskType:
        canonical = member.to_canonical()
        assert canonical, f"{member.name} produced an empty canonical name"
        assert _is_registered(
            registry, canonical
        ), f"{member.name} -> {canonical!r} is not a registered type (orphan)"


def test_round_trip_is_canonical_stable():
    for member in PatternTaskType:
        canonical = member.to_canonical()
        back = PatternTaskType.from_canonical(canonical)
        assert back is not None
        # Multiple members may share a canonical (bug_fix/issue_resolution -> edit);
        # require only that the round trip is canonical-stable.
        assert back.to_canonical() == canonical


def test_orphans_remap_to_closest_core_type():
    # The genuine orphans (no core alias, not a registered type) route through the
    # adapter override map onto the closest core type.
    assert PatternTaskType.BUG_FIX.to_canonical() == "edit"
    assert PatternTaskType.ISSUE_RESOLUTION.to_canonical() == "edit"
    assert PatternTaskType.DATA_ANALYSIS.to_canonical() == "analyze"
    assert PatternTaskType.SECURITY.to_canonical() == "analyze"
    assert PatternTaskType.EXPLAIN.to_canonical() == "analyze"
    assert PatternTaskType.GENERAL_QUERY.to_canonical() == "general"


def test_existing_core_aliases_resolve_without_overrides():
    # implement/plan/architecture are already aliases in the core registry — they must
    # resolve through it, not via an adapter-local override.
    assert PatternTaskType.IMPLEMENT.to_canonical() == "create"
    assert PatternTaskType.PLAN.to_canonical() == "design"
    assert PatternTaskType.ARCHITECTURE.to_canonical() == "design"


def test_vertical_scoped_members_resolve_to_themselves():
    # Vertical-scoped members keep their own identity (registered under a vertical),
    # rather than being force-collapsed onto a core type.
    for member in (
        PatternTaskType.KUBERNETES,
        PatternTaskType.TERRAFORM,
        PatternTaskType.INFRASTRUCTURE,
        PatternTaskType.CODE_GENERATION,
        PatternTaskType.DATA_PROFILING,
        PatternTaskType.VISUALIZATION,
    ):
        assert member.to_canonical() == member.value


def test_core_members_are_identity():
    for member in (
        PatternTaskType.EDIT,
        PatternTaskType.SEARCH,
        PatternTaskType.CREATE,
        PatternTaskType.ANALYZE,
        PatternTaskType.DESIGN,
        PatternTaskType.GENERAL,
        PatternTaskType.ACTION,
        PatternTaskType.DEBUG,
        PatternTaskType.TEST,
    ):
        assert member.to_canonical() == member.value


def test_refactor_resolves_to_edit_via_registry_alias_precedence():
    # Pre-existing registry behavior: "refactor" is an alias of "edit" (in EDIT's
    # alias set) and ``resolve_alias`` consults aliases before core types, so it
    # collapses onto "edit" even though a standalone "refactor" core type exists. The
    # adapter faithfully reflects the single registry authority rather than overriding it.
    assert PatternTaskType.REFACTOR.to_canonical() == "edit"


def test_from_canonical_unknown_returns_none():
    assert PatternTaskType.from_canonical("definitely_not_a_task_type") is None


def test_canonicalize_helper_applies_pattern_overrides():
    from victor.classification.pattern_registry import _CANONICAL_OVERRIDES

    assert canonicalize_task_type("bug_fix", overrides=_CANONICAL_OVERRIDES) == "edit"
    assert canonicalize_task_type("kubernetes", overrides=_CANONICAL_OVERRIDES) == "kubernetes"
