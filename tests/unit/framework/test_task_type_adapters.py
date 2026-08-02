# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""tool-supply P7 — task-type adapter enums collapse onto the canonical registry.

Covers the two coarse runtime adapters (TrackerTaskType, ClassifierTaskType). The
larger ``pattern_registry.TaskType`` (with vertical-scoped types) is covered by the
P7 follow-on in
``tests/unit/classification/test_pattern_registry_tasktype_adapter.py``.
"""

from __future__ import annotations

import pytest

from victor.agent.unified_classifier import ClassifierTaskType
from victor.agent.unified_task_tracker import TrackerTaskType
from victor.framework.task_types import canonicalize_task_type, get_task_type_registry

ADAPTERS = [TrackerTaskType, ClassifierTaskType]


@pytest.mark.parametrize("adapter", ADAPTERS)
def test_every_member_maps_to_registered_canonical(adapter):
    registry = get_task_type_registry()
    for member in adapter:
        canonical = member.to_canonical()
        assert canonical, f"{adapter.__name__}.{member.name} produced empty canonical"
        assert (
            registry.get(canonical) is not None
        ), f"{adapter.__name__}.{member.name} -> {canonical!r} is not a registered type (orphan)"


@pytest.mark.parametrize("adapter", ADAPTERS)
def test_round_trip_is_idempotent(adapter):
    for member in adapter:
        canonical = member.to_canonical()
        back = adapter.from_canonical(canonical)
        assert back is not None
        # Multiple members may share a canonical; require canonical-stable round trip.
        assert back.to_canonical() == canonical


def test_tracker_values_are_already_canonical():
    # TrackerTaskType values are canonical names — to_canonical is identity here.
    for member in TrackerTaskType:
        assert member.to_canonical() == member.value


def test_classifier_coarse_remaps_through_registry_aliases():
    # The coarse classifier labels resolve via the registry's single alias authority.
    assert ClassifierTaskType.ANALYSIS.to_canonical() == "analyze"
    assert ClassifierTaskType.GENERATION.to_canonical() == "create_simple"
    assert ClassifierTaskType.DEFAULT.to_canonical() == "general"
    assert ClassifierTaskType.EDIT.to_canonical() == "edit"


def test_from_canonical_unknown_returns_none():
    assert TrackerTaskType.from_canonical("definitely_not_a_task_type") is None


def test_canonicalize_helper_applies_overrides_then_aliases():
    # Override pre-map wins; otherwise the registry alias map resolves.
    assert canonicalize_task_type("analysis") == "analyze"  # alias
    assert canonicalize_task_type("edit") == "edit"  # already canonical
    assert canonicalize_task_type("bug_fix", overrides={"bug_fix": "debug"}) == "debug"
