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

"""FEP-0009 Phase 1 — the SDK Tool Contract (data-only, frozen, version-mirrored)."""

from __future__ import annotations

import dataclasses

import pytest

from victor_contracts.tools import (
    AccessMode,
    CONTRACT,
    CostTier,
    DangerLevel,
    ExecutionCategory,
    ToolCategory,
    ToolContract,
)

# Values MUST mirror the framework enums (victor/tools/enums.py and the ToolCategory
# vocabulary in victor/framework/tools.py) so the framework bridge maps by value.
EXPECTED_VALUES = {
    AccessMode: {"readonly", "write", "execute", "network", "mixed"},
    DangerLevel: {"safe", "low", "medium", "high", "critical"},
    ExecutionCategory: {"read_only", "write", "compute", "network", "execute", "mixed"},
    CostTier: {"free", "low", "medium", "high"},
    ToolCategory: {
        "core",
        "filesystem",
        "git",
        "search",
        "web",
        "database",
        "docker",
        "testing",
        "refactoring",
        "documentation",
        "analysis",
        "communication",
        "notebook",
        "task_management",
        "verification",
        "custom",
    },
}


@pytest.mark.parametrize("enum_cls,values", list(EXPECTED_VALUES.items()))
def test_enum_values_mirror_framework(enum_cls, values):
    assert {member.value for member in enum_cls} == values


@pytest.mark.parametrize("enum_cls", list(EXPECTED_VALUES.keys()))
def test_enums_are_str_valued(enum_cls):
    # str-valued so they serialize/compare as plain strings across the package boundary.
    for member in enum_cls:
        assert isinstance(member, str)
        assert member == member.value


def test_contract_is_frozen_and_hashable():
    c = ToolContract()
    with pytest.raises(dataclasses.FrozenInstanceError):
        c.category = ToolCategory.GIT  # type: ignore[misc]
    # Hashable (frozen + tuple fields) — usable as a dict key / set member.
    assert hash(c) == hash(ToolContract())


def test_defaults_are_conservative():
    c = ToolContract()
    assert c.category == ToolCategory.CUSTOM
    assert c.access_mode == AccessMode.READONLY
    assert c.danger_level == DangerLevel.SAFE
    assert c.execution_category == ExecutionCategory.READ_ONLY
    assert c.cost_tier == CostTier.LOW
    assert c.keywords == () and c.use_cases == () and c.task_types == () and c.stages == ()


def test_no_priority_or_engine_fields():
    # FEP-0009 Q2: declarable-intent only; ranking/loop-detection knobs stay internal.
    fields = {f.name for f in dataclasses.fields(ToolContract)}
    for forbidden in (
        "priority",
        "priority_hints",
        "mandatory_keywords",
        "signature_params",
    ):
        assert forbidden not in fields


def test_category_value_accepts_enum_or_str():
    assert ToolContract(category=ToolCategory.GIT).category_value() == "git"
    assert ToolContract(category="my_custom").category_value() == "my_custom"


def test_capability_contract_gate():
    assert CONTRACT.name == "tools"
    assert CONTRACT.version == 1
    assert CONTRACT.min_sdk_version == ">=0.7.1"


def test_exports_available_from_package_root():
    import victor_contracts as vc

    assert vc.ToolContract is ToolContract
    assert vc.ToolCategory is ToolCategory
    assert vc.TOOL_CONTRACT is CONTRACT
