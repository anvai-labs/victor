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

"""Guard: the ToolCategory enum and tool_categories.yaml are consistent views.

The ``ToolCategory`` enum is the single identity authority for category *names*.
``tool_categories.yaml`` is a derived view that supplies membership / descriptions
/ presets but may only name categories that exist in the enum. These tests pin the
two vocabularies equal so the historical drift (the YAML declared ``notebook`` /
``task_management`` / ``verification`` while the enum did not, so they were silently
dropped from the built-in fallback) cannot reappear.
"""

from __future__ import annotations

from victor.config.tool_categories import (
    load_presets,
    load_tool_categories,
)
from victor.framework.tools import (
    ToolCategory,
    _get_builtin_category_tools,
)


def _enum_vocabulary() -> set[str]:
    """Canonical category names. ``REFACTOR`` is an alias of ``REFACTORING`` and
    is collapsed by the set comprehension (same value)."""
    return {category.value for category in ToolCategory}


def test_enum_and_yaml_vocabularies_match() -> None:
    """The enum vocabulary and the YAML category keys must be identical.

    This is the core anti-drift guard: neither source may introduce a category
    name the other does not know about.
    """
    enum_names = _enum_vocabulary()
    yaml_names = set(load_tool_categories().keys())

    only_in_enum = enum_names - yaml_names
    only_in_yaml = yaml_names - enum_names

    assert not only_in_enum, f"Categories in enum but missing from YAML: {sorted(only_in_enum)}"
    assert not only_in_yaml, f"Categories in YAML but missing from enum: {sorted(only_in_yaml)}"


def test_yaml_presets_reference_known_categories() -> None:
    """Every category named by a YAML preset must exist in the enum vocabulary.

    Presets are a derived grouping; they cannot reference a category identity that
    does not exist.
    """
    enum_names = _enum_vocabulary()
    presets = load_presets()

    for preset_name, preset in presets.items():
        for category_name in preset.get("categories", []):
            assert (
                category_name in enum_names
            ), f"Preset '{preset_name}' references unknown category '{category_name}'"


def test_reconciled_categories_are_not_silently_dropped() -> None:
    """The previously-dropped categories now flow into the built-in fallback.

    Before the collapse, ``_load_builtin_category_tools`` hit ``ValueError`` on these
    three YAML categories and discarded them. They are now enum members, so their
    declared YAML membership is live on the built-in fallback path.
    """
    builtin = _get_builtin_category_tools()
    yaml_categories = load_tool_categories()

    for category in (
        ToolCategory.NOTEBOOK,
        ToolCategory.TASK_MANAGEMENT,
        ToolCategory.VERIFICATION,
    ):
        assert category in builtin, f"{category.value} missing from built-in category tools"
        # The built-in fallback must carry exactly what the YAML declares.
        assert builtin[category] == yaml_categories[category.value]
